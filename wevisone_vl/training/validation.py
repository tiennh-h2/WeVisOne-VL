"""Free-running validation against held-out query targets, using live weights."""
from __future__ import annotations

import random

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from ..feature_bridge import (LastHiddenCapture, VisualFeatureCapture,
                              per_sample_premerge_feature_maps, premerge_grid_hw)
from .quality import detection_scores, foreground_scores, selection_score, parse_predictions


class GreedyConfidence:
    """Record one scalar per step, rather than retaining vocabulary-sized scores.

    Called after generation logits processors (e.g. repetition penalty); greedy
    decoding selects argmax. This is a token likelihood proxy, not calibrated
    detector objectness. Object scores use the four coordinate token likelihoods.
    """
    def __init__(self):
        self.values = []

    def __call__(self, input_ids, scores):
        self.values.append((scores.float().max(-1).values - scores.float().logsumexp(-1)).detach())
        return scores

@torch.no_grad()
def predict_generated(model, processor, sample, accelerator, options):
    """Generate without target tokens; replay only generated tokens for masks."""
    from accelerate.utils import send_to_device

    image = sample['image']
    messages = [
        {'role': 'system', 'content': 'You are a helpful assistant.'},
        {'role': 'user', 'content': [{'type': 'image', 'image': image},
                                     {'type': 'text', 'text': sample['prompt']}]},
    ]
    prefix = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = send_to_device(dict(processor(text=[prefix], images=[image],
                                          return_tensors='pt', padding=False)), accelerator.device)
    recorder = GreedyConfidence()
    with accelerator.autocast():
        sequences = model.vlm.generate(
            **inputs, max_new_tokens=options['max_new_tokens'], do_sample=False,
            num_beams=1, num_return_sequences=1, use_cache=True, repetition_penalty=options['repetition_penalty'],
            logits_processor=[recorder], return_dict_in_generate=False)
    new_ids = sequences[0, inputs['input_ids'].shape[1]:].tolist()
    response = processor.tokenizer.decode(new_ids, skip_special_tokens=False,
                                         clean_up_tokenization_spaces=False)
    log_probs = torch.cat(recorder.values).float().cpu().tolist() if recorder.values else []
    boxes, scores = parse_predictions(response, new_ids, log_probs, processor.tokenizer,
                                     sample['task'] == 'segmentation')
    masks = []
    if sample['task'] == 'segmentation' and boxes:
        # Reject malformed OBJ alignment rather than decoding the wrong object.
        text = processor.apply_chat_template(messages + [{'role': 'assistant', 'content': response}],
                                             tokenize=False, add_generation_prompt=False)
        teacher = send_to_device(dict(processor(text=[text], images=[image], return_tensors='pt',
                                                padding=False)), accelerator.device)
        positions = (teacher['input_ids'][0] == model.seg_token_id).nonzero().flatten()
        if len(positions) != len(boxes):
            # Invalid boxes/OBJ spans cannot be safely aligned. Count as no masks.
            return boxes, scores, [], len(new_ids) >= options['max_new_tokens'], True
        with accelerator.autocast():
            with VisualFeatureCapture(model.backbone, model.cfg.vision_feature_layers) as visual:
                with LastHiddenCapture(model.backbone) as hidden:
                    model.vlm(**teacher, use_cache=False, return_dict=True, output_hidden_states=False)
            if hidden.hidden is None:
                raise RuntimeError('generated response hidden states were not captured')
            grid = teacher['image_grid_thw']
            features = per_sample_premerge_feature_maps(visual.premerge_features,
                        model.cfg.vision_feature_layers, grid, model.spatial_merge_size)[0]
            size = model.cfg.cnn_image_size
            rgb = np.asarray(image.resize((size, size), Image.Resampling.BICUBIC), dtype=np.float32)/255
            cnn_image = torch.from_numpy(rgb.copy()).permute(2, 0, 1)[None].to(accelerator.device)
            box_tensor = torch.tensor(boxes, dtype=torch.float32, device=accelerator.device)
            for start in range(0, len(boxes), options['decoder_batch_size']):
                end = start + options['decoder_batch_size']
                prediction = model.mask_decoder(
                    image=cnn_image, visual_features_2d=features,
                    t_seg=hidden.hidden[0, positions[start:end]], boxes01=box_tensor[start:end],
                    grid_hw=premerge_grid_hw(grid)[0])
                logits = prediction['mask_logits']
                if logits.ndim == 4 and logits.shape[1] == 1:
                    logits = logits[:, 0]
                if logits.ndim != 3:
                    raise ValueError(f'unexpected mask shape: {tuple(logits.shape)}')
                probabilities = F.interpolate(logits.float().sigmoid()[None],
                                    (image.height, image.width), mode='bilinear', align_corners=False)[0]
                # Union at original resolution; transfer binary masks only.
                union = (probabilities >= options['mask_threshold']).any(dim=0).cpu().numpy()
                if masks:
                    masks[0] |= union
                else:
                    masks.append(union)
    return boxes, scores, masks, len(new_ids) >= options['max_new_tokens'], False


def summarize(rows, max_detections):
    result = {'query_count': len(rows),
              'truncated_queries': sum(r['truncated'] for r in rows),
              'malformed_mask_queries': sum(r['malformed'] for r in rows)}
    seg = [r for r in rows if r['task'] == 'segmentation']
    det = [r for r in rows if r['task'] == 'detection']
    if seg:
        result.update(miou=float(np.mean([r['iou'] for r in seg])),
                      mdice=float(np.mean([r['dice'] for r in seg])))
    if det:
        result.update(detection_scores(det, max_detections))
    return result


@torch.no_grad()
def evaluate_generation(model, datasets, processor, accelerator, cfg):
    from accelerate.utils import gather_object

    options = cfg['generation_validation']
    original_training = model.training
    model.eval()
    unwrapped = accelerator.unwrap_model(model)
    local = []
    try:
        for dataset in datasets:
            indices = list(range(len(dataset)))
            random.Random(cfg['seed']).shuffle(indices)
            limit = cfg['validation_samples_per_source']
            indices = sorted(indices[:limit] if limit else indices)
            for index in indices[accelerator.process_index::accelerator.num_processes]:
                sample = dataset[index]
                boxes, scores, masks, truncated, malformed = predict_generated(
                    unwrapped, processor, sample, accelerator, options)
                row = {'source': dataset.name, 'index': index, 'task': sample['task'],
                       'query': sample['query'], 'boxes': sample['boxes'],
                       'pred_boxes': boxes, 'scores': scores,
                       'truncated': truncated, 'malformed': malformed}
                if sample['task'] == 'segmentation':
                    row['iou'], row['dice'] = foreground_scores(masks, sample['masks'],
                                                (sample['image'].height, sample['image'].width))
                local.append(row)
        # Gather compact per-query records, never full-resolution masks.
        rows = sorted(gather_object(local), key=lambda r: (r['source'], r['index']))
        sources = {ds.name: summarize([r for r in rows if r['source'] == ds.name],
                                      options['max_detections']) for ds in datasets}
        # Namespace source/query pairs for overall category averaging.
        overall = summarize([{**r, 'query': (r['source'], r['query'])} for r in rows],
                            options['max_detections'])
        overall['score'] = selection_score(overall, cfg['checkpoint_metric'])
        return {'sources': sources, 'overall': overall}
    finally:
        model.train(original_training)
