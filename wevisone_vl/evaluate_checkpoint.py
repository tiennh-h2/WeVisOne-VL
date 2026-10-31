"""Run from the repository root: python evaluate_checkpoint.py --help."""
import argparse
import csv
import json
import random
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description='Full validation and repetition-penalty comparison; no training.')
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--config', help='Optional YAML supplying validation paths and generation settings only')
    p.add_argument('--output', required=True)
    p.add_argument('--penalties', nargs='+', type=float, default=[1.05, 1.0])
    args = p.parse_args()
    import torch
    from accelerate import Accelerator
    from accelerate.utils import gather_object, send_to_device
    from wevisone_vl.training.runner import load_config, load_components
    from wevisone_vl.training.data import CocoTaskDataset
    from wevisone_vl.training.validation import predict_generated
    from wevisone_vl.training.quality import detection_scores

    checkpoint = Path(args.checkpoint).resolve()
    cfg = load_config(checkpoint / 'training_config.yaml')
    if args.config:
        supplied = load_config(Path(args.config))
        cfg['validation'] = supplied['validation']
        cfg['generation_validation'] = supplied['generation_validation']
        cfg["validation_samples_per_source"] = supplied["validation_samples_per_source"]
    datasets = [CocoTaskDataset(s, cfg['seed']) for s in cfg['validation']]
    if not datasets or any(ds.task != 'detection' for ds in datasets):
        raise ValueError('This evaluator requires nonempty detection-only validation sources')
    accelerator = Accelerator(mixed_precision=cfg['optimization']['mixed_precision'])
    out = Path(args.output)
    if accelerator.is_main_process:
        out.mkdir(parents=True, exist_ok=False)
    accelerator.wait_for_everyone()
    model, processor, _ = load_components(cfg)
    weights = checkpoint / 'pytorch_model.bin'
    if not weights.is_file():
        raise FileNotFoundError(f'Expected trainer checkpoint weights: {weights}')
    model.load_state_dict(torch.load(weights, map_location='cpu', weights_only=True), strict=True)
    model.to(accelerator.device).eval()
    summary = []
    for penalty in args.penalties:
        options = dict(cfg['generation_validation'], repetition_penalty=penalty)
        local = []
        for ds in datasets:

            indices = list(range(len(ds)))
            random.Random(cfg["seed"]).shuffle(indices)

            limit = cfg.get("validation_samples_per_source", 32)
            indices = sorted(indices[:limit] if limit else indices)

            for index in indices[accelerator.process_index::accelerator.num_processes]:
                sample = ds[index]
                boxes, scores, _, truncated, malformed = predict_generated(model, processor, sample, accelerator, options)
                local.append(dict(source=ds.name, index=index, query=sample['query'],
                    image_id=sample['image_id'], file_name=sample['file_name'],
                    boxes=sample['boxes'], pred_boxes=boxes, scores=scores,
                    truncated=truncated, malformed=malformed))
                if (index // accelerator.num_processes + 1) % 25 == 0:
                    print(f'rank={accelerator.process_index} penalty={penalty} source={ds.name} index={index}/{len(ds)}', flush=True)
        rows = sorted(gather_object(local), key=lambda r: (r['source'], r['index']))
        if accelerator.is_main_process:
            tag = str(penalty).replace('.', '_')
            with (out / f'predictions_penalty_{tag}.jsonl').open('w') as f:
                for row in rows:
                    f.write(json.dumps(row) + '\n')
            groups = [('overall', [{**r, 'query': (r['source'], r['query'])} for r in rows])]
            for source in sorted({r['source'] for r in rows}):
                subset = [r for r in rows if r['source'] == source]
                groups.append((source, subset))
                for query in sorted({r['query'] for r in subset}):
                    groups.append((f'{source}/{query}', [r for r in subset if r['query'] == query]))
            for name, subset in groups:
                result = dict(penalty=penalty, group=name, query_count=len(subset),
                    gt_count=sum(len(r['boxes']) for r in subset),
                    prediction_count=sum(len(r['pred_boxes']) for r in subset),
                    truncated_queries=sum(r['truncated'] for r in subset),
                    **detection_scores(subset, options['max_detections']))
                summary.append(result)
                print(json.dumps(result), flush=True)
            (out / 'summary.json').write_text(json.dumps(summary, indent=2))
            with (out / 'summary.csv').open('w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(summary[0]))
                writer.writeheader()
                writer.writerows(summary)
            (out / 'evaluation_settings.json').write_text(json.dumps(dict(checkpoint=str(checkpoint),
                max_pixels=cfg['max_pixels'], generation=options, validation=cfg['validation'],
                penalties=args.penalties, world_size=accelerator.num_processes), indent=2))
        accelerator.wait_for_everyone()


if __name__ == '__main__':
    main()
