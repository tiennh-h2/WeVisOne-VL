"""CPU metrics and checkpoint policy for generated predictions (no torch import)."""
from __future__ import annotations

import contextlib
import io
import math
import re

import numpy as np


def foreground_scores(predictions, targets, shape, threshold=.5):
    """Binary foreground union IoU/Dice per query; empty/empty is one.

    This is semantic overlap, not instance mask AP. Every generated mask is used,
    so missed/extra foreground affects the result, independent of object order.
    """
    def union(masks):
        result = np.zeros(shape, dtype=bool)
        for mask in masks:
            mask = np.asarray(mask)
            if mask.shape != tuple(shape) or not np.isfinite(mask).all():
                raise ValueError('invalid mask shape or nonfinite mask')
            result |= mask >= threshold
        return result
    pred, gt = union(predictions), union(targets)
    intersection = int((pred & gt).sum())
    area = int(pred.sum()) + int(gt.sum())
    return (intersection / (area - intersection) if area > intersection else 1.,
            2 * intersection / area if area else 1.)


def detection_scores(rows, max_detections=100):
    """COCO bbox AP@[.50:.95], AP50/AP75 and AR@max_detections.

    Each query gets a synthetic image ID, preventing phrase/category records on
    the same original image from interfering. Query strings are category IDs.
    Boxes are normalized xyxy and evaluated in a common 1000x1000 space.
    """
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    categories = {name: i + 1 for i, name in enumerate(sorted({r['query'] for r in rows}))}
    images, annotations, predictions = [], [], []
    def xywh(box):
        x1, y1, x2, y2 = box
        return [1000*x1, 1000*y1, 1000*(x2-x1), 1000*(y2-y1)]
    for image_id, row in enumerate(rows, 1):
        images.append({'id': image_id, 'width': 1000, 'height': 1000})
        category = categories[row['query']]
        for box in row['boxes']:
            bbox = xywh(box)
            annotations.append({'id': len(annotations)+1, 'image_id': image_id,
                                'category_id': category, 'bbox': bbox,
                                'area': bbox[2]*bbox[3], 'iscrowd': 0})
        if len(row['pred_boxes']) != len(row['scores']):
            raise ValueError('prediction/score count mismatch')
        for box, score in zip(row['pred_boxes'], row['scores']):
            predictions.append({'image_id': image_id, 'category_id': category,
                                'bbox': xywh(box), 'score': float(score)})
    if not annotations:
        # AP and AR have no definition when the evaluated subset has no GT.
        return {'map': None, 'map50': None, 'map75': None, 'mar': None}
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO()
        gt.dataset = {'images': images, 'annotations': annotations,
                      'categories': [{'id': v, 'name': k} for k, v in categories.items()], 'info': {}}
        gt.createIndex()
        if predictions:
            dt = gt.loadRes(predictions)
        else:
            dt = COCO()
            dt.dataset = {**gt.dataset, 'annotations': []}
            dt.createIndex()
        evaluator = COCOeval(gt, dt, 'bbox')
        evaluator.params.maxDets = [1, min(10, max_detections), max_detections]
        evaluator.evaluate()
        evaluator.accumulate()
    # Read tensors directly; COCO summarize() hardcodes maxDets=100 for AP.
    precision = evaluator.eval['precision'][:, :, :, 0, -1]
    recall = evaluator.eval['recall'][:, :, 0, -1]
    def mean_valid(value):
        value = value[value >= 0]
        return float(value.mean()) if value.size else None
    return {'map': mean_valid(precision), 'map50': mean_valid(precision[0]),
            'map75': mean_valid(precision[5]), 'mar': mean_valid(recall)}


def selection_score(metrics, metric='auto'):
    if metric != 'auto':
        value = metrics.get(metric)
        if value is None or not math.isfinite(value):
            raise ValueError(f'checkpoint metric {metric!r} unavailable; check validation tasks/targets')
        return value
    groups = [('miou', 'mdice'), ('map', 'mar')]
    scores = [sum(metrics[k] for k in group)/len(group) for group in groups
              if all(metrics.get(k) is not None for k in group)]
    if not scores:
        raise ValueError('no generated quality metrics available for checkpoint selection')
    return sum(scores)/len(scores)


def update_bests(best, metrics, checkpoint):
    result = {key: dict(value) for key, value in best.items()}
    for key in ('miou', 'mdice', 'map', 'mar', 'score'):
        value = metrics.get(key)
        if value is not None:
            if not math.isfinite(value):
                raise FloatingPointError(f'nonfinite validation {key}')
            if key not in result or value > result[key]['value']:
                result[key] = {'value': value, 'checkpoint': checkpoint}
    return result


BOX_BODY = re.compile(r'<box>(.*?)(?:</box>|$)', re.S | re.I)
OBJECT = re.compile(r'<(\d{1,4})><(\d{1,4})><(\d{1,4})><(\d{1,4})>(<OBJ>)?', re.I)


def parse_predictions(response, token_ids, log_probs, tokenizer, segmentation):
    boxes, scores = [], []
    cursor = 0
    response = response.split('<|im_end|>', 1)[0].split('<|endoftext|>', 1)[0]
    for body in BOX_BODY.findall(response):
        for match in OBJECT.finditer(body):
            values = list(map(int, match.groups()[:4]))
            x1, y1, x2, y2 = values
            if segmentation and not match.group(5):
                continue
            if not (0 <= x1 < x2 <= 1000 and 0 <= y1 < y2 <= 1000):
                continue
            coordinate_ids = [tokenizer.convert_tokens_to_ids(f'<{value}>') for value in values]
            # These released coordinate tokens are individual vocabulary entries.
            start = next((i for i in range(cursor, len(token_ids)-3)
                          if token_ids[i:i+4] == coordinate_ids), None)
            if start is None:
                raise RuntimeError('generated coordinate tokens could not be aligned to confidence scores')
            cursor = start + 4
            boxes.append([v/1000 for v in values])
            scores.append(math.exp(sum(log_probs[start:start+4])/4))
    return boxes, scores

