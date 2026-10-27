#!/usr/bin/env python3
"""Evaluate an exported WeVisOne-VL model against test/images + test/masks.

Run from the WeVisOne-VL repository root. Uses free-running generation, not
teacher forcing or ground-truth boxes. All predicted instances form one wall
mask. mDice/mIoU average foreground scores over images (not background classes).
Both masks empty => 1; only one empty => 0. No smoothing alters the scores.
Dependencies: normal WeVisOne-VL inference environment plus numpy and Pillow.
"""
import argparse
import csv
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image

EXTENSIONS = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.webp'}
DEFAULT_TEST = '/data1/workspace/tien.nguyen/project/common/data/ykkap/yap-3/wall_seg_crop/test'


def score_masks(pred, gt):
    pred, gt = np.asarray(pred, dtype=bool), np.asarray(gt, dtype=bool)
    if pred.shape != gt.shape:
        raise ValueError(f'Prediction/GT shape mismatch: {pred.shape} vs {gt.shape}')
    tp = int(np.count_nonzero(pred & gt))
    fp = int(np.count_nonzero(pred & ~gt))
    fn = int(np.count_nonzero(~pred & gt))
    return {'dice': 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.0,
            'iou': tp / (tp + fp + fn) if tp + fp + fn else 1.0,
            'tp': tp, 'fp': fp, 'fn': fn,
            'gt_pixels': tp + fn, 'pred_pixels': tp + fp}


def union_predictions(masks, shape, threshold):
    union = np.zeros(shape, dtype=bool)
    for mask in masks:
        array = np.asarray(mask)
        if array.shape != shape:
            raise ValueError(f'Expected original-resolution mask {shape}; got {array.shape}')
        if not np.isfinite(array).all() or (array < 0).any() or (array > 1).any():
            raise ValueError('Expected finite mask probabilities in [0, 1]')
        union |= array >= threshold
    return union


def load_gt(path, shape, threshold, invert):
    with Image.open(path) as opened:
        array = np.array(opened)
    if array.ndim == 3:
        rgb = array[..., :3]
        if not np.all(rgb == rgb[..., :1]):
            raise ValueError(f'Expected grayscale or label mask: {path}')
        array = rgb[..., 0]
    if array.shape != shape:
        raise ValueError(f'GT/image shape mismatch for {path}: {array.shape} vs {shape}')
    foreground = array > threshold
    return ~foreground if invert else foreground


def aggregate(rows):
    return {'num_images': len(rows),
            'mDice': float(np.mean([r['dice'] for r in rows])) if rows else None,
            'mIoU': float(np.mean([r['iou'] for r in rows])) if rows else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, help='Exported model directory or model ID')
    parser.add_argument('--images', type=Path, default=Path(DEFAULT_TEST) / 'images')
    parser.add_argument('--masks', type=Path, default=Path(DEFAULT_TEST) / 'masks')
    parser.add_argument('--output', type=Path, default=Path('outputs/wall_test_eval'))
    parser.add_argument('--query', default='wall')
    parser.add_argument('--threshold', type=float, default=.5, help='Prediction probability threshold')
    parser.add_argument('--gt-threshold', type=float, default=0, help='GT wall = value > threshold')
    parser.add_argument('--invert-gt', action='store_true', help='GT wall = value <= gt-threshold')
    parser.add_argument('--device', default='auto', help='auto, cuda:0, cpu, etc.')
    parser.add_argument('--dtype', choices=['bfloat16', 'float16', 'float32'], default='bfloat16')
    parser.add_argument('--min-pixels', type=int, default=16384)
    parser.add_argument('--max-pixels', type=int, default=1048576)
    parser.add_argument('--max-new-tokens', type=int, default=256)
    parser.add_argument('--limit', type=int, default=0, help='First N images; 0 = whole test set')
    parser.add_argument('--save-masks', action='store_true', help='Save binary prediction PNGs')
    args = parser.parse_args()
    if not 0 < args.threshold <= 1 or args.limit < 0 or args.max_new_tokens <= 0:
        parser.error('Require 0 < threshold <= 1, limit >= 0, max-new-tokens > 0')
    if args.min_pixels <= 0 or args.max_pixels < args.min_pixels:
        parser.error('Require 0 < min-pixels <= max-pixels')
    args.images, args.masks = args.images.expanduser().resolve(), args.masks.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    model_path = Path(args.model).expanduser()
    if model_path.is_dir():
        args.model = str(model_path.resolve())
        if (model_path / 'training_config.yaml').is_file():
            parser.error('This is a training checkpoint. Export it using '
                         'python -m wevisone_vl.training.export --checkpoint CHECKPOINT --output EXPORT')
    if not args.images.is_dir() or not args.masks.is_dir():
        parser.error('Both --images and --masks must be existing directories')
    images = sorted(p for p in args.images.rglob('*') if p.is_file() and p.suffix.lower() in EXTENSIONS)
    if args.limit:
        images = images[:args.limit]
    if not images:
        parser.error('No test images found')
    mask_index = {}
    for path in sorted(args.masks.rglob('*')):
        if path.is_file() and path.suffix.lower() in EXTENSIONS:
            key = path.relative_to(args.masks).with_suffix('').as_posix()
            if key in mask_index:
                raise ValueError(f'Ambiguous GT mask stem: {key}')
            mask_index[key] = path
    pairs = []
    for path in images:
        relative = path.relative_to(args.images)
        key = relative.with_suffix('').as_posix()
        if key not in mask_index:
            raise FileNotFoundError(f'Missing GT mask for {path}')
        pairs.append((path, relative, mask_index[key]))

    from wevisone_vl import WeVisOneVL
    model = WeVisOneVL(args.model, backend='transformers', device=args.device, dtype=args.dtype,
                      min_pixels=args.min_pixels, max_pixels=args.max_pixels)
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    with (args.output / 'per_image.csv').open('w', newline='', encoding='utf-8') as csv_file, \
         (args.output / 'responses.jsonl').open('w', encoding='utf-8') as response_file:
        fields = ['file_name', 'dice', 'iou', 'tp', 'fp', 'fn', 'gt_pixels', 'pred_pixels',
                  'num_pred_masks', 'seconds']
        writer = csv.DictWriter(csv_file, fieldnames=fields)
        writer.writeheader()
        for index, (path, relative, gt_path) in enumerate(pairs, 1):
            with Image.open(path) as opened:
                image = opened.convert('RGB')
            shape = (image.height, image.width)
            gt = load_gt(gt_path, shape, args.gt_threshold, args.invert_gt)
            started = time.perf_counter()
            result = model.predict(image, task='segmentation', query=args.query,
                                   max_new_tokens=args.max_new_tokens, temperature=0.0)
            pred = union_predictions(result.masks, shape, args.threshold)
            row = dict(file_name=relative.as_posix(), **score_masks(pred, gt),
                       num_pred_masks=len(result.masks), seconds=time.perf_counter() - started)
            rows.append(row)
            writer.writerow(row)
            csv_file.flush()
            response_file.write(json.dumps({'file_name': relative.as_posix(),
                                            'response': result.response}, ensure_ascii=False) + '\n')
            response_file.flush()
            if args.save_masks:
                # Include the original extension to avoid a.jpg / a.png collisions.
                target = args.output / 'pred_masks' / (relative.as_posix() + '.png')
                target.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(pred.astype(np.uint8) * 255).save(target)
            print(f'[{index}/{len(pairs)}] {relative}: Dice={row["dice"]:.4f} '
                  f'IoU={row["iou"]:.4f} masks={row["num_pred_masks"]}', flush=True)
    tp, fp, fn = (sum(r[key] for r in rows) for key in ('tp', 'fp', 'fn'))
    summary = dict(aggregate(rows),
                   micro_Dice=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.0,
                   micro_IoU=tp / (tp + fp + fn) if tp + fp + fn else 1.0,
                   nonempty_gt=aggregate([r for r in rows if r['gt_pixels'] > 0]),
                   num_empty_gt=sum(r['gt_pixels'] == 0 for r in rows),
                   num_no_pred_masks=sum(r['num_pred_masks'] == 0 for r in rows),
                   empty_policy='both empty = 1; only one empty = 0',
                   evaluation='free-running wall foreground union at original image resolution',
                   arguments={key: str(value) if isinstance(value, Path) else value
                              for key, value in vars(args).items()})
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(f'\nmDice={summary["mDice"]:.6f}  mIoU={summary["mIoU"]:.6f}')
    print(f'Results: {args.output}')


if __name__ == '__main__':
    main()
