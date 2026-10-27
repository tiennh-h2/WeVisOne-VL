#!/usr/bin/env python3
"""Convert train/valid/test/{images,masks} for the attached WeVisOne-VL trainer.

Install: pip install numpy Pillow pycocotools PyYAML
Each nonempty image gets ONE wall instance (all disconnected regions included).
Default mask convention: 0 = background, any positive value = wall.
Images are referenced in place; no image or mask pixels are modified.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import yaml
from PIL import Image
from pycocotools import mask as coco_mask

EXTENSIONS = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.webp'}


def convert_split(root, output, split, threshold, invert):
    image_root = root / split / 'images'
    mask_root = root / split / 'masks'
    if not image_root.is_dir() or not mask_root.is_dir():
        raise ValueError(f'Missing images/masks directories in {root / split}')
    images = sorted(p for p in image_root.rglob('*')
                    if p.is_file() and p.suffix.lower() in EXTENSIONS)
    if not images:
        raise ValueError(f'No images in {image_root}')
    # Match relative paths without extensions: images/foo.jpg -> masks/foo.png.
    masks = {}
    for path in sorted(mask_root.rglob('*')):
        if path.is_file() and path.suffix.lower() in EXTENSIONS:
            key = path.relative_to(mask_root).with_suffix('').as_posix()
            if key in masks:
                raise ValueError(f'Ambiguous mask stem: {key}')
            masks[key] = path
    doc = {'images': [], 'annotations': [],
           'categories': [{'id': 0, 'name': 'wall'}]}
    for image_id, path in enumerate(images, 1):
        relative = path.relative_to(image_root)
        key = relative.with_suffix('').as_posix()
        if key not in masks:
            raise ValueError(f'Missing mask for {path}; expected {mask_root / key}.*')
        with Image.open(path) as image:
            width, height = image.size
        with Image.open(masks[key]) as image:
            array = np.array(image)
        if array.ndim == 3:
            # Accept grayscale RGB/RGBA; ignore alpha, never count it as wall.
            rgb = array[..., :3]
            if not np.all(rgb == rgb[..., :1]):
                raise ValueError(f'Expected grayscale/label mask, got RGB colors: {masks[key]}')
            array = rgb[..., 0]
        if array.shape != (height, width):
            raise ValueError(f'Mask/image size mismatch: {masks[key]}: '
                             f'{array.shape} vs {(height, width)}')
        foreground = array > threshold
        if invert:
            foreground = ~foreground
        doc['images'].append({'id': image_id, 'file_name': relative.as_posix(),
                              'width': width, 'height': height})
        if foreground.any():
            rle = coco_mask.encode(np.asfortranarray(foreground, dtype=np.uint8))
            area = int(coco_mask.area(rle))
            bbox = coco_mask.toBbox(rle).tolist()  # COCO xywh in pixels
            rle['counts'] = rle['counts'].decode('ascii')
            doc['annotations'].append({
                'id': len(doc['annotations']) + 1, 'image_id': image_id,
                'category_id': 0, 'segmentation': rle, 'bbox': bbox,
                'area': area, 'iscrowd': 0,
            })
    destination = output / split / '_annotations.coco.json'
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(doc, ensure_ascii=False), encoding='utf-8')
    positives = len(doc['annotations'])
    print(f'{split}: {len(images)} images, {positives} wall instances, '
          f'{len(images) - positives} empty masks -> {destination}')
    return {'name': f'wall_{split}', 'task': 'segmentation',
            'images': str(image_root), 'annotations': str(destination),
            'negative_queries_per_image': 1}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True,
                        help='Dataset root containing train/, valid/, test/')
    parser.add_argument('--output', type=Path, required=True,
                        help='Output COCO JSON files and wall_finetune.yaml')
    parser.add_argument('--model', default='tencent/WeVisOne-VL-4B',
                        help='Released model ID or absolute local model directory')
    parser.add_argument('--threshold', type=float, default=0,
                        help='Wall = mask > threshold (default: any nonzero value)')
    parser.add_argument('--invert', action='store_true',
                        help='Use mask <= threshold as wall, e.g. black walls on white')
    args = parser.parse_args()
    root, output = args.root.expanduser().resolve(), args.output.expanduser().resolve()
    sources = {split: convert_split(root, output, split, args.threshold, args.invert)
               for split in ('train', 'valid', 'test')}
    # Other optimizer/backbone/loss options use the supplied trainer's defaults.
    config = {
        'model': args.model, 'output_dir': str(output / 'training_output'),
        'seed': 42, 'max_steps': 2000,
        'train': [dict(sources['train'], weight=1.0)],
        'validation': [sources['valid']],
        'max_masks_per_sample': 1,
    }
    config_path = output / 'wall_finetune.yaml'
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')
    print(f'Config: {config_path}')
    print('Test annotations are separate and are not included in training/validation.')
    print('Run from your patched WeVisOne-VL repository:')
    print(f'python -m wevisone_vl.training --config "{config_path}" --dry-run')
    print(f'python -m wevisone_vl.training --config "{config_path}"')


if __name__ == '__main__':
    main()
