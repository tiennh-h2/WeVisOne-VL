"""Strict COCO query construction; annotation identity survives serialization."""
from __future__ import annotations

import json
import math
import random
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from ..prompts import build_prompt


def make_target(query: str, boxes: list, segmentation: bool) -> str:
    if not query.strip() or re.search(r"<(?:/?(?:ref|box)|OBJ|\d+|\|[^>]*\|)>", query, re.I):
        raise ValueError("query is empty or contains reserved model tokens")
    objects = []
    for box in boxes:
        if len(box) != 4 or not all(math.isfinite(float(v)) for v in box):
            raise ValueError(f"invalid normalized box: {box}")
        x1, y1, x2, y2 = map(float, box)
        if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
            raise ValueError(f"invalid normalized box: {box}")
        bins = [min(1000, max(0, int(v * 1000))) for v in box]
        # Preserve tiny instances instead of silently producing degenerate boxes.
        bins[0], bins[1] = min(bins[0], 999), min(bins[1], 999)
        bins[2], bins[3] = max(bins[2], bins[0] + 1), max(bins[3], bins[1] + 1)
        objects.append("".join(f"<{v}>" for v in bins) + ("<OBJ>" if segmentation else ""))
    return f"<ref>{query}</ref><box>{','.join(objects) if objects else 'None'}</box>"


def decode_instance(segmentation, height: int, width: int) -> np.ndarray:
    from pycocotools import mask as coco_mask

    if not segmentation:
        raise ValueError("missing segmentation for a segmentation-task instance")
    if isinstance(segmentation, list):
        if any(len(poly) < 6 or len(poly) % 2 for poly in segmentation):
            raise ValueError("invalid segmentation polygon")
        rle = coco_mask.merge(coco_mask.frPyObjects(segmentation, height, width))
    elif isinstance(segmentation, dict):
        rle = dict(segmentation)
        if isinstance(rle.get("counts"), list):
            rle = coco_mask.frPyObjects(rle, height, width)
        elif isinstance(rle.get("counts"), str):
            rle["counts"] = rle["counts"].encode("ascii")
    else:
        raise ValueError("segmentation must be COCO polygons or RLE")
    mask = coco_mask.decode(rle)
    if mask.ndim == 3:
        mask = mask.max(axis=2)
    if mask.shape != (height, width) or not mask.any():
        raise ValueError(f"empty or wrong-size segmentation: {mask.shape}, expected {(height, width)}")
    return (mask > 0).astype(np.uint8)


class CocoTaskDataset(Dataset):
    """One query/image per record, all matching instances in its response.

    ``queries_jsonl`` optionally adds explicit phrase-to-annotation-ID records.
    These supplied ID sets must be exhaustive for each phrase; no fuzzy text
    matching or phrase rewriting is used to choose targets.
    """

    def __init__(self, spec: dict, seed: int = 42):
        self.spec = dict(spec)
        self.name = str(spec.get("name", Path(spec["annotations"]).stem))
        self.task = spec["task"]
        if self.task not in {"detection", "segmentation"}:
            raise ValueError("dataset task must be detection or segmentation")
        self.root = Path(spec["images"]).expanduser()
        doc = json.loads(Path(spec["annotations"]).expanduser().read_text(encoding="utf-8"))
        self.images = {im["id"]: im for im in doc["images"]}
        self.categories = {cat["id"]: cat["name"] for cat in doc["categories"]}
        self.annotations = {ann["id"]: ann for ann in doc["annotations"]}
        if len(self.annotations) != len(doc["annotations"]) or len(self.images) != len(doc["images"]):
            raise ValueError("duplicate image or annotation IDs")
        selected = list(spec.get("category_ids", self.categories))
        if not selected or any(cat not in self.categories for cat in selected):
            raise ValueError("category_ids must select existing categories")
        # Partition mixed-supervision COCO sources without rewriting annotations.
        # Exclusions also apply to absent-category queries below. IDs absent from
        # a split are harmless, so one exclusion list can be shared across splits.
        excluded = set(spec.get("exclude_category_ids", []))
        selected = [cat for cat in selected if cat not in excluded]
        if not selected:
            raise ValueError("exclude_category_ids removes all selected categories")
        groups = defaultdict(list)
        crowds = set()
        for ann in self.annotations.values():
            if ann["image_id"] not in self.images or ann["category_id"] not in self.categories:
                raise ValueError(f"annotation {ann['id']} references unknown image/category")
            key = ann["image_id"], ann["category_id"]
            if ann.get("iscrowd", 0):
                crowds.add(key)
            else:
                groups[key].append(ann["id"])
        negatives = int(spec.get("negative_queries_per_image", 0))
        if negatives < 0:
            raise ValueError("negative_queries_per_image must be nonnegative")
        rng = random.Random(seed)
        self.records = []
        for image_id in self.images:
            for cat in selected:
                key = image_id, cat
                if groups[key] and key not in crowds:
                    self.records.append({"image_id": image_id, "query": self.categories[cat],
                                         "annotation_ids": groups[key]})
            absent = [cat for cat in selected if not groups[image_id, cat]
                      and (image_id, cat) not in crowds]
            for cat in rng.sample(absent, min(negatives, len(absent))):
                self.records.append({"image_id": image_id, "query": self.categories[cat],
                                     "annotation_ids": []})
        if spec.get("queries_jsonl"):
            with Path(spec["queries_jsonl"]).expanduser().open(encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    image_id, ids = record["image_id"], record["annotation_ids"]
                    if image_id not in self.images or len(set(ids)) != len(ids):
                        raise ValueError("phrase query has unknown image or duplicate annotation IDs")
                    if any(i not in self.annotations or self.annotations[i]["image_id"] != image_id
                           or self.annotations[i].get("iscrowd", 0) for i in ids):
                        raise ValueError("phrase query references wrong-image, unknown or crowd annotation")
                    self.records.append({"image_id": image_id, "query": record["query"],
                                         "annotation_ids": ids})
        if not self.records:
            raise ValueError(f"source {self.name!r} has no usable queries")
        for record in self.records:
            make_target(record["query"], [], self.task == "segmentation")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        info = self.images[record["image_id"]]
        with Image.open(self.root / info["file_name"]) as opened:
            image = opened.convert("RGB")
        width, height = image.size
        if (width, height) != (info["width"], info["height"]):
            raise ValueError(f"image size mismatch for {info['file_name']}: {image.size}")
        instances = []
        for ann_id in record["annotation_ids"]:
            ann = self.annotations[ann_id]
            mask = decode_instance(ann.get("segmentation"), height, width) if self.task == "segmentation" else None
            if ann.get("bbox") is not None:
                x, y, w, h = map(float, ann["bbox"])
                if w <= 0 or h <= 0 or not all(math.isfinite(v) for v in (x, y, w, h)):
                    raise ValueError(f"invalid COCO bbox for annotation {ann_id}")
                box = [max(0, x)/width, max(0, y)/height,
                       min(width, x+w)/width, min(height, y+h)/height]
            elif mask is not None:
                ys, xs = np.where(mask)
                box = [xs.min()/width, ys.min()/height, (xs.max()+1)/width, (ys.max()+1)/height]
            else:
                raise ValueError(f"missing bbox for detection annotation {ann_id}")
            # Validate before sorting and serialization.
            make_target(record["query"], [box], self.task == "segmentation")
            instances.append((box, ann_id, mask))
        instances.sort(key=lambda item: (item[0][1], item[0][0], item[1]))
        boxes = [item[0] for item in instances]
        masks = None
        if self.task == "segmentation":
            masks = np.stack([item[2] for item in instances]) if instances else np.zeros((0, height, width), np.uint8)
        return {"image": image, "source": self.name, "task": self.task,
                "image_id": record["image_id"], "file_name": info["file_name"],
                "query": record["query"], "annotation_ids": [item[1] for item in instances],
                "prompt": build_prompt(self.task, record["query"]),
                "response": make_target(record["query"], boxes, self.task == "segmentation"),
                "boxes": boxes, "masks": masks}
