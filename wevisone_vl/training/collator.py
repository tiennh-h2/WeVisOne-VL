"""Teacher-forced conversations with assistant-only token supervision."""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from ..prompts import build_messages
from .losses import resize_binary_targets


def assistant_labels(input_ids: torch.Tensor, prefix_ids: torch.Tensor,
                     max_length: int) -> torch.Tensor:
    if input_ids.shape[1] > max_length:
        raise ValueError(f"sequence length {input_ids.shape[1]} exceeds max_length={max_length}; "
                         "reduce image resolution or use smaller tiles; targets are never truncated")
    length = prefix_ids.shape[1]
    if length >= input_ids.shape[1] or not torch.equal(input_ids[:, :length], prefix_ids):
        raise ValueError("assistant prefix does not match full tokenized conversation")
    labels = input_ids.clone()
    labels[:, :length] = -100
    return labels


class TrainingCollator:
    def __init__(self, processor, seg_token_id: int, cnn_size: int = 1024,
                 max_length: int = 8192, fast_prefix: bool = True):
        self.processor = processor
        self.seg_token_id = seg_token_id
        self.cnn_size = cnn_size
        self.max_length = max_length
        self.fast_prefix = fast_prefix

    def __call__(self, samples: list[dict]) -> dict:
        if len(samples) != 1:
            raise ValueError("one image/query per microbatch is required; use gradient accumulation")
        sample = samples[0]
        image = sample["image"]
        messages = build_messages(image, sample["prompt"])
        prefix = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        text = self.processor.apply_chat_template(
            messages + [{"role": "assistant", "content": sample["response"]}],
            tokenize=False, add_generation_prompt=False)
        # Process each complete conversation to account for expanded image tokens.
        inputs = dict(self.processor(text=[text], images=[image], return_tensors="pt", padding=False))
        image_token = getattr(self.processor, "image_token", None)
        grid = inputs.get("image_grid_thw")
        image_processor = getattr(self.processor, "image_processor", None)
        merge = getattr(image_processor, "merge_size", None)
        if self.fast_prefix and image_token and prefix.count(image_token) == 1 and grid is not None and merge:
            # Qwen expands the placeholder using this same image grid. Tokenize
            # text only, avoiding another complete resize/patch extraction pass.
            image_tokens = int(grid[0].prod()) // int(merge) ** 2
            expanded = prefix.replace(image_token, image_token * image_tokens)
            prefix_inputs = self.processor.tokenizer([expanded], return_tensors="pt", padding=False)
        else:
            prefix_inputs = self.processor(text=[prefix], images=[image], return_tensors="pt", padding=False)
        inputs["labels"] = assistant_labels(inputs["input_ids"], prefix_inputs["input_ids"], self.max_length)
        loss_positions = (inputs["labels"][0, 1:] != -100).nonzero().flatten()
        if not len(loss_positions):
            raise ValueError("sample has no supervised next-token targets")
        obj_positions = ((inputs["input_ids"][0] == self.seg_token_id)
                         & (inputs["labels"][0] != -100)).nonzero().flatten()
        obj_count = len(obj_positions)
        expected = len(sample["boxes"]) if sample["task"] == "segmentation" else 0
        if obj_count != expected:
            raise ValueError(f"OBJ alignment: token count {obj_count} != target instances {expected}")
        masks = None
        if sample["masks"] is not None:
            masks = torch.from_numpy(np.asarray(sample["masks"], dtype=np.float32))
            if len(masks):
                masks = resize_binary_targets(masks, (self.cnn_size, self.cnn_size)).bool()
        rgb = None
        if sample["task"] == "segmentation" and expected:
            decoder_image = image.resize((self.cnn_size, self.cnn_size), Image.Resampling.BICUBIC)
            rgb = torch.from_numpy(np.asarray(decoder_image, dtype=np.float32).copy() / 255).permute(2, 0, 1)[None]
        return {"inputs": inputs, "loss_positions": loss_positions, "obj_positions": obj_positions, "task": sample["task"],
                "boxes": torch.tensor(sample["boxes"], dtype=torch.float32).reshape(-1, 4),
                "masks": masks, "cnn_image": rgb,
                "source": sample["source"], "query": sample["query"],
                "image_id": sample["image_id"]}
