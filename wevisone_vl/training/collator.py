"""Teacher-forced conversations with assistant-only token supervision."""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image

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
                 max_length: int = 8192):
        self.processor = processor
        self.seg_token_id = seg_token_id
        self.cnn_size = cnn_size
        self.max_length = max_length

    def __call__(self, samples: list[dict]) -> dict:
        if len(samples) != 1:
            raise ValueError("one image/query per microbatch is required; use gradient accumulation")
        sample = samples[0]
        image = sample["image"]
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": [{"type": "image", "image": image},
                                          {"type": "text", "text": sample["prompt"]}]},
        ]
        prefix = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        text = self.processor.apply_chat_template(
            messages + [{"role": "assistant", "content": sample["response"]}],
            tokenize=False, add_generation_prompt=False)
        # Process each complete conversation to account for expanded image tokens.
        inputs = dict(self.processor(text=[text], images=[image], return_tensors="pt", padding=False))
        prefix_inputs = self.processor(text=[prefix], images=[image], return_tensors="pt", padding=False)
        inputs["labels"] = assistant_labels(inputs["input_ids"], prefix_inputs["input_ids"], self.max_length)
        obj_count = int(((inputs["input_ids"] == self.seg_token_id) & (inputs["labels"] != -100)).sum())
        expected = len(sample["boxes"]) if sample["task"] == "segmentation" else 0
        if obj_count != expected:
            raise ValueError(f"OBJ alignment: token count {obj_count} != target instances {expected}")
        masks = None
        if sample["masks"] is not None:
            masks = torch.from_numpy(np.asarray(sample["masks"], dtype=np.float32))
            if len(masks):
                masks = resize_binary_targets(masks, (self.cnn_size, self.cnn_size))
        decoder_image = image.resize((self.cnn_size, self.cnn_size), Image.Resampling.BICUBIC)
        rgb = torch.from_numpy(np.asarray(decoder_image, dtype=np.float32).copy() / 255).permute(2, 0, 1)[None]
        return {"inputs": inputs, "task": sample["task"],
                "boxes": torch.tensor(sample["boxes"], dtype=torch.float32).reshape(-1, 4),
                "masks": masks, "cnn_image": rgb,
                "source": sample["source"], "query": sample["query"],
                "image_id": sample["image_id"]}
