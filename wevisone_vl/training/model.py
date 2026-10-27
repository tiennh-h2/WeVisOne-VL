"""Differentiable training forward; never calls the inference-mode engine."""
from __future__ import annotations

from contextlib import nullcontext

import torch
from torch import nn

from ..config import Qwen3VLSegConfig
from ..feature_bridge import (LastHiddenCapture, VisualFeatureCapture,
                              per_sample_premerge_feature_maps, premerge_grid_hw)
from .losses import binary_mask_loss


class MixedTaskModel(nn.Module):
    def __init__(self, vlm, mask_decoder, cfg: Qwen3VLSegConfig, seg_token_id: int,
                 spatial_merge_size: int, ntp_weight: float = 1.0,
                 segmentation_weight: float = 10.0, focal_weight: float = 20.0,
                 dice_weight: float = 1.0, max_masks_per_sample: int = 24):
        super().__init__()
        self.vlm = vlm
        self.mask_decoder = mask_decoder.float()
        self.cfg = cfg
        self.seg_token_id = seg_token_id
        self.spatial_merge_size = spatial_merge_size
        self.ntp_weight = ntp_weight
        self.segmentation_weight = segmentation_weight
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.max_masks_per_sample = max_masks_per_sample

    @property
    def backbone(self):
        return self.vlm.get_base_model() if hasattr(self.vlm, "get_base_model") else self.vlm

    def forward(self, batch):
        inputs = batch["inputs"]
        positions = ((inputs["input_ids"][0] == self.seg_token_id)
                     & (inputs["labels"][0] != -100)).nonzero().flatten()
        pixel = batch["task"] == "segmentation" and len(positions) > 0
        if batch["task"] == "segmentation":
            masks = batch["masks"]
            if masks is None or len(positions) != len(batch["boxes"]) or len(positions) != len(masks):
                raise ValueError("OBJ/box/mask alignment mismatch")
        visual_context = VisualFeatureCapture(self.backbone, self.cfg.vision_feature_layers) if pixel else nullcontext()
        hidden_context = LastHiddenCapture(self.backbone) if pixel else nullcontext()
        with visual_context as visual, hidden_context as hidden:
            output = self.vlm(**inputs, use_cache=False, return_dict=True, output_hidden_states=False)
        ntp = output.loss
        result = {"loss": self.ntp_weight * ntp, "ntp_loss": ntp.detach(),
                  "seg_loss": ntp.detach().new_zeros(()),
                  "focal_loss": ntp.detach().new_zeros(()),
                  "dice_loss": ntp.detach().new_zeros(()),
                  "mask_dice": ntp.detach().new_zeros(()), "mask_count": 0}
        if not pixel:
            return result
        if hidden.hidden is None:
            raise RuntimeError("LM hidden state was not captured")
        grid = inputs["image_grid_thw"]
        features = per_sample_premerge_feature_maps(
            visual.premerge_features, self.cfg.vision_feature_layers,
            grid, self.spatial_merge_size)[0]
        count = len(positions)
        indices = torch.arange(count, device=positions.device)
        if 0 < self.max_masks_per_sample < count:
            if self.training:
                indices = torch.randperm(count, device=positions.device)[:self.max_masks_per_sample].sort().values
            else:
                indices = torch.linspace(0, count - 1, self.max_masks_per_sample,
                                         device=positions.device).long()
        queries = hidden.hidden[0, positions[indices]]
        prediction = self.mask_decoder(
            image=batch["cnn_image"], visual_features_2d=features, t_seg=queries,
            boxes01=batch["boxes"][indices], grid_hw=premerge_grid_hw(grid)[0])
        losses = binary_mask_loss(prediction["mask_logits"], batch["masks"][indices],
                                  self.focal_weight, self.dice_weight)
        result.update({"loss": self.ntp_weight * ntp + self.segmentation_weight * losses["loss"],
                       "seg_loss": losses["loss"].detach(),
                       "focal_loss": losses["focal_loss"].detach(),
                       "dice_loss": losses["dice_loss"].detach(),
                       "mask_dice": losses["mask_dice"].detach(), "mask_count": len(indices)})
        return result


def configure_backbone(vlm, options: dict):
    """LM LoRA with optional vision-block LoRA, or full vision fine-tuning."""
    vision_lora = options.get('vision_lora', False)
    if not isinstance(vision_lora, bool):
        raise ValueError('vision_lora must be a boolean')
    if vision_lora and (not options.get('lora', True) or options.get('train_vision', False)):
        raise ValueError('vision_lora requires lora: true and train_vision: false')
    root = getattr(vlm, "model", vlm)
    visual = getattr(root, "visual", None)
    if visual is None:
        raise RuntimeError("cannot locate Qwen3-VL visual module")
    if options.get("lora", True):
        from peft import LoraConfig, get_peft_model

        targets = [name for name, module in vlm.named_modules()
                   if isinstance(module, nn.Linear) and ".language_model." in name
                   and name.rsplit(".", 1)[-1] in {"q_proj", "k_proj", "v_proj", "o_proj",
                                                      "gate_proj", "up_proj", "down_proj"}]
        if not targets:
            raise RuntimeError("no language-model LoRA projection targets found")
        if vision_lora:
            blocks = getattr(visual, 'blocks', None)
            if blocks is None:
                raise RuntimeError('cannot locate vision transformer blocks for vision_lora')
            block_linears = {id(module) for module in blocks.modules() if isinstance(module, nn.Linear)}
            vision_targets = [name for name, module in vlm.named_modules()
                              if id(module) in block_linears]
            if not vision_targets:
                raise RuntimeError('no vision-block linear targets found for vision_lora')
            targets.extend(vision_targets)
        vlm = get_peft_model(vlm, LoraConfig(
            r=int(options.get("lora_rank", 16)), lora_alpha=int(options.get("lora_alpha", 32)),
            lora_dropout=float(options.get("lora_dropout", .05)), target_modules=targets,
            bias="none", task_type=None))
    else:
        vlm.requires_grad_(True)
    if vision_lora:
        # PEFT froze the base parameters and enabled the injected adapters.
        # Do not freeze visual wholesale here: that would freeze its LoRA too.
        for name, parameter in visual.named_parameters():
            parameter.requires_grad_('.lora_A.' in name or '.lora_B.' in name)
    else:
        visual.requires_grad_(bool(options.get("train_vision", False)))
    if options.get("lora", True) and options.get("train_embeddings", False):
        vlm.get_input_embeddings().requires_grad_(True)
        vlm.get_output_embeddings().requires_grad_(True)
    return vlm
