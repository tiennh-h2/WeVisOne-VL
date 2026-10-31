"""Differentiable training forward; never calls the inference-mode engine."""
from __future__ import annotations

from contextlib import nullcontext
import inspect
import warnings
from functools import partial

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from ..config import Qwen3VLSegConfig
from ..feature_bridge import (LastHiddenCapture, VisualFeatureCapture,
                              per_sample_premerge_feature_maps, premerge_grid_hw)
from .losses import binary_mask_loss


class MixedTaskModel(nn.Module):
    def __init__(self, vlm, mask_decoder, cfg: Qwen3VLSegConfig, seg_token_id: int,
                 spatial_merge_size: int, ntp_weight: float = 1.0,
                 segmentation_weight: float = 10.0, focal_weight: float = 20.0,
                 dice_weight: float = 1.0, max_masks_per_sample: int = 24,
                 selective_logits: bool = True, decoder_checkpointing: bool = False,
                 refinement_chunk_size: int = 0):
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
        self.decoder_checkpointing = decoder_checkpointing
        if type(refinement_chunk_size) is not int or refinement_chunk_size < 0:
            raise ValueError("refinement_chunk_size must be an integer >= 0")
        self.refinement_chunk_size = refinement_chunk_size
        base = self.backbone
        self.selective_logits = selective_logits and "logits_to_keep" in inspect.signature(base.forward).parameters
        if selective_logits and not self.selective_logits:
            warnings.warn("backbone lacks logits_to_keep; using full vocabulary logits", stacklevel=2)

    @property
    def backbone(self):
        return self.vlm.get_base_model() if hasattr(self.vlm, "get_base_model") else self.vlm

    def forward(self, batch):
        if batch.get('generation_step', False):
            # Must enter through the FSDP root forward on EVERY rank. Calling
            # unwrapped.vlm.generate() bypasses root parameter all-gathers.
            arguments = dict(batch['inputs'])
            arguments.pop('use_cache', None)
            return self.vlm(**arguments, use_cache=True, return_dict=True,
                            output_hidden_states=False, logits_to_keep=1)
        inputs = batch["inputs"]
        positions = batch.get("obj_positions")
        if positions is None:
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
            if self.selective_logits:
                # State i predicts label i+1. Drop ignored positions BEFORE the
                # vocabulary projection; retain all supervised assistant tokens.
                positions_for_loss = batch.get("loss_positions")
                if positions_for_loss is None:
                    positions_for_loss = (inputs["labels"][0, 1:] != -100).nonzero().flatten()
                if not len(positions_for_loss):
                    raise ValueError("sample has no supervised next-token targets")
                arguments = {key: value for key, value in inputs.items() if key != "labels"}
                output = self.vlm(**arguments, logits_to_keep=positions_for_loss,
                                  use_cache=False, return_dict=True, output_hidden_states=False)
                targets = inputs["labels"][:, positions_for_loss + 1]
                ntp = F.cross_entropy(output.logits.float().reshape(-1, output.logits.shape[-1]),
                                     targets.reshape(-1))
            else:
                output = self.vlm(**inputs, use_cache=False, return_dict=True, output_hidden_states=False)
                ntp = output.loss
        del output
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
        targets = batch["masks"][indices]
        boxes = batch["boxes"][indices]
        grid_hw = premerge_grid_hw(grid)[0]

        def decode_loss(image, visual_features, query, box, target):
            prediction = self.mask_decoder(
                image=image, visual_features_2d=visual_features, t_seg=query,
                boxes01=box, grid_hw=grid_hw)
            # Checkpoint the loss too: do not retain full-resolution mask logits.
            return binary_mask_loss(prediction["mask_logits"], target,
                                    self.focal_weight, self.dice_weight)

        args = (batch["cnn_image"], features, queries, boxes, targets)
        if self.refinement_chunk_size:
            losses = self.mask_decoder.forward_loss(
                image=batch["cnn_image"], visual_features_2d=features, t_seg=queries,
                boxes01=boxes, grid_hw=grid_hw, targets=targets,
                loss_fn=partial(binary_mask_loss, focal_weight=self.focal_weight,
                                dice_weight=self.dice_weight),
                chunk_size=self.refinement_chunk_size,
                checkpointing=self.decoder_checkpointing)
        elif self.decoder_checkpointing and self.training and torch.is_grad_enabled():
            losses = checkpoint(decode_loss, *args, use_reentrant=False)
        else:
            losses = decode_loss(*args)
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
    if 'train_merger' in options:
        for name, module in visual.named_modules():
            if 'merger' in name.split('.')[-1]:
                module.requires_grad_(options['train_merger'])
    if options.get("lora", True) and options.get("train_embeddings", False):
        vlm.get_input_embeddings().requires_grad_(True)
        vlm.get_output_embeddings().requires_grad_(True)
    return vlm


def configure_language_checkpointing(vlm, enabled=False):
    """Checkpoint only LM blocks; merger captures stay outside recomputation.

    Non-reentrant checkpointing preserves differentiable hook outputs and works
    with frozen embeddings. Vision checkpointing deliberately remains disabled.
    """
    if hasattr(vlm, "gradient_checkpointing_disable"):
        vlm.gradient_checkpointing_disable()
    if not enabled:
        return
    base = vlm.get_base_model() if hasattr(vlm, "get_base_model") else vlm
    root = getattr(base, "model", base)
    language = getattr(root, "language_model", None)
    if language is None or not hasattr(language, "gradient_checkpointing_enable"):
        raise RuntimeError("language_checkpointing requires a Qwen language_model with checkpointing support")
    language.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})


def configure_vision_checkpointing(vlm, enabled=False):
    if not enabled:
        return
    base = vlm.get_base_model() if hasattr(vlm, 'get_base_model') else vlm
    visual = base.model.visual
    if not hasattr(visual, 'gradient_checkpointing_enable'):
        raise RuntimeError('this Transformers vision module does not support gradient checkpointing')
    visual.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
