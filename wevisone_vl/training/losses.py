"""Binary instance losses, evaluated in FP32 for numerical stability."""
import torch
import torch.nn.functional as F


def resize_binary_targets(targets, size):
    if tuple(targets.shape[-2:]) == tuple(size):
        return targets.float()
    # Any occupied input area marks the output pixel as foreground. Ordinary
    # nearest downsampling can erase thin strokes or entire tiny instances.
    return (F.interpolate(targets[:, None].float(), size=size, mode="area")[:, 0] > 0).float()


def binary_mask_loss(logits, targets, focal_weight=20.0, dice_weight=1.0,
                     alpha=0.25, gamma=2.0):
    if logits.ndim == 4 and logits.shape[1] == 1:
        logits = logits[:, 0]
    if logits.ndim != 3 or targets.ndim != 3 or len(logits) != len(targets) or not len(logits):
        raise ValueError("mask loss expects nonempty aligned [N,H,W] tensors")
    logits = logits.float()
    targets = resize_binary_targets(targets, logits.shape[-2:])
    probabilities = logits.sigmoid()
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = probabilities * targets + (1 - probabilities) * (1 - targets)
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    focal = (alpha_t * (1 - p_t).pow(gamma) * ce).flatten(1).mean(1).mean()
    p, t = probabilities.flatten(1), targets.flatten(1)
    dice = (1 - (2 * (p * t).sum(1) + 1) / (p.sum(1) + t.sum(1) + 1)).mean()
    with torch.no_grad():
        binary = (p >= .5).float()
        hard_dice = ((2 * (binary * t).sum(1) + 1) / (binary.sum(1) + t.sum(1) + 1)).mean()
    return {"loss": focal_weight * focal + dice_weight * dice,
            "focal_loss": focal, "dice_loss": dice, "mask_dice": hard_dice}
