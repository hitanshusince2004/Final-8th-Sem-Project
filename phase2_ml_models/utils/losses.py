"""
Glacier Melting Detection — Loss Functions
==========================================
Custom loss functions for imbalanced glacier segmentation:
  - DiceLoss            — overlap-based, handles class imbalance
  - FocalLoss           — down-weights easy negatives
  - DiceBCELoss         — combined Dice + Binary Cross-Entropy
  - DiceFocalLoss       — combined Dice + Focal (for DeepLabv3+)
  - WeightedCELoss      — class-weighted cross-entropy
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Soft Dice Loss for multi-class segmentation.
    Computes mean Dice across all classes.

    Args:
        smooth:      smoothing constant to avoid division by zero
        ignore_index: class index to ignore (e.g., boundary pixels)
    """

    def __init__(self, smooth=1.0, ignore_index=None, per_class=False):
        super().__init__()
        self.smooth       = smooth
        self.ignore_index = ignore_index
        self.per_class    = per_class

    def forward(self, logits, targets):
        """
        logits:  (B, C, H, W) — raw model outputs
        targets: (B, H, W)    — integer class labels
        """
        num_classes = logits.shape[1]
        probs       = F.softmax(logits, dim=1)

        # One-hot encode targets: (B, C, H, W)
        targets_oh  = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()

        # Flatten spatial dims
        probs_flat  = probs.view(probs.shape[0], num_classes, -1)      # (B, C, N)
        targets_flat = targets_oh.view(targets_oh.shape[0], num_classes, -1)  # (B, C, N)

        intersection = (probs_flat * targets_flat).sum(dim=2)           # (B, C)
        union        = probs_flat.sum(dim=2) + targets_flat.sum(dim=2)  # (B, C)

        dice_per_class = (2 * intersection + self.smooth) / (union + self.smooth)

        if self.per_class:
            return 1.0 - dice_per_class.mean(dim=0)   # (C,) per-class Dice loss

        return 1.0 - dice_per_class.mean()


class BinaryDiceLoss(nn.Module):
    """Dice loss for binary segmentation (single-channel output)."""

    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        """
        logits:  (B, H, W) or (B, 1, H, W) — raw outputs
        targets: (B, H, W) — binary labels (0 or 1)
        """
        probs   = torch.sigmoid(logits.squeeze(1))
        targets = targets.float()

        probs_f  = probs.contiguous().view(probs.shape[0], -1)
        targets_f = targets.contiguous().view(targets.shape[0], -1)

        inter = (probs_f * targets_f).sum(dim=1)
        union = probs_f.sum(dim=1) + targets_f.sum(dim=1)

        dice = (2 * inter + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


class FocalLoss(nn.Module):
    """
    Focal Loss — Lin et al. 2017.
    Reduces relative loss for well-classified examples, focusing on hard negatives.

    Args:
        gamma:  focusing parameter (2.0 recommended)
        alpha:  class-weighting tensor of shape (C,)
        reduction: 'mean' or 'sum'
    """

    def __init__(self, gamma=2.0, alpha=None, reduction="mean"):
        super().__init__()
        self.gamma     = gamma
        self.alpha     = alpha    # tensor (C,)
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        logits:  (B, C, H, W)
        targets: (B, H, W) long
        """
        B, C, H, W = logits.shape
        log_probs   = F.log_softmax(logits, dim=1)
        probs       = torch.exp(log_probs)

        # Gather log-probs of the true class: (B, H, W)
        log_p_t = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        p_t     = probs.gather(1,     targets.unsqueeze(1)).squeeze(1)

        focal_weight = (1 - p_t) ** self.gamma

        if self.alpha is not None:
            alpha = self.alpha.to(logits.device)
            alpha_t = alpha[targets]
            focal_weight = focal_weight * alpha_t

        loss = -focal_weight * log_p_t

        if self.reduction == "mean":
            return loss.mean()
        return loss.sum()


class DiceBCELoss(nn.Module):
    """
    Dice + Binary Cross-Entropy — primary loss for U-Net binary segmentation.
    dice_weight + bce_weight should sum to 1.0.
    """

    def __init__(self, dice_weight=0.6, bce_weight=0.4, smooth=1.0):
        super().__init__()
        self.dice      = DiceLoss(smooth=smooth)
        self.bce       = nn.CrossEntropyLoss()
        self.dw        = dice_weight
        self.bw        = bce_weight

    def forward(self, logits, targets):
        return self.dw * self.dice(logits, targets) + self.bw * self.bce(logits, targets)


class DiceFocalLoss(nn.Module):
    """
    Dice + Focal Loss — primary loss for DeepLabv3+ multi-class segmentation.
    Handles severe class imbalance (glacier vs land).
    """

    def __init__(self, dice_weight=0.5, focal_weight=0.5,
                 gamma=2.0, class_weights=None, smooth=1.0):
        super().__init__()
        alpha = torch.tensor(class_weights) if class_weights else None
        self.dice   = DiceLoss(smooth=smooth)
        self.focal  = FocalLoss(gamma=gamma, alpha=alpha)
        self.dw     = dice_weight
        self.fw     = focal_weight

    def forward(self, logits, targets):
        return self.dw * self.dice(logits, targets) + self.fw * self.focal(logits, targets)


class WeightedCELoss(nn.Module):
    """Cross-entropy with per-class weights — for CNN classification."""

    def __init__(self, class_weights=None):
        super().__init__()
        weights = torch.tensor(class_weights).float() if class_weights else None
        self.ce = nn.CrossEntropyLoss(weight=weights)

    def forward(self, logits, targets):
        return self.ce(logits, targets)


def get_loss(name, config):
    """Factory: return the right loss object based on model config."""
    loss_name = config.get("loss", "ce")

    if loss_name == "dice_bce":
        return DiceBCELoss(
            dice_weight=config.get("dice_weight", 0.6),
            bce_weight =config.get("bce_weight",  0.4),
        )
    elif loss_name == "dice_focal":
        from config.ml_config import CLASS_WEIGHTS
        return DiceFocalLoss(
            dice_weight  = config.get("dice_weight",   0.5),
            focal_weight = config.get("focal_weight",  0.5),
            gamma        = config.get("focal_gamma",   2.0),
            class_weights= CLASS_WEIGHTS,
        )
    elif loss_name == "focal":
        from config.ml_config import CLASS_WEIGHTS
        return FocalLoss(gamma=config.get("focal_gamma", 2.0),
                         alpha=torch.tensor(CLASS_WEIGHTS))
    elif loss_name == "dice":
        return DiceLoss()
    else:
        from config.ml_config import CLASS_WEIGHTS
        return WeightedCELoss(class_weights=CLASS_WEIGHTS)
