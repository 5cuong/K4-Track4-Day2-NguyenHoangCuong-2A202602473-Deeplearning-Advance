"""Classification losses and label-aware Mixup/CutMix batch operations."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_criterion(kind: str = "ce", **kw):
    """Return a scalar-loss module for CE, label smoothing, focal, or weighted CE."""
    if kind == "ce":
        return nn.CrossEntropyLoss()
    if kind == "ls":
        return LabelSmoothingCE(smoothing=kw.get("smoothing", 0.1))
    if kind == "focal":
        return FocalLoss(gamma=kw.get("gamma", 2.0), alpha=kw.get("alpha"))
    if kind == "ce_weighted":
        weight = kw.get("weight")
        if weight is None:
            raise ValueError("ce_weighted requires a class-weight tensor.")
        return nn.CrossEntropyLoss(weight=weight)
    raise ValueError(f"Unknown loss kind: {kind!r}")


class LabelSmoothingCE(nn.Module):
    """Cross-entropy with uniform label smoothing."""

    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        if not 0.0 <= smoothing < 1.0:
            raise ValueError("smoothing must be in [0, 1).")
        self.loss = nn.CrossEntropyLoss(label_smoothing=float(smoothing))

    def forward(self, logits, target):
        return self.loss(logits, target)


class FocalLoss(nn.Module):
    """Multiclass focal loss; gamma=0 and alpha=None reduce exactly to CE."""

    def __init__(self, gamma: float = 2.0, alpha=None):
        super().__init__()
        if gamma < 0:
            raise ValueError("gamma must be non-negative.")
        self.gamma = float(gamma)
        if alpha is None:
            self.register_buffer("alpha", None)
        else:
            alpha_tensor = torch.as_tensor(alpha, dtype=torch.float32)
            if alpha_tensor.ndim == 0:
                if alpha_tensor.item() <= 0:
                    raise ValueError("Scalar alpha must be positive.")
            elif (alpha_tensor < 0).any():
                raise ValueError("Class alpha weights must be non-negative.")
            self.register_buffer("alpha", alpha_tensor)

    def forward(self, logits, target):
        target = target.long()
        log_probs = F.log_softmax(logits, dim=1)
        log_pt = log_probs.gather(1, target.unsqueeze(1)).squeeze(1)
        pt = log_pt.exp()
        loss = -((1.0 - pt).clamp_min(0.0).pow(self.gamma)) * log_pt
        if self.alpha is not None:
            if self.alpha.ndim == 0:
                alpha_t = self.alpha
            else:
                if self.alpha.numel() != logits.shape[1]:
                    raise ValueError(
                        f"alpha has {self.alpha.numel()} values for {logits.shape[1]} classes."
                    )
                alpha_t = self.alpha.to(logits.device)[target]
            loss = loss * alpha_t
        return loss.mean()


def class_weights(counts, beta: float = 0.0):
    """Compute normalized inverse-frequency or effective-number weights from TRAIN counts."""
    import numpy as np
    import pandas as pd

    if isinstance(counts, pd.Series):
        values = counts.reindex(range(9), fill_value=0).to_numpy(dtype=np.float64)
    elif isinstance(counts, dict):
        values = np.asarray([counts.get(i, 0) for i in range(9)], dtype=np.float64)
    else:
        values = np.asarray(counts, dtype=np.float64).reshape(-1)
    if values.size != 9 or not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("counts must contain nine finite, positive training-class counts.")
    if not 0.0 <= beta < 1.0:
        raise ValueError("beta must be in [0, 1).")
    if beta == 0:
        weights = 1.0 / values
    else:
        log_beta = np.log(beta)
        denominator = -np.expm1(values * log_beta)
        weights = (1.0 - beta) / denominator
    weights /= weights.mean()
    return torch.as_tensor(weights, dtype=torch.float32)


def mix_batch(x, y, alpha: float = 1.0, mode: str = "cutmix"):
    """Apply Mixup or CutMix and return mixed images with both targets and lambda."""
    if alpha <= 0:
        raise ValueError("alpha must be positive.")
    if mode not in {"mixup", "cutmix"}:
        raise ValueError("mode must be mixup or cutmix.")
    if x.ndim != 4 or y.ndim != 1 or x.shape[0] != y.shape[0]:
        raise ValueError("Expected x=(N,C,H,W), y=(N,), with matching batch sizes.")
    if x.shape[0] < 2:
        return x, (y, y, 1.0)

    lam = float(torch.distributions.Beta(alpha, alpha).sample().item())
    permutation = torch.randperm(x.shape[0], device=x.device)
    y_a, y_b = y, y[permutation]
    if mode == "mixup":
        return lam * x + (1.0 - lam) * x[permutation], (y_a, y_b, lam)

    _, _, height, width = x.shape
    cut_ratio = (1.0 - lam) ** 0.5
    cut_w, cut_h = int(width * cut_ratio), int(height * cut_ratio)
    center_x = int(torch.randint(width, (1,), device=x.device).item())
    center_y = int(torch.randint(height, (1,), device=x.device).item())
    x1, x2 = max(center_x - cut_w // 2, 0), min(center_x + (cut_w + 1) // 2, width)
    y1, y2 = max(center_y - cut_h // 2, 0), min(center_y + (cut_h + 1) // 2, height)
    mixed = x.clone()
    mixed[:, :, y1:y2, x1:x2] = x[permutation, :, y1:y2, x1:x2]
    actual_lam = 1.0 - ((x2 - x1) * (y2 - y1) / float(width * height))
    return mixed, (y_a, y_b, actual_lam)


def mixed_loss(criterion, logits, targets):
    """Combine losses using the actual Mixup/CutMix area coefficient."""
    if not isinstance(targets, (tuple, list)) or len(targets) != 3:
        return criterion(logits, targets)
    y_a, y_b, lam = targets
    lam = float(lam)
    return lam * criterion(logits, y_a) + (1.0 - lam) * criterion(logits, y_b)
