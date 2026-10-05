"""Validation-only TTA, probability aggregation, calibration, and Conv-BN fusion."""
from __future__ import annotations

import numpy as np


def _as_logits(output):
    if isinstance(output, (tuple, list)):
        output = output[0]
    if isinstance(output, dict):
        if "logits" not in output:
            raise ValueError("Model output dictionary has no 'logits' entry.")
        output = output["logits"]
    return output


def predict_logits(model, loader, device, view=None, amp: bool = False):
    """Return filenames, true labels, and logits in the loader's stable row order."""
    import torch

    device = torch.device(device)
    model.eval()
    names, labels, batches = [], [], []
    with torch.inference_mode():
        for images, target, filenames in loader:
            images = images.to(device, non_blocking=True)
            if view is not None:
                images = view(images)
            with torch.autocast(
                device_type="cuda", dtype=torch.float16,
                enabled=(device.type == "cuda" and bool(amp))
            ):
                logits = _as_logits(model(images))
            batches.append(logits.float().cpu().numpy())
            labels.append(target.cpu().numpy())
            names.extend(str(filename) for filename in filenames)
    if not batches:
        raise ValueError("Cannot infer on an empty data loader.")
    return names, np.concatenate(labels).astype(np.int64), np.concatenate(batches, axis=0)


def predict_multi_logits(model, loader, device, view_builder, space: str = "logit",
                         amp: bool = False):
    """Run a batch-to-views callable and aggregate its predictions in a chosen space.

    For probability averaging the returned score matrix is log(mean probability), so
    callers can still pass a finite score matrix to the temperature scaler.
    """
    import torch

    if space not in {"prob", "logit"}:
        raise ValueError("space must be prob or logit.")
    device = torch.device(device)
    model.eval()
    names, labels, output_batches = [], [], []
    with torch.inference_mode():
        for images, target, filenames in loader:
            images = images.to(device, non_blocking=True)
            views = view_builder(images)
            if not isinstance(views, (list, tuple)) or not views:
                raise ValueError("view_builder must return a non-empty list of image batches.")
            logits = []
            for view in views:
                with torch.autocast(
                    device_type="cuda", dtype=torch.float16,
                    enabled=(device.type == "cuda" and bool(amp))
                ):
                    logits.append(_as_logits(model(view)).float())
            if space == "logit":
                combined = torch.stack(logits).mean(dim=0)
            else:
                probabilities = torch.stack([value.softmax(dim=1) for value in logits]).mean(dim=0)
                combined = probabilities.clamp_min(1e-15).log()
            output_batches.append(combined.cpu().numpy())
            labels.append(target.cpu().numpy())
            names.extend(str(filename) for filename in filenames)
    if not output_batches:
        raise ValueError("Cannot infer on an empty data loader.")
    return names, np.concatenate(labels).astype(np.int64), np.concatenate(output_batches, axis=0)


def view_identity(x):
    return x


def view_hflip(x):
    """Horizontal flip over the width axis of an NCHW tensor."""
    import torch
    if x.ndim != 4:
        raise ValueError("Expected an NCHW image batch.")
    return torch.flip(x, dims=(-1,))


def views_multicrop(x, crop: int):
    """Return five crops (four corners and center) from a square or rectangular batch."""
    import torch.nn.functional as F

    if x.ndim != 4 or crop <= 0:
        raise ValueError("Expected NCHW input and a positive crop size.")
    _, _, height, width = x.shape
    if crop > height or crop > width:
        x = F.interpolate(
            x, size=(max(height, crop), max(width, crop)),
            mode="bilinear", align_corners=False,
        )
        _, _, height, width = x.shape
    offsets = [
        (0, 0),
        (0, width - crop),
        (height - crop, 0),
        (height - crop, width - crop),
        ((height - crop) // 2, (width - crop) // 2),
    ]
    return [x[:, :, top:top + crop, left:left + crop] for top, left in offsets]


def views_multiscale(x, sizes):
    """Resize a batch to each requested square size (model-dependent support)."""
    import torch.nn.functional as F

    sizes = [int(size) for size in sizes]
    if x.ndim != 4 or not sizes or any(size <= 0 for size in sizes):
        raise ValueError("Expected NCHW input and a non-empty list of positive sizes.")
    return [
        F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
        for size in sizes
    ]


def aggregate_views(logits_per_view, space: str = "prob"):
    """Average model predictions in probability or logit space and normalize rows."""
    import torch

    if space not in {"prob", "logit"}:
        raise ValueError("space must be 'prob' or 'logit'.")
    if not logits_per_view:
        raise ValueError("At least one view is required.")
    arrays = [torch.as_tensor(value, dtype=torch.float64) for value in logits_per_view]
    shape = arrays[0].shape
    if len(shape) != 2 or any(value.shape != shape for value in arrays):
        raise ValueError("Every view must have the same (N, K) shape.")
    if not all(torch.isfinite(value).all() for value in arrays):
        raise ValueError("View logits contain NaN or infinite values.")
    if space == "prob":
        probs = torch.stack([value.softmax(dim=1) for value in arrays]).mean(dim=0)
    else:
        probs = torch.stack(arrays).mean(dim=0).softmax(dim=1)
    probs = probs / probs.sum(dim=1, keepdim=True).clamp_min(1e-15)
    return probs.cpu().numpy()


def ensemble_probs(list_of_probs):
    """Average probability matrices from models evaluated on the same ordered rows."""
    if not list_of_probs:
        raise ValueError("At least one probability matrix is required.")
    arrays = [np.asarray(value, dtype=np.float64) for value in list_of_probs]
    shape = arrays[0].shape
    if len(shape) != 2 or any(value.shape != shape for value in arrays):
        raise ValueError("Every model must have the same (N, K) probability shape.")
    if any(not np.isfinite(value).all() or (value < 0).any() for value in arrays):
        raise ValueError("Probabilities must be finite and non-negative.")
    probs = np.mean(arrays, axis=0)
    totals = probs.sum(axis=1, keepdims=True)
    if (totals <= 0).any():
        raise ValueError("Probability row sums must be positive.")
    return probs / totals


def fit_temperature(val_logits, val_labels) -> float:
    """Fit one positive scalar temperature by minimizing validation negative log-likelihood."""
    import torch
    import torch.nn.functional as F

    logits = torch.as_tensor(val_logits, dtype=torch.float64, device="cpu")
    labels = torch.as_tensor(val_labels, dtype=torch.long, device="cpu")
    if logits.ndim != 2 or labels.ndim != 1 or logits.shape[0] != labels.shape[0]:
        raise ValueError("Expected logits=(N,K) and labels=(N,) with matching N.")
    if logits.shape[0] == 0 or not torch.isfinite(logits).all():
        raise ValueError("Validation logits must be non-empty and finite.")
    log_temperature = torch.zeros((), dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS(
        [log_temperature], lr=0.1, max_iter=100, line_search_fn="strong_wolfe"
    )

    def closure():
        optimizer.zero_grad(set_to_none=True)
        temperature = log_temperature.clamp(-5.0, 5.0).exp()
        loss = F.cross_entropy(logits / temperature, labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_temperature.detach().clamp(-5.0, 5.0).exp().item())


def apply_temperature(logits, T: float):
    """Convert logits to normalized probabilities using a fitted positive temperature."""
    import torch

    if not np.isfinite(T) or T <= 0:
        raise ValueError("Temperature T must be finite and greater than zero.")
    values = torch.as_tensor(logits, dtype=torch.float64)
    if values.ndim != 2 or not torch.isfinite(values).all():
        raise ValueError("logits must be a finite (N, K) matrix.")
    return values.div(float(T)).softmax(dim=1).cpu().numpy()


def fuse_conv_bn(model):
    """Fuse adjacent Conv2d/BatchNorm2d pairs in evaluation mode and report max error."""
    import torch
    import torch.nn as nn
    from torch.nn.utils.fusion import fuse_conv_bn_eval

    model.eval()
    cfg = getattr(model, "pretrained_cfg", {}) or {}
    input_size = cfg.get("input_size", (3, 224, 224))
    height, width = int(input_size[-2]), int(input_size[-1])
    parameter = next(model.parameters())
    device = parameter.device
    dtype = parameter.dtype
    cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()] \
        if device.type == "cuda" else []

    def forward_logits(batch):
        output = model(batch)
        return _as_logits(output)

    with torch.random.fork_rng(devices=cuda_devices), torch.inference_mode():
        sample = torch.randn((1, 3, height, width), device=device, dtype=dtype)
        before = forward_logits(sample).float()
        fused_count = 0

        def recurse(parent):
            nonlocal fused_count
            for child in list(parent.children()):
                recurse(child)
            children = list(parent.named_children())
            for index in range(1, len(children)):
                previous_name, previous = children[index - 1]
                current_name, current = children[index]
                if isinstance(previous, nn.Conv2d) and isinstance(current, nn.BatchNorm2d):
                    setattr(parent, previous_name, fuse_conv_bn_eval(previous, current))
                    setattr(parent, current_name, nn.Identity())
                    fused_count += 1

        recurse(model)
        if fused_count == 0:
            model.lab_bn_fused = False
            print("No adjacent Conv2d/BatchNorm2d pairs were found.")
            return model
        after = forward_logits(sample).float()
        max_error = float((before - after).abs().max().item())
        model.lab_bn_fused = True
        model.lab_bn_fuse_max_error = max_error
        print(f"Fused {fused_count} Conv-BN pairs; max absolute output error={max_error:.3g}")
    return model
