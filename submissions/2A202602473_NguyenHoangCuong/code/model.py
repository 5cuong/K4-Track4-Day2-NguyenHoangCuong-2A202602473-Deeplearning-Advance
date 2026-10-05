"""timm model construction, freezing, optimizer groups, and complexity estimates."""
from __future__ import annotations

import warnings

SUGGESTED_BACKBONES = {
    "resnet50": "resnet50",
    "resnext50": "resnext50_32x4d",
    "convnext_tiny": "convnext_tiny",
    "deit_small": "deit_small_patch16_224",
    "swin_tiny": "swin_tiny_patch4_window7_224",
    "efficientnet_b0": "efficientnet_b0",
    "mobilenetv3": "mobilenetv3_large_100",
}


def _classifier_module(model):
    getter = getattr(model, "get_classifier", None)
    if callable(getter):
        classifier = getter()
        if classifier is not None:
            return classifier
    for name in ("classifier", "fc", "head", "head.fc"):
        current = model
        try:
            for part in name.split("."):
                current = getattr(current, part)
            if current is not None:
                return current
        except AttributeError:
            continue
    raise ValueError(f"Could not locate the classifier head for {type(model).__name__}.")


def build_model(name: str, pretrained: bool = True, num_classes: int = 9,
                drop_rate: float = 0.0, init: str = "finetune"):
    """Create a timm classifier and optionally freeze its feature extractor."""
    if init not in {"scratch", "frozen", "finetune"}:
        raise ValueError("init must be scratch, frozen, or finetune.")
    if num_classes <= 1:
        raise ValueError("num_classes must be greater than one.")
    import timm

    model_name = SUGGESTED_BACKBONES.get(name, name)
    use_pretrained = bool(pretrained) and init != "scratch"
    model = timm.create_model(
        model_name, pretrained=use_pretrained, num_classes=num_classes, drop_rate=drop_rate
    )
    model.lab_model_name = model_name
    model.lab_pretrained = use_pretrained
    model.lab_pretrained_cfg = dict(getattr(model, "pretrained_cfg", {}) or {})
    if init == "frozen":
        freeze_backbone(model)
    return model


def freeze_backbone(model) -> None:
    """Freeze all feature parameters while leaving the classifier head trainable."""
    classifier = _classifier_module(model)
    head_ids = {id(parameter) for parameter in classifier.parameters()}
    if not head_ids:
        raise ValueError("The classifier head has no parameters.")
    for parameter in model.parameters():
        parameter.requires_grad_(id(parameter) in head_ids)
    model.lab_frozen = True
    model.lab_head_parameter_ids = head_ids


def param_groups(model, lr_backbone: float, lr_head: float, weight_decay: float):
    """Split trainable parameters into backbone matrices, backbone norm/bias, and head."""
    if min(lr_backbone, lr_head) < 0 or weight_decay < 0:
        raise ValueError("Learning rates and weight decay must be non-negative.")
    classifier = _classifier_module(model)
    head_ids = {id(parameter) for parameter in classifier.parameters()}
    groups = {
        "backbone_decay": [],
        "backbone_no_decay": [],
        "head": [],
    }
    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue
        if id(parameter) in head_ids:
            groups["head"].append(parameter)
        elif parameter.ndim <= 1:
            groups["backbone_no_decay"].append(parameter)
        else:
            groups["backbone_decay"].append(parameter)

    specs = (
        (groups["backbone_decay"], lr_backbone, weight_decay),
        (groups["backbone_no_decay"], lr_backbone, 0.0),
        (groups["head"], lr_head, weight_decay),
    )
    return [
        {"params": params, "lr": lr, "weight_decay": decay}
        for params, lr, decay in specs if params
    ]


def count_params(model) -> float:
    """Count all model parameters, including frozen ones, in millions."""
    return sum(parameter.numel() for parameter in model.parameters()) / 1_000_000.0


def count_gmacs(model, img_size: int = 224) -> float:
    """Estimate MACs with fvcore; return GMAC (one multiply-accumulate is one MAC)."""
    if img_size <= 0:
        raise ValueError("img_size must be positive.")
    try:
        import torch
        from collections import Counter
        from fvcore.nn import FlopCountAnalysis
    except ImportError as exc:
        raise RuntimeError(
            "Install fvcore to record GMACs (pip install fvcore); no partial estimate is returned."
        ) from exc

    parameter = next(model.parameters())
    device, was_training = parameter.device, model.training
    model.eval()
    sample = torch.zeros((1, 3, img_size, img_size), device=device, dtype=parameter.dtype)
    try:
        with torch.inference_mode(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            analysis = FlopCountAnalysis(model, sample)
            def matmul_macs(inputs, outputs):
                left = inputs[0].type().sizes()
                output = outputs[0].type().sizes()
                if not left or not output or left[-1] is None or any(dim is None for dim in output):
                    return Counter()
                output_elements = 1
                for dimension in output:
                    output_elements *= int(dimension)
                return Counter({"matmul": output_elements * int(left[-1])})

            analysis.set_op_handle(
                "aten::matmul", matmul_macs,
                "aten::bmm", matmul_macs,
                "aten::mm", matmul_macs,
            )
            analysis.unsupported_ops_warnings(False)
            analysis.uncalled_modules_warnings(False)
            macs = float(analysis.total())
            model.lab_gmac_unsupported_ops = sorted(analysis.unsupported_ops())
    except Exception as exc:
        raise RuntimeError(
            f"fvcore could not estimate MACs for {getattr(model, 'lab_model_name', type(model).__name__)} "
            f"at {img_size}px: {exc}"
        ) from exc
    finally:
        model.train(was_training)
    if macs <= 0:
        raise RuntimeError("fvcore returned a non-positive MAC count.")
    model.lab_gmac_method = "fvcore FlopCountAnalysis"
    return macs / 1_000_000_000.0
