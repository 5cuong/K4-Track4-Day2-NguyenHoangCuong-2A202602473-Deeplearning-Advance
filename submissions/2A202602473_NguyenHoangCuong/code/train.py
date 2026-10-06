"""One reproducible training path for backbone sweeps, ablations, and final runs."""
from __future__ import annotations

import argparse
import copy
import importlib
import importlib.util
import json
import os
import random
import sys
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

import dataset as data_module
import inference as inference_module
import losses as loss_module
import model as model_module


@dataclass
class Config:
    exp_id: str = "T00"
    seed: int = 0
    fold: int = 0
    backbone: str = "resnet50"
    pretrained: bool = True
    init: str = "finetune"
    drop_rate: float = 0.0
    img_size: int = 224
    aug: str = "basic"
    sampler: str | None = None
    mix: str | None = None
    mix_alpha: float = 1.0
    loss: str = "ce"
    label_smoothing: float = 0.0
    focal_gamma: float = 2.0
    class_weight_beta: float | None = None
    optimizer: str = "adamw"
    epochs: int = 12
    batch_size: int = 64
    lr_backbone: float = 1e-4
    lr_head: float = 1e-3
    weight_decay: float = 0.05
    warmup_epochs: float = 1.0
    ema_decay: float | None = None
    amp: bool = True
    channels_last: bool = True
    grad_clip_norm: float = 1.0
    num_workers: int = 2
    images_dir: str = "data/images"
    labels_dir: str = "data/labels"
    out_dir: str = "runs"
    pred_dir: str = "predictions"
    curves_dir: str = "curves"
    save_test_predictions: bool = False
    save_uncalibrated_predictions: bool = False
    temperature_scaling: bool = False
    tta: str = "identity"             # identity | hflip | multicrop | multiscale
    tta_space: str = "prob"            # prob | logit
    tta_scales: str = "224,256"
    resume: bool = True


_RUNTIME_FIELDS = {
    "save_test_predictions", "save_uncalibrated_predictions",
    "temperature_scaling", "tta", "tta_space", "tta_scales", "resume",
}


def run_dir(cfg: Config) -> Path:
    return Path(cfg.out_dir) / cfg.exp_id / f"seed{cfg.seed}"


def pred_path(cfg: Config, split: str) -> Path:
    if split not in {"val", "test"}:
        raise ValueError("split must be val or test.")
    return Path(cfg.pred_dir) / f"{cfg.exp_id}_seed{cfg.seed}_{split}.csv"


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, PyTorch, CUDA, and deterministic backend behavior."""
    if seed < 0:
        raise ValueError("seed must be non-negative.")
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        torch.use_deterministic_algorithms(True)


def build_optimizer(model, cfg: Config):
    """Build AdamW or SGD using the backbone/head parameter groups."""
    groups = model_module.param_groups(
        model, cfg.lr_backbone, cfg.lr_head, cfg.weight_decay
    )
    if cfg.optimizer.lower() == "adamw":
        return torch.optim.AdamW(groups, betas=(0.9, 0.999))
    if cfg.optimizer.lower() == "sgd":
        return torch.optim.SGD(groups, momentum=0.9, nesterov=True)
    raise ValueError("optimizer must be adamw or sgd.")


def build_scheduler(optimizer, cfg: Config, steps_per_epoch: int):
    """Linear warmup, followed by cosine decay, updated once per optimizer step."""
    if steps_per_epoch <= 0 or cfg.epochs <= 0:
        raise ValueError("Training must contain at least one step and one epoch.")
    total_steps = cfg.epochs * steps_per_epoch
    warmup_steps = min(total_steps, max(1, int(round(cfg.warmup_epochs * steps_per_epoch))))

    def scale(step):
        if step < warmup_steps:
            return max(1e-3, float(step + 1) / warmup_steps)
        remaining = max(1, total_steps - warmup_steps)
        progress = min(1.0, max(0.0, (step - warmup_steps) / remaining))
        return 0.5 * (1.0 + float(np.cos(np.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, scale)


class EMA:
    """Exponential moving average of model parameters and floating-point buffers."""

    def __init__(self, model, decay: float):
        if not 0.0 < decay < 1.0:
            raise ValueError("EMA decay must be in (0, 1).")
        self.decay = float(decay)
        self.shadow = copy.deepcopy(model).eval()
        self.shadow.requires_grad_(False)
        self.steps = 0

    @torch.no_grad()
    def update(self, model) -> None:
        source = model.state_dict()
        target = self.shadow.state_dict()
        for name, averaged in target.items():
            current = source[name].detach().to(device=averaged.device)
            if averaged.is_floating_point():
                averaged.lerp_(current, 1.0 - self.decay)
            else:
                averaged.copy_(current)
        self.steps += 1

    def copy_to(self, model) -> None:
        model.load_state_dict(self.shadow.state_dict())


def _set_train_mode(model, frozen: bool) -> None:
    if not frozen:
        model.train()
        return
    classifier = model_module._classifier_module(model)
    model.eval()
    classifier.train()


def train_one_epoch(model, loader, criterion, optimizer, scheduler, scaler, cfg: Config,
                    device, ema: EMA | None = None) -> dict:
    """Run one epoch with optional AMP, Mixup/CutMix, gradient clipping, and EMA."""
    amp_enabled = bool(cfg.amp and device.type == "cuda")
    _set_train_mode(model, frozen=cfg.init == "frozen")
    total_loss = 0.0
    seen = 0
    skipped_optimizer_steps = 0
    started = time.perf_counter()
    for images, target, _filenames in loader:
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        if cfg.channels_last and device.type == "cuda":
            images = images.contiguous(memory_format=torch.channels_last)

        mixed_targets = target
        if cfg.mix:
            images, mixed_targets = loss_module.mix_batch(
                images, target, alpha=cfg.mix_alpha, mode=cfg.mix
            )
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type="cuda", dtype=torch.float16, enabled=amp_enabled
        ):
            logits = model(images)
            loss = loss_module.mixed_loss(criterion, logits, mixed_targets)
        scaler.scale(loss).backward()
        if cfg.grad_clip_norm is not None and cfg.grad_clip_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
        scale_before = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        optimizer_step_skipped = scaler.get_scale() < scale_before
        if optimizer_step_skipped:
            skipped_optimizer_steps += 1
        else:
            scheduler.step()
            if ema is not None:
                ema.update(model)
        batch_size = int(target.shape[0])
        total_loss += float(loss.detach().item()) * batch_size
        seen += batch_size

    return {
        "train_loss": total_loss / max(1, seen),
        "lr_backbone": float(optimizer.param_groups[0]["lr"]),
        "epoch_seconds": time.perf_counter() - started,
        "skipped_optimizer_steps": skipped_optimizer_steps,
    }


@torch.inference_mode()
def evaluate(model, loader, criterion, device):
    """Evaluate without augmentation or gradients; return stable filenames and raw logits."""
    device = torch.device(device)
    model.eval()
    names, labels, logits_list = [], [], []
    total_loss = 0.0
    seen = 0
    for images, target, filenames in loader:
        images = images.to(device, non_blocking=True)
        target_device = target.to(device, non_blocking=True)
        output = model(images)
        loss = criterion(output, target_device)
        total_loss += float(loss.item()) * int(target.shape[0])
        seen += int(target.shape[0])
        names.extend(str(filename) for filename in filenames)
        labels.append(target.cpu().numpy())
        logits_list.append(output.float().cpu().numpy())
    if not logits_list:
        raise ValueError("Cannot evaluate an empty data loader.")
    return (
        names,
        np.concatenate(labels).astype(np.int64),
        np.concatenate(logits_list, axis=0),
        total_loss / max(1, seen),
    )


def plot_curves(history: list[dict], path: str | Path, title: str) -> None:
    """Save a readable per-run figure with train/validation loss and validation metrics."""
    import matplotlib.pyplot as plt

    if not history:
        raise ValueError("history must contain at least one epoch.")
    frame = pd.DataFrame(history)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    epochs = frame["epoch"].to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(epochs, frame["train_loss"], marker="o", label="Train loss")
    axes[0].plot(epochs, frame["val_loss"], marker="o", label="Validation loss")
    axes[0].set(title="Loss", xlabel="Epoch", ylabel="Cross-entropy")
    axes[0].legend()
    axes[1].plot(epochs, frame["val_macro_f1"], marker="o", label="Validation macro-F1")
    if "val_top1" in frame:
        axes[1].plot(epochs, frame["val_top1"], marker=".", label="Validation top-1")
    axes[1].set(title="Validation quality", xlabel="Epoch", ylabel="Score")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(destination, dpi=160, bbox_inches="tight")
    plt.close(fig)


def parse_overrides(pairs: list[str]) -> dict:
    """Parse KEY=VALUE overrides and coerce values from Config type annotations."""
    from typing import get_type_hints, get_origin, get_args, Union

    hints = get_type_hints(Config)
    allowed = {field.name for field in fields(Config)}
    result = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"Expected KEY=VALUE, got {pair!r}.")
        key, raw = pair.split("=", 1)
        key, raw = key.strip(), raw.strip()
        if key not in allowed:
            raise ValueError(f"Unknown Config field {key!r}; valid fields: {sorted(allowed)}")
        annotation = hints[key]
        choices = get_args(annotation)
        optional = type(None) in choices
        base = next((item for item in choices if item is not type(None)), annotation)
        if raw.lower() in {"none", "null"} and optional:
            value = None
        elif base is bool:
            if raw.lower() not in {"true", "false", "1", "0", "yes", "no"}:
                raise ValueError(f"{key} expects a boolean, got {raw!r}.")
            value = raw.lower() in {"true", "1", "yes"}
        elif base is int:
            value = int(raw)
        elif base is float:
            value = float(raw)
        elif base is str:
            value = raw
        else:
            value = raw
        result[key] = value
    return result


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.device):
        return str(value)
    return str(value)


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_torch_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _load_eval_module():
    candidates = [Path(__file__).resolve().parents[3], Path.cwd(), *Path.cwd().parents]
    for root in candidates:
        if (root / "eval.py").is_file():
            root_string = str(root.resolve())
            if root_string not in sys.path:
                sys.path.insert(0, root_string)
            return importlib.import_module("eval")
    raise FileNotFoundError("Could not locate the repository's unchanged eval.py.")


def _training_fingerprint(cfg: Config) -> dict:
    return {
        key: value for key, value in asdict(cfg).items()
        if key not in _RUNTIME_FIELDS
    }


def _load_existing_run(cfg: Config, output: Path, eval_module, device):
    config_path = output / "config.json"
    if not config_path.exists():
        return None
    existing = json.loads(config_path.read_text(encoding="utf-8"))
    if existing.get("training_fingerprint") != _training_fingerprint(cfg):
        raise ValueError(
            f"{output} already belongs to a different training configuration; "
            "choose a new exp_id or seed."
        )
    done_path = output / "done.json"
    if not done_path.exists():
        return None
    summary = json.loads(done_path.read_text(encoding="utf-8"))
    best_path = output / "checkpoint_best.pt"
    checkpoint = torch.load(best_path, map_location=device, weights_only=True)
    model = model_module.build_model(
        cfg.backbone, pretrained=False, num_classes=9,
        drop_rate=cfg.drop_rate, init=cfg.init,
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    if cfg.channels_last and device.type == "cuda":
        model.to(memory_format=torch.channels_last)
    return summary, model


def _metrics(eval_module, labels, logits):
    probs = torch.as_tensor(logits, dtype=torch.float64).softmax(dim=1).numpy()
    predicted = probs.argmax(axis=1)
    return eval_module.compute_metrics(labels, predicted, probs), probs


def _predict_configured(model, loader, device, cfg: Config):
    if cfg.tta == "identity":
        names, labels, logits, _ = evaluate(
            model, loader, torch.nn.CrossEntropyLoss(), device
        )
        return names, labels, logits
    if cfg.tta == "hflip":
        builder = lambda images: [images, inference_module.view_hflip(images)]
    elif cfg.tta == "multicrop":
        builder = lambda images: inference_module.views_multicrop(images, cfg.img_size)
    elif cfg.tta == "multiscale":
        scales = [int(value.strip()) for value in cfg.tta_scales.split(",") if value.strip()]
        builder = lambda images: inference_module.views_multiscale(images, scales)
    else:
        raise ValueError("tta must be identity, hflip, multicrop, or multiscale.")
    return inference_module.predict_multi_logits(
        model, loader, device, builder, space=cfg.tta_space
    )


def _write_predictions(eval_module, path, names, labels, probs):
    return str(eval_module.save_predictions(path, names, labels, probs))


def run(cfg: Config) -> dict:
    """Train/select by validation macro-F1 and optionally perform one final test pass."""
    if cfg.epochs <= 0 or cfg.batch_size <= 0:
        raise ValueError("epochs and batch_size must be positive.")
    if cfg.fold != 0:
        raise ValueError("The required submission uses the official fold 0.")
    if cfg.save_test_predictions and not (cfg.exp_id == "T00" or cfg.exp_id.startswith("F")):
        raise ValueError("Test predictions are reserved for the locked T00 baseline or F* final runs.")
    if cfg.save_uncalibrated_predictions and not cfg.save_test_predictions:
        raise ValueError("Uncalibrated test predictions require save_test_predictions=True.")
    if cfg.tta not in {"identity", "hflip", "multicrop", "multiscale"}:
        raise ValueError("tta must be identity, hflip, multicrop, or multiscale.")
    if cfg.tta_space not in {"prob", "logit"}:
        raise ValueError("tta_space must be prob or logit.")
    if cfg.save_test_predictions and pred_path(cfg, "test").exists():
        output = run_dir(cfg)
        config_path = output / "config.json"
        done_path = output / "done.json"
        if config_path.is_file() and done_path.is_file():
            saved_config = json.loads(config_path.read_text(encoding="utf-8"))
            saved_summary = json.loads(done_path.read_text(encoding="utf-8"))
            if (
                saved_config.get("training_fingerprint") == _training_fingerprint(cfg)
                and saved_summary.get("test_prediction") == str(pred_path(cfg, "test"))
            ):
                return saved_summary
        raise FileExistsError(
            f"Test prediction already exists at {pred_path(cfg, 'test')}; it will not be evaluated or overwritten."
        )

    set_seed(cfg.seed)
    output = run_dir(cfg)
    output.mkdir(parents=True, exist_ok=True)
    eval_module = _load_eval_module()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    existing = _load_existing_run(cfg, output, eval_module, device) if cfg.resume else None
    completed_model = None
    if existing is not None:
        summary, completed_model = existing
    else:
        summary = None

    if summary is None:
        train_df, val_df, test_df = data_module.load_split(cfg.labels_dir, cfg.fold)
        split_report = data_module.check_split(train_df, val_df, test_df, cfg.images_dir)
        train_transform = data_module.build_transforms(True, cfg.img_size, cfg.aug)
        eval_transform = data_module.build_transforms(False, cfg.img_size)
        train_loader = data_module.make_loader(
            train_df, cfg.images_dir, train_transform, cfg.batch_size, True,
            sampler=cfg.sampler, num_workers=cfg.num_workers,
        )
        val_loader = data_module.make_loader(
            val_df, cfg.images_dir, eval_transform, cfg.batch_size, False,
            num_workers=cfg.num_workers,
        )

        model = model_module.build_model(
            cfg.backbone, pretrained=cfg.pretrained, num_classes=9,
            drop_rate=cfg.drop_rate, init=cfg.init,
        ).to(device)
        if cfg.channels_last and device.type == "cuda":
            model.to(memory_format=torch.channels_last)
        criterion_weight = None
        if cfg.loss in {"ce_weighted", "focal"} and cfg.class_weight_beta is not None:
            counts = train_df["Label"].astype(int).value_counts().reindex(range(9), fill_value=0)
            criterion_weight = loss_module.class_weights(counts, cfg.class_weight_beta).to(device)
        if cfg.loss == "ce_weighted":
            criterion = loss_module.build_criterion("ce_weighted", weight=criterion_weight)
        elif cfg.loss == "focal":
            criterion = loss_module.build_criterion(
                "focal", gamma=cfg.focal_gamma, alpha=criterion_weight
            )
        elif cfg.loss == "ls":
            criterion = loss_module.build_criterion(
                "ls", smoothing=cfg.label_smoothing
            )
        else:
            criterion = loss_module.build_criterion(cfg.loss)
        criterion = criterion.to(device)

        optimizer = build_optimizer(model, cfg)
        scheduler = build_scheduler(optimizer, cfg, len(train_loader))
        amp_enabled = bool(cfg.amp and device.type == "cuda")
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
        ema = EMA(model, cfg.ema_decay) if cfg.ema_decay is not None else None
        history_path = output / "history.csv"
        history = pd.read_csv(history_path).to_dict("records") if history_path.exists() else []
        best_score = -float("inf")
        best_epoch = None
        start_epoch = 0
        last_path = output / "checkpoint_last.pt"

        config_path = output / "config.json"
        fingerprint = _training_fingerprint(cfg)
        if config_path.exists():
            existing_config = json.loads(config_path.read_text(encoding="utf-8"))
            if existing_config.get("training_fingerprint") != fingerprint:
                raise ValueError(
                    f"{output} already belongs to a different training configuration; "
                    "choose a new exp_id or seed."
                )
        else:
            pretrained_cfg = getattr(model, "lab_pretrained_cfg", {})
            _atomic_json(config_path, {
                "config": asdict(cfg),
                "training_fingerprint": fingerprint,
                "pretrained_cfg": pretrained_cfg,
                "torch": torch.__version__,
                "device": str(device),
            })

        if cfg.resume and last_path.exists():
            checkpoint = torch.load(last_path, map_location=device, weights_only=True)
            model.load_state_dict(checkpoint["model_state"])
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            scheduler.load_state_dict(checkpoint["scheduler_state"])
            scaler.load_state_dict(checkpoint["scaler_state"])
            if ema is not None and checkpoint.get("ema_state") is not None:
                ema.shadow.load_state_dict(checkpoint["ema_state"])
                ema.steps = int(checkpoint.get("ema_steps", 0))
            start_epoch = int(checkpoint["epoch"]) + 1
            best_score = float(checkpoint.get("best_macro_f1", best_score))
            best_epoch = checkpoint.get("best_epoch")
            if not history:
                history = checkpoint.get("history", [])

        best_path = output / "checkpoint_best.pt"
        for epoch in range(start_epoch, cfg.epochs):
            train_stats = train_one_epoch(
                model, train_loader, criterion, optimizer, scheduler, scaler,
                cfg, device, ema=ema,
            )
            evaluation_model = ema.shadow if ema is not None else model
            val_names, val_labels, val_logits, val_loss = evaluate(
                evaluation_model, val_loader, criterion, device
            )
            metrics, val_probs = _metrics(eval_module, val_labels, val_logits)
            row = {
                "epoch": epoch + 1,
                **train_stats,
                "val_loss": val_loss,
                "val_macro_f1": float(metrics["macro_f1"]),
                "val_top1": float(metrics["top1"]),
                "val_balanced_acc": float(metrics["balanced_acc"]),
                "val_ece": float(metrics["ece"]),
                "lr_head": float(optimizer.param_groups[-1]["lr"]),
            }
            history.append(row)
            pd.DataFrame(history).to_csv(history_path, index=False)
            if metrics["macro_f1"] > best_score:
                best_score = float(metrics["macro_f1"])
                best_epoch = epoch + 1
                best_state = copy.deepcopy(evaluation_model.state_dict())
                _atomic_torch_save({
                    "model_state": best_state,
                    "best_epoch": best_epoch,
                    "best_macro_f1": best_score,
                    "ema": ema is not None,
                }, best_path)
                np.savez_compressed(
                    output / "best_val_logits.npz",
                    filenames=np.asarray(val_names, dtype=str),
                    y_true=val_labels,
                    logits=val_logits,
                )
            _atomic_torch_save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "scaler_state": scaler.state_dict(),
                "ema_state": ema.shadow.state_dict() if ema is not None else None,
                "ema_steps": ema.steps if ema is not None else 0,
                "best_macro_f1": best_score,
                "best_epoch": best_epoch,
                "history": history,
            }, last_path)

        if best_epoch is None:
            if not best_path.exists():
                raise RuntimeError(
                    f"No new epoch was run and no best checkpoint exists at {best_path}."
                )
            best_checkpoint = torch.load(best_path, map_location=device, weights_only=True)
            best_score = float(best_checkpoint["best_macro_f1"])
            best_epoch = int(best_checkpoint["best_epoch"])
        best_checkpoint = torch.load(best_path, map_location=device, weights_only=True)
        model.load_state_dict(best_checkpoint["model_state"])
        if cfg.channels_last and device.type == "cuda":
            model.to(memory_format=torch.channels_last)
        val_blob = np.load(output / "best_val_logits.npz", allow_pickle=False)
        val_names = val_blob["filenames"].tolist()
        val_labels = val_blob["y_true"]
        val_logits = val_blob["logits"]
        metrics, _ = _metrics(eval_module, val_labels, val_logits)

        if history:
            curve_path = Path(cfg.curves_dir) / f"{cfg.exp_id}_seed{cfg.seed}.png"
            plot_curves(
                history, curve_path,
                f"{cfg.exp_id} | {cfg.backbone} | seed {cfg.seed}",
            )
        try:
            gmacs = model_module.count_gmacs(model, cfg.img_size)
            gmac_error = None
        except Exception as exc:
            gmacs = None
            gmac_error = str(exc)
        summary = {
            "exp_id": cfg.exp_id,
            "seed": cfg.seed,
            "backbone": cfg.backbone,
            "pretrained_tag": (
                getattr(model, "lab_pretrained_cfg", {}).get("tag")
                or getattr(model, "lab_pretrained_cfg", {}).get("hf_hub_id")
            ),
            "best_epoch": int(best_epoch),
            "macro_f1_val": float(metrics["macro_f1"]),
            "top1_val": float(metrics["top1"]),
            "balanced_acc_val": float(metrics["balanced_acc"]),
            "ece_val": float(metrics["ece"]),
            "best_val_loss": float(pd.DataFrame(history).loc[
                pd.DataFrame(history)["epoch"] == int(best_epoch), "val_loss"
            ].iloc[0]) if history and (pd.DataFrame(history)["epoch"] == int(best_epoch)).any() else None,
            "train_seconds_per_epoch_mean": float(
                pd.DataFrame(history).loc[
                    pd.DataFrame(history)["epoch"] <= int(best_epoch), "epoch_seconds"
                ].mean()
            ) if history else None,
            "parameters_m": float(model_module.count_params(model)),
            "gmac": gmacs,
            "gmac_error": gmac_error,
            "gmac_unsupported_ops": getattr(model, "lab_gmac_unsupported_ops", []),
            "img_size": cfg.img_size,
            "temperature": 1.0,
            "temperature_scaling": False,
            "val_prediction": str(pred_path(cfg, "val")),
            "split": split_report,
            "config": asdict(cfg),
        }
        _atomic_json(output / "done.json", summary)
        completed_model = model

    if cfg.tta == "identity":
        val_blob = np.load(run_dir(cfg) / "best_val_logits.npz", allow_pickle=False)
        val_names = val_blob["filenames"].tolist()
        val_labels = val_blob["y_true"]
        val_logits = val_blob["logits"]
    else:
        _, val_df, _ = data_module.load_split(cfg.labels_dir, cfg.fold)
        eval_size = cfg.img_size + 32 if cfg.tta == "multicrop" else cfg.img_size
        val_loader = data_module.make_loader(
            val_df, cfg.images_dir, data_module.build_transforms(False, eval_size),
            cfg.batch_size, False, num_workers=cfg.num_workers,
        )
        val_names, val_labels, val_logits = _predict_configured(
            completed_model, val_loader, device, cfg
        )
    uncalibrated = torch.as_tensor(val_logits, dtype=torch.float64).softmax(dim=1).numpy()
    temperature = (
        inference_module.fit_temperature(val_logits, val_labels)
        if cfg.temperature_scaling else 1.0
    )
    calibrated = inference_module.apply_temperature(val_logits, temperature)
    metrics = eval_module.compute_metrics(val_labels, calibrated.argmax(axis=1), calibrated)
    _write_predictions(eval_module, pred_path(cfg, "val"), val_names, val_labels, calibrated)
    if cfg.temperature_scaling:
        _write_predictions(
            eval_module,
            Path(cfg.pred_dir) / f"{cfg.exp_id}_seed{cfg.seed}_val_uncal.csv",
            val_names, val_labels, uncalibrated,
        )
    summary.update({
        "macro_f1_val": float(metrics["macro_f1"]),
        "top1_val": float(metrics["top1"]),
        "balanced_acc_val": float(metrics["balanced_acc"]),
        "ece_val": float(metrics["ece"]),
        "temperature": float(temperature),
        "temperature_scaling": bool(cfg.temperature_scaling),
        "tta": cfg.tta,
        "tta_space": cfg.tta_space,
        "config": asdict(cfg),
    })
    _atomic_json(run_dir(cfg) / "done.json", summary)

    if cfg.save_test_predictions:
        test_path = pred_path(cfg, "test")
        _, _, test_df = data_module.load_split(cfg.labels_dir, cfg.fold)
        eval_size = cfg.img_size + 32 if cfg.tta == "multicrop" else cfg.img_size
        test_loader = data_module.make_loader(
            test_df, cfg.images_dir,
            data_module.build_transforms(False, eval_size),
            cfg.batch_size, False, num_workers=cfg.num_workers,
        )
        test_names, test_labels, test_logits = _predict_configured(
            completed_model, test_loader, device, cfg
        )
        uncalibrated = torch.as_tensor(test_logits, dtype=torch.float64).softmax(dim=1).numpy()
        calibrated = inference_module.apply_temperature(
            test_logits, float(summary.get("temperature", 1.0))
        )
        _write_predictions(eval_module, test_path, test_names, test_labels, calibrated)
        np.savez_compressed(
            run_dir(cfg) / "test_logits.npz",
            filenames=np.asarray(test_names, dtype=str),
            y_true=test_labels,
            logits=test_logits,
        )
        if cfg.save_uncalibrated_predictions:
            _write_predictions(
                eval_module,
                Path(cfg.pred_dir) / f"{cfg.exp_id}_seed{cfg.seed}_test_uncal.csv",
                test_names, test_labels, uncalibrated,
            )
        summary["test_prediction"] = str(test_path)
        _atomic_json(run_dir(cfg) / "done.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one DeepWeeds experiment.")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()
    cfg = Config(**parse_overrides(args.set))
    print(json.dumps(run(cfg), indent=2, ensure_ascii=False, default=_json_default))


if __name__ == "__main__":
    main()
