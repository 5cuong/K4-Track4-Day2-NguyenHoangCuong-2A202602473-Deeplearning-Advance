"""Synchronized inference latency measurement with warmup and percentile reports."""
from __future__ import annotations

import time


def bench(fn, warmup: int = 10, iters: int = 100, sync=None) -> dict:
    """Measure a callable in milliseconds after warmup; require rubric-compliant counts."""
    import numpy as np

    if warmup < 10:
        raise ValueError("At least 10 warmup calls are required.")
    if iters < 50:
        raise ValueError("At least 50 timed iterations are required.")
    for _ in range(warmup):
        fn()
    if sync is not None:
        sync()
    samples = []
    for _ in range(iters):
        if sync is not None:
            sync()
        start = time.perf_counter()
        fn()
        if sync is not None:
            sync()
        samples.append((time.perf_counter() - start) * 1000.0)
    values = np.asarray(samples, dtype=np.float64)
    return {
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "mean": float(values.mean()),
        "n": int(iters),
        "warmup": int(warmup),
    }


def _prepare_model_input(model, batch_size, img_size, dtype, device):
    import copy
    import torch

    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    if dtype not in {"fp32", "amp", "fp16"}:
        raise ValueError("dtype must be fp32, amp, or fp16.")
    if dtype in {"amp", "fp16"} and device.type != "cuda":
        raise ValueError(f"{dtype} latency requires a CUDA device.")
    measured_model = copy.deepcopy(model) if dtype == "fp16" else model
    measured_model.to(device).eval()
    if dtype == "fp16":
        measured_model.half()
    sample = torch.randn(batch_size, 3, img_size, img_size, device=device)
    if dtype == "fp16":
        sample = sample.half()
    enabled = dtype == "amp"

    def call():
        with torch.inference_mode(), torch.autocast(
            device_type="cuda", dtype=torch.float16, enabled=enabled
        ):
            measured_model(sample)

    sync = torch.cuda.synchronize if device.type == "cuda" else None
    gpu = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    return measured_model, sample, call, sync, gpu, device


def latency_report(model, batch_size: int, img_size: int, dtype: str = "fp32", device: str = "cuda",
                   warmup: int = 10, iters: int = 100) -> dict:
    """Measure a batch-one or batch-N model forward with CUDA synchronization when applicable."""
    import torch

    measured_model, sample, call, sync, gpu, device_obj = _prepare_model_input(
        model, batch_size, img_size, dtype, device
    )
    timing = bench(call, warmup=warmup, iters=iters, sync=sync)
    p50 = timing["p50"]
    return {
        "gpu": gpu,
        "dtype": dtype,
        "batch": int(batch_size),
        "img_size": int(img_size),
        "p50": p50,
        "p95": timing["p95"],
        "p99": timing["p99"],
        "mean": timing["mean"],
        "n": timing["n"],
        "warmup": timing["warmup"],
        "images_per_s": float(batch_size / (p50 / 1000.0)) if p50 > 0 else float("inf"),
        "torch": torch.__version__,
        "bn_fused": bool(getattr(measured_model, "lab_bn_fused", False)),
        "includes_preprocessing": False,
        "device": str(device_obj),
    }


def tta_latency(model, k_views: int, **kw) -> dict:
    """Measure actual K-forward TTA cost on the same batch (not an extrapolation)."""
    import torch

    if k_views <= 0:
        raise ValueError("k_views must be positive.")
    batch_size = int(kw.pop("batch_size", 1))
    img_size = int(kw.pop("img_size", 224))
    dtype = kw.pop("dtype", "fp32")
    device = kw.pop("device", "cuda")
    warmup = int(kw.pop("warmup", 10))
    iters = int(kw.pop("iters", 100))
    if kw:
        raise TypeError(f"Unexpected latency arguments: {sorted(kw)}")

    measured_model, sample, _, sync, gpu, device_obj = _prepare_model_input(
        model, batch_size, img_size, dtype, device
    )

    def call():
        with torch.inference_mode(), torch.autocast(
            device_type="cuda", dtype=torch.float16, enabled=dtype == "amp"
        ):
            for _ in range(k_views):
                measured_model(sample)

    timing = bench(call, warmup=warmup, iters=iters, sync=sync)
    p50 = timing["p50"]
    return {
        "gpu": gpu,
        "dtype": dtype,
        "batch": batch_size,
        "img_size": img_size,
        "k_views": int(k_views),
        "p50": p50,
        "p95": timing["p95"],
        "p99": timing["p99"],
        "mean": timing["mean"],
        "n": timing["n"],
        "warmup": timing["warmup"],
        "images_per_s": float(batch_size / (p50 / 1000.0)) if p50 > 0 else float("inf"),
        "torch": torch.__version__,
        "bn_fused": bool(getattr(measured_model, "lab_bn_fused", False)),
        "includes_preprocessing": False,
        "device": str(device_obj),
    }
