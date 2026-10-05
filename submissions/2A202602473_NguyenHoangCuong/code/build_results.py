"""Build results.xlsx and a fact-based report from the saved training artifacts."""
from __future__ import annotations

import importlib
import json
import math
import re
import sys
from pathlib import Path
from statistics import mean, stdev

import numpy as np

SUBMISSION_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = SUBMISSION_DIR.parents[1]
RUNS_DIR = SUBMISSION_DIR / "runs"
PREDICTIONS_DIR = SUBMISSION_DIR / "predictions"
WORKBOOK = SUBMISSION_DIR / "results.xlsx"
REPORT = SUBMISSION_DIR / "report.md"

HEADERS = {
    "Backbones": ["exp_id", "backbone", "pretrained_tag", "parameters_m", "GMAC",
        "resolution_px", "epochs", "seed", "macro_f1_val", "top1_val",
        "train_seconds_per_epoch", "latency_batch1_ms", "notes"],
    "Training": ["exp_id", "backbone", "changed_axis", "difference_from_T00", "seed",
        "macro_f1_val", "top1_val", "delta_macro_f1_vs_T00", "rare_class_f1", "notes"],
    "Inference": ["exp_id", "method", "model_checkpoint", "K", "macro_f1_val", "top1_val",
        "ECE_val", "latency_p50_ms_batch1", "latency_p95_ms_batch1",
        "latency_p99_ms_batch1", "throughput_images_s", "relative_cost_vs_I00",
        "GPU", "dtype", "BN_fused", "notes"],
    "Final": ["exp_id", "configuration", "seed", "macro_f1_val", "macro_f1_test",
        "top1_test", "ECE_test_uncal", "ECE_test", "macro_f1_test_mean",
        "macro_f1_test_std", "top1_test_mean", "top1_test_std",
        "ECE_test_uncal_mean", "ECE_test_uncal_std", "ECE_test_mean",
        "ECE_test_std", "notes"],
    "PerClass": ["exp_id", "class", "test_support", "precision_mean", "precision_std",
        "recall_mean", "recall_std", "f1_mean", "f1_std"],
    "Latency": ["configuration", "GPU", "dtype", "batch", "BN_fused", "p50_ms",
        "p95_ms", "p99_ms", "images_per_s", "resolution_px",
        "includes_preprocessing", "torch"],
    "Summary": ["exp_id", "backbone", "num_seeds", "macro_f1_val_mean",
        "macro_f1_val_std", "top1_val_mean", "top1_val_std", "parameters_m",
        "GMAC", "latency_p95_ms_batch1", "notes"],
}


def load_eval():
    sys.path.insert(0, str(REPO_DIR.resolve()))
    return importlib.import_module("eval")


def load_json(path, default):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else default


def finite_round(value):
    return round(float(value), 6) if value is not None and np.isfinite(value) else None


def mean_std(values):
    values = [float(x) for x in values if x is not None and np.isfinite(x)]
    return (mean(values), stdev(values) if len(values) > 1 else None) if values else (None, None)


def build_rows():
    ev = load_eval()
    summaries = []
    if RUNS_DIR.is_dir():
        for path in sorted(RUNS_DIR.glob("*/seed*/done.json")):
            row = load_json(path, {})
            row["_run_dir"] = path.parent
            summaries.append(row)
    inference_rows = load_json(SUBMISSION_DIR / "inference_results.json", [])
    manifest = load_json(SUBMISSION_DIR / "experiment_manifest.json", {})
    backbone_latency = load_json(SUBMISSION_DIR / "backbone_latency.json", {})

    reference_test = SUBMISSION_DIR / "data/labels/test_subset0.csv"
    if not reference_test.is_file():
        reference_test = REPO_DIR / "data/labels/test_subset0.csv"
    test_metrics = {}
    uncalibrated_ece = {}
    if PREDICTIONS_DIR.is_dir():
        for path in sorted(PREDICTIONS_DIR.glob("*_seed*_test.csv")):
            match = re.fullmatch(r"(.+)_seed(\d+)_test", path.stem)
            if not match:
                continue
            exp_id, seed = match.group(1), int(match.group(2))
            pred = ev.read_pred(str(path))
            if reference_test.is_file():
                ev.check_against_csv(pred, str(reference_test), "test")
            metrics = ev.compute_metrics(pred.y_true, pred.y_pred, pred.probs)
            test_metrics[(exp_id, seed)] = (pred, metrics)
        for path in sorted(PREDICTIONS_DIR.glob("*_seed*_test_uncal.csv")):
            match = re.fullmatch(r"(.+)_seed(\d+)_test_uncal", path.stem)
            if not match:
                continue
            exp_id, seed = match.group(1), int(match.group(2))
            pred = ev.read_pred(str(path))
            if reference_test.is_file():
                ev.check_against_csv(pred, str(reference_test), "test")
            metrics = ev.compute_metrics(pred.y_true, pred.y_pred, pred.probs)
            uncalibrated_ece[(exp_id, seed)] = finite_round(metrics["ece"])

    i00 = next((r for r in inference_rows if r.get("exp_id") == "I00"), {})
    by_seed = {(r["exp_id"], int(r["seed"])): r for r in summaries}
    baseline = {int(r["seed"]): r for r in summaries if r["exp_id"] == "T00"}
    backbones, training, final = [], [], []
    final_metrics = {}
    for run in summaries:
        exp_id = run["exp_id"]
        cfg = run.get("config", {})
        seed = int(run.get("seed", 0))
        common = {
            "exp_id": exp_id, "backbone": run.get("backbone"),
            "pretrained_tag": run.get("pretrained_tag"),
            "parameters_m": finite_round(run.get("parameters_m")),
            "GMAC": finite_round(run.get("gmac")),
            "resolution_px": run.get("img_size"), "epochs": cfg.get("epochs"),
            "seed": seed, "macro_f1_val": finite_round(run.get("macro_f1_val")),
            "top1_val": finite_round(run.get("top1_val")),
            "train_seconds_per_epoch": finite_round(run.get("train_seconds_per_epoch_mean")),
            "latency_batch1_ms": finite_round(backbone_latency.get(exp_id, {}).get("p50")) if exp_id.startswith("B") else None,
            "notes": run.get("gmac_error") or ", ".join(run.get("gmac_unsupported_ops", [])),
        }
        if exp_id.startswith("B"):
            backbones.append(common)
        if exp_id.startswith("T"):
            base = baseline.get(seed)
            delta = (
                run.get("macro_f1_val", 0) - base.get("macro_f1_val", 0)
                if base is not None else None
            )
            meta = manifest.get(exp_id, {})
            training.append({
                "exp_id": exp_id, "backbone": run.get("backbone"),
                "changed_axis": meta.get("axis", "baseline" if exp_id == "T00" else ""),
                "difference_from_T00": meta.get("difference", ""),
                "seed": seed, "macro_f1_val": common["macro_f1_val"],
                "top1_val": common["top1_val"], "delta_macro_f1_vs_T00": finite_round(delta),
                "rare_class_f1": "", "notes": meta.get("notes", ""),
            })
        scored = test_metrics.get((exp_id, seed))
        if scored and (exp_id == "T00" or exp_id.startswith("F")):
            pred, metrics = scored
            ece_uncal = uncalibrated_ece.get((exp_id, seed))
            if ece_uncal is None and float(run.get("temperature", 1.0)) == 1.0:
                # With T=1, the saved test probabilities are the uncalibrated probabilities.
                ece_uncal = finite_round(metrics["ece"])
            final_metrics.setdefault(exp_id, []).append((run, pred, metrics))
            final.append({
                "exp_id": exp_id,
                "configuration": (
                    f"{run.get('backbone')} | {cfg.get('loss')} | {cfg.get('aug')} | "
                    f"T={run.get('temperature', 1.0):.4g}"
                ),
                "seed": seed, "macro_f1_val": common["macro_f1_val"],
                "macro_f1_test": finite_round(metrics["macro_f1"]),
                "top1_test": finite_round(metrics["top1"]),
                "ECE_test_uncal": ece_uncal,
                "ECE_test": finite_round(metrics["ece"]),
                "macro_f1_test_mean": None, "macro_f1_test_std": None,
                "top1_test_mean": None, "top1_test_std": None,
                "ECE_test_uncal_mean": None, "ECE_test_uncal_std": None,
                "ECE_test_mean": None, "ECE_test_std": None,
                "notes": "Full test fold; configuration selected with validation only.",
            })

    for exp_id, items in final_metrics.items():
        ref = next(row for row in final if row["exp_id"] == exp_id and isinstance(row["seed"], int))
        f1m, f1s = mean_std([m["macro_f1"] for _, _, m in items])
        accm, accs = mean_std([m["top1"] for _, _, m in items])
        ecem, eces = mean_std([m["ece"] for _, _, m in items])
        ece_uncal_m, ece_uncal_s = mean_std([
            uncalibrated_ece.get((exp_id, int(run.get("seed", 0))))
            if uncalibrated_ece.get((exp_id, int(run.get("seed", 0)))) is not None
            else (finite_round(metrics["ece"]) if float(run.get("temperature", 1.0)) == 1.0 else None)
            for run, _, metrics in items
        ])
        final.append({
            "exp_id": exp_id, "configuration": ref["configuration"], "seed": "mean ± std",
            "macro_f1_val": None, "macro_f1_test": None, "top1_test": None, "ECE_test": None,
            "ECE_test_uncal": None,
            "macro_f1_test_mean": finite_round(f1m), "macro_f1_test_std": finite_round(f1s),
            "top1_test_mean": finite_round(accm), "top1_test_std": finite_round(accs),
            "ECE_test_uncal_mean": finite_round(ece_uncal_m),
            "ECE_test_uncal_std": finite_round(ece_uncal_s),
            "ECE_test_mean": finite_round(ecem), "ECE_test_std": finite_round(eces),
            "notes": f"Aggregate over {len(items)} seed(s); sample std uses ddof=1.",
        })

    final_candidates = [(key, items) for key, items in final_metrics.items() if key.startswith("F")]
    best_id = max(
        final_candidates,
        key=lambda pair: mean([float(item[0]["macro_f1_val"]) for item in pair[1]]),
    )[0] if final_candidates else None
    perclass = []
    for exp_id in ("T00", best_id):
        if not exp_id or exp_id not in final_metrics:
            continue
        items = final_metrics[exp_id]
        for index, name in enumerate(ev.CLASS_NAMES):
            supports = np.bincount(items[0][1].y_true, minlength=9)
            columns = {
                key: np.asarray([item[2][key][index] for item in items], dtype=float)
                for key in ("precision", "recall", "f1")
            }
            row = {"exp_id": exp_id, "class": name, "test_support": int(supports[index])}
            for key, values in columns.items():
                row[f"{key}_mean"] = finite_round(values.mean())
                row[f"{key}_std"] = finite_round(values.std(ddof=1)) if len(values) > 1 else None
            perclass.append(row)

    inference, latency = [], []
    for row in inference_rows:
        inference.append({
            "exp_id": row.get("exp_id", ""), "method": row.get("method", ""),
            "model_checkpoint": row.get("model_checkpoint", ""), "K": row.get("K"),
            "macro_f1_val": finite_round(row.get("macro_f1_val")),
            "top1_val": finite_round(row.get("top1_val")),
            "ECE_val": finite_round(row.get("ECE_val")),
            "latency_p50_ms_batch1": finite_round(row.get("latency_p50_ms")),
            "latency_p95_ms_batch1": finite_round(row.get("latency_p95_ms")),
            "latency_p99_ms_batch1": finite_round(row.get("latency_p99_ms")),
            "throughput_images_s": finite_round(row.get("throughput_images_s")),
            "relative_cost_vs_I00": finite_round(row.get("relative_cost_vs_I00")),
            "GPU": row.get("GPU", ""), "dtype": row.get("dtype", ""),
            "BN_fused": row.get("BN_fused", False), "notes": row.get("notes", ""),
        })
        latency.append({
            "configuration": row.get("method", ""), "GPU": row.get("GPU", ""),
            "dtype": row.get("dtype", ""), "batch": row.get("batch", 1),
            "BN_fused": row.get("BN_fused", False),
            "p50_ms": finite_round(row.get("latency_p50_ms")),
            "p95_ms": finite_round(row.get("latency_p95_ms")),
            "p99_ms": finite_round(row.get("latency_p99_ms")),
            "images_per_s": finite_round(row.get("throughput_images_s")),
            "resolution_px": row.get("img_size"),
            "includes_preprocessing": row.get("includes_preprocessing", False),
            "torch": row.get("torch", ""),
        })

    grouped = {}
    for run in summaries:
        grouped.setdefault(run["exp_id"], []).append(run)
    summary_rows = []
    for exp_id, items in grouped.items():
        f1m, f1s = mean_std([r.get("macro_f1_val") for r in items])
        accm, accs = mean_std([r.get("top1_val") for r in items])
        first = items[0]
        summary_rows.append({
            "exp_id": exp_id, "backbone": first.get("backbone"), "num_seeds": len(items),
            "macro_f1_val_mean": finite_round(f1m), "macro_f1_val_std": finite_round(f1s),
            "top1_val_mean": finite_round(accm), "top1_val_std": finite_round(accs),
            "parameters_m": finite_round(first.get("parameters_m")),
            "GMAC": finite_round(first.get("gmac")),
            "latency_p95_ms_batch1": next(
                (finite_round(r.get("latency_p95_ms")) for r in inference_rows if r.get("exp_id") == "I00"),
                None,
            ),
            "notes": manifest.get(exp_id, {}).get("notes", ""),
        })
    summary_rows.sort(
        key=lambda row: row["macro_f1_val_mean"] if row["macro_f1_val_mean"] is not None else -1,
        reverse=True,
    )
    return {
        "Backbones": backbones, "Training": training, "Inference": inference,
        "Final": final, "PerClass": perclass, "Latency": latency,
        "Summary": summary_rows[:10], "final_metrics": final_metrics,
        "best_final_id": best_id, "inference_rows": inference_rows,
        "summaries": summaries,
    }


def write_workbook(data):
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter

    workbook = Workbook()
    workbook.remove(workbook.active)
    for name, headers in HEADERS.items():
        sheet = workbook.create_sheet(name)
        sheet.append(headers)
        for cell in sheet[1]:
            cell.fill = PatternFill("solid", fgColor="1F4E78")
            cell.font = Font(color="FFFFFF", bold=True)
            cell.alignment = Alignment(wrap_text=True, vertical="center")
        for row in data[name]:
            sheet.append([row.get(header) for header in headers])
        sheet.freeze_panes = "A2"
        sheet.auto_filter.ref = sheet.dimensions
        sheet.row_dimensions[1].height = 34
        sheet.sheet_view.showGridLines = False
        for col, header in enumerate(headers, start=1):
            values = [
                str(sheet.cell(r, col).value or "")
                for r in range(2, sheet.max_row + 1)
            ]
            width = min(44, max(12, len(header) + 2, max(map(len, values), default=0) + 2))
            sheet.column_dimensions[get_column_letter(col)].width = width
    for sheet_name, metric in (
        ("Backbones", "macro_f1_val"), ("Training", "macro_f1_val"),
        ("Inference", "macro_f1_val"), ("Summary", "macro_f1_val_mean"),
    ):
        sheet = workbook[sheet_name]
        col = HEADERS[sheet_name].index(metric) + 1
        values = [
            sheet.cell(r, col).value for r in range(2, sheet.max_row + 1)
            if isinstance(sheet.cell(r, col).value, (float, int))
        ]
        if values:
            best = max(values)
            for r in range(2, sheet.max_row + 1):
                cell = sheet.cell(r, col)
                if cell.value == best:
                    cell.fill = PatternFill("solid", fgColor="C6E0B4")
                    cell.font = Font(bold=True)
    workbook.save(WORKBOOK)


def write_report(data):
    final_metrics = data["final_metrics"]
    if not final_metrics:
        REPORT.write_text(
            """# Báo cáo Lab Day 2 — DeepWeeds

> **Chờ chạy thí nghiệm thật.** Chưa có run log và test predictions hợp lệ để báo cáo metric. Báo cáo không điền số dự đoán hay mượn số từ bài báo.

## Tóm tắt
Chưa thể chốt mô hình hoặc báo cáo macro-F1, top-1, ECE trên test. Chạy code/lab_day2.ipynb, hoàn thành fold 0, lưu log, curves và predictions; sau đó chạy code/build_results.py --report.

## Dữ liệu và thiết lập
- Dataset: DeepWeeds, split fold 0 nguyên bản.
- Train cập nhật trọng số; validation chọn model/checkpoint/inference/temperature.
- Test chỉ chạy sau khi khóa cấu hình, đúng một lần mỗi seed.
- Bảng lớp, kiểm tra overlap/tổng ảnh và phần cứng được lấy từ output thực của notebook.

## Kết quả
Các bảng backbone/training/inference/final và ma trận lỗi sẽ được sinh từ log/predictions thật. Mốc bài báo chỉ là tài liệu tham khảo, không phải kết quả của bài này.

## Kết luận và hạn chế
Chưa có số liệu để chọn cấu hình tốt nhất, so sánh baseline, báo cáo lớp Chinee apple/Snake weed hay kết luận latency. Cần ít nhất ba seed cho final và baseline. Fold 0 được chia ngẫu nhiên, không theo địa điểm; điểm test có thể lạc quan với mùa/vùng mới.

## Việc cần làm
1. Tải và kiểm tra MD5 của images.zip cùng các CSV fold 0.
2. Chạy các bước notebook trên GPU; chỉ validation được dùng để lựa chọn.
3. Chạy final và T00 đủ ba seed, sinh predictions, curves, xlsx và báo cáo.
4. Tạo link notebook Colab/Kaggle đã chia sẻ rồi thêm link vào README submission.
""",
            encoding="utf-8",
        )
        return

    best_id = data["best_final_id"]
    best = final_metrics.get(best_id, [])
    baseline = final_metrics.get("T00", [])
    best_mean, best_std = mean_std([item[2]["macro_f1"] for item in best])
    base_mean, base_std = mean_std([item[2]["macro_f1"] for item in baseline])
    delta = best_mean - base_mean if best_mean is not None and base_mean is not None else None
    noise = max(value or 0 for value in (best_std, base_std))
    if delta is None:
        conclusion = "Chưa đủ hai prediction groups cho kết luận so với baseline."
    elif delta > noise:
        conclusion = f"Final hơn baseline Δ={delta:.4f}; Δ lớn hơn std lớn nhất {noise:.4f}."
    elif delta > 0:
        conclusion = f"Final hơn baseline Δ={delta:.4f}, nhưng chênh lệch chưa vượt std lớn nhất {noise:.4f}."
    else:
        conclusion = f"Chưa thấy final cải thiện macro-F1 (Δ={delta:.4f})."

    final_rows = [
        row for row in data["Final"] if row["seed"] == "mean ± std"
        and row["exp_id"] in {"T00", best_id}
    ]
    inference_rows = data["inference_rows"]
    best_inference = max(
        (row for row in inference_rows if row.get("macro_f1_val") is not None),
        key=lambda row: row["macro_f1_val"], default=None,
    )
    runtime = next(
        (row for row in inference_rows if row.get("batch", 1) == 1 and row.get("latency_p95_ms") is not None),
        None,
    )
    split_summary = next(
        (row.get("split") for row in data["summaries"] if row.get("split")), None
    )
    best_test_items = final_metrics.get(best_id, []) if best_id else []
    combined_confusion = (
        np.sum([item[2]["confusion"] for item in best_test_items], axis=0)
        if best_test_items else None
    )
    lines = [
        "# Báo cáo Lab Day 2 — DeepWeeds",
        "",
        "## Tóm tắt",
        "",
        f"- Cấu hình final được chọn bằng validation: {best_id or 'chưa có F*'}.",
        f"- Final macro-F1 test: {best_mean:.4f} ± {best_std if best_std is not None else float('nan'):.4f}." if best_mean is not None else "- Chưa có final test.",
        f"- T00 baseline macro-F1 test: {base_mean:.4f} ± {base_std if base_std is not None else float('nan'):.4f}." if base_mean is not None else "- Chưa có baseline test.",
        f"- {conclusion}",
        "",
        "## Dữ liệu và thiết lập",
        "",
        "Dùng DeepWeeds fold 0 nguyên bản, 9 lớp. Train chỉ cập nhật trọng số; validation chọn backbone, siêu tham số, checkpoint, suy luận và temperature. Không gộp validation vào train. Test chỉ dùng sau khi khóa cấu hình, một lượt cho mỗi seed. Metric chính là macro-F1; báo cáo thêm top-1, balanced accuracy, ECE 15-bin, per-class và latency.",
        "",
        f"Số cấu hình train đã lưu: {len(data['summaries'])}. Split sizes: {split_summary.get('n') if split_summary else 'chưa có'}.",
        f"Hardware/runtime: {runtime.get('GPU') if runtime else 'chưa đo'}; torch version được lưu trong mỗi run config. Input preprocessing: resize + center crop, ImageNet mean/std.",
        "",
        "Thứ tự lớp theo eval.py: Chinee Apple, Lantana, Parkinsonia, Parthenium, Prickly Acacia, Rubber Vine, Siam Weed, Snake Weed, Negatives.",
        "",
        "| Class | Train | Validation | Test |",
        "|---|---:|---:|---:|",
        "",
        "## Backbone",
        "",
        "| ID | Backbone | Tag | Params M | GMAC | Val macro-F1 | Val top-1 | sec/epoch |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    if split_summary and split_summary.get("per_class"):
        split_classes = split_summary["per_class"]
        class_lines = [
            f"| {class_name} | {split_classes['train'].get(class_name, 0)} | "
            f"{split_classes['val'].get(class_name, 0)} | {split_classes['test'].get(class_name, 0)} |"
            for class_name in load_eval().CLASS_NAMES
        ]
        if (SUBMISSION_DIR / "class_distribution.png").is_file():
            class_lines += ["", "![Fold 0 class distribution](class_distribution.png)"]
        if (SUBMISSION_DIR / "eda_samples.png").is_file():
            class_lines += ["", "![Three training samples per class](eda_samples.png)"]
        if (SUBMISSION_DIR / "augmented_batch.png").is_file():
            class_lines += ["", "Augmentation preview:", "", "![Augmented batch](augmented_batch.png)"]
        split_index = lines.index("## Backbone")
        lines[split_index:split_index] = class_lines + [""]
    for row in data["Backbones"]:
        lines.append("| " + " | ".join(str(row.get(k) if row.get(k) is not None else "—") for k in (
            "exp_id", "backbone", "pretrained_tag", "parameters_m", "GMAC",
            "macro_f1_val", "top1_val", "train_seconds_per_epoch",
        )) + " |")
    lines += ["", "## Công thức huấn luyện", "",
        "| ID | Backbone | Trục | Khác T00 | Val macro-F1 | Δ vs T00 |",
        "|---|---|---|---|---:|---:|"]
    for row in data["Training"]:
        lines.append("| " + " | ".join(str(row.get(k) if row.get(k) is not None else "—") for k in (
            "exp_id", "backbone", "changed_axis", "difference_from_T00",
            "macro_f1_val", "delta_macro_f1_vs_T00",
        )) + " |")
    lines += ["", "Các sweep thường dùng một seed để sàng lọc; chưa đủ căn cứ xem chênh lệch nhỏ hơn biến thiên seed là cải thiện chắc chắn.", "",
        "## Suy luận và latency", "",
        "| ID | Phương pháp | K | Val macro-F1 | ECE val | p50 ms | p95 ms | p99 ms |",
        "|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in data["Inference"]:
        lines.append("| " + " | ".join(str(row.get(k) if row.get(k) is not None else "—") for k in (
            "exp_id", "method", "K", "macro_f1_val", "ECE_val",
            "latency_p50_ms_batch1", "latency_p95_ms_batch1", "latency_p99_ms_batch1",
        )) + " |")
    if (SUBMISSION_DIR / "accuracy_latency_tradeoff.png").is_file():
        lines += ["", "Validation accuracy-latency trade-off:", "", "![Validation accuracy-latency trade-off](accuracy_latency_tradeoff.png)"]
    lines += ["", (
        f"Validation inference tốt nhất trong các phương pháp đã ghi là {best_inference.get('method')}."
        if best_inference else "Chưa có phép đo inference."
    ), (
        f"Latency batch-1 p95: {runtime.get('latency_p95_ms')} ms trên {runtime.get('GPU')}."
        if runtime else "Chưa có latency batch-1."
    ), "Đo latency cần warmup ít nhất 10 lần, synchronize GPU và ít nhất 50 lần đo; notebook lưu rõ GPU/dtype/batch/resolution và việc tính preprocessing.", "",
        "## Chung kết và per-class", "",
        "| ID | Seed | Macro-F1 test mean | std | Top-1 mean | std | ECE uncal mean | std | ECE calibrated mean | std |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in final_rows:
        lines.append("| " + " | ".join(str(row.get(k) if row.get(k) is not None else "—") for k in (
            "exp_id", "seed", "macro_f1_test_mean", "macro_f1_test_std",
            "top1_test_mean", "top1_test_std", "ECE_test_uncal_mean",
            "ECE_test_uncal_std", "ECE_test_mean", "ECE_test_std",
        )) + " |")
    if best_id:
        lines += ["", f"Confusion matrix totals for {best_id} are recalculated from all saved seed prediction files by eval.py.", ""]
        if combined_confusion is not None:
            names = load_eval().CLASS_NAMES
            lines.append("| Actual / Predicted | " + " | ".join(names) + " |")
            lines.append("|---|" + "---|" * len(names))
            for class_name, row_values in zip(names, combined_confusion):
                lines.append("| " + class_name + " | " + " | ".join(str(int(value)) for value in row_values) + " |")
        lines += ["", "Per-class metrics over final seeds:", "",
            "| Class | Support | Precision mean | Recall mean | Recall std | F1 mean | F1 std |",
            "|---|---:|---:|---:|---:|---:|---:|"]
        for row in data["PerClass"]:
            if row["exp_id"] == best_id:
                lines.append("| " + " | ".join(str(row.get(k) if row.get(k) is not None else "—") for k in (
                    "class", "test_support", "precision_mean", "recall_mean", "recall_std", "f1_mean", "f1_std",
                )) + " |")
        lines += ["", "The two required hard classes are Chinee Apple and Snake Weed; review their recall and the largest off-diagonal confusion entries together."]
        if (SUBMISSION_DIR / "confusion_matrix_F01.png").is_file():
            lines += ["", "![F01 test confusion matrix](confusion_matrix_F01.png)"]
        if (SUBMISSION_DIR / "error_examples_F01.png").is_file():
            lines += ["", "Misclassified test examples from seed 0 (opened only after model selection):", "", "![F01 test error examples](error_examples_F01.png)"]
    lines += ["", conclusion, "",
        "## Kết luận và hạn chế", "",
        "Kết luận về cấu hình dựa trên validation; các chỉ số test chỉ mô tả cấu hình đã khóa. Một fold chia ngẫu nhiên, không theo địa điểm, có thể làm điểm lạc quan ở vùng, mùa hoặc điều kiện ánh sáng khác. Báo cáo độ ổn định bằng mean ± sample std qua số seed thực tế; nếu thiếu ba seed, kết luận đó còn hạn chế.", "",
        f"Phương pháp có macro-F1 val cao nhất đã đo: {best_inference.get('method') if best_inference else 'chưa có'}.",
        f"Cấu hình có p95 ≤ 100 ms batch 1 đã đo: {runtime.get('method') if runtime and runtime.get('latency_p95_ms', 1e9) <= 100 else 'chưa ghi nhận'}.",
        "",
        "## Tái lập",
        "",
        "Mọi số liệu trong báo cáo lấy từ runs/*/seed*/done.json, history.csv, inference_results.json và predictions/*_test.csv. File eval.py của repo gốc được dùng để đối chiếu và tính lại chỉ số. Các run thiếu log hoặc prediction không được dùng làm bằng chứng.",
    ]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build():
    data = build_rows()
    write_workbook(data)
    write_report(data)
    return {
        "workbook": str(WORKBOOK), "report": str(REPORT),
        "sheets": list(HEADERS), "run_count": len(data["summaries"]),
        "final_test_groups": list(data["final_metrics"]),
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate workbook/report from real run artifacts.")
    parser.add_argument("--report", action="store_true", help="rebuild report.md as well as results.xlsx")
    parser.parse_args()
    print(json.dumps(build(), indent=2, ensure_ascii=True))
