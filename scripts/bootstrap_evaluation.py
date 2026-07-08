#!/usr/bin/env python3
"""Image-level bootstrap evaluation for deterministic COCO prediction files.

This script does not train detectors, rerun detector inference, or perform
classical k-fold validation. It resamples validation images with replacement,
creates temporary COCO annotation/prediction subsets for each bootstrap sample,
and evaluates existing fused COCO detection JSON files with COCOeval.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

METRIC_NAMES = (
    "AP",
    "AP50",
    "AP75",
    "AP_small",
    "AP_medium",
    "AP_large",
    "AR1",
    "AR10",
    "AR100",
    "AR_small",
    "AR_medium",
    "AR_large",
)

# Existing reproduction outputs. Additional *_predictions.json files are also
# accepted, but these mappings produce paper-style method names when present.
PREDICTION_FILE_METHODS = {
    "wbf_predictions.json": "WBF",
    "awbf_predictions.json": "AWBF",
    "incremental_awbf_predictions.json": "Incremental_AWBF",
    "awbf_competition_predictions.json": "AWBF-competition",
    "awbf_negotiation_predictions.json": "AWBF-Negotiation",
    "awbf_competition_incremental_state_predictions.json": "AWBF-competition-IncrementalState",
    "awbf_negotiation_incremental_state_predictions.json": "AWBF-Negotiation-IncrementalState",
}


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f)


def load_predictions(predictions_dir: Path) -> Dict[str, List[dict]]:
    """Load existing COCO detection JSON files keyed by method name."""
    predictions: Dict[str, List[dict]] = {}
    for path in sorted(predictions_dir.glob("*_predictions.json")):
        method = PREDICTION_FILE_METHODS.get(path.name, path.stem.replace("_predictions", ""))
        rows = load_json(path)
        if not isinstance(rows, list):
            raise ValueError(f"Prediction file {path} must contain a COCO detection list")
        predictions[method] = rows
    if not predictions:
        raise FileNotFoundError(f"No *_predictions.json files found in {predictions_dir}")
    if "WBF" not in predictions:
        raise FileNotFoundError(
            f"WBF baseline file is required for paired deltas; expected {predictions_dir / 'wbf_predictions.json'}"
        )
    return predictions


def index_by_image(rows: Iterable[Mapping]) -> Dict[int, List[dict]]:
    by_image: Dict[int, List[dict]] = defaultdict(list)
    for row in rows:
        by_image[int(row["image_id"])].append(dict(row))
    return by_image


def build_bootstrap_subset(
    coco: Mapping,
    predictions: Sequence[Mapping],
    sampled_image_ids: Sequence[int],
) -> Tuple[dict, List[dict]]:
    """Duplicate sampled images with fresh IDs so sampling with replacement is valid.

    COCO JSON image IDs must be unique, so each bootstrap draw receives a new
    synthetic image ID while preserving the original image metadata, annotations,
    categories, and matching prediction rows.
    """
    images_by_id = {int(img["id"]): dict(img) for img in coco.get("images", [])}
    anns_by_image = index_by_image(coco.get("annotations", []))
    preds_by_image = index_by_image(predictions)

    subset_images: List[dict] = []
    subset_annotations: List[dict] = []
    subset_predictions: List[dict] = []
    ann_id = 1

    for draw_index, original_id in enumerate(sampled_image_ids, start=1):
        if original_id not in images_by_id:
            raise KeyError(f"Sampled image_id {original_id} is missing from annotations")
        new_image_id = draw_index
        image = dict(images_by_id[original_id])
        image["id"] = new_image_id
        image["bootstrap_original_image_id"] = original_id
        subset_images.append(image)

        for ann in anns_by_image.get(original_id, []):
            new_ann = dict(ann)
            new_ann["id"] = ann_id
            new_ann["image_id"] = new_image_id
            subset_annotations.append(new_ann)
            ann_id += 1

        for pred in preds_by_image.get(original_id, []):
            new_pred = dict(pred)
            new_pred["image_id"] = new_image_id
            subset_predictions.append(new_pred)

    subset_coco = {
        "info": dict(coco.get("info", {})),
        "licenses": list(coco.get("licenses", [])),
        "images": subset_images,
        "annotations": subset_annotations,
        "categories": list(coco.get("categories", [])),
    }
    return subset_coco, subset_predictions


def evaluate_coco_metrics(annotations_json: Path, detections_json: Path) -> Dict[str, float]:
    """Evaluate COCO metrics with correct standard COCO recall names."""
    # pycocotools.loadRes cannot load an empty detection list because it
    # inspects the first row. Treat no detections as zero metric values for the
    # bootstrap replicate instead of crashing the whole analysis.
    if not load_json(detections_json):
        return {metric: 0.0 for metric in METRIC_NAMES}

    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    coco_gt = COCO(str(annotations_json))
    coco_dt = coco_gt.loadRes(str(detections_json))
    evaluator = COCOeval(coco_gt, coco_dt, "bbox")
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    stats = evaluator.stats
    return {
        "AP": float(stats[0]),
        "AP50": float(stats[1]),
        "AP75": float(stats[2]),
        "AP_small": float(stats[3]),
        "AP_medium": float(stats[4]),
        "AP_large": float(stats[5]),
        "AR1": float(stats[6]),
        "AR10": float(stats[7]),
        "AR100": float(stats[8]),
        "AR_small": float(stats[9]),
        "AR_medium": float(stats[10]),
        "AR_large": float(stats[11]),
    }


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    weight = pos - lo
    return ordered[lo] * (1.0 - weight) + ordered[hi] * weight


def summarize_values(values: Sequence[float]) -> Dict[str, float]:
    values = [float(v) for v in values]
    return {
        "mean": statistics.fmean(values) if values else float("nan"),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "ci95_low": percentile(values, 0.025),
        "ci95_high": percentile(values, 0.975),
        "n": len(values),
    }


def summarize_metrics(results: Mapping[str, Sequence[Mapping[str, float]]]) -> Dict[str, Dict[str, Dict[str, float]]]:
    summary: Dict[str, Dict[str, Dict[str, float]]] = {}
    for method, rows in results.items():
        summary[method] = {}
        for metric in METRIC_NAMES:
            summary[method][metric] = summarize_values([row[metric] for row in rows if metric in row])
    return summary


def summarize_deltas(
    results: Mapping[str, Sequence[Mapping[str, float]]],
    baseline: str = "WBF",
    tolerance: float = 1e-12,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    if baseline not in results:
        raise KeyError(f"Baseline {baseline!r} is required for paired deltas")
    deltas: Dict[str, Dict[str, Dict[str, float]]] = {}
    baseline_rows = results[baseline]
    for method, rows in results.items():
        if method == baseline:
            continue
        deltas[method] = {}
        for metric in METRIC_NAMES:
            values = [float(row[metric]) - float(base[metric]) for row, base in zip(rows, baseline_rows)]
            summary = summarize_values(values)
            n = len(values)
            summary.update({
                "proportion_gt_wbf": sum(v > tolerance for v in values) / n if n else float("nan"),
                "proportion_eq_wbf": sum(abs(v) <= tolerance for v in values) / n if n else float("nan"),
                "proportion_lt_wbf": sum(v < -tolerance for v in values) / n if n else float("nan"),
            })
            deltas[method][metric] = summary
    return deltas


def write_summary_csv(path: Path, summary: Mapping[str, Mapping[str, Mapping[str, float]]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "metric", "mean", "std", "ci95_low", "ci95_high", "n"])
        writer.writeheader()
        for method, metrics in summary.items():
            for metric, values in metrics.items():
                writer.writerow({"method": method, "metric": metric, **values})


def write_deltas_csv(path: Path, deltas: Mapping[str, Mapping[str, Mapping[str, float]]]) -> None:
    fieldnames = [
        "method",
        "metric",
        "mean_delta",
        "std_delta",
        "ci95_low",
        "ci95_high",
        "proportion_gt_wbf",
        "proportion_eq_wbf",
        "proportion_lt_wbf",
        "n",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for method, metrics in deltas.items():
            for metric, values in metrics.items():
                writer.writerow({
                    "method": method,
                    "metric": metric,
                    "mean_delta": values["mean"],
                    "std_delta": values["std"],
                    "ci95_low": values["ci95_low"],
                    "ci95_high": values["ci95_high"],
                    "proportion_gt_wbf": values["proportion_gt_wbf"],
                    "proportion_eq_wbf": values["proportion_eq_wbf"],
                    "proportion_lt_wbf": values["proportion_lt_wbf"],
                    "n": values["n"],
                })


def fmt(value: float, digits: int = 4) -> str:
    if value != value:
        return "nan"
    return f"{value:.{digits}f}"


def write_markdown_report(path: Path, config: Mapping, summary: Mapping, deltas: Mapping) -> None:
    lines = [
        "# Bootstrap Evaluation",
        "",
        "This report is an image-level bootstrap stability analysis for deterministic post-processing outputs.",
        "It is **not** classical k-fold cross-validation: no detector is retrained, no detector inference is rerun, and only existing fused COCO detection JSON files are evaluated.",
        "",
        "## Configuration",
        "",
        f"- Annotations: `{config['annotations']}`",
        f"- Predictions directory: `{config['predictions_dir']}`",
        f"- Number of bootstrap samples: {config['num_samples']}",
        f"- Sample size per bootstrap replicate: {config['sample_size']}",
        f"- Seed: {config['seed']}",
        "",
        "## Metric summary",
        "",
        "| Method | Metric | Mean | Std | 95% CI | N |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for method, metrics in summary.items():
        for metric in METRIC_NAMES:
            values = metrics[metric]
            lines.append(
                f"| {method} | {metric} | {fmt(values['mean'])} | {fmt(values['std'])} | "
                f"[{fmt(values['ci95_low'])}, {fmt(values['ci95_high'])}] | {values['n']} |"
            )
    lines.extend([
        "",
        "## Paired deltas against WBF",
        "",
        "Positive deltas mean the method exceeded WBF on the same bootstrap image sample. Do not claim statistical significance when the 95% confidence interval overlaps zero.",
        "",
        "| Method | Metric | Mean delta | Std delta | 95% CI | > WBF | = WBF | < WBF |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for method, metrics in deltas.items():
        for metric in METRIC_NAMES:
            values = metrics[metric]
            lines.append(
                f"| {method} | {metric} | {fmt(values['mean'])} | {fmt(values['std'])} | "
                f"[{fmt(values['ci95_low'])}, {fmt(values['ci95_high'])}] | "
                f"{fmt(values['proportion_gt_wbf'], 3)} | {fmt(values['proportion_eq_wbf'], 3)} | {fmt(values['proportion_lt_wbf'], 3)} |"
            )
    lines.extend([
        "",
        "## Interpretation guidance",
        "",
        "- This bootstrap estimates stability across validation images for deterministic fusion/post-processing outputs.",
        "- It is not a substitute for training-time k-fold validation because the detectors are fixed.",
        "- Paired deltas should be interpreted cautiously; if the delta confidence interval includes zero, report the comparison as inconclusive rather than statistically significant.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def write_latex_table(path: Path, summary: Mapping, deltas: Mapping) -> None:
    lines = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Method & AP mean & AP 95\% CI & $\Delta$AP vs WBF & $P(\Delta\mathrm{AP}>0)$ \\",
        r"\midrule",
    ]
    for method, metrics in summary.items():
        ap = metrics["AP"]
        if method == "WBF":
            delta_text = "--"
            prob_text = "--"
        else:
            delta = deltas.get(method, {}).get("AP", {})
            delta_text = fmt(delta.get("mean", float("nan")))
            prob_text = fmt(delta.get("proportion_gt_wbf", float("nan")), 3)
        lines.append(
            f"{method} & {fmt(ap['mean'])} & [{fmt(ap['ci95_low'])}, {fmt(ap['ci95_high'])}] & {delta_text} & {prob_text} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def run_bootstrap(args, evaluator: Callable[[Path, Path], Dict[str, float]] = evaluate_coco_metrics) -> dict:
    annotations_path = Path(args.annotations)
    predictions_dir = Path(args.predictions_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    coco = load_json(annotations_path)
    image_ids = [int(img["id"]) for img in coco.get("images", [])]
    if not image_ids:
        raise ValueError(f"No images found in annotations file {annotations_path}")
    predictions = load_predictions(predictions_dir)
    rng = random.Random(args.seed)
    sample_size = int(args.sample_size) if args.sample_size else len(image_ids)
    if sample_size <= 0:
        raise ValueError("--sample-size must be positive")

    results: Dict[str, List[Dict[str, float]]] = {method: [] for method in predictions}
    with tempfile.TemporaryDirectory(prefix="bootstrap_eval_", dir=output_dir) as tmp_name:
        tmp_dir = Path(tmp_name)
        for sample_index in range(1, int(args.num_samples) + 1):
            sampled = [rng.choice(image_ids) for _ in range(sample_size)]
            sample_ann_path = tmp_dir / f"sample_{sample_index:04d}_annotations.json"
            print(f"bootstrap sample {sample_index}/{args.num_samples}: {sample_size} images", flush=True)
            for method, rows in predictions.items():
                subset_coco, subset_predictions = build_bootstrap_subset(coco, rows, sampled)
                if not sample_ann_path.exists():
                    write_json(sample_ann_path, subset_coco)
                pred_path = tmp_dir / f"sample_{sample_index:04d}_{method}_predictions.json"
                write_json(pred_path, subset_predictions)
                metrics = evaluator(sample_ann_path, pred_path)
                results[method].append({metric: float(metrics[metric]) for metric in METRIC_NAMES})

    summary = summarize_metrics(results)
    deltas = summarize_deltas(results, baseline="WBF", tolerance=float(args.tolerance))
    config = {
        "annotations": str(annotations_path),
        "predictions_dir": str(predictions_dir),
        "output_dir": str(output_dir),
        "num_samples": int(args.num_samples),
        "sample_size": sample_size,
        "seed": int(args.seed),
        "metrics": list(METRIC_NAMES),
        "analysis_type": "image-level bootstrap stability analysis for deterministic post-processing outputs",
        "not_classical_k_fold": True,
    }
    report = {"config": config, "summary": summary, "paired_deltas_vs_wbf": deltas}

    write_json(output_dir / "bootstrap_report.json", report)
    write_summary_csv(output_dir / "bootstrap_summary.csv", summary)
    write_deltas_csv(output_dir / "bootstrap_deltas.csv", deltas)
    write_markdown_report(output_dir / "BOOTSTRAP_EVALUATION.md", config, summary, deltas)
    write_latex_table(output_dir / "bootstrap_table.tex", summary, deltas)
    return report


def parse_args(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", default="data/coco/annotations/instances_val2017.json", help="COCO annotations JSON")
    parser.add_argument("--predictions-dir", default="outputs/reproduction_benchmark", help="Directory containing existing *_predictions.json files")
    parser.add_argument("--output-dir", default="outputs/bootstrap", help="Directory for bootstrap reports")
    parser.add_argument("--num-samples", type=int, default=100, help="Number of bootstrap replicates")
    parser.add_argument("--sample-size", type=int, default=5000, help="Images sampled with replacement per replicate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--tolerance", type=float, default=1e-12, help="Tolerance for equality proportions in paired deltas")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    run_bootstrap(args)
    print(f"Wrote bootstrap outputs to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
