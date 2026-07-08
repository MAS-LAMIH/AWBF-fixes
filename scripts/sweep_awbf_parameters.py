#!/usr/bin/env python3
"""Sweep AWBF competition/negotiation parameters and export CSV/plots."""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.reproduce_paper_results import (  # noqa: E402
    export_coco_detections,
    fuse_all,
    load_benchmark_predictions,
    load_coco_image_sizes,
    load_predictions,
    sample_predictions,
)
from wbf_agents.awbf import evaluate_coco  # noqa: E402

T_VALUES = [round(i / 10.0, 2) for i in range(11)]
W_VALUES = [round(i * 0.05, 2) for i in range(21)]
RMAX_VALUES = [1, 2, 3, 5, 10]
METRIC_COLUMNS = [
    "AP",
    "AP50",
    "AP75",
    "AP_small",
    "AP_medium",
    "AP_large",
    "AR",
    "AR50",
    "AR75",
    "AR_small",
    "AR_medium",
    "AR_large",
]


def _base_args(cli_args, method):
    return SimpleNamespace(
        methods=[method],
        progress_interval=10**9,
        profile=False,
        disable_preclustering=cli_args.disable_preclustering,
        iou_threshold=cli_args.iou_threshold,
        score_threshold=cli_args.score_threshold,
        cooperation_threshold=cli_args.cooperation_threshold,
        negotiation_threshold=cli_args.negotiation_threshold,
        rounds=cli_args.rounds,
        weight=cli_args.weight,
        incremental_cluster_state=False,
    )


def load_inputs(args):
    if args.sample:
        return sample_predictions()
    if args.benchmark_dir:
        return load_benchmark_predictions(args.benchmark_dir)
    if args.predictions_dir:
        return load_predictions(args.predictions_dir)
    raise SystemExit("Provide --sample, --benchmark-dir, or --predictions-dir")


def evaluate_outputs(outputs, method, annotations, image_sizes, bbox_scale, output_dir, run_name):
    detections = outputs[method]
    prediction_path = output_dir / f"{run_name}_{method}.json"
    export_coco_detections(detections, str(prediction_path), image_sizes=image_sizes, bbox_scale=bbox_scale)
    if not annotations:
        return {"detections": sum(len(v) for v in detections.values())}
    return evaluate_coco(annotations, str(prediction_path))


def run_sweep(args):
    by_model = load_inputs(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_sizes = load_coco_image_sizes(args.annotations) if args.annotations else None
    rows = []

    for value in T_VALUES:
        run_args = _base_args(args, "AWBF-competition")
        run_args.cooperation_threshold = value
        outputs, _timings, flow = fuse_all(by_model, run_args)
        metrics = evaluate_outputs(outputs, "AWBF-competition", args.annotations, image_sizes, args.bbox_scale, output_dir, f"T_{value:.2f}")
        rows.append(_row("T", value, "AWBF-competition", metrics, flow["AWBF-competition"]))

    for value in W_VALUES:
        run_args = _base_args(args, "AWBF-Negotiation")
        run_args.weight = value
        outputs, _timings, flow = fuse_all(by_model, run_args)
        metrics = evaluate_outputs(outputs, "AWBF-Negotiation", args.annotations, image_sizes, args.bbox_scale, output_dir, f"W_{value:.2f}")
        rows.append(_row("W", value, "AWBF-Negotiation", metrics, flow["AWBF-Negotiation"]))

    for value in RMAX_VALUES:
        run_args = _base_args(args, "AWBF-Negotiation")
        run_args.rounds = value
        outputs, _timings, flow = fuse_all(by_model, run_args)
        metrics = evaluate_outputs(outputs, "AWBF-Negotiation", args.annotations, image_sizes, args.bbox_scale, output_dir, f"Rmax_{value}")
        rows.append(_row("Rmax", value, "AWBF-Negotiation", metrics, flow["AWBF-Negotiation"]))

    csv_path = output_dir / "PARAMETER_SWEEP_RESULTS.csv"
    write_csv(csv_path, rows)
    write_plots(output_dir / "plots", rows)
    return csv_path


def _row(parameter, value, method, metrics, flow):
    row = {
        "parameter": parameter,
        "value": value,
        "method": method,
        "output_detections": flow.get("output_detections", 0),
        "detections_removed": flow.get("detections_removed", 0),
        "average_confidence": flow.get("average_confidence", 0.0),
    }
    for metric in METRIC_COLUMNS:
        row[metric] = metrics.get(metric, "")
    return row


def write_csv(path, rows):
    fieldnames = ["parameter", "value", "method", *METRIC_COLUMNS, "output_detections", "detections_removed", "average_confidence"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_plots(plot_dir, rows):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    plot_dir.mkdir(parents=True, exist_ok=True)
    specs = [
        ("T", ["AP", "AR", "AP_small", "AP_medium", "AP_large", "AR_small", "AR_medium", "AR_large"]),
        ("W", ["AP", "AR"]),
        ("Rmax", ["AP", "AR"]),
    ]
    for parameter, metrics in specs:
        subset = [row for row in rows if row["parameter"] == parameter]
        if not subset:
            continue
        x = [float(row["value"]) for row in subset]
        for metric in metrics:
            y = [row.get(metric) for row in subset]
            if any(value == "" for value in y):
                continue
            plt.figure()
            plt.plot(x, [float(value) for value in y], marker="o")
            plt.xlabel(parameter)
            plt.ylabel(metric)
            plt.title(f"{metric} vs {parameter}")
            plt.grid(True)
            plt.savefig(plot_dir / f"{metric}_vs_{parameter}.png", bbox_inches="tight")
            plt.close()


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-dir")
    parser.add_argument("--predictions-dir")
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--annotations")
    parser.add_argument("--bbox-scale", choices=["auto", "normalized", "pixel"], default="auto")
    parser.add_argument("--output-dir", default="outputs/parameter_sweep")
    parser.add_argument("--iou-threshold", type=float, default=0.55)
    parser.add_argument("--score-threshold", type=float, default=0.0)
    parser.add_argument("--cooperation-threshold", type=float, default=0.5)
    parser.add_argument("--negotiation-threshold", type=float, default=0.05)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--weight", type=float, default=0.3)
    parser.add_argument("--disable-preclustering", action="store_true")
    args = parser.parse_args(argv)
    csv_path = run_sweep(args)
    print(f"Wrote parameter sweep results to {csv_path}")


if __name__ == "__main__":
    main()
