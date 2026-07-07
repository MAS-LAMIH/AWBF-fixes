#!/usr/bin/env python3
"""Reproduce/smoke-test the PAPER_SPEC AWBF evaluation pipeline.

Full mode expects COCO annotations and one JSON prediction file per detector in
--predictions-dir. Each prediction JSON must be COCO detection format with
image_id, category_id, bbox (xywh), and score. The script fuses per image and
exports WBF, AWBF, AWBF-competition, and AWBF-Negotiation results, then evaluates
with pycocotools when annotations are supplied.
"""
from __future__ import annotations
import argparse, csv, json, math, os, sys, time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wbf_agents.awbf import Detection, awbf_competition, awbf_negotiation, cluster_detections_by_label_and_iou, cluster_stats, convert_coco_detection_bboxes, decentralized_wbf, incremental_awbf, incremental_decentralized_wbf, evaluate_coco, export_coco_detections, load_coco_image_sizes
from scripts.download_benchmark import EXPECTED_FILES, ensure_benchmark, find_expected_file, validate_benchmark_files

PAPER_TARGETS={
 'WBF': {'AP':0.673,'AP50':0.894,'AP75':0.709},
 'AWBF': {'AP':0.610,'AP50':0.660,'AP75':0.625},
 'AWBF-competition': {'AP':0.651,'AP50':0.666,'AP75':0.590},
 'AWBF-competition-IncrementalState': {'AP': None, 'AP50': None, 'AP75': None},
 'AWBF-Negotiation': {'AP':0.626,'AP50':0.684,'AP75':0.640},
 'AWBF-Negotiation-IncrementalState': {'AP': None, 'AP50': None, 'AP75': None},
 'Incremental_AWBF': {'AP': None, 'AP50': None, 'AP75': None},
}

OUTPUT_FILENAMES = {
    'WBF': 'wbf_predictions.json',
    'AWBF': 'awbf_predictions.json',
    'AWBF-competition': 'awbf_competition_predictions.json',
    'AWBF-competition-IncrementalState': 'awbf_competition_incremental_state_predictions.json',
    'AWBF-Negotiation': 'awbf_negotiation_predictions.json',
    'AWBF-Negotiation-IncrementalState': 'awbf_negotiation_incremental_state_predictions.json',
    'Incremental_AWBF': 'incremental_awbf_predictions.json',
}
METHOD_NAMES = tuple(PAPER_TARGETS.keys())
METHOD_CHOICES = ('all',) + METHOD_NAMES

def resolve_methods(requested_methods):
    requested_methods = requested_methods or ['all']
    if 'all' in requested_methods:
        return list(METHOD_NAMES)
    selected = []
    for method in requested_methods:
        if method not in METHOD_NAMES:
            raise ValueError(f"Unknown method {method!r}. Choose from: {', '.join(METHOD_CHOICES)}")
        if method not in selected:
            selected.append(method)
    return selected

def format_duration(seconds):
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, sec = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes}m {sec}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes}m"

def estimate_eta(start_time, completed, total):
    if completed <= 0:
        return "unknown"
    elapsed = time.perf_counter() - start_time
    remaining = max(0, total - completed)
    return format_duration((elapsed / completed) * remaining)


def _sorted_detections_for_compare(detections):
    return sorted(
        detections,
        key=lambda d: (d.label, round(d.score, 12), tuple(round(v, 12) for v in d.box)),
    )

def compare_method_outputs(outputs, method_a, method_b, tolerance=1e-9):
    image_ids = sorted(set(outputs.get(method_a, {})) | set(outputs.get(method_b, {})))
    count_a = sum(len(outputs.get(method_a, {}).get(image_id, [])) for image_id in image_ids)
    count_b = sum(len(outputs.get(method_b, {}).get(image_id, [])) for image_id in image_ids)
    max_coordinate_difference = 0.0
    max_score_difference = 0.0
    compared = 0
    for image_id in image_ids:
        left = _sorted_detections_for_compare(outputs.get(method_a, {}).get(image_id, []))
        right = _sorted_detections_for_compare(outputs.get(method_b, {}).get(image_id, []))
        for det_a, det_b in zip(left, right):
            compared += 1
            max_coordinate_difference = max(max_coordinate_difference, *(abs(det_a.box[i] - det_b.box[i]) for i in range(4)))
            max_score_difference = max(max_score_difference, abs(det_a.score - det_b.score))
    identical = count_a == count_b and max_coordinate_difference == 0.0 and max_score_difference == 0.0
    nearly_identical = count_a == count_b and max_coordinate_difference <= tolerance and max_score_difference <= tolerance
    if identical:
        status = "identical"
    elif nearly_identical:
        status = "nearly identical"
    else:
        status = "different"
    return {
        "method_a": method_a,
        "method_b": method_b,
        "detection_count_a": count_a,
        "detection_count_b": count_b,
        "count_difference": count_b - count_a,
        "detections_compared": compared,
        "max_coordinate_difference": max_coordinate_difference,
        "max_score_difference": max_score_difference,
        "equivalence": status,
    }

def count_method_detections(outputs):
    return {method: sum(len(dets) for dets in images.values()) for method, images in outputs.items()}

def build_method_equivalence_audit(outputs):
    pairs = [
        ("WBF", "AWBF"),
        ("AWBF", "Incremental_AWBF"),
        ("AWBF-competition", "AWBF-competition-IncrementalState"),
        ("AWBF-Negotiation", "AWBF-Negotiation-IncrementalState"),
    ]
    return {f"{a} vs {b}": compare_method_outputs(outputs, a, b) for a, b in pairs if a in outputs and b in outputs}

def write_method_equivalence_audit(path, audit, method_counts):
    lines = [
        "# Method Equivalence Audit",
        "",
        "This file is generated by `scripts/reproduce_paper_results.py` for the current input data.",
        "",
        "## Summary",
        "",
        "| Comparison | Count A | Count B | Max coordinate difference | Max score difference | Result |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    if audit:
        for name, row in audit.items():
            lines.append(
                f"| {name} | {row['detection_count_a']} | {row['detection_count_b']} | "
                f"{row['max_coordinate_difference']:.12g} | {row['max_score_difference']:.12g} | {row['equivalence']} |"
            )
    else:
        lines.append("| No selected method pairs are available for equivalence comparison |  |  |  |  | skipped |")
    lines.extend(["", "## Detection counts", ""])
    for method, count in method_counts.items():
        lines.append(f"- `{method}`: {count}")
    lines.extend([
        "",
        "## Interpretation",
        "",
        "- `WBF` and `AWBF` are expected to match in this implementation because `AWBF` is the paper-style decentralized execution of the same confidence-weighted cluster fusion used by WBF.",
        "- `Incremental_AWBF` is expected to match `AWBF` coordinates and scores when cluster membership is fixed; it changes execution style, not the weighted-average formula.",
        "- `AWBF-competition-IncrementalState` and `AWBF-Negotiation-IncrementalState` are reported separately because incrementally collapsing post-interaction cluster state can change detection counts and coordinates compared with paper-style outputs.",
        "",
    ])
    Path(path).write_text("\n".join(lines))

def xywh_to_xyxy(b): return (float(b[0]), float(b[1]), float(b[0])+float(b[2]), float(b[1])+float(b[3]))

def load_predictions(predictions_dir):
    by_model={}
    for path in sorted(Path(predictions_dir).glob('*.json')):
        rows=json.loads(path.read_text())
        per_image=defaultdict(list)
        for r in rows:
            per_image[int(r['image_id'])].append(Detection(xywh_to_xyxy(r['bbox']), float(r['score']), int(r['category_id']), path.stem))
        by_model[path.stem]=per_image
    if not by_model: raise FileNotFoundError(f'No *.json prediction files found in {predictions_dir}')
    return by_model

def load_benchmark_predictions(benchmark_dir):
    extract_dir = Path(benchmark_dir) / 'benchmark'
    print(f"Loading benchmark CSV files from {extract_dir}", flush=True)
    validate_benchmark_files(extract_dir)
    by_model = {}
    for index, expected in enumerate(EXPECTED_FILES, start=1):
        path = find_expected_file(extract_dir, expected)
        model_name = Path(expected).stem
        print(f"  loading {index}/{len(EXPECTED_FILES)}: {expected}", flush=True)
        per_image = defaultdict(list)
        with open(path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            required = {'img_id', 'label', 'score', 'x1', 'x2', 'y1', 'y2'}
            missing = required.difference(reader.fieldnames or [])
            if missing:
                raise ValueError(f"Benchmark file {path} is missing columns: {', '.join(sorted(missing))}")
            for row in reader:
                per_image[int(row['img_id'])].append(
                    Detection(
                        (float(row['x1']), float(row['y1']), float(row['x2']), float(row['y2'])),
                        float(row['score']),
                        int(float(row['label'])),
                        model_name,
                    )
                )
        detections = sum(len(items) for items in per_image.values())
        print(f"    loaded {detections} detections across {len(per_image)} images", flush=True)
        by_model[model_name] = per_image
    print(f"Loaded {len(by_model)} benchmark CSV files", flush=True)
    return by_model


def load_coco_metadata(annotations_json):
    if not annotations_json:
        return None
    with open(annotations_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    images = {int(img["id"]): {"width": float(img["width"]), "height": float(img["height"])} for img in data.get("images", [])}
    categories = {int(cat["id"]) for cat in data.get("categories", [])}
    return {"images": images, "categories": categories, "annotations": annotations_json}

def validate_coco_detection_rows(rows, coco_meta=None):
    images = (coco_meta or {}).get("images", {})
    categories = (coco_meta or {}).get("categories", set())
    stats = {
        "total_rows": len(rows),
        "valid_rows": 0,
        "invalid_rows": 0,
        "missing_image_ids": [],
        "invalid_category_ids": [],
        "invalid_scores": 0,
        "invalid_bboxes": 0,
        "non_pixel_or_tiny_bboxes": 0,
        "normalized_looking_bboxes": 0,
        "clipped_or_out_of_bounds_bboxes": 0,
        "empty_or_nonpositive_bboxes_removed": 0,
    }
    missing_images = set()
    invalid_categories = set()
    valid_rows = []
    for row in rows:
        valid = True
        image_id = int(row.get("image_id"))
        category_id = int(row.get("category_id"))
        score = float(row.get("score", -1.0))
        bbox = [float(v) for v in row.get("bbox", [])]
        if images and image_id not in images:
            missing_images.add(image_id)
            valid = False
        if categories and category_id not in categories:
            invalid_categories.add(category_id)
            valid = False
        if not (0.0 <= score <= 1.0) or not math.isfinite(score):
            stats["invalid_scores"] += 1
            valid = False
        if len(bbox) != 4 or not all(math.isfinite(v) for v in bbox):
            stats["invalid_bboxes"] += 1
            valid = False
        else:
            x, y, w, h = bbox
            if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 and 0.0 <= w <= 1.0 and 0.0 <= h <= 1.0:
                stats["normalized_looking_bboxes"] += 1
            if w <= 0.0 or h <= 0.0:
                stats["invalid_bboxes"] += 1
                stats["empty_or_nonpositive_bboxes_removed"] += 1
                valid = False
            elif w <= 1.0 or h <= 1.0:
                stats["non_pixel_or_tiny_bboxes"] += 1
            if images and image_id in images:
                width = images[image_id]["width"]
                height = images[image_id]["height"]
                if x < 0.0 or y < 0.0 or x + w > width + 1e-6 or y + h > height + 1e-6:
                    stats["clipped_or_out_of_bounds_bboxes"] += 1
        if valid:
            valid_rows.append(row)
    stats["missing_image_ids"] = sorted(missing_images)
    stats["invalid_category_ids"] = sorted(invalid_categories)
    stats["valid_rows"] = len(valid_rows)
    stats["invalid_rows"] = len(rows) - len(valid_rows)
    return valid_rows, stats

def initialize_flow_stats(methods):
    return {
        method: {
            "images": 0,
            "input_detections": 0,
            "output_detections": 0,
            "score_sum": 0.0,
            "clusters": 0,
            "clustered_detections": 0,
            "largest_cluster_size": 0,
        }
        for method in methods
    }

def update_flow_stats(flow_stats, method, input_count, output_detections, stats):
    row = flow_stats[method]
    output_count = len(output_detections)
    row["images"] += 1
    row["input_detections"] += input_count
    row["output_detections"] += output_count
    row["score_sum"] += sum(d.score for d in output_detections)
    row["clusters"] += int(stats["clusters"])
    row["clustered_detections"] += input_count
    row["largest_cluster_size"] = max(row["largest_cluster_size"], int(stats["largest_cluster_size"]))

def finalize_flow_stats(flow_stats):
    finalized = {}
    for method, row in flow_stats.items():
        images = row["images"] or 1
        clusters = row["clusters"] or 1
        output = row["output_detections"]
        input_count = row["input_detections"]
        finalized[method] = {
            "input_detections": input_count,
            "output_detections": output,
            "detections_per_image": output / images,
            "average_confidence": (row["score_sum"] / output) if output else 0.0,
            "number_of_clusters": row["clusters"],
            "average_cluster_size": row["clustered_detections"] / clusters,
            "largest_cluster_size": row["largest_cluster_size"],
            "detections_removed": max(0, input_count - output),
            "detections_fused": max(0, input_count - output),
            "detections_retained": output,
        }
    return finalized

def _metric_value(metrics, key):
    value = metrics.get(key)
    return value if isinstance(value, (int, float)) else None

def _fmt(value):
    if value is None:
        return "n/a"
    return f"{float(value):.6f}"

def write_markdown_table(path, title, headers, rows, intro=None):
    lines = [f"# {title}", ""]
    if intro:
        lines.extend([intro, ""])
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    lines.extend("| " + " | ".join(str(cell) for cell in row) + " |" for row in rows)
    Path(path).write_text("\n".join(lines) + "\n")

def write_evaluation_validation_report(path, validation_reports, bbox_reports):
    rows = []
    for method, stats in validation_reports.items():
        bbox = bbox_reports.get(method, {})
        rows.append([
            method,
            stats.get("total_rows", 0),
            stats.get("valid_rows", 0),
            stats.get("invalid_rows", 0),
            bbox.get("boxes_converted_to_pixel", "n/a"),
            bbox.get("boxes_left_unchanged", "n/a"),
            len(stats.get("missing_image_ids", [])),
            len(stats.get("invalid_category_ids", [])),
            stats.get("invalid_scores", 0),
            stats.get("empty_or_nonpositive_bboxes_removed", 0),
            stats.get("normalized_looking_bboxes", 0),
        ])
    write_markdown_table(
        path,
        "Evaluation Validation",
        ["method", "rows", "valid", "invalid", "converted_to_pixel", "left_unchanged", "missing_images", "invalid_categories", "invalid_scores", "empty_bboxes_removed", "normalized_looking_bboxes"],
        rows,
        "Validation checks COCO image/category IDs, score range, pixel-space xywh bboxes, and removal of invalid rows before COCOeval.",
    )

def write_detection_flow_report(path, flow_stats):
    rows = []
    for method, stats in flow_stats.items():
        rows.append([
            method,
            stats["input_detections"],
            stats["output_detections"],
            _fmt(stats["detections_per_image"]),
            _fmt(stats["average_confidence"]),
            stats["number_of_clusters"],
            _fmt(stats["average_cluster_size"]),
            stats["largest_cluster_size"],
            stats["detections_removed"],
            stats["detections_fused"],
            stats["detections_retained"],
        ])
    write_markdown_table(path, "Detection Flow Report", ["method", "input detections", "output detections", "detections/image", "avg confidence", "clusters", "avg cluster size", "largest cluster", "removed", "fused", "retained"], rows)

def write_paper_results_comparison(path, paper_targets, metrics):
    keys = ["AP", "AP50", "AP75", "AR", "AR50", "AR75", "AP_small", "AP_medium", "AP_large", "AR_small", "AR_medium", "AR_large"]
    rows = []
    for method, reproduced in metrics.items():
        if not isinstance(reproduced, dict) or "evaluation" in reproduced:
            continue
        target = paper_targets.get(method, {})
        row = [method]
        for key in keys:
            paper = target.get(key)
            actual = _metric_value(reproduced, key)
            delta = actual - paper if paper is not None and actual is not None else None
            row.extend([_fmt(paper), _fmt(actual), _fmt(delta)])
        rows.append(row)
    headers = ["method"]
    for key in keys:
        headers.extend([f"paper {key}", f"reproduced {key}", f"{key} delta"])
    write_markdown_table(path, "Paper Results Comparison", headers, rows, "`n/a` means the paper target table in this repository did not provide that metric for the method.")

def build_recall_audit(metrics, flow_stats):
    valid_metrics = {m: v for m, v in metrics.items() if isinstance(v, dict) and "AR" in v}
    baseline = valid_metrics.get("WBF")
    best_ar_method = max(valid_metrics, key=lambda m: valid_metrics[m].get("AR", float("-inf"))) if valid_metrics else None
    rows = []
    for method, values in valid_metrics.items():
        ar = values.get("AR")
        ap = values.get("AP")
        ar_delta = ar - baseline.get("AR") if baseline and ar is not None else None
        ap_delta = ap - baseline.get("AP") if baseline and ap is not None else None
        output = flow_stats.get(method, {}).get("output_detections", "n/a")
        rows.append([method, _fmt(ap), _fmt(ar), _fmt(ap_delta), _fmt(ar_delta), output, _fmt(values.get("AR_medium")), _fmt(values.get("AR_large"))])
    return best_ar_method, rows

def write_recall_argument_audit(path, metrics, flow_stats):
    best_ar_method, rows = build_recall_audit(metrics, flow_stats)
    lines = ["# Recall Argument Audit", ""]
    if not rows:
        lines.extend(["COCO metrics were not available, so the recall-oriented argument cannot be evaluated from this run.", ""])
    else:
        lines.extend([
            f"Best AR method: `{best_ar_method}`.",
            "",
            "The table compares AP/AR deltas against WBF. Positive AR delta with smaller or negative AP delta indicates recall-oriented behavior rather than precision-oriented behavior.",
            "",
            "| method | AP | AR | AP delta vs WBF | AR delta vs WBF | output detections | AR medium | AR large |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ])
        lines.extend("| " + " | ".join(str(cell) for cell in row) + " |" for row in rows)
        lines.append("")
    Path(path).write_text("\n".join(lines))

def write_reproduction_limitations(path, args, metrics, validation_reports):
    limitations = []
    if not args.annotations:
        limitations.append("COCO annotations were not supplied, so AP/AR could not be computed.")
    if not metrics or not any(isinstance(v, dict) and "AP" in v for v in metrics.values()):
        limitations.append("No computed COCO AP/AR metrics are available for comparison.")
    for method, stats in validation_reports.items():
        if stats.get("invalid_rows", 0):
            limitations.append(f"{method} had {stats['invalid_rows']} invalid prediction rows removed before evaluation.")
        if stats.get("invalid_category_ids"):
            limitations.append(f"{method} used category IDs not present in the annotation file: {stats['invalid_category_ids'][:10]}")
    lines = ["# Reproduction Limitations", "", "## Inputs", "", f"- Benchmark directory: `{args.benchmark_dir or 'n/a'}`", f"- Annotation file: `{args.annotations or 'n/a'}`", f"- bbox scale mode: `{args.bbox_scale}`", f"- IoU threshold: `{args.iou_threshold}`", f"- Score threshold: `{args.score_threshold}`", "", "## Limitations", ""]
    if limitations:
        lines.extend(f"- {item}" for item in limitations)
    else:
        lines.append("- No blocking validation limitations were detected by the reproduction script.")
    lines.extend(["", "## Likely causes of mismatch from the paper", "", "- Detector provenance and detector training/inference settings may differ from the original paper.", "- Benchmark CSV labels/scores may not encode every detector calibration detail used in the paper.", "- Implementation details for AWBF competition and negotiation are reconstructed from the paper/spec and may not match unreleased code exactly.", "- Parameter choices such as IoU threshold, negotiation rounds, and attack/defense thresholds can change AP/AR tradeoffs.", "", "The qualitative recall argument is considered supported only when nonzero COCO AR is computed and AR trends are explicitly compared against AP trends in `RECALL_ARGUMENT_AUDIT.md`."])
    Path(path).write_text("\n".join(lines) + "\n")

def sample_predictions():
    return {'det_a':{1:[Detection((0,0,1,1),.9,1,'det_a'), Detection((2,2,3,3),.7,2,'det_a')]}, 'det_b':{1:[Detection((.1,0,1.1,1),.8,1,'det_b')]}}

def detections_by_source(detections):
    grouped = defaultdict(list)
    for det in detections:
        grouped[det.source].append(det)
    return grouped

def fuse_wbf_clusters(clusters, iou_threshold, score_threshold):
    fused = []
    for cluster in clusters:
        fused.extend(decentralized_wbf(detections_by_source(cluster), iou_threshold, score_threshold))
    return fused

def fuse_incremental_wbf_clusters(clusters, iou_threshold, score_threshold):
    fused = []
    for cluster in clusters:
        fused.extend(incremental_decentralized_wbf(detections_by_source(cluster), iou_threshold, score_threshold))
    return fused

def run_clustered_strategy(clusters, strategy):
    fused = []
    for cluster in clusters:
        fused.extend(strategy(cluster))
    return fused

def fuse_all(by_model, args):
    selected_methods = resolve_methods(args.methods)
    selected = set(selected_methods)
    image_ids=sorted({i for model in by_model.values() for i in model})
    outputs={k:{} for k in selected_methods}
    timings={k: 0.0 for k in selected_methods}
    flow_stats = initialize_flow_stats(selected_methods)
    total = len(image_ids)
    interval = max(1, args.progress_interval)
    run_start = time.perf_counter()
    print(f"Fusing predictions for {total} images using {len(by_model)} detector/model files", flush=True)
    for index, image_id in enumerate(image_ids, start=1):
        per_model={m: imgs.get(image_id, []) for m, imgs in by_model.items()}
        flat=[d for ds in per_model.values() for d in ds]
        box_count = len(flat)
        if args.disable_preclustering:
            clusters = [flat]
        else:
            clusters = cluster_detections_by_label_and_iou(flat, args.iou_threshold)
        stats = cluster_stats(flat, clusters)
        should_report = args.profile or index == 1 or index == total or index % interval == 0
        if should_report:
            print(
                f"image {index}/{total} image_id={image_id} boxes={box_count} "
                f"clusters={int(stats['clusters'])} largest_cluster={int(stats['largest_cluster_size'])} "
                f"avg_cluster={stats['average_cluster_size']:.1f} "
                f"pair_comparisons_global={int(stats['global_pair_comparisons'])} "
                f"pair_comparisons_clustered={int(stats['clustered_pair_comparisons'])} "
                f"comparisons_avoided={int(stats['comparisons_avoided_estimate'])} "
                f"elapsed={format_duration(time.perf_counter() - run_start)} "
                f"ETA={estimate_eta(run_start, index - 1, total) if index > 1 else 'unknown'}",
                flush=True,
            )

        if 'WBF' in selected:
            start = time.perf_counter()
            if args.disable_preclustering:
                outputs['WBF'][image_id]=decentralized_wbf(per_model,args.iou_threshold,args.score_threshold)
            else:
                outputs['WBF'][image_id]=fuse_wbf_clusters(clusters,args.iou_threshold,args.score_threshold)
            timings['WBF'] += time.perf_counter() - start
            update_flow_stats(flow_stats, 'WBF', box_count, outputs['WBF'][image_id], stats)

        if 'AWBF' in selected:
            start = time.perf_counter()
            if args.disable_preclustering:
                outputs['AWBF'][image_id]=decentralized_wbf(per_model,args.iou_threshold,args.score_threshold)
            else:
                outputs['AWBF'][image_id]=fuse_wbf_clusters(clusters,args.iou_threshold,args.score_threshold)
            timings['AWBF'] += time.perf_counter() - start
            update_flow_stats(flow_stats, 'AWBF', box_count, outputs['AWBF'][image_id], stats)

        if 'Incremental_AWBF' in selected:
            start = time.perf_counter()
            if args.disable_preclustering:
                outputs['Incremental_AWBF'][image_id]=incremental_decentralized_wbf(per_model,args.iou_threshold,args.score_threshold)
            else:
                outputs['Incremental_AWBF'][image_id]=fuse_incremental_wbf_clusters(clusters,args.iou_threshold,args.score_threshold)
            timings['Incremental_AWBF'] += time.perf_counter() - start
            update_flow_stats(flow_stats, 'Incremental_AWBF', box_count, outputs['Incremental_AWBF'][image_id], stats)

        comp_clusters = None
        comp_elapsed = 0.0
        if 'AWBF-competition' in selected or 'AWBF-competition-IncrementalState' in selected:
            start = time.perf_counter()
            if args.disable_preclustering:
                comp_clusters=[awbf_competition(flat,args.iou_threshold,args.cooperation_threshold)]
            else:
                comp_clusters=[awbf_competition(c,args.iou_threshold,args.cooperation_threshold) for c in clusters]
            comp_elapsed = time.perf_counter() - start
            if 'AWBF-competition' in selected:
                outputs['AWBF-competition'][image_id]=[d for cluster in comp_clusters for d in cluster]
                timings['AWBF-competition'] += comp_elapsed
                update_flow_stats(flow_stats, 'AWBF-competition', box_count, outputs['AWBF-competition'][image_id], stats)

        if 'AWBF-competition-IncrementalState' in selected:
            start = time.perf_counter()
            outputs['AWBF-competition-IncrementalState'][image_id]=incremental_awbf([cluster for cluster in comp_clusters if cluster])
            timings['AWBF-competition-IncrementalState'] += comp_elapsed + (time.perf_counter() - start)
            update_flow_stats(flow_stats, 'AWBF-competition-IncrementalState', box_count, outputs['AWBF-competition-IncrementalState'][image_id], stats)

        def negotiation_progress(event):
            if args.profile:
                elapsed = time.perf_counter() - run_start
                print(
                    f"  negotiation image {index}/{total} round={event['pass']}/? "
                    f"remaining_boxes={event['remaining_boxes']} "
                    f"comparisons={event['total_comparisons']} "
                    f"last_round_comparisons={event['pass_comparisons']} "
                    f"elapsed={format_duration(elapsed)} ETA={estimate_eta(run_start, index - 1, total) if index > 1 else 'unknown'}",
                    flush=True,
                )

        neg_clusters = None
        neg_elapsed = 0.0
        if 'AWBF-Negotiation' in selected or 'AWBF-Negotiation-IncrementalState' in selected:
            start = time.perf_counter()
            if args.disable_preclustering:
                neg_clusters=[awbf_negotiation(
                    flat,
                    args.iou_threshold,
                    args.rounds,
                    args.weight,
                    args.negotiation_threshold,
                    progress_callback=negotiation_progress if args.profile else None,
                )]
            else:
                neg_clusters=[awbf_negotiation(
                    c,
                    args.iou_threshold,
                    args.rounds,
                    args.weight,
                    args.negotiation_threshold,
                    progress_callback=negotiation_progress if args.profile else None,
                ) for c in clusters]
            neg_elapsed = time.perf_counter() - start
            if 'AWBF-Negotiation' in selected:
                outputs['AWBF-Negotiation'][image_id]=[d for cluster in neg_clusters for d in cluster]
                timings['AWBF-Negotiation'] += neg_elapsed
                update_flow_stats(flow_stats, 'AWBF-Negotiation', box_count, outputs['AWBF-Negotiation'][image_id], stats)

        if 'AWBF-Negotiation-IncrementalState' in selected:
            start = time.perf_counter()
            outputs['AWBF-Negotiation-IncrementalState'][image_id]=incremental_awbf([cluster for cluster in neg_clusters if cluster])
            timings['AWBF-Negotiation-IncrementalState'] += neg_elapsed + (time.perf_counter() - start)
            update_flow_stats(flow_stats, 'AWBF-Negotiation-IncrementalState', box_count, outputs['AWBF-Negotiation-IncrementalState'][image_id], stats)

        if args.profile:
            print(
                f"  timings image {index}/{total}: "
                + ", ".join(f"{method}={format_duration(timings[method])} total" for method in selected_methods),
                flush=True,
            )
    print("Fusion complete", flush=True)
    print("Fusion timing summary:", flush=True)
    for method, elapsed in timings.items():
        print(f"  {method}: {format_duration(elapsed)}", flush=True)
    return outputs, timings, finalize_flow_stats(flow_stats)

def main(argv=None):
    ap=argparse.ArgumentParser()
    ap.add_argument('--annotations'); ap.add_argument('--predictions-dir'); ap.add_argument('--output-dir',default='outputs/reproduction')
    ap.add_argument('--bbox-scale', choices=['auto', 'normalized', 'pixel'], default='auto', help='Scale COCO xywh boxes: auto detects normalized boxes, normalized always scales, pixel leaves unchanged')
    ap.add_argument('--evaluate-predictions', nargs='+', help='Evaluate one or more existing COCO detection JSON files and skip benchmark loading/fusion')
    ap.add_argument('--download-benchmark', action='store_true', help='Download/extract the WBF benchmark if required files are missing')
    ap.add_argument('--benchmark-zip', help='Use a manually downloaded local benchmark.zip instead of downloading from GitHub')
    ap.add_argument('--benchmark-dir', help='Directory containing benchmark.zip and benchmark/ extraction')
    ap.add_argument('--sample', action='store_true'); ap.add_argument('--iou-threshold',type=float,default=.55); ap.add_argument('--score-threshold',type=float,default=0.0)
    ap.add_argument('--cooperation-threshold',type=float,default=.5); ap.add_argument('--negotiation-threshold',type=float,default=.05); ap.add_argument('--rounds',type=int,default=5); ap.add_argument('--weight',type=float,default=.3)
    ap.add_argument('--progress-interval', type=int, default=100, help='Print fusion progress every N images')
    ap.add_argument('--profile', action='store_true', help='Print detailed per-image/per-strategy timing and AWBF negotiation progress')
    ap.add_argument('--disable-preclustering', action='store_true', help='Disable label/IoU pre-clustering and run previous global per-image fusion behavior')
    ap.add_argument('--incremental-cluster-state', action='store_true', help='Deprecated compatibility flag; incremental-state competition/negotiation variants are selected with --methods')
    ap.add_argument('--methods', nargs='+', choices=METHOD_CHOICES, default=['all'], help='Fusion methods to run. Use one or more method names, or all. Default: all')
    args=ap.parse_args(argv)
    if args.evaluate_predictions:
        if not args.annotations:
            raise SystemExit("--annotations is required with --evaluate-predictions")
        image_sizes = load_coco_image_sizes(args.annotations)
        coco_meta = load_coco_metadata(args.annotations)
        report = {"annotations": args.annotations, "bbox_scale_mode": args.bbox_scale, "metrics": {}, "bbox_reports": {}, "evaluation_validation": {}}
        eval_output_dir = Path(args.output_dir) if args.output_dir != 'outputs/reproduction' else Path('outputs')
        eval_output_dir.mkdir(parents=True, exist_ok=True)
        for prediction_json in args.evaluate_predictions:
            with open(prediction_json, "r", encoding="utf-8") as f:
                original_rows = json.load(f)
            converted_rows, bbox_stats = convert_coco_detection_bboxes(original_rows, image_sizes=image_sizes, bbox_scale=args.bbox_scale)
            valid_rows, validation_stats = validate_coco_detection_rows(converted_rows, coco_meta)
            eval_file = prediction_json
            if bbox_stats["boxes_converted_to_pixel"] > 0 or len(valid_rows) != len(converted_rows):
                suffix = "pixel_xywh" if bbox_stats["boxes_converted_to_pixel"] > 0 else "validated"
                eval_file_path = eval_output_dir / f"{Path(prediction_json).stem}_{suffix}.json"
                eval_file = eval_file_path.as_posix()
                with open(eval_file_path, "w", encoding="utf-8") as f:
                    json.dump(valid_rows, f, indent=2)
                bbox_stats["converted_prediction_file"] = eval_file
            print(f"Evaluating predictions: {eval_file}", flush=True)
            metrics = evaluate_coco(args.annotations, eval_file)
            report["metrics"][prediction_json] = metrics
            report["bbox_reports"][prediction_json] = bbox_stats
            report["evaluation_validation"][prediction_json] = validation_stats
            print(f"Metrics for {prediction_json}:", flush=True)
            for key, value in metrics.items():
                print(f"  {key}: {value:.6f}", flush=True)
        write_evaluation_validation_report(eval_output_dir / "EVALUATION_VALIDATION.md", report["evaluation_validation"], report["bbox_reports"])
        report_path = eval_output_dir / "evaluation_report.json"
        report_path.write_text(json.dumps(report, indent=2))
        print(f"Saved evaluation report to {report_path}", flush=True)
        return report
    if args.sample:
        by_model=sample_predictions()
    elif args.download_benchmark or args.benchmark_zip or args.benchmark_dir:
        benchmark_dir = args.benchmark_dir or 'data/benchmark'
        benchmark_extract = Path(benchmark_dir) / 'benchmark'
        try:
            validate_benchmark_files(benchmark_extract)
        except FileNotFoundError:
            ensure_benchmark(benchmark_dir, benchmark_zip=args.benchmark_zip)
        by_model=load_benchmark_predictions(benchmark_dir)
    else:
        if not args.predictions_dir: raise SystemExit('--predictions-dir is required unless --sample, --benchmark-dir, --download-benchmark, or --benchmark-zip is used')
        by_model=load_predictions(args.predictions_dir)
    os.makedirs(args.output_dir,exist_ok=True)
    image_sizes = load_coco_image_sizes(args.annotations) if args.annotations else None
    coco_meta = load_coco_metadata(args.annotations) if args.annotations else None
    selected_methods=resolve_methods(args.methods); outputs,timings,flow_stats=fuse_all(by_model,args); method_audit=build_method_equivalence_audit(outputs); method_counts=count_method_detections(outputs); report={'paper_targets':PAPER_TARGETS,'selected_methods':selected_methods,'metrics':{},'bbox_scale_mode':args.bbox_scale,'bbox_reports':{},'evaluation_validation':{},'timings_seconds':timings,'method_detection_counts':method_counts,'detection_flow':flow_stats,'method_equivalence_audit':method_audit,'notes':[]}
    for method,dets in outputs.items():
        pred_path=os.path.join(args.output_dir,OUTPUT_FILENAMES[method])
        total_detections = sum(len(v) for v in dets.values())
        print(f"Exporting {method}: {total_detections} detections -> {pred_path}", flush=True)
        rows, bbox_stats = export_coco_detections(dets,pred_path,image_sizes=image_sizes,bbox_scale=args.bbox_scale,return_stats=True)
        report['bbox_reports'][method] = bbox_stats
        valid_rows, validation_stats = validate_coco_detection_rows(rows, coco_meta)
        report['evaluation_validation'][method] = validation_stats
        if len(valid_rows) != len(rows):
            print(f"Validation removed {len(rows) - len(valid_rows)} invalid rows for {method} before evaluation/export", flush=True)
            with open(pred_path, "w", encoding="utf-8") as f:
                json.dump(valid_rows, f, indent=2)
        if args.annotations:
            print(f"Evaluating {method} with COCO annotations {args.annotations}", flush=True)
            metrics=evaluate_coco(args.annotations,pred_path)
            report['metrics'][method]=metrics
        else:
            report['metrics'][method]={'detections':sum(len(v) for v in dets.values()),'evaluation':'skipped: provide --annotations for COCO metrics'}
    if args.sample: report['notes'].append('Sample mode validates fusion/export only; it cannot reproduce paper metrics without COCO annotations and detector predictions.')
    if not args.annotations:
        message = 'COCO AP/AR metrics cannot be computed because annotations are missing.' if (args.download_benchmark or args.benchmark_zip or args.benchmark_dir) else 'COCO AP/AR evaluation skipped because --annotations was not supplied; fusion/export does not require full COCO annotations.'
        report['notes'].append(message)
    write_method_equivalence_audit(Path(args.output_dir, 'METHOD_EQUIVALENCE_AUDIT.md'), method_audit, method_counts)
    write_evaluation_validation_report(Path(args.output_dir, 'EVALUATION_VALIDATION.md'), report['evaluation_validation'], report['bbox_reports'])
    write_detection_flow_report(Path(args.output_dir, 'DETECTION_FLOW_REPORT.md'), flow_stats)
    write_paper_results_comparison(Path(args.output_dir, 'PAPER_RESULTS_COMPARISON.md'), PAPER_TARGETS, report['metrics'])
    write_recall_argument_audit(Path(args.output_dir, 'RECALL_ARGUMENT_AUDIT.md'), report['metrics'], flow_stats)
    write_reproduction_limitations(Path(args.output_dir, 'REPRODUCTION_LIMITATIONS.md'), args, report['metrics'], report['evaluation_validation'])
    Path(args.output_dir,'reproduction_report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
if __name__=='__main__': main()
