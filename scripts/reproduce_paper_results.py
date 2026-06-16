#!/usr/bin/env python3
"""Reproduce/smoke-test the PAPER_SPEC AWBF evaluation pipeline.

Full mode expects COCO annotations and one JSON prediction file per detector in
--predictions-dir. Each prediction JSON must be COCO detection format with
image_id, category_id, bbox (xywh), and score. The script fuses per image and
exports WBF, AWBF, AWBF-competition, and AWBF-Negotiation results, then evaluates
with pycocotools when annotations are supplied.
"""
from __future__ import annotations
import argparse, csv, json, os, sys, time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wbf_agents.awbf import Detection, awbf_competition, awbf_negotiation, cluster_detections_by_label_and_iou, cluster_stats, convert_coco_detection_bboxes, decentralized_wbf, evaluate_coco, export_coco_detections, load_coco_image_sizes
from scripts.download_benchmark import EXPECTED_FILES, ensure_benchmark, find_expected_file, validate_benchmark_files

PAPER_TARGETS={
 'WBF': {'AP':0.673,'AP50':0.894,'AP75':0.709},
 'AWBF': {'AP':0.610,'AP50':0.660,'AP75':0.625},
 'AWBF-competition': {'AP':0.651,'AP50':0.666,'AP75':0.590},
 'AWBF-Negotiation': {'AP':0.626,'AP50':0.684,'AP75':0.640},
}

OUTPUT_FILENAMES = {
    'WBF': 'wbf_predictions.json',
    'AWBF': 'awbf_predictions.json',
    'AWBF-competition': 'awbf_competition_predictions.json',
    'AWBF-Negotiation': 'awbf_negotiation_predictions.json',
}

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

def run_clustered_strategy(clusters, strategy):
    fused = []
    for cluster in clusters:
        fused.extend(strategy(cluster))
    return fused

def fuse_all(by_model, args):
    image_ids=sorted({i for model in by_model.values() for i in model})
    outputs={k:{} for k in PAPER_TARGETS}
    timings={k: 0.0 for k in PAPER_TARGETS}
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

        start = time.perf_counter()
        if args.disable_preclustering:
            outputs['WBF'][image_id]=decentralized_wbf(per_model,args.iou_threshold,args.score_threshold)
        else:
            outputs['WBF'][image_id]=fuse_wbf_clusters(clusters,args.iou_threshold,args.score_threshold)
        timings['WBF'] += time.perf_counter() - start

        start = time.perf_counter()
        if args.disable_preclustering:
            outputs['AWBF'][image_id]=decentralized_wbf(per_model,args.iou_threshold,args.score_threshold)
        else:
            outputs['AWBF'][image_id]=fuse_wbf_clusters(clusters,args.iou_threshold,args.score_threshold)
        timings['AWBF'] += time.perf_counter() - start

        start = time.perf_counter()
        if args.disable_preclustering:
            outputs['AWBF-competition'][image_id]=awbf_competition(flat,args.iou_threshold,args.cooperation_threshold)
        else:
            outputs['AWBF-competition'][image_id]=run_clustered_strategy(clusters, lambda c: awbf_competition(c,args.iou_threshold,args.cooperation_threshold))
        timings['AWBF-competition'] += time.perf_counter() - start

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

        start = time.perf_counter()
        if args.disable_preclustering:
            outputs['AWBF-Negotiation'][image_id]=awbf_negotiation(
                flat,
                args.iou_threshold,
                args.rounds,
                args.weight,
                args.negotiation_threshold,
                progress_callback=negotiation_progress if args.profile else None,
            )
        else:
            outputs['AWBF-Negotiation'][image_id]=run_clustered_strategy(
                clusters,
                lambda c: awbf_negotiation(
                    c,
                    args.iou_threshold,
                    args.rounds,
                    args.weight,
                    args.negotiation_threshold,
                    progress_callback=negotiation_progress if args.profile else None,
                ),
            )
        timings['AWBF-Negotiation'] += time.perf_counter() - start

        if args.profile:
            print(
                f"  timings image {index}/{total}: "
                f"WBF={format_duration(timings['WBF'])} total, "
                f"AWBF={format_duration(timings['AWBF'])} total, "
                f"Competition={format_duration(timings['AWBF-competition'])} total, "
                f"Negotiation={format_duration(timings['AWBF-Negotiation'])} total",
                flush=True,
            )
    print("Fusion complete", flush=True)
    print("Fusion timing summary:", flush=True)
    for method, elapsed in timings.items():
        print(f"  {method}: {format_duration(elapsed)}", flush=True)
    return outputs, timings

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
    args=ap.parse_args(argv)
    if args.evaluate_predictions:
        if not args.annotations:
            raise SystemExit("--annotations is required with --evaluate-predictions")
        image_sizes = load_coco_image_sizes(args.annotations)
        report = {"annotations": args.annotations, "bbox_scale_mode": args.bbox_scale, "metrics": {}, "bbox_reports": {}}
        eval_output_dir = Path(args.output_dir) if args.output_dir != 'outputs/reproduction' else Path('outputs')
        eval_output_dir.mkdir(parents=True, exist_ok=True)
        for prediction_json in args.evaluate_predictions:
            with open(prediction_json, "r", encoding="utf-8") as f:
                original_rows = json.load(f)
            converted_rows, bbox_stats = convert_coco_detection_bboxes(original_rows, image_sizes=image_sizes, bbox_scale=args.bbox_scale)
            eval_file = prediction_json
            if bbox_stats["boxes_converted_to_pixel"] > 0:
                eval_file = str(eval_output_dir / f"{Path(prediction_json).stem}_pixel_xywh.json")
                with open(eval_file, "w", encoding="utf-8") as f:
                    json.dump(converted_rows, f, indent=2)
                bbox_stats["converted_prediction_file"] = eval_file
            print(f"Evaluating predictions: {eval_file}", flush=True)
            metrics = evaluate_coco(args.annotations, eval_file)
            report["metrics"][prediction_json] = metrics
            report["bbox_reports"][prediction_json] = bbox_stats
            print(f"Metrics for {prediction_json}:", flush=True)
            for key, value in metrics.items():
                print(f"  {key}: {value:.6f}", flush=True)
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
    outputs,timings=fuse_all(by_model,args); report={'paper_targets':PAPER_TARGETS,'metrics':{},'bbox_scale_mode':args.bbox_scale,'bbox_reports':{},'timings_seconds':timings,'notes':[]}
    for method,dets in outputs.items():
        pred_path=os.path.join(args.output_dir,OUTPUT_FILENAMES[method])
        total_detections = sum(len(v) for v in dets.values())
        print(f"Exporting {method}: {total_detections} detections -> {pred_path}", flush=True)
        _rows, bbox_stats = export_coco_detections(dets,pred_path,image_sizes=image_sizes,bbox_scale=args.bbox_scale,return_stats=True)
        report['bbox_reports'][method] = bbox_stats
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
    Path(args.output_dir,'reproduction_report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
if __name__=='__main__': main()
