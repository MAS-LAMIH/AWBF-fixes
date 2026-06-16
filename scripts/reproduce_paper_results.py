#!/usr/bin/env python3
"""Reproduce/smoke-test the PAPER_SPEC AWBF evaluation pipeline.

Full mode expects COCO annotations and one JSON prediction file per detector in
--predictions-dir. Each prediction JSON must be COCO detection format with
image_id, category_id, bbox (xywh), and score. The script fuses per image and
exports WBF, AWBF, AWBF-competition, and AWBF-Negotiation results, then evaluates
with pycocotools when annotations are supplied.
"""
from __future__ import annotations
import argparse, csv, json, os, sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wbf_agents.awbf import Detection, awbf_competition, awbf_negotiation, decentralized_wbf, evaluate_coco, export_coco_detections
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
    validate_benchmark_files(extract_dir)
    by_model = {}
    for expected in EXPECTED_FILES:
        path = find_expected_file(extract_dir, expected)
        model_name = Path(expected).stem
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
        by_model[model_name] = per_image
    return by_model

def sample_predictions():
    return {'det_a':{1:[Detection((0,0,1,1),.9,1,'det_a'), Detection((2,2,3,3),.7,2,'det_a')]}, 'det_b':{1:[Detection((.1,0,1.1,1),.8,1,'det_b')]}}

def fuse_all(by_model, args):
    image_ids=sorted({i for model in by_model.values() for i in model})
    outputs={k:{} for k in PAPER_TARGETS}
    for image_id in image_ids:
        per_model={m: imgs.get(image_id, []) for m, imgs in by_model.items()}
        flat=[d for ds in per_model.values() for d in ds]
        outputs['WBF'][image_id]=decentralized_wbf(per_model,args.iou_threshold,args.score_threshold)
        outputs['AWBF'][image_id]=decentralized_wbf(per_model,args.iou_threshold,args.score_threshold)
        outputs['AWBF-competition'][image_id]=awbf_competition(flat,args.iou_threshold,args.cooperation_threshold)
        outputs['AWBF-Negotiation'][image_id]=awbf_negotiation(flat,args.iou_threshold,args.rounds,args.weight,args.negotiation_threshold)
    return outputs

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--annotations'); ap.add_argument('--predictions-dir'); ap.add_argument('--output-dir',default='outputs/reproduction')
    ap.add_argument('--download-benchmark', action='store_true', help='Download/extract the WBF benchmark if required files are missing')
    ap.add_argument('--benchmark-zip', help='Use a manually downloaded local benchmark.zip instead of downloading from GitHub')
    ap.add_argument('--benchmark-dir', help='Directory containing benchmark.zip and benchmark/ extraction')
    ap.add_argument('--sample', action='store_true'); ap.add_argument('--iou-threshold',type=float,default=.55); ap.add_argument('--score-threshold',type=float,default=0.0)
    ap.add_argument('--cooperation-threshold',type=float,default=.5); ap.add_argument('--negotiation-threshold',type=float,default=.05); ap.add_argument('--rounds',type=int,default=5); ap.add_argument('--weight',type=float,default=.3)
    args=ap.parse_args()
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
    outputs=fuse_all(by_model,args); report={'paper_targets':PAPER_TARGETS,'metrics':{},'notes':[]}
    for method,dets in outputs.items():
        pred_path=os.path.join(args.output_dir,OUTPUT_FILENAMES[method])
        export_coco_detections(dets,pred_path)
        if args.annotations:
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
