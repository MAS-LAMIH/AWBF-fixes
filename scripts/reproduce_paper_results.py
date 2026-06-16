#!/usr/bin/env python3
"""Reproduce/smoke-test the PAPER_SPEC AWBF evaluation pipeline.

Full mode expects COCO annotations and one JSON prediction file per detector in
--predictions-dir. Each prediction JSON must be COCO detection format with
image_id, category_id, bbox (xywh), and score. The script fuses per image and
exports WBF, AWBF, AWBF-competition, and AWBF-Negotiation results, then evaluates
with pycocotools when annotations are supplied.
"""
from __future__ import annotations
import argparse, json, os, sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wbf_agents.awbf import Detection, awbf_competition, awbf_negotiation, decentralized_wbf, evaluate_coco, export_coco_detections

PAPER_TARGETS={
 'WBF': {'AP':0.673,'AP50':0.894,'AP75':0.709},
 'AWBF': {'AP':0.610,'AP50':0.660,'AP75':0.625},
 'AWBF-competition': {'AP':0.651,'AP50':0.666,'AP75':0.590},
 'AWBF-Negotiation': {'AP':0.626,'AP50':0.684,'AP75':0.640},
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
    ap.add_argument('--sample', action='store_true'); ap.add_argument('--iou-threshold',type=float,default=.55); ap.add_argument('--score-threshold',type=float,default=0.0)
    ap.add_argument('--cooperation-threshold',type=float,default=.5); ap.add_argument('--negotiation-threshold',type=float,default=.05); ap.add_argument('--rounds',type=int,default=5); ap.add_argument('--weight',type=float,default=.3)
    args=ap.parse_args()
    if args.sample: by_model=sample_predictions()
    else:
        if not args.predictions_dir: raise SystemExit('--predictions-dir is required unless --sample is used')
        by_model=load_predictions(args.predictions_dir)
    os.makedirs(args.output_dir,exist_ok=True)
    outputs=fuse_all(by_model,args); report={'paper_targets':PAPER_TARGETS,'metrics':{},'notes':[]}
    for method,dets in outputs.items():
        pred_path=os.path.join(args.output_dir,f'{method}.json')
        export_coco_detections(dets,pred_path)
        if args.annotations:
            metrics=evaluate_coco(args.annotations,pred_path)
            report['metrics'][method]=metrics
        else:
            report['metrics'][method]={'detections':sum(len(v) for v in dets.values()),'evaluation':'skipped: provide --annotations for COCO metrics'}
    if args.sample: report['notes'].append('Sample mode validates fusion/export only; it cannot reproduce paper metrics without COCO annotations and detector predictions.')
    Path(args.output_dir,'reproduction_report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
if __name__=='__main__': main()
