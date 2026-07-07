import json

import pytest

from scripts.reproduce_paper_results import (
    build_recall_audit,
    load_coco_metadata,
    validate_coco_detection_rows,
    write_paper_results_comparison,
    write_recall_argument_audit,
)
from wbf_agents.awbf import evaluate_coco


def _tiny_annotations(path):
    data = {
        "info": {"description": "tiny"},
        "licenses": [],
        "images": [{"id": 1, "width": 100, "height": 100}],
        "categories": [{"id": 3, "name": "car"}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 3, "bbox": [10, 10, 20, 20], "area": 400, "iscrowd": 0}
        ],
    }
    path.write_text(json.dumps(data))


def test_category_image_score_and_bbox_validation(tmp_path):
    ann = tmp_path / "ann.json"
    _tiny_annotations(ann)
    meta = load_coco_metadata(ann)
    rows = [
        {"image_id": 1, "category_id": 3, "bbox": [10, 10, 20, 20], "score": 0.9},
        {"image_id": 999, "category_id": 3, "bbox": [10, 10, 20, 20], "score": 0.9},
        {"image_id": 1, "category_id": 999, "bbox": [10, 10, 20, 20], "score": 0.9},
        {"image_id": 1, "category_id": 3, "bbox": [10, 10, 0, 20], "score": 0.9},
        {"image_id": 1, "category_id": 3, "bbox": [10, 10, 20, 20], "score": 1.5},
    ]
    valid, stats = validate_coco_detection_rows(rows, meta)
    assert valid == [rows[0]]
    assert stats["missing_image_ids"] == [999]
    assert stats["invalid_category_ids"] == [999]
    assert stats["invalid_scores"] == 1
    assert stats["empty_or_nonpositive_bboxes_removed"] == 1
    assert stats["invalid_rows"] == 4


def test_nonzero_metric_sanity_on_tiny_coco_fixture(tmp_path):
    pytest.importorskip("pycocotools")
    ann = tmp_path / "ann.json"
    pred = tmp_path / "pred.json"
    _tiny_annotations(ann)
    pred.write_text(json.dumps([{"image_id": 1, "category_id": 3, "bbox": [10, 10, 20, 20], "score": 0.99}]))
    metrics = evaluate_coco(str(ann), str(pred))
    assert metrics["AP50"] > 0
    assert metrics["AR50"] > 0
    assert metrics["AR75"] > 0
    assert "AR_maxDets10" in metrics
    assert "AR_maxDets1" in metrics


def test_recall_ap_comparison_report_generation(tmp_path):
    metrics = {
        "WBF": {"AP": 0.5, "AR": 0.4, "AR_medium": 0.45, "AR_large": 0.5},
        "AWBF-Negotiation": {"AP": 0.48, "AR": 0.46, "AR_medium": 0.50, "AR_large": 0.55},
    }
    flow = {"WBF": {"output_detections": 10}, "AWBF-Negotiation": {"output_detections": 12}}
    best, rows = build_recall_audit(metrics, flow)
    assert best == "AWBF-Negotiation"
    assert rows[1][3] == "-0.020000"
    assert rows[1][4] == "0.060000"
    recall_path = tmp_path / "RECALL_ARGUMENT_AUDIT.md"
    comparison_path = tmp_path / "PAPER_RESULTS_COMPARISON.md"
    write_recall_argument_audit(recall_path, metrics, flow)
    write_paper_results_comparison(comparison_path, {"WBF": {"AP": 0.6, "AP50": 0.7, "AP75": 0.65}}, metrics)
    assert "Best AR method" in recall_path.read_text()
    assert "Paper Results Comparison" in comparison_path.read_text()
