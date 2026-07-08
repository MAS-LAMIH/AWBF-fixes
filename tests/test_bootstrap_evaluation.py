import csv
import json
from types import SimpleNamespace

from scripts import bootstrap_evaluation as boot


def _write_json(path, data):
    path.write_text(json.dumps(data), encoding="utf-8")


def _metrics_from_prediction_count(_ann_path, pred_path):
    rows = json.loads(pred_path.read_text())
    value = min(1.0, len(rows) / 10.0)
    return {metric: value for metric in boot.METRIC_NAMES}


def test_build_bootstrap_subset_duplicates_sampled_images_with_new_ids():
    coco = {
        "images": [{"id": 139, "width": 640, "height": 426}],
        "annotations": [{"id": 7, "image_id": 139, "category_id": 1, "bbox": [1, 2, 3, 4], "area": 12, "iscrowd": 0}],
        "categories": [{"id": 1, "name": "object"}],
    }
    preds = [{"image_id": 139, "category_id": 1, "bbox": [1, 2, 3, 4], "score": 0.9}]
    subset_coco, subset_preds = boot.build_bootstrap_subset(coco, preds, [139, 139])
    assert [img["id"] for img in subset_coco["images"]] == [1, 2]
    assert [ann["image_id"] for ann in subset_coco["annotations"]] == [1, 2]
    assert [pred["image_id"] for pred in subset_preds] == [1, 2]
    assert subset_coco["images"][0]["bootstrap_original_image_id"] == 139


def test_run_bootstrap_writes_reports_and_paired_deltas(tmp_path):
    annotations = tmp_path / "instances_val2017.json"
    predictions_dir = tmp_path / "predictions"
    output_dir = tmp_path / "bootstrap"
    predictions_dir.mkdir()
    _write_json(
        annotations,
        {
            "images": [{"id": 1, "width": 10, "height": 10}, {"id": 2, "width": 10, "height": 10}],
            "annotations": [],
            "categories": [{"id": 1, "name": "object"}],
        },
    )
    _write_json(predictions_dir / "wbf_predictions.json", [{"image_id": 1, "category_id": 1, "bbox": [1, 1, 2, 2], "score": 0.9}])
    _write_json(
        predictions_dir / "awbf_predictions.json",
        [
            {"image_id": 1, "category_id": 1, "bbox": [1, 1, 2, 2], "score": 0.9},
            {"image_id": 2, "category_id": 1, "bbox": [2, 2, 2, 2], "score": 0.8},
        ],
    )
    args = SimpleNamespace(
        annotations=str(annotations),
        predictions_dir=str(predictions_dir),
        output_dir=str(output_dir),
        num_samples=3,
        sample_size=2,
        seed=42,
        tolerance=1e-12,
    )
    report = boot.run_bootstrap(args, evaluator=_metrics_from_prediction_count)
    assert (output_dir / "bootstrap_report.json").is_file()
    assert (output_dir / "BOOTSTRAP_EVALUATION.md").is_file()
    assert (output_dir / "bootstrap_summary.csv").is_file()
    assert (output_dir / "bootstrap_deltas.csv").is_file()
    assert (output_dir / "bootstrap_table.tex").is_file()
    assert "AWBF" in report["paired_deltas_vs_wbf"]
    rows = list(csv.DictReader(open(output_dir / "bootstrap_deltas.csv", newline="", encoding="utf-8")))
    assert {"method", "metric", "mean_delta", "proportion_gt_wbf"}.issubset(rows[0])
    assert any(row["metric"] == "AR100" for row in rows)
