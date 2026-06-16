import json

import pytest

from wbf_agents.awbf import Detection, convert_coco_detection_bboxes, export_coco_detections
import scripts.reproduce_paper_results as reproduce

IMAGE_ID = 139
WIDTH = 640
HEIGHT = 426
NORM_XYWH = [0.6423535, 0.3707462, 0.08414556, 0.32648011]
EXPECTED_PIXEL = [411.10624, 157.9378812, 53.8531584, 139.08052686]


def assert_bbox_close(actual, expected=EXPECTED_PIXEL):
    assert actual == pytest.approx(expected, abs=1e-3)


def test_export_normalized_xyxy_to_pixel_xywh(tmp_path):
    x, y, w, h = NORM_XYWH
    det = Detection((x, y, x + w, y + h), 0.9, 1)
    out = tmp_path / "pred.json"
    rows, stats = export_coco_detections(
        {IMAGE_ID: [det]},
        str(out),
        image_sizes={IMAGE_ID: (WIDTH, HEIGHT)},
        bbox_scale="normalized",
        return_stats=True,
    )
    assert_bbox_close(rows[0]["bbox"])
    assert stats["boxes_converted_to_pixel"] == 1
    assert stats["boxes_left_unchanged"] == 0
    assert json.loads(out.read_text())[0]["bbox"] == rows[0]["bbox"]


def test_pixel_mode_leaves_values_unchanged():
    rows, stats = convert_coco_detection_bboxes(
        [{"image_id": IMAGE_ID, "category_id": 1, "bbox": EXPECTED_PIXEL, "score": 0.9}],
        image_sizes={IMAGE_ID: (WIDTH, HEIGHT)},
        bbox_scale="pixel",
    )
    assert_bbox_close(rows[0]["bbox"])
    assert stats["boxes_converted_to_pixel"] == 0
    assert stats["boxes_left_unchanged"] == 1


def test_missing_image_id_warns_and_records_missing_id():
    with pytest.warns(UserWarning, match="Missing image_id"):
        rows, stats = convert_coco_detection_bboxes(
            [{"image_id": IMAGE_ID, "category_id": 1, "bbox": NORM_XYWH, "score": 0.9}],
            image_sizes={},
            bbox_scale="normalized",
        )
    assert rows[0]["bbox"] == NORM_XYWH
    assert stats["missing_image_ids"] == [IMAGE_ID]
    assert stats["boxes_left_unchanged"] == 1


def test_pixel_mode_warns_when_values_look_normalized():
    with pytest.warns(UserWarning, match="normalized while --bbox-scale pixel"):
        _rows, stats = convert_coco_detection_bboxes(
            [{"image_id": IMAGE_ID, "category_id": 1, "bbox": NORM_XYWH, "score": 0.9}],
            image_sizes={IMAGE_ID: (WIDTH, HEIGHT)},
            bbox_scale="pixel",
        )
    assert stats["boxes_converted_to_pixel"] == 0


def test_direct_evaluation_writes_converted_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    pred = tmp_path / "wbf_predictions.json"
    ann = tmp_path / "instances_val2017.json"
    pred.write_text(json.dumps([{"image_id": IMAGE_ID, "category_id": 1, "bbox": NORM_XYWH, "score": 0.9}]))
    ann.write_text(json.dumps({"images": [{"id": IMAGE_ID, "width": WIDTH, "height": HEIGHT}]}))

    evaluated_files = []

    def fake_evaluate(_annotations, detections):
        evaluated_files.append(detections)
        converted = json.loads((tmp_path / detections).read_text() if not str(detections).startswith(str(tmp_path)) else open(detections).read())
        assert_bbox_close(converted[0]["bbox"])
        return {"AP": 0.5}

    monkeypatch.setattr(reproduce, "evaluate_coco", fake_evaluate)
    report = reproduce.main([
        "--evaluate-predictions",
        str(pred),
        "--annotations",
        str(ann),
        "--bbox-scale",
        "auto",
        "--output-dir",
        "outputs/eval_wbf",
    ])
    converted_file = tmp_path / "outputs" / "eval_wbf" / "wbf_predictions_pixel_xywh.json"
    assert converted_file.is_file()
    assert report["bbox_reports"][str(pred)]["converted_prediction_file"] == "outputs/eval_wbf/wbf_predictions_pixel_xywh.json"
    assert evaluated_files == ["outputs/eval_wbf/wbf_predictions_pixel_xywh.json"]
