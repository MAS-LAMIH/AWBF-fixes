import json

import pytest

import scripts.reproduce_paper_results as reproduce


def test_direct_evaluation_requires_annotations(tmp_path):
    pred = tmp_path / "pred.json"
    pred.write_text("[]")
    with pytest.raises(SystemExit, match="--annotations is required"):
        reproduce.main(["--evaluate-predictions", str(pred)])


def test_direct_evaluation_skips_fuse_all(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    pred1 = tmp_path / "wbf_predictions.json"
    pred2 = tmp_path / "awbf_predictions.json"
    ann = tmp_path / "instances_val2017.json"
    pred1.write_text("[]")
    pred2.write_text("[]")
    ann.write_text("{}")

    def fail_fuse_all(*_args, **_kwargs):
        raise AssertionError("fuse_all should not be called in direct evaluation mode")

    calls = []

    def fake_evaluate(annotations, detections):
        calls.append((annotations, detections))
        return {"AP": 0.1, "AP50": 0.2}

    monkeypatch.setattr(reproduce, "fuse_all", fail_fuse_all)
    monkeypatch.setattr(reproduce, "evaluate_coco", fake_evaluate)

    report = reproduce.main([
        "--evaluate-predictions",
        str(pred1),
        str(pred2),
        "--annotations",
        str(ann),
    ])

    assert calls == [(str(ann), str(pred1)), (str(ann), str(pred2))]
    assert report["metrics"][str(pred1)] == {"AP": 0.1, "AP50": 0.2}
    assert "evaluation_validation" in report
    assert (tmp_path / "outputs" / "EVALUATION_VALIDATION.md").is_file()
    saved = json.loads((tmp_path / "outputs" / "evaluation_report.json").read_text())
    assert saved == report
