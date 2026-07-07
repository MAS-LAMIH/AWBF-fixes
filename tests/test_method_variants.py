from types import SimpleNamespace

import pytest

from scripts.reproduce_paper_results import build_method_equivalence_audit, fuse_all
from wbf_agents.awbf import Detection


def _args(**overrides):
    defaults = dict(
        progress_interval=100,
        profile=False,
        disable_preclustering=False,
        iou_threshold=0.5,
        score_threshold=0.0,
        cooperation_threshold=0.5,
        negotiation_threshold=0.05,
        rounds=5,
        weight=0.3,
        incremental_cluster_state=False,
        methods=["all"],
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def test_incremental_awbf_matches_awbf_when_cluster_membership_is_fixed():
    by_model = {
        "a": {1: [Detection((0.0, 0.0, 1.0, 1.0), 0.9, 1, "a")]},
        "b": {1: [Detection((0.1, 0.0, 1.1, 1.0), 0.6, 1, "b")]},
    }
    outputs, _timings, _flow = fuse_all(by_model, _args())
    audit = build_method_equivalence_audit(outputs)
    row = audit["AWBF vs Incremental_AWBF"]
    assert row["detection_count_a"] == row["detection_count_b"] == 1
    assert row["max_coordinate_difference"] == pytest.approx(0.0)
    assert row["max_score_difference"] == pytest.approx(0.0)
    assert row["equivalence"] in {"identical", "nearly identical"}



def test_incremental_awbf_matches_awbf_for_chained_precluster_that_wbf_splits():
    by_model = {
        "a": {1: [Detection((0.0, 0.0, 1.0, 1.0), 0.9, 1, "a")]},
        "b": {1: [Detection((0.2, 0.0, 1.2, 1.0), 0.8, 1, "b")]},
        "c": {1: [Detection((0.45, 0.0, 1.45, 1.0), 0.7, 1, "c")]},
    }
    outputs, _timings, _flow = fuse_all(by_model, _args())
    audit = build_method_equivalence_audit(outputs)
    row = audit["AWBF vs Incremental_AWBF"]
    assert row["detection_count_a"] == row["detection_count_b"] == 2
    assert row["max_coordinate_difference"] == pytest.approx(0.0)
    assert row["max_score_difference"] == pytest.approx(0.0)

def test_incremental_state_variants_may_change_detection_count_without_overwriting_paper_outputs():
    by_model = {
        "a": {1: [Detection((0.0, 0.0, 1.0, 1.0), 0.9, 1, "a")]},
        "b": {1: [Detection((3.0, 3.0, 4.0, 4.0), 0.7, 1, "b")]},
    }
    outputs, _timings, _flow = fuse_all(by_model, _args(disable_preclustering=True))
    assert len(outputs["AWBF-competition"][1]) == 2
    assert len(outputs["AWBF-competition-IncrementalState"][1]) == 1
    assert len(outputs["AWBF-Negotiation"][1]) == 2
    assert len(outputs["AWBF-Negotiation-IncrementalState"][1]) == 1


def test_incremental_cluster_state_flag_does_not_overwrite_paper_style_outputs():
    by_model = {
        "a": {1: [Detection((0.0, 0.0, 1.0, 1.0), 0.9, 1, "a")]},
        "b": {1: [Detection((3.0, 3.0, 4.0, 4.0), 0.7, 1, "b")]},
    }
    outputs_default, _, _ = fuse_all(by_model, _args(disable_preclustering=True, incremental_cluster_state=False))
    outputs_flag, _, _ = fuse_all(by_model, _args(disable_preclustering=True, incremental_cluster_state=True))
    assert outputs_default["AWBF-competition"][1] == outputs_flag["AWBF-competition"][1]
    assert outputs_default["AWBF-Negotiation"][1] == outputs_flag["AWBF-Negotiation"][1]
    assert "AWBF-competition-IncrementalState" in outputs_flag
    assert "AWBF-Negotiation-IncrementalState" in outputs_flag
