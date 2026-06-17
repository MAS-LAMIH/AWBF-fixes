import itertools

import pytest

from wbf_agents.awbf import (
    Detection,
    confidence_weighted_box,
    incremental_awbf,
    incremental_fuse_detections,
)


def test_incremental_fusion_matches_one_shot_coordinates_and_score():
    detections = [
        Detection((0.0, 0.0, 1.0, 1.0), 0.9, 1),
        Detection((0.2, 0.0, 1.2, 1.0), 0.3, 1),
        Detection((0.4, 0.0, 1.4, 1.0), 0.6, 1),
    ]
    incremental = incremental_fuse_detections(detections)
    one_shot = confidence_weighted_box(detections)
    assert incremental.box == pytest.approx(one_shot.box)
    assert incremental.score == pytest.approx(one_shot.score)
    assert incremental.source == "incremental_awbf"


def test_incremental_fusion_order_sensitivity_is_only_floating_point_noise():
    detections = [
        Detection((0.0, 0.0, 1.0, 1.0), 0.9, 1),
        Detection((0.2, 0.0, 1.2, 1.0), 0.3, 1),
        Detection((0.4, 0.0, 1.4, 1.0), 0.6, 1),
    ]
    expected = confidence_weighted_box(detections)
    for order in itertools.permutations(detections):
        out = incremental_fuse_detections(list(order))
        assert out.box == pytest.approx(expected.box)
        assert out.score == pytest.approx(expected.score)


def test_incremental_awbf_fuses_each_cluster_independently():
    clusters = [
        [Detection((0.0, 0.0, 1.0, 1.0), 0.9, 1), Detection((0.2, 0.0, 1.2, 1.0), 0.3, 1)],
        [Detection((2.0, 2.0, 3.0, 3.0), 0.8, 1)],
    ]
    outputs = incremental_awbf(clusters)
    assert len(outputs) == 2
    assert outputs[0].source == "incremental_awbf"
    assert outputs[1].box == (2.0, 2.0, 3.0, 3.0)


def test_incremental_score_bookkeeping_with_zero_weights():
    detections = [Detection((0.0, 0.0, 1.0, 1.0), 0.0, 1), Detection((1.0, 1.0, 2.0, 2.0), 0.0, 1)]
    out = incremental_fuse_detections(detections)
    assert out.box == pytest.approx((0.5, 0.5, 1.5, 1.5))
    assert out.score == pytest.approx(0.0)
