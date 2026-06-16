from wbf_agents.awbf import (
    Detection,
    awbf_competition,
    awbf_negotiation,
    cluster_detections_by_label_and_iou,
)


def test_different_labels_form_different_clusters():
    detections = [Detection((0, 0, 1, 1), 0.9, 1), Detection((0, 0, 1, 1), 0.8, 2)]
    clusters = cluster_detections_by_label_and_iou(detections, iou_thr=0.5)
    assert len(clusters) == 2
    assert sorted(cluster[0].label for cluster in clusters) == [1, 2]


def test_non_overlapping_same_label_boxes_form_different_clusters():
    detections = [Detection((0, 0, 1, 1), 0.9, 1), Detection((2, 2, 3, 3), 0.8, 1)]
    clusters = cluster_detections_by_label_and_iou(detections, iou_thr=0.5)
    assert len(clusters) == 2


def test_overlapping_same_label_boxes_cluster_together():
    detections = [Detection((0, 0, 1, 1), 0.9, 1), Detection((0.1, 0.1, 1.1, 1.1), 0.8, 1)]
    clusters = cluster_detections_by_label_and_iou(detections, iou_thr=0.5)
    assert len(clusters) == 1
    assert len(clusters[0]) == 2


def test_competition_only_happens_inside_clusters():
    detections = [
        Detection((0, 0, 1, 1), 0.9, 1),
        Detection((0, 0, 1, 1), 0.1, 1),
        Detection((2, 2, 3, 3), 0.8, 1),
    ]
    clusters = cluster_detections_by_label_and_iou(detections, iou_thr=0.5)
    outputs = []
    for cluster in clusters:
        outputs.extend(awbf_competition(cluster, iou_thr=0.5, threshold=0.0))
    assert len(outputs) == 2
    assert any(det.box == (2, 2, 3, 3) for det in outputs)


def test_negotiation_only_happens_inside_clusters():
    detections = [
        Detection((0, 0, 1, 1), 0.9, 1),
        Detection((0, 0, 1, 1), 0.85, 1),
        Detection((2, 2, 3, 3), 0.8, 1),
    ]
    clusters = cluster_detections_by_label_and_iou(detections, iou_thr=0.5)
    outputs = []
    for cluster in clusters:
        outputs.extend(awbf_negotiation(cluster, iou_thr=0.5, rmax=1, w=0.3, threshold=1.0))
    assert len(outputs) == 2
    assert any(det.box == (2, 2, 3, 3) for det in outputs)
