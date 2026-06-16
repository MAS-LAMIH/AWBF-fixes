import pytest
from wbf_agents.awbf import Detection, decentralized_wbf

def test_wbf_confidence_weighted_average_and_deterministic_score():
    out=decentralized_wbf({'a':[Detection((0,0,1,1),0.75,1)],'b':[Detection((0.2,0,1.2,1),0.25,1)]}, iou_thr=0.5)
    assert len(out)==1
    assert out[0].box[0] == pytest.approx(0.05)
    assert out[0].score == pytest.approx(0.5)

def test_wbf_different_labels_and_below_iou_not_fused():
    out=decentralized_wbf({'a':[Detection((0,0,1,1),.9,1), Detection((0,0,1,1),.8,2)], 'b':[Detection((2,2,3,3),.7,1)]}, iou_thr=0.5)
    assert len(out)==3
