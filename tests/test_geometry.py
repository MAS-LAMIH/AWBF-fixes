import pytest
from wbf_agents.awbf import iou, iob

def test_iou_identical_nonoverlap_partial_zero_area():
    assert iou([0,0,1,1],[0,0,1,1]) == pytest.approx(1.0)
    assert iou([0,0,1,1],[2,2,3,3]) == 0.0
    assert iou([0,0,2,2],[1,1,3,3]) == pytest.approx(1/7)
    assert iou([0,0,0,1],[0,0,1,1]) == 0.0

def test_iob_is_asymmetric():
    small=[1,1,2,2]; large=[0,0,4,4]
    assert iob(small, large) == pytest.approx(1.0)
    assert iob(large, small) == pytest.approx(1/16)
