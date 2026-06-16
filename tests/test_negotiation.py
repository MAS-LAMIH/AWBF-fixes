from wbf_agents.awbf import Detection, negotiate_pair

def test_rmax_one_w_zero_and_larger_w_behavior():
    a=Detection((0,0,1,1),.4,1); b=Detection((.2,0,1.2,1),.9,1)
    static, updates0 = negotiate_pair(a,b,rmax=3,w=0.0,threshold=0.0)
    assert updates0 >= 1 and static.box == a.box
    moved, updates1 = negotiate_pair(a,b,rmax=1,w=0.4,threshold=0.0)
    assert updates1 == 1 and moved.box[0] > a.box[0]

def test_terminates_when_no_bid_improves_or_close_fuses():
    a=Detection((0,0,1,1),.8,1); b=Detection((0,0,1,1),.81,1)
    out, updates = negotiate_pair(a,b,rmax=5,w=.3,threshold=1.0)
    assert out.source == 'fused' and updates == 0
