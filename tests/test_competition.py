from wbf_agents.awbf import Detection, compete_pair

def test_stronger_attack_removes_weaker_box():
    a=Detection((0,0,2,2),.9,1); b=Detection((0,0,1,1),.1,1)
    assert compete_pair(a,b,threshold=0.0)==[a]

def test_stronger_defense_keeps_attacked_box():
    a=Detection((0,0,1,1),.1,1); b=Detection((0,0,2,2),.9,1)
    assert compete_pair(a,b,threshold=0.0)==[b]

def test_close_scores_threshold_one_fuses_threshold_zero_competes():
    a=Detection((0,0,1,1),.6,1); b=Detection((0,0,1,1),.55,1)
    assert len(compete_pair(a,b,threshold=1.0))==1 and compete_pair(a,b,threshold=1.0)[0].source=='fused'
    assert compete_pair(a,b,threshold=0.0)==[a]
