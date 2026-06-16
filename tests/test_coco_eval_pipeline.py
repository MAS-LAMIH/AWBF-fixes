import json
from wbf_agents.awbf import Detection, export_coco_detections

def test_prediction_export_valid_coco_detection_json(tmp_path):
    path=tmp_path/'pred.json'
    rows=export_coco_detections({1:[Detection((10,20,30,50),1.2,3)]}, str(path))
    loaded=json.loads(path.read_text())
    assert loaded == rows
    row=loaded[0]
    assert row['category_id']==3
    assert row['bbox']==[10.0,20.0,20.0,30.0]
    assert 0 <= row['score'] <= 1
