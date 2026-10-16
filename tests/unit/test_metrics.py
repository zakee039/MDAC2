import pytest
from mdac2.evaluation.standard_metrics import event_scores,tempo_scores,boundary_scores


def test_tolerance_sensitivity_and_no_trim():
    r=event_scores([.1,.6,1.1],[.16,.66,1.16]).payload
    assert r['scores']['baseline']['f1']==1 and r['scores']['strict']['f1']==0
    assert event_scores([.1],[.1],kind='downbeat').payload['scores']['baseline']['f1']==1
    assert event_scores([1],[1.04],kind='onset').payload['scores']['strict']['f1']==0


def test_missing_empty_and_malformed_are_distinct():
    missing=event_scores([1,2],None);empty=event_scores([1,2],[])
    assert missing.payload['status']=='missing' and empty.payload['status']=='empty'
    assert missing.artifact_hash!=empty.artifact_hash and missing.payload['scores']['baseline']['f1']==0
    assert event_scores([],[]).payload['scores']['baseline']['f1']==0
    for malformed in ([2,1],[1,1],[float('nan')],[True]):
        with pytest.raises(ValueError):event_scores(malformed,[])


def test_tempo_alias_does_not_hide_acc1_and_boundaries():
    r=tempo_scores(120,60).payload['scores']['baseline'];assert r['acc1']==0 and r['acc2_half_double']==1
    assert tempo_scores(120,None).payload['status']=='missing'
    assert boundary_scores([0,2,4],[0,2,4]).payload['scores']['baseline']['f1']==1
    assert boundary_scores([0,2,4],[]).payload['scores']['baseline']['f1']==0
    with pytest.raises(ValueError):boundary_scores([0,2,4],[0,2,5])
