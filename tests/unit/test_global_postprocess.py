import numpy as np
import pytest
from mdac2.data_preparation.training_chart import TrainingEvent
from mdac2.data_preparation.global_postprocess import (align_holds,move_hold_clicks,
    normalize_ms,avoidance_violations,on_ms,prioritize_exclusive,issues)
from mdac2.data_preparation import v2_vocal_bias


def e(key,kind,start,end=None,lane='ground',group=None):
    return TrainingEvent(key,kind,('ground','air') if kind=='gemini' else (lane,),
        start,start if end is None else end,(key,),'test','test',kind=='extra_heavy',group)


def test_storage_retains_non_5ms_points():
    es=[e('h','hold',.301,1.503),e('n','light',2.007)]
    assert normalize_ms(es,4)==es
    assert all(on_ms(t) for x in es for t in (x.start,x.end))


def test_endpoint_average_uses_1ms():
    es=[e('h','hold',.301,1.501),e('j','hold',.400,1.600,lane='air')]
    out,_=align_holds(es,3)
    assert {x.start for x in out}=={.351}
    assert {x.end for x in out}=={1.551}


def test_note_anchor_preserves_exact_millisecond():
    es=[e('h','hold',.301,1.501),e('n','light',1.587,lane='air')]
    out,_=align_holds(es,3)
    assert out[0].end==1.587


def test_all_interior_clicks_not_only_dense_move():
    es=[e('h','hold',1,4),e('a','light',1.1),e('b','heavy',3.1)]
    out,r=move_hold_clicks(es,6)
    assert [x.lanes for x in out[1:]]==[('air',),('air',)]
    assert not avoidance_violations(out) and len(r['edits'])==2


def test_post_tail_500ms_inclusive_but_not_501():
    es=[e('h','hold',1,2),e('head','light',1),e('a','light',2),e('b','light',2.5),e('c','light',2.501,lane='air')]
    # The 1ms neighbour makes b's target illegal; test boundary independently.
    for t,expected in [(2.,'air'),(2.5,'air'),(2.501,'ground'),(1.,'air')]:
        out,_=move_hold_clicks([es[0],e('n','light',t)],4)
        assert out[-1].lanes==(expected,)


def test_gemini_becomes_opposite_light():
    out,r=move_hold_clicks([e('h','hold',1,3),e('g','gemini',2,group='pair')],5)
    assert out[-1].kind=='light' and out[-1].lanes==('air',) and out[-1].group_id is None
    assert r['edits'][0]['gemini_to_opposite_light']


def test_original_target_wins_without_merge():
    out,r=move_hold_clicks([e('h','hold',1,3),e('a','light',2),e('b','light',2,lane='air')],5)
    assert len(out)==2 and out[-1].source_ids==('b',)
    assert r['dropped'][0]['event_ids']==['a']


def test_two_held_lanes_discard_click_not_holds():
    es=[e('a','hold',1,3),e('b','hold',1,3,lane='air'),e('g','gemini',2,group='g')]
    out,r=move_hold_clicks(es,5)
    assert out==es[:2] and r['dropped'][0]['reason']=='both_lanes_guarded'


def test_double_heavy_becomes_opposite_single_heavy():
    es=[e('h','hold',1,3),e('a','heavy',2,group='double'),e('b','heavy',2,lane='air',group='double')]
    out,r=move_hold_clicks(es,5)
    assert len(out)==2 and out[-1].kind=='heavy' and out[-1].lanes==('air',)
    assert out[-1].group_id is None and r['edits'][0]['double_heavy_to_opposite_heavy']


@pytest.mark.parametrize('a,b,t',[(3,1.8,2),(1.8,3,2),(1.8,1.9,2)])
def test_any_two_lane_guard_combination_discards(a,b,t):
    holds=[e('g','hold',1,a),e('a','hold',1,b,lane='air')]
    out,r=move_hold_clicks(holds+[e('n','light',t)],5)
    assert set(x.event_id for x in out)=={'g','a'} and len(r['dropped'])==1


@pytest.mark.parametrize('kind,offset',[('light',0),('heavy',0),('light',.02),('light',-.02)])
def test_original_target_nearby_or_other_type_discards_transfer(kind,offset):
    original=e('z','light',2+offset,lane='air')
    es=[e('h','hold',1,3),e('a',kind,2),original]
    out,r=move_hold_clicks(es,5)
    assert original in out and not any(x.event_id=='a' for x in out)
    assert r['dropped'][0]['blockers']==['z']


def test_target_beyond_20ms_allows_transfer():
    out,_=move_hold_clicks([e('h','hold',1,3),e('n','light',2),e('o','light',2.021,lane='air')],5)
    assert len(out)==3 and not avoidance_violations(out)


def test_original_reservation_is_input_order_independent():
    es=[e('h','hold',1,3),e('n','light',2),e('o','heavy',2,lane='air')]
    a,_=move_hold_clicks(es,5);b,_=move_hold_clicks(list(reversed(es)),5)
    assert a==b and es[-1] in a


def test_exclusive_cuts_both_holds_and_removes_other_strikes():
    es=[e('g','hold',1,3),e('a','hold',1,4,lane='air'),e('n','light',2),e('x','extra_heavy',2)]
    out,r=move_hold_clicks(es,5)
    assert {x.end for x in out if x.kind=='hold'}=={2}
    assert [x.kind for x in out if x.kind!='hold']==['extra_heavy']
    assert not issues(out,5) and not avoidance_violations(out)
    assert len(r['exclusive']['edits'])==2


def test_exclusive_discards_collapsed_hold_and_keeps_future_events():
    es=[e('h','hold',1.99,3),e('x','extra_heavy',2),e('future','light',2.1)]
    out,r=prioritize_exclusive(es,5)
    assert [x.event_id for x in out]==['x','future']
    assert r['dropped'][0]['reason']=='exclusive_leaves_under_30ms_hold'


def test_exclusive_nearby_duplicate_and_double_heavy_cleanup():
    es=[e('x','extra_heavy',2),e('y','extra_heavy',2.01,lane='air'),
        e('a','heavy',2.015,group='d'),e('b','heavy',2.015,lane='air',group='d')]
    out,_=move_hold_clicks(es,5)
    assert [x.event_id for x in out]==['x']


def test_double_heavy_outside_guard_remains_double():
    es=[e('a','heavy',2,group='d'),e('b','heavy',2,lane='air',group='d')]
    out,_=move_hold_clicks(es,5)
    assert out==es


def test_alignment_then_transfer_includes_new_tail_anchor():
    es=[e('h','hold',1,2),e('n','light',2.09)]
    aligned,_=align_holds(es,5)
    out,_=move_hold_clicks(aligned,5)
    assert out[0].end==2.09 and out[1].lanes==('air',)


def test_vocal_boost_adds_supported_weak_peak_without_removing_instrument(monkeypatch):
    n=200;p=np.zeros(n,np.float32);p[50]=.20;p[150]=.6
    support=np.zeros(n);support[50]=1
    monkeypatch.setattr(v2_vocal_bias,'vocal_evidence',lambda f,policy:dict(vocal_mode=np.ones(n,bool),support=support))
    times,scores,r=v2_vocal_bias.select(np.zeros((3,96,n)),dict(probability=p,offsets=np.zeros(n)),2)
    assert times==[.5,1.5] and r['added']==1
    assert scores['probability'][150]==p[150] and r['removed_by_local_peak_competition']==0


def test_no_voice_exact_v2_scores(monkeypatch):
    n=200;p=np.linspace(0,1,n,dtype=np.float32)
    monkeypatch.setattr(v2_vocal_bias,'vocal_evidence',lambda f,policy:dict(vocal_mode=np.zeros(n,bool),support=np.ones(n)))
    _,scores,r=v2_vocal_bias.select(np.zeros((3,96,n)),dict(probability=p,offsets=np.zeros(n)),2)
    assert np.array_equal(scores['probability'],p) and r['boosted_frames']==0
