from mdac2.application.studio_timing_v4 import apply
from mdac2.data_preparation.training_chart import TrainingEvent


def event(event_id, kind, start, lane='ground', end=None):
    lanes = ('ground', 'air') if kind == 'gemini' else (lane,)
    return TrainingEvent(event_id, kind, lanes, start,
                         start if end is None else end, (event_id,),
                         'v4-test', 'test', kind == 'extra_heavy', None)


def config(enabled):
    return dict(enabled=enabled, bpm=120., offset_ms=0., tap_latency_ms=0., subdivisions=12)


def test_bpm_same_beat_four_chain_uses_special_uniform_slots():
    events = [event(str(i), 'light', t) for i, t in enumerate((.01, .19, .25, .31))]
    out, report = apply(events, 2, config(True))
    assert [round(e.start, 3) for e in out] == [0., .125, .25, .375]
    assert report['uniformization']['groups']


def test_no_bpm_four_chain_keeps_endpoints_and_adds_midpoint():
    events = [event(str(i), 'light', t) for i, t in enumerate((.01, .09, .17, .31))]
    out, report = apply(events, 2, config(False))
    assert [round(e.start, 3) for e in out] == [.01, .16, .31]
    assert report['four_chain_reduction']


def test_gemini_wins_against_ordinary_same_lane_conflict():
    out, _ = apply([event('g', 'gemini', 1.), event('n', 'light', 1.05)], 2, config(False))
    assert [(e.event_id, e.kind) for e in out] == [('g', 'gemini')]


def test_acoustic_preprocessing_does_not_erase_gemini_for_exclusive():
    from mdac2.data_preparation.global_postprocess_v4 import prioritize_v4_exclusive
    events = [event('g', 'gemini', 1.), event('x', 'extra_heavy', 1.01)]
    out, report = prioritize_v4_exclusive(events, 3.)
    assert [e.event_id for e in out] == ['g']
    assert report['dropped'][0]['reason'] == 'gemini_priority'


def test_tail_guard_converts_one_blocked_gemini_lane_to_heavy():
    events = [event('h', 'hold', 1., end=2.), event('g', 'gemini', 2.2)]
    out, _ = apply(events, 4, config(False))
    gemini = next(e for e in out if e.event_id == 'g')
    assert gemini.kind == 'heavy' and gemini.lanes == ('air',)


def test_hold_interior_uses_strict_same_lane_spacing():
    events = [event('h', 'hold', 1., end=3.),
              event('a', 'light', 1.4), event('b', 'light', 1.8)]
    out, _ = apply(events, 4, config(False))
    clicks = [e for e in out if e.kind != 'hold' and e.lanes == ('ground',)]
    assert all(abs(a.start - b.start) >= .5 - 1e-8 for i, a in enumerate(clicks) for b in clicks[i + 1:])


import pytest
from mdac2.data_preparation.playability_audit_v4 import audit_playability_v4
from mdac2.data_preparation.six_export import write_bms, read_bms, semantics


def checked(events, cfg, duration=5):
    out, report = apply(events, duration, cfg)
    assert audit_playability_v4(out, duration, timing=cfg)['engineering_playtest_gate_passed']
    if out:
        assert semantics(out) == semantics(read_bms(write_bms(out, duration), duration))
    return out, report


@pytest.mark.parametrize('bpm,offset', [(120,0),(127.823,194),(60,100),(399.9,-73)])
def test_full_four_and_three_chains_pass_final_gate(bpm, offset):
    cfg=dict(config(True),bpm=bpm,offset_ms=offset)
    for count,slots in [(3,[0,4,8]),(4,[0,3,6,9])]:
        times=[round((offset/1000+60/bpm*(4+s/12))*1000)/1000 for s in slots]
        out,_=checked([event(str(i),'light',t) for i,t in enumerate(times)],cfg,10)
        assert len(out)==count and all(e.lanes==('ground',) for e in out)
        assert [e.start for e in out]==times


def test_both_lanes_blocked_discards_instead_of_flip_loop():
    out,r=checked([event('a','light',1),event('b','light',1,'air'),event('c','light',1.05)],config(False))
    assert len(out)==2 and r['conflict_resolution']['dropped']


def test_rounded_beat_head_and_partial_first_beat():
    out,_=checked([event('a','light',.469)],dict(config(True),bpm=127.823))
    assert out[0].start==.469
    out,_=checked([event('a','light',.05)],dict(config(True),offset_ms=100))
    assert 0<=out[0].start<.1


def test_tail_guard_includes_next_hold_head():
    out,_=checked([event('h','hold',1,end=2),event('next','hold',2.1,end=3.5)],config(False))
    assert next(e for e in out if e.event_id=='next').lanes==('air',)


def test_hold_interior_four_exception_does_not_override_one_beat():
    es=[event('h','hold',0,end=3)]+[event(str(i),'light',t) for i,t in enumerate([1,1.125,1.25,1.375])]
    out,_=checked(es,config(True))
    clicks=[e for e in out if e.kind!='hold' and e.lanes==('ground',)]
    assert all(b.start-a.start>=.5 for a,b in zip(clicks,clicks[1:]))
    bad=[event('h','hold',0,end=3),event('a','light',1),event('b','light',1.25)]
    assert not audit_playability_v4(bad,5,timing=config(True))['engineering_playtest_gate_passed']


def test_audit_uses_local_tempo_and_detects_slow_segment_conflict():
    cfg=dict(config(True),bpm=60,segments=[dict(start_ms=1000,bpm=180,offset_ms=1000,tap_latency_ms=0)])
    checked([event('a','light',1),event('b','light',1.2)],cfg)
    cfg['bpm']=180;cfg['segments'][0]['bpm']=60
    assert not audit_playability_v4([event('a','light',1),event('b','light',1.2)],5,timing=cfg)['engineering_playtest_gate_passed']


def test_no_bpm_legal_four_is_preserved_and_gemini_reduces_to_three():
    es=[event(str(i),'light',t) for i,t in enumerate([0,.15,.3,.45])]
    out,r=checked(es,config(False));assert out==es and not r['four_chain_reduction']
    es=[event(str(i),'gemini',t) for i,t in enumerate([0,.1,.2,.3])]
    out,_=checked(es,config(False))
    assert [e.start for e in out]==[0,.15,.3] and all(e.kind=='gemini' for e in out)


def test_hold_head_participates_in_uniformization():
    out,_=checked([event('h','hold',.25,end=2),event('a','light',.35)],config(True))
    assert [e.start for e in out]==[0,.25]


def test_gemini_one_or_both_tail_lanes_blocked():
    es=[event('h','hold',0,end=2),event('g','gemini',2.2)]
    out,_=checked(es,config(False));assert [(e.kind,e.lanes) for e in out if e.event_id=='g']==[('heavy',('air',))]
    out,_=checked(es+[event('ha','hold',0,'air',end=2)],config(False))
    assert all(e.event_id!='g' for e in out)


def test_randomized_resolution_and_bms_roundtrip():
    import random
    rng=random.Random(20261020)
    for enabled in (False,True):
        for run in range(15):
            cfg=dict(config(enabled),bpm=rng.choice([60,120,127.823,180]),offset_ms=rng.choice([0,194,-50]))
            es=[]
            for i in range(30):
                t=round(rng.uniform(0,7),3);kind=rng.choice(['light','heavy','gemini','hold'])
                es.append(event(str(i),kind,t,rng.choice(['ground','air']),end=round(t+.8,3) if kind=='hold' else None))
            checked(es,cfg,8)
