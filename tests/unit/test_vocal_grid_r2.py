from dataclasses import replace
import json

import numpy as np
import pytest

from mdac2.data_preparation.training_chart import TrainingEvent
from mdac2.data_preparation.vocal_priority_r2 import LookaheadPolicy,lookahead_mode,select_onsets
from mdac2.data_preparation.fixed_grid_r2 import grid_tick,on_grid,snap_events,align_holds,issues
from mdac2.data_preparation.six_grid_export_r2 import write_bms,read_bms,semantics


def event(i,start,end=None,lane='ground',kind='light',group=None):
    return TrainingEvent(str(i),kind,('ground','air') if kind=='gemini' else (lane,),start,
        start if end is None else end,(str(i),),'test','test',kind=='extra_heavy',group)


def test_lookahead_confirms_from_start_and_stops_at_end_without_two_seconds():
    raw=np.zeros(500,bool);raw[100:250]=True
    mode,coverage,decisions=lookahead_mode(raw)
    assert not mode[:100].any() and mode[100:250].all() and not mode[250:].any()
    assert coverage[100]==1 and decisions[0]['accepted']


def test_short_residue_and_internal_gap_and_end_of_song():
    raw=np.zeros(500,bool);raw[30:34]=True;raw[100:150]=True;raw[155:200]=True;raw[480:]=True
    mode,_,_=lookahead_mode(raw)
    assert not mode[:100].any()
    assert mode[100:200].all() and not mode[200:480].any()
    assert mode[480:].all()  # actual available future, no artificial zero padding


def test_lookahead_still_preserves_baseline_without_vocals():
    f=np.zeros((3,96,500),np.float32);f[0]=f[2]=9
    p=np.zeros(500,np.float32);p[::25]=.7
    times,a,r=select_onsets(f,dict(probability=p,offsets=np.zeros(500)),5.)
    assert times==[i*.25 for i in range(20)] and not a['vocal_mode'].any()
    json.dumps(r)


def test_sustained_vocal_only_has_attack_not_volume_clock():
    f=np.zeros((3,96,500),np.float32);f[0]=f[2]=9;f[1,10:75,100:300]=8
    p=np.zeros(500,np.float32);p[::20]=.7
    times,a,r=select_onsets(f,dict(probability=p,offsets=np.zeros(500)),5.)
    assert a['vocal_mode'][150:250].all()
    assert not any(1.2<t<2.8 for t in times)
    assert not a['vocal_mode'][310:].any()
    assert any(3.2<t<4.8 for t in times)


@pytest.mark.parametrize('time,expected',[(.00249,0),(.0025,1),(.0075,2),(1.003,201),(3.156,631)])
def test_grid_half_up(time,expected):
    assert grid_tick(time)==expected


def test_clicks_and_holds_snap_and_coincident_groups_stay_together():
    es=[event('a',.103),event('b',.103,lane='air',kind='heavy'),event('h',.103,.507,kind='hold')]
    out,r=snap_events(es,1.)
    assert all(e.start==.105 for e in out) and out[-1].end==.505
    assert all(on_grid(t) for e in out for t in [e.start,e.end])
    assert not issues(out,1.)


def test_rounding_collision_repairs_without_deleting_or_changing_type():
    es=[event('a',0.),event('b',.021)]
    out,r=snap_events(es,1.)
    assert [e.start for e in out]==[0.,.025] and r['repairs']
    assert len(out)==2 and not issues(out,1.)


@pytest.mark.parametrize('lane',['ground','air'])
@pytest.mark.parametrize('end,click,expected',[(.4,.5,.5),(.5,.4,.4)])
def test_hold_tail_extends_or_shortens_to_click_on_either_lane(lane,end,click,expected):
    es=[event('h',.1,end,kind='hold'),event('a',click,lane=lane)]
    out,r=align_holds(es,2.)
    assert out[0].end==expected and out[1]==es[1]
    assert not r['old_250ms_tail_extension']


def test_hold_head_moves_to_note():
    es=[event('h',.3,.8,kind='hold'),event('a',.4,lane='air')]
    out,_=align_holds(es,2.)
    assert out[0].start==.4 and out[0].end==.8


@pytest.mark.parametrize('kind',['heads','tails','handoff'])
def test_three_hold_pair_relations_take_grid_midpoint(kind):
    if kind=='heads':
        es=[event('a',.3,.8,kind='hold'),event('b',.4,1.2,lane='air',kind='hold')]
    elif kind=='tails':
        es=[event('a',0.,.3,kind='hold'),event('b',.15,.4,lane='air',kind='hold')]
    else:
        es=[event('a',0.,.3,kind='hold'),event('b',.4,.8,kind='hold')]
    out,_=align_holds(es,2.)
    a=out[0].start if kind=='heads' else out[0].end
    b=out[1].end if kind=='tails' else out[1].start
    assert a==b==.35
    assert semantics(out)==semantics(read_bms(write_bms(out,2.),2.))


def test_note_anchor_wins_and_no_250ms_extension():
    es=[event('h',0.,.3,kind='hold'),event('h2',.1,.4,lane='air',kind='hold'),event('n',.35)]
    out,_=align_holds(es,1.)
    assert out[0].end==out[1].end==.35
    es=[event('h',0.,.3,kind='hold'),event('n',.5)]
    out,_=align_holds(es,1.)
    assert out==es


def test_cannot_collapse_short_hold_or_extend_through_exclusive_attack():
    es=[event('h',.3,.35,kind='hold'),event('n',.32)]
    out,_=align_holds(es,1.)
    assert out[0].end-out[0].start>=.03-1e-9 and not issues(out,1.)
    es=[event('h',0.,.3,kind='hold'),event('x',.34,kind='extra_heavy'),event('n',.4)]
    out,r=align_holds(es,1.)
    assert out==es and r['rejected']


def test_no_transitive_alignment_beyond_original_span():
    es=[event('h',.3,.8,kind='hold'),event('j',.39,.48,lane='air',kind='hold')]
    out,r=align_holds(es,1.)
    assert all(max(e['sources'])-min(e['sources'])<=.1000001 for e in r['edits'])
    assert out[1].end-out[1].start>=.03-1e-9


def test_already_aligned_group_is_not_split_and_export_has_200_cells():
    es=[event('a',.1,.5,kind='hold',group='both'),event('b',.1,.5,lane='air',kind='hold',group='both'),event('n',.55)]
    out,_=align_holds(es,1.)
    assert out[0].end==out[1].end==.55
    text=write_bms(out,1.)
    assert '#BPM 240' in text
    assert all(len(line.split(':')[1])==400 for line in text.splitlines() if ':' in line and line[4:6]!='01')
    assert semantics(out)==semantics(read_bms(text,1.))
    with pytest.raises(ValueError):write_bms([event('off',.103)],1.)


def test_gemini_and_double_heavy_remain_distinct_after_export():
    es=[event('g',.1,kind='gemini'),event('h1',.3,kind='heavy',group='double'),event('h2',.3,lane='air',kind='heavy',group='double')]
    assert semantics(es)==semantics(read_bms(write_bms(es,1.),1.))
