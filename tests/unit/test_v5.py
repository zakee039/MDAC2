import json
import hashlib
from pathlib import Path
import numpy as np
import pytest
from mdac2.application.studio_timing import settings
from mdac2.application.strategies.v5 import dense_groups, run, soft_edges
from mdac2.application.strategies.v5_holds import detect, refine
from mdac2.application.strategies.v5_events import SpecialEvent, issues, from_dict
from mdac2.application.strategies.v5_export import write_bms, read_bms, semantics
from mdac2.application.strategies.v5_spacing import conflict, hold_edge_conflict, resolve
from mdac2.data_preparation.training_chart import TrainingEvent


def event(key,t,lane='ground',kind='light',end=None):
    return TrainingEvent(key,kind,('ground','air') if kind=='gemini' else (lane,),t,
                         t if end is None else end,(key,),'01','test',kind=='extra_heavy')


def features():
    f=np.zeros((3,96,600),np.float32)
    f[1,12:17,100:230]=8
    f[0,12:17,100:230]=8
    f[1,38:43,300:390]=8
    f[0,38:43,300:390]=8
    return f


@pytest.fixture
def raw_boss_span(monkeypatch):
    """Neutralise the span tiers so the density/merge machinery stays observable.

    With the tiers active, short intervals get their end rewritten and mid-span
    intervals are demoted, which hides the merge and threshold rules.
    """
    monkeypatch.setattr('mdac2.application.strategies.v5_density.BOSS_SHORT_SPAN_MS',0)
    monkeypatch.setattr('mdac2.application.strategies.v5_density.BOSS_DEMOTE_SPAN_MS',(-1,-1))


def test_independent_tracks_extend_short_hold_and_add_second_phrase():
    f=features();f[0,68:73,450:550]=8
    es,r=refine([event('short',1.05,kind='hold',end=1.25)],f,6.)
    assert next(e for e in es if e.event_id=='short').end==pytest.approx(2.32,abs=.04)
    assert any(2.9<e.start<3.1 and e.end>3.8 for e in es if e.kind=='hold')
    assert any(4.4<e.start<4.6 and e.end>5.4 for e in es if e.kind=='hold')
    assert {r['source'] for r in r['regions']}=={'vocals','mix'}


def test_silence_short_attacks_and_changing_pitch_do_not_create_holds():
    f=np.zeros((3,96,600),np.float32)
    assert not detect(f,6.)[0]
    for i in range(10,590,20):f[:2,(i//20)%80,i:i+5]=8
    assert not detect(f,6.)[0]


def test_dense_group_counts_whole_segment_and_gemini_only_once(raw_boss_span):
    es=[event('a',1),event('b',1.05,kind='gemini'),event('c',1.1),event('d',1.075,'air'),event('far',3)]
    out,r=dense_groups(es,settings())
    boss=next(e for e in out if e.kind=='masher')
    assert boss.source_note_count==4 and boss.target_hits==8
    assert (boss.start,boss.end)==(1,1.1)
    assert {e.event_id for e in out if e.kind!='masher'}=={'far'}
    assert set(r[0]['source_ids'])=={'a','b','c','d'}


def test_overlapping_dense_spans_merge_across_lanes(raw_boss_span):
    es=[event('a',1),event('b',1.05),event('c',1.025,'air'),event('d',1.075,'air')]
    out,_=dense_groups(es,settings())
    # Two discarded / two retained exceeds the 80% threshold.
    assert len(out)==1 and out[0].source_note_count==4


@pytest.mark.parametrize('times,expected_boss,expected_retained',[
    ([1,1.06],True,1), # 1 / 1 > 80%
    ([1,1.06,1.12],False,2), # 1 / 2 below 80%
    ([1,1.06,1.12,1.18],True,2), # 2 / 2 > 80%
    ([1,1.06,1.12,1.18,1.24],False,3), # 2 / 3 between 50% and 80%
    ([1,1.06,1.12,1.18,1.24,1.3,1.36],False,4), # 3 / 4 between 50% and 80%
    ([1,1.06,1.12,1.18,1.24,1.3,1.36,1.42],True,4), # 4 / 4 > 80%
    ([1,1.06,1.12,1.18,1.24,1.3,1.36,1.42,1.48],False,5), # 4 / 5 == 80%
])
def test_boss_requires_strict_discard_to_retain_threshold(raw_boss_span,times,expected_boss,expected_retained):
    audit={};es=[event(str(i),t) for i,t in enumerate(times)]
    # The threshold is spacing-independent; pin 100 ms so the 60 ms grid applies.
    out,_=dense_groups(es,settings(),audit=audit,spacing=.1)
    assert any(e.kind=='masher' for e in out)==expected_boss
    decision=audit['segments'][0]
    assert decision['retained']==expected_retained
    assert decision['discarded']==len(times)-expected_retained
    if not expected_boss:
        assert len(out)==expected_retained
        assert all(e in es for e in out) # deletion only: no flip, retiming or type change


@pytest.mark.parametrize('gap_ms,count',[(499,1),(500,2),(501,2)])
def test_boss_merge_uses_end_to_start_and_strict_500ms(raw_boss_span,gap_ms,count):
    first=[event('a'+str(i),1+i*.04) for i in range(3)]
    start=round(1.08+gap_ms/1000,3)
    second=[event('b'+str(i),round(start+i*.04,3),'air') for i in range(3)]
    out,_=dense_groups(first+second,settings())
    bosses=[e for e in out if e.kind=='masher']
    assert len(bosses)==count
    assert sum(e.source_note_count for e in bosses)==6
    assert sum(e.target_hits for e in bosses)==12


def test_chained_boss_merge_counts_bridge_notes_once(raw_boss_span):
    es=[event(f'{j}-{i}',round(1+j*.5+i*.04,3)) for j in range(3) for i in range(3)]
    es.append(event('bridge',1.3,'air'))
    out,_=dense_groups(es,settings())
    assert len(out)==1
    assert (out[0].start,out[0].end)==(1,2.08)
    assert out[0].source_note_count==10 and out[0].target_hits==20
    assert len(set(out[0].source_ids))==10


def test_sparse_hold_interior_is_not_used_to_inflate_density_ratio():
    es=[event('h',.5,kind='hold',end=2),event('a',1),event('b',1.09),event('c',1.18)]
    audit={};out,_=dense_groups(es,settings(),audit=audit)
    assert not any(e.kind=='masher' for e in out)
    assert audit['segments'][0]['discard_to_retain_ratio']==.5


def test_valid_four_per_beat_is_not_dense():
    out,r=dense_groups([event(str(i),1+i*.125) for i in range(4)],dict(settings(),enabled=True))
    assert len(out)==4 and not r


@pytest.mark.parametrize('enabled',[False,True])
def test_hold_interior_keeps_notes_and_forbids_boss_near_it(enabled):
    es=[event('hold',.5,kind='hold',end=2.5),event('single',1.3,'air'),
        event('after',4,'air')]
    f=np.zeros((3,96,600),np.float32)
    pred=dict(probability=np.zeros(600),offsets=np.zeros(600))
    out,r,g,a=run(es,f,pred,6.,dict(switch_hazard=np.full((4,16),.4).tolist()),
                  dict(settings(),enabled=enabled),options=dict(spacing_ms=125,seed=0))
    assert a['engineering_playtest_gate_passed'],a
    # The hold no longer blanks its window: the opposite-lane note survives, either
    # in place or rerouted onto the sustained lane by the 50% rule.
    single=[e for e in out if e.event_id=='single']
    assert len(single)==1
    assert out and semantics(out)==semantics(read_bms(write_bms(out,6.),6.))


def test_hold_reroute_keeps_at_most_one_interior_hit():
    es=[event('hold',1,kind='hold',end=3),event('x',1.5,'air'),event('y',1.8,'air'),event('z',2.6,'air')]
    f=np.zeros((3,96,600),np.float32)
    pred=dict(probability=np.zeros(600),offsets=np.zeros(600))
    out,r,_,_=run(es,f,pred,6.,dict(switch_hazard=np.full((4,16),.4).tolist()),settings(),
                  options=dict(spacing_ms=125,seed=7))
    inside=[e for e in out if e.kind!='hold' and 1<e.start<3 and e.lanes==('ground',)]
    assert len(inside)<=1
    reroute=r['hold_reroute']
    assert reroute['minimal_hold_seconds']==.5 and reroute['first_probability']==.5
    assert reroute['rerouted_count']<=1


def test_boss_is_suppressed_within_500ms_of_a_hold(raw_boss_span):
    hold=event('h',.5,kind='hold',end=1.)
    near=dense_groups([hold]+[event(str(i),1.4+i*.04) for i in range(4)],settings(),holds=[hold])[0]
    far=dense_groups([hold]+[event(str(i),1.6+i*.04) for i in range(4)],settings(),holds=[hold])[0]
    assert not any(e.kind=='masher' for e in near)
    assert any(e.kind=='masher' for e in far)


def test_short_boss_is_kept_and_normalised_to_100ms():
    # 4 notes 40 ms apart: merged span 120 ms -> short accent tier.
    es=[event(f'n{i}',round(1+i*.04,3)) for i in range(4)]
    audit={};out,report=dense_groups(es,settings(),audit=audit)
    boss=next(e for e in out if e.kind=='masher')
    assert boss.start==1 and boss.end==pytest.approx(1.1)
    assert report[0]['span_ms']==120 and report[0]['action']=='boss'
    assert audit['short_bosses_normalised']==1 and audit['boss_span_restored_notes']==0


def test_mid_span_boss_falls_back_to_discard_mode():
    # 8 notes 40 ms apart: merged span 280 ms -> demoted tier.
    es=[event(f'n{i}',round(1+i*.04,3)) for i in range(8)]
    audit={};out,report=dense_groups(es,settings(),audit=audit)
    assert not any(e.kind=='masher' for e in out)
    assert report[0]['action']=='discard_only_boss_span' and report[0]['span_ms']==280
    # Survivors of _thin return as single notes; only the excess is dropped.
    assert {e.event_id for e in out}==set(report[0]['retained_ids'])
    assert 0<len(out)<len(es)
    assert audit['boss_span_restored_notes']==len(out)


def test_long_boss_beyond_two_seconds_is_untouched():
    # Three dense bursts 400 ms apart chain into one interval longer than 2 s.
    es=[event(f'n{index}-{i}',round(1+i*.4+index*.04,3)) for i in range(8) for index in range(3)]
    out,report=dense_groups(es,settings())
    boss=next(e for e in out if e.kind=='masher')
    assert report[0]['span_ms']>2000
    assert boss.end==pytest.approx(report[0]['end']) and boss.end-report[0]['start']>2


def test_refused_boss_never_deletes_notes_that_density_kept():
    es=[event(chr(97+i),round(1+i*.04,3)) for i in range(8)]
    out,report=dense_groups(es,settings(),spacing=.1)
    assert not any(e.kind=='masher' for e in out)
    kept=set(report[0]['retained_ids'])
    assert 0<len(kept)<8 and {e.event_id for e in out}==kept


def test_spacing_slider_and_doubled_hold_interior():
    ok=[event('a',1),event('b',1.14)]
    tight=[event('a',1),event('b',1.1)]
    assert not conflict(ok[0],ok[1],'ground',[],settings())
    assert conflict(tight[0],tight[1],'ground',[],settings())
    # The slider widens and narrows the ordinary floor.
    assert not conflict(tight[0],tight[1],'ground',[],settings(),spacing=.075)
    wide=[event('a',1),event('b',1.17)]
    assert conflict(wide[0],wide[1],'ground',[],settings(),spacing=.175)
    flipped=resolve(tight,6.,settings())[0]
    # 100 ms is illegal on one lane at 125 ms, so the pair splits across lanes.
    assert len(flipped)==2 and {e.lanes for e in flipped}=={('ground',),('air',)}
    # Interior notes inside one hold use twice the ordinary floor: 250 ms at 125.
    hold=event('h',.5,kind='hold',end=2)
    assert conflict(event('x',1),event('y',1.2),'ground',[hold],settings())
    assert not conflict(event('x',1),event('y',1.26),'ground',[hold],settings())

def test_quiet_edges_become_music_but_middle_and_silence_do_not():
    f=np.full((3,96,2000),np.log1p(100.),np.float32)
    f[0,:,50:150]=np.log1p(1.)
    f[0,:,950:1050]=np.log1p(1.)
    f[0,:,1850:1950]=0
    out,_=soft_edges([event('edge',1),event('middle',10),event('silence',19)],f,20.)
    assert [e.kind for e in out]==['music','light','light']


def test_special_export_and_identity_preserve_target_but_not_loader_directive():
    boss=SpecialEvent('boss','masher',('air',),1.,2.,('a','b','c'),'0G','test',source_note_count=3,target_hits=6)
    music=SpecialEvent('music','music',('ground',),3.,3.,('m',),'23','test')
    es=[event('hold',.5,kind='hold',end=2.5),boss,music]
    text=write_bms(es,5.)
    assert '// MDAC2_V5_MASHER' in text and '#TARGET_HITS' not in text
    assert semantics(read_bms(text,5.))==semantics(es)
    assert from_dict(boss.to_dict())==boss
    with pytest.raises(ValueError):read_bms(text.replace('#BPM 120','#BPM 130'),5.)
    with pytest.raises(ValueError):read_bms('\n'.join(x for x in text.splitlines() if not x.startswith('// MDAC2_V5_MASHER ')),5.)


def test_independent_bms_parser_sees_exact_masher_pair_and_music_code(tmp_path):
    from mdac2.data_preparation.community import inspect_bms
    from mdac2.data_preparation.community_labels import SourceTiming
    boss=SpecialEvent('boss','masher',('air',),1.137,2.251,('a','b'),'0G','test',source_note_count=2,target_hits=4)
    music=SpecialEvent('music','music',('ground',),3.123,3.123,('m',),'23','test')
    p=tmp_path/'chart.bms';p.write_text(write_bms([boss,music],5.),encoding='utf-8')
    raw=inspect_bms(p);assert not raw['errors'];timing=SourceTiming(raw)
    marks=[(c['code'],c['channel'],float(timing.seconds(c))) for c in raw['cells'] if c.get('code') in ('0G','23')]
    assert sorted(marks,key=lambda item:item[2])==[('0G','53',1.137),('0G','53',2.251),('23','14',3.123)]


def test_validator_allows_other_lane_music_inside_hold():
    music=SpecialEvent('music','music',('air',),1.,1.,('m',),'23','test')
    assert not issues([event('h',.5,kind='hold',end=2),music],3)


def test_bpm_snapping_keeps_second_hold_one_floor_after_the_dual_tail():
    es=[event('g',1,kind='hold',end=2.41),event('a',1,'air','hold',2.41),
        event('next',2.911,'air','hold',3.54)]
    f=np.zeros((3,96,600),np.float32)
    pred=dict(probability=np.zeros(600),offsets=np.zeros(600))
    out,_,_,audit=run(es,f,pred,6.,dict(switch_hazard=np.full((4,16),.4).tolist()),dict(settings(),enabled=True))
    assert audit['engineering_playtest_gate_passed']
    h=next(e for e in out if e.event_id=='next')
    # The hold tail is now just an ordinary note: one 125 ms floor, not 500 ms.
    assert h.start==2.917 and h.end==3.542
    assert h.start-2.41>=.125


def test_hold_tail_blocks_notes_only_within_one_floor():
    hold=event('h',.5,kind='hold',end=2)
    assert hold_edge_conflict(event('x',2.05),'ground',[hold]) is not None
    assert hold_edge_conflict(event('y',2.2),'ground',[hold]) is None
    # The other lane is unrestricted by the hold endpoint.
    assert hold_edge_conflict(event('z',2.05,'air'),'air',[hold]) is None
    assert hold_edge_conflict(hold,'ground',[hold]) is None


def test_smoothing_stays_finite_after_loud_to_silent_transition():
    f=np.zeros((3,96,600),np.float32)
    f[:2,:,:70]=np.linspace(.1,20,70)
    with np.errstate(invalid='raise'):detect(f,6.)


def test_target_rating_and_old_rating_are_separate():
    from mdac2.application.studio_difficulty_v5 import estimate
    from mdac2.application.studio_difficulty import estimate as old
    boss=SpecialEvent('boss','masher',('air',),1.,2.,('a','b'),'0G','test',source_note_count=2,target_hits=4)
    result=estimate(dict(strategy_id='v5',duration=6.,events=[boss.to_dict()]))
    assert result['target_hits']==4 and not result['target_hits_game_enforced']
    chart=dict(strategy_id='v4',duration=6.,events=[event('a',1).to_dict()])
    assert estimate(chart)==old(chart)


def test_v5_manifest_binds_new_sources_and_preserves_frozen_parent():
    from mdac2.application.studio_strategy_catalog import STRATEGIES
    root=Path(__file__).resolve().parents[2]
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    m=json.loads((root/STRATEGIES['v5']['manifest']).read_text())
    assert m['base_strategy_sha256']==sha(root/'recipes/strategies/v4.json')
    for name,item in m['postprocessing']['sources'].items():assert sha(root/name)==item['sha256'],name
    registry=json.loads((root/'models/registry.json').read_text(encoding='utf-8'))
    assert registry['current_strategy']=='v4'
    assert registry['strategies']['v5']['manifest_sha256']==sha(root/STRATEGIES['v5']['manifest'])
