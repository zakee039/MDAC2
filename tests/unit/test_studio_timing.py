import math
import random
import pytest
from mdac2.application.studio_timing import settings,fit_beats,snap,rebase
from mdac2.data_preparation.training_chart import TrainingEvent
from mdac2.data_preparation.six_export import write_bms,read_bms,semantics
from mdac2.data_preparation.global_postprocess import avoidance_violations


def event(id,t,lane='ground',kind='light',end=None,group=None):
    return TrainingEvent(id,kind,('ground','air') if kind=='gemini' else (lane,),t,t if end is None else end,
        (id,),'01','fixture',kind=='extra_heavy',group)


def test_disabled_grid_preserves_frozen_timing_and_validates_settings():
    es=[event('a',.137)]
    assert snap(es,2,None)[0]==es
    for bad in [dict(settings(),bpm=float('nan')),dict(settings(),bpm=0),dict(settings(),enabled='yes'),dict(settings(),offset_ms=None)]:
        with pytest.raises(ValueError):settings(bad)


def test_nearby_same_lane_heavy_is_downgraded_without_bpm_and_retained():
    es=[event('light',1.0),event('heavy',1.15,kind='heavy'),event('far',1.25,kind='heavy')]
    out,report=snap(es,3,None)
    assert [(e.event_id,e.kind) for e in out]==[('light','light'),('heavy','light'),('far','heavy')]
    assert report['heavy_conflicts']['mode']=='nearby_200ms'
    assert report['heavy_conflicts']['changed_count']==1
    duplicate,_=snap([event('light',1.0),event('heavy',1.005,kind='heavy')],2,None)
    assert [(e.event_id,e.kind) for e in duplicate]==[('light','light'),('heavy','light')]


def test_same_beat_heavy_is_downgraded_even_when_more_than_200ms_apart():
    cfg=dict(settings(),enabled=True,bpm=60,subdivisions=12)
    out,report=snap([event('light',1.01),event('heavy',1.79,kind='heavy')],3,cfg)
    assert [e.kind for e in out]==['light','light']
    assert report['heavy_conflicts']['mode']=='same_beat'


def test_nearby_heavy_in_different_bpm_beats_is_kept():
    cfg=dict(settings(),enabled=True,bpm=60,subdivisions=12)
    out,_=snap([event('light',.91),event('heavy',1.08,kind='heavy')],3,cfg)
    assert [e.kind for e in out]==['light','heavy']


def test_twelve_subdivisions_preserve_halves_thirds_quarters_and_dense_notes():
    cfg=dict(settings(),enabled=True)
    assert cfg['subdivisions']==12
    times=[0,.125,.25,.375]  # four-note groups use positions 1, 4, 7, 10
    es=[event(str(i),t+.003) for i,t in enumerate(times)]
    out,report=snap(es,2,cfg)
    assert len(out)==len(es) and report['positions_per_beat']==12
    for e,t in zip(out,times):assert abs(e.start-t)<=.000500001
    actual={e.start for e in out}
    assert {0.,.125,.25,.375}<=actual
    assert semantics(out)==semantics(read_bms(write_bms(out,2),2))
    legacy={k:v for k,v in cfg.items() if k!='subdivisions'}
    assert settings(legacy)['subdivisions']==4
    assert snap(es,2,legacy)[1]['positions_per_beat']==4


def test_bpm_grid_uniformizes_supported_same_track_groups_and_drops_overdense_groups():
    cfg=dict(settings(),enabled=True,bpm=120,subdivisions=12)
    es=[event('a',.01),event('b',.19),event('hold',.21,kind='hold',end=.8),
        event('other-1',.01,'air'),event('other-2',.19,'air')]
    out,report=snap(es,2,cfg)
    ground=sorted(e.start for e in out if 'ground' in e.lanes)
    assert ground[:3]==[0.0,.167,.333]
    assert next(e for e in out if e.event_id=='hold').start==.333
    assert report['uniformization']['moved']>=1
    assert report['uniformization']['groups']

    dense=[event(str(i),.01+i*.03) for i in range(5)]+[event('air',.4,'air')]
    out,report=snap(dense,2,cfg)
    assert [e.event_id for e in out]==['air']
    assert set(report['uniformization']['dropped'])=={str(i) for i in range(5)}


def test_snaps_endpoints_resolves_collisions_and_special_truncation():
    es=[event('hold',.1,kind='hold',end=1.4),event('tap',.38),event('gemini',.65,kind='gemini'),
        event('special',1.02,'air','extra_heavy'),event('late',1.81,'air'),event('duplicate',1.8,'air')]
    out,report=snap(es,3,dict(settings(),enabled=True,bpm=120,subdivisions=4))
    assert not avoidance_violations(out)
    hold=next(e for e in out if e.event_id=='hold')
    assert hold.start==.125 and hold.end==1.
    assert all(t/.125==round(t/.125) for e in out for t in (e.start,e.end))
    assert any(e.event_id=='special' for e in out)
    assert len([e for e in out if e.event_id in ('late','duplicate')])==1
    assert report['collisions']['rejected_count']>=1
    assert semantics(out)==semantics(read_bms(write_bms(out,3),3))


def test_double_heavy_atomic_and_gemini_remain_distinct():
    es=[event('ha',.263,'air','heavy',group='pair'),event('hg',.263,'ground','heavy',group='pair'),event('g',1.13,kind='gemini')]
    out,_=snap(es,3,dict(settings(),enabled=True))
    assert sum(e.kind=='heavy' for e in out)==2 and sum(e.kind=='gemini' for e in out)==1
    assert out[0].start==out[1].start==.25


def test_short_holds_drop_and_arbitrary_bpm_respects_ms_storage():
    es=[event('short',.251,kind='hold',end=.26),event('a',1.128),event('b',2.934,'air')]
    out,report=snap(es,4,dict(settings(),enabled=True,bpm=137,offset_ms=-63))
    assert report['dropped'][0]['event_id']=='short'
    for e in out:
        for t in (e.start,e.end):
            n=round((t+.063)/(5/137));assert abs(t-(-.063+n*5/137))<=.000500001
    assert semantics(out)==semantics(read_bms(write_bms(out,4),4))


def test_grid_boundaries_and_random_collision_cases():
    rng=random.Random(98)
    for bpm in (30,97.3,139,240,400):
        es=[event(str(i),round(rng.random()*8,3),'air' if i%2 else 'ground') for i in range(60)]
        out,_=snap(es,8,dict(settings(),enabled=True,bpm=bpm,offset_ms=83))
        assert all(0<=e.start<=e.end<=8 for e in out)
        assert semantics(out)==semantics(read_bms(write_bms(out,8),8))


def test_fit_missing_beats_and_variable_tempo_warning():
    beats=[.12+i*.5 for i in range(40) if i not in (4,5,21)]
    fit=fit_beats(beats)
    assert abs(fit['bpm']-120)<.01 and abs(fit['offset_ms']-120)<.01 and fit['p95_ms']<1
    assert fit['needs_review']
    with pytest.raises(ValueError):fit_beats([1,1,1,2])
    with pytest.raises(ValueError):fit_beats([])


def test_tempo_change_snaps_crossing_hold_and_roundtrips_bms():
    cfg=dict(settings(),enabled=True,bpm=120,segments=[dict(start_ms=2000,bpm=60,offset_ms=2000,tap_latency_ms=0)])
    out,report=snap([event('hold',1.89,kind='hold',end=2.35),event('after',3.13,'air')],4,cfg)
    hold=next(e for e in out if e.event_id=='hold')
    assert hold.start==1.875 and hold.end==2.333
    assert next(e for e in out if e.event_id=='after').start==3.167
    assert report['policy']=='studio-tempo-map-v1'
    assert semantics(out)==semantics(read_bms(write_bms(out,4),4))
    # A trim can move a formerly later segment before zero: it owns the crop start.
    cfg['segments']=[dict(start_ms=-1000,bpm=60,offset_ms=-1000,tap_latency_ms=0)]
    assert snap([event('a',.13)],2,cfg)[0][0].start==.167
    with pytest.raises(ValueError):settings({**cfg,'segments':cfg['segments']*2})


def test_trim_rebase_preserves_phase_and_restores_hidden_points():
    cfg=dict(settings(),offset_ms=120,segments=[dict(start_ms=5000,bpm=90,offset_ms=5100,tap_latency_ms=0)])
    shifted=rebase(cfg,10000)
    assert shifted['offset_ms']==-9880 and shifted['segments'][0]['start_ms']==-5000
    assert rebase(shifted,-10000)==cfg
