import math
import random
import pytest
from mdac2.application.studio_timing import settings,fit_beats,snap
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


def test_twelve_subdivisions_preserve_halves_thirds_quarters_and_dense_notes():
    cfg=dict(settings(),enabled=True)
    assert cfg['subdivisions']==12
    times=[i/24 for i in range(13)]  # one 120 BPM beat, all twelve positions
    es=[event(str(i),t+.003) for i,t in enumerate(times)]
    out,report=snap(es,2,cfg)
    assert len(out)==len(es) and report['positions_per_beat']==12
    for e,t in zip(out,times):assert abs(e.start-t)<=.000500001
    actual={e.start for e in out}
    assert {0.,.125,.167,.25,.333,.375,.5}<=actual
    assert semantics(out)==semantics(read_bms(write_bms(out,2),2))
    legacy={k:v for k,v in cfg.items() if k!='subdivisions'}
    assert settings(legacy)['subdivisions']==4
    assert snap(es,2,legacy)[1]['positions_per_beat']==4


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
