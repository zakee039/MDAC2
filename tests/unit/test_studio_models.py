import numpy as np
import pytest
from mdac2.application.studio_models import select_onsets,run_strategy
from mdac2.application.studio_timing import settings,snap
from mdac2.data_preparation.training_chart import TrainingEvent
from mdac2.data_preparation.global_postprocess import avoidance_violations
from mdac2.data_preparation.six_export import write_bms,read_bms,semantics


@pytest.mark.parametrize('model',['r2','v2','v3','v4'])
@pytest.mark.parametrize('enabled',[False,True])
def test_each_complete_strategy_roundtrips(model, enabled):
    f=np.zeros((3,96,600),np.float32)
    prediction=dict(probability=np.zeros(600,np.float32),offsets=np.zeros(600,np.float32))
    events=[TrainingEvent('h','hold',('ground',),.3,1.5,('h',),'01','test'),
            TrainingEvent('n','light',('air',),2.,2.,('n',),'01','test')]
    out,report,grid,audit=run_strategy(model,events,f,prediction,6.,
        dict(switch_hazard=np.full((4,16),.4).tolist()),dict(settings(),enabled=enabled))
    assert audit['engineering_playtest_gate_passed'] and grid['enabled']==enabled
    assert semantics(out)==semantics(read_bms(write_bms(out,6.),6.))


def test_strategy_entrypoints_and_registrations_are_independent():
    import hashlib
    import json
    from pathlib import Path
    from importlib import import_module
    from mdac2.application.studio_models import STRATEGIES
    root=Path(__file__).resolve().parents[2]
    registry=json.loads((root/'models/registry.json').read_text(encoding='utf-8'))
    paths=[s['manifest'] for s in STRATEGIES.values()]
    assert len(set(paths))==4
    for sid,spec in STRATEGIES.items():
        path=root/spec['manifest']
        assert registry['strategies'][sid]['manifest_sha256']==hashlib.sha256(path.read_bytes()).hexdigest()
        manifest=json.loads(path.read_text(encoding='utf-8'))
        assert manifest['strategy_id']==sid
        for name,item in manifest['postprocessing']['sources'].items():
            assert hashlib.sha256((root/name).read_bytes()).hexdigest()==item['sha256'],name
        module=import_module('mdac2.application.strategies.'+sid+('.pipeline' if sid!='v4' else ''))
        assert module.run.__module__.startswith('mdac2.application.strategies.'+sid)
        if sid!='v4':
            assert module.process.__module__=='mdac2.application.strategies.'+sid+'.postprocess'
            assert module.snap.__module__=='mdac2.application.strategies.'+sid+'.timing'


@pytest.mark.parametrize('model',['r2','v2','v3'])
def test_profile_selection_global_rules_and_optional_grid(model):
    if model=='v2':pytest.importorskip('torch')  # Frozen inference imports torch; verified separately in GPU runtime.
    f=np.zeros((3,96,600),np.float32);f[0]=f[2]=9
    p=np.zeros(600,np.float32);p[::20]=.7
    prediction=dict(probability=p,offsets=np.full(600,.3,np.float32))
    times,density,routing=select_onsets(model,f,prediction,6.)
    assert len(times)==30 and times[1]==.203  # no old 5ms quantization
    f[1,10:75,100:300]=8
    voiced,_,_=select_onsets(model,f,prediction,6.)
    if model=='r2':assert not any(1.2<t<2.8 for t in voiced)
    else:assert any(1.2<t<2.8 for t in voiced)
    f[1]=0
    def event(key,kind,start,end):
        return TrainingEvent(key,kind,('ground',),start,end,(key,),'01','test',False,None)
    events=[event('hold','hold',.303,1.503),event('tap','light',.803,.803)]
    out,report,_,audit=run_strategy(model,events,f,density,6.,dict(switch_hazard=np.full((4,16),.4).tolist()),settings())
    assert audit['engineering_playtest_gate_passed']
    assert report['policy']['version']==('playtest-postprocess-r1.1' if model=='v2' else 'global-postprocess-1ms-v2')
    if model!='v2':
        assert not avoidance_violations(out)
        assert all(e.lanes==('air',) for e in out if e.event_id=='tap')
    assert snap(out,6.,settings())[0]==out
    aligned,grid=snap(out,6.,dict(settings(),enabled=True,bpm=120))
    assert grid['positions_per_beat']==12 and not avoidance_violations(aligned)
    assert semantics(aligned)==semantics(read_bms(write_bms(aligned,6.),6.))
