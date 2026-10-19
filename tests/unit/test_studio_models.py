import numpy as np
import pytest
from mdac2.application.studio_models import select_onsets,postprocess
from mdac2.application.studio_timing import settings,snap
from mdac2.data_preparation.training_chart import TrainingEvent
from mdac2.data_preparation.global_postprocess import avoidance_violations
from mdac2.data_preparation.six_export import write_bms,read_bms,semantics


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
    out,report=postprocess(events,f,density,6.,dict(switch_hazard=np.full((4,16),.4).tolist()))
    assert report['policy']['version']=='global-postprocess-1ms-v2'
    assert not avoidance_violations(out)
    assert all(e.lanes==('air',) for e in out if e.event_id=='tap')
    assert snap(out,6.,settings())[0]==out
    aligned,grid=snap(out,6.,dict(settings(),enabled=True,bpm=120))
    assert grid['positions_per_beat']==12 and not avoidance_violations(aligned)
    assert semantics(aligned)==semantics(read_bms(write_bms(aligned,6.),6.))
