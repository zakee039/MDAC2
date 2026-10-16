from dataclasses import replace
from fractions import Fraction as F
from test_upload import setup,audio
from mdac2.api.manual import attach_manual
from mdac2.application.manual_workflow import ManualWorkflow,timing_data
from mdac2.domain.chart import ChartRevision,ChartEvent
from mdac2.domain.timing import TimingMap,Anchor
from mdac2.domain.timing_edits import TimingRevision


def initialized(tmp_path):
    repo,store,api,project=setup(tmp_path)
    workflow=ManualWorkflow(repo,store);attach_manual(api.app,workflow)
    assert api.post(f'/api/v1/projects/{project}/audio',content=audio(96000),headers={'Content-Type':'audio/wav'}).status_code==201
    chart=ChartRevision('c','t',(ChartEvent('e',F(1),('ground',)),))
    timing=TimingRevision('t',TimingMap((Anchor(0,0),Anchor(4,2)),2))
    body=dict(expected_revision=None,command_id='cmd',chart=chart.to_dict(),timing=timing_data(timing),position_policy=None,preview_fingerprint=None)
    url=f'/api/v1/projects/{project}/manual'
    assert api.post(url,json=body).status_code==200
    return repo,store,api,project,url,chart,timing


def test_persistent_authority_and_conflicting_save(tmp_path):
    repo,store,api,project,url,chart,timing=initialized(tmp_path)
    reopened=ManualWorkflow(repo,store).load(project)
    assert reopened==(chart,timing)
    assert api.get(url).json()['timing']==timing_data(timing)
    body=dict(expected_revision=None,command_id='other',chart=replace(chart,revision_id='other').to_dict(),timing=timing_data(timing),position_policy=None,preview_fingerprint=None)
    assert api.post(url,json=body).status_code==409
    assert repo.get_project(project).head_revision=='c'


def test_preview_confirm_seconds_migration_and_stale_fingerprint(tmp_path):
    repo,store,api,project,url,chart,old=initialized(tmp_path)
    new=TimingRevision('t2',TimingMap((Anchor(0,0),Anchor(8,2)),2))
    preview=api.post(url+'/preview',json=dict(expected_revision='c',timing=timing_data(new),position_policy='keep_audio_time'))
    assert preview.status_code==200
    data=preview.json();assert data['chart']['events'][0]['q']=='2'
    body=dict(expected_revision='c',command_id='cmd2',chart=data['chart'],timing=timing_data(new),position_policy='keep_audio_time',preview_fingerprint='bad')
    assert api.post(url,json=body).status_code==422
    body['preview_fingerprint']=data['fingerprint'];assert api.post(url,json=body).status_code==200
    assert api.get(url).json()['chart']['timing_revision']=='t2'


def test_reference_insert_failure_rolls_back_head(tmp_path,monkeypatch):
    repo,store,api,project,url,chart,timing=initialized(tmp_path)
    with repo._connection() as c:
        c.execute("CREATE TRIGGER refuse_timing BEFORE INSERT ON revision_timing BEGIN SELECT RAISE(ABORT,'injected'); END")
    body=dict(expected_revision='c',command_id='cmd2',chart=replace(chart,revision_id='c2').to_dict(),timing=timing_data(timing),position_policy=None,preview_fingerprint=None)
    assert api.post(url,json=body).status_code==409
    assert repo.get_project(project).head_revision=='c'
