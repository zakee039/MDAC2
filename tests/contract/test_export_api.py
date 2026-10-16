"""Real child-process export, persistent delivery, and authenticated downloads."""
from pathlib import Path
from dataclasses import replace
import io
import time
import zipfile
from fastapi.testclient import TestClient
from test_upload import audio
from mdac2.server import build_app
from mdac2.application.manual_workflow import timing_data
from mdac2.domain.chart import ChartRevision, ChartEvent
from mdac2.domain.timing import TimingMap, Anchor
from mdac2.domain.timing_edits import TimingRevision
from mdac2.application.export_runtime import ExportRuntime
from mdac2.adapters.audio_io import AudioDecoder


def setup(tmp_path):
    root=Path(__file__).parents[2]
    app=build_app(root,'t'*40,runtime_root=tmp_path)
    api=TestClient(app,base_url='http://127.0.0.1:8000',client=('127.0.0.1',123),headers={'Authorization':'Bearer '+'t'*40})
    project=api.post('/api/v1/projects',json={'title':'Synthetic'}).json()['project_id']
    url=f'/api/v1/projects/{project}'
    assert api.post(url+'/audio',content=audio(96000),headers={'Content-Type':'audio/wav'}).status_code==201
    timing=TimingRevision('t',TimingMap((Anchor(0,0),Anchor(4,2)),2))
    chart=ChartRevision('c','t',(ChartEvent('e',1,('ground',)),))
    body=dict(expected_revision=None,command_id='cmd',chart=chart.to_dict(),timing=timing_data(timing),position_policy=None,preview_fingerprint=None)
    assert api.post(url+'/manual',json=body).status_code==200
    binary=root/'tools/.deps/ffmpeg/extracted/ffmpeg-9.0.1-essentials_build/bin'
    worker=ExportRuntime(app.state.repository,app.state.store,AudioDecoder(binary/'ffmpeg.exe',binary/'ffprobe.exe'),tmp_path/'queue.sqlite')
    return api,url,worker,body


def test_real_worker_export_download_and_reopen(tmp_path):
    api,url,worker,_=setup(tmp_path)
    job=api.post(url+'/exports',json={'expected_revision':'c'})
    assert job.status_code==202
    job=job.json();jid=job['job_id']
    assert api.post(url+'/exports',json={'expected_revision':'c'}).json()['job_id']==jid
    assert api.get('/api/v1/jobs/'+jid+'/export').status_code==422
    worker.tick()
    state=api.get('/api/v1/jobs/'+jid).json()
    assert state['state']=='succeeded',state
    result=api.get('/api/v1/jobs/'+jid+'/export').json()
    download='/api/v1/exports/'+result['export_id']+'/download'
    assert api.get(download,headers={'Authorization':''}).status_code==401
    package=api.get(download)
    assert package.status_code==200
    with zipfile.ZipFile(io.BytesIO(package.content)) as z:
        assert z.testzip() is None and len(z.namelist())==5
    assert result['manifest']['chart_revision']=='c'
    assert result['manifest']['game_verified'] is False
    reopened=build_app(Path(__file__).parents[2],'t'*40,runtime_root=tmp_path)
    client=TestClient(reopened,base_url='http://127.0.0.1:8000',client=('127.0.0.1',123),headers={'Authorization':'Bearer '+'t'*40})
    assert client.get(download).content==package.content
    worker.close()


def test_cancel_and_stale_export(tmp_path):
    api,url,worker,body=setup(tmp_path)
    job=api.post(url+'/exports',json={'expected_revision':'c'}).json()
    assert api.post('/api/v1/jobs/'+job['job_id']+'/cancel',json={'run_id':job['run_id']}).status_code==200
    worker.tick()
    assert worker.backend.get_job(job['job_id']).state=='cancelled'
    body.update(expected_revision='c',command_id='cmd2');body['chart']['revision_id']='c2'
    assert api.post(url+'/manual',json=body).status_code==200
    assert api.post(url+'/exports',json={'expected_revision':'c'}).status_code==409
    assert api.get('/api/v1/exports/'+'a'*64+'/download').status_code==404
    worker.close()


def test_expired_worker_recovery_and_static_boundary(tmp_path):
    api,url,worker,_=setup(tmp_path)
    job=api.post(url+'/exports',json={'expected_revision':'c'}).json()
    worker.backend.claim(job['job_id'],job['run_id'],'dead',now=time.time()-60,lease_seconds=1)
    worker.tick()
    run=worker.backend.get_job(job['job_id']).runs[-1]
    assert run.state=='interrupted' and run.exit_confirmed
    assert api.get('/',headers={'Authorization':''}).status_code==200
    assert api.get(url,headers={'Authorization':''}).status_code==401
    assert api.get('/',headers={'Origin':'https://evil.example'}).status_code==403
    worker.close()

def test_lost_delivery_recovered_on_restart_and_bad_child_fails(tmp_path):
    api,url,worker,body=setup(tmp_path)
    body.update(expected_revision='c',command_id='accent');body['chart']['revision_id']='accent'
    body['chart']['events'][0]['accent']=True
    assert api.post(url+'/manual',json=body).status_code==200
    job=api.post(url+'/exports',json={'expected_revision':'accent'}).json()
    worker.backend.dispatch(worker.transport)
    assert worker.transport.huey.dequeue() is not None  # simulate crash before claim
    assert worker.reconciler.recover_queued_delivery()==1
    worker.tick()
    assert worker.backend.get_job(job['job_id']).state=='failed'
    assert api.get('/api/v1/jobs/'+job['job_id']+'/export').status_code==422
    assert worker.backend.get_project(url.rsplit('/',1)[1]).head_revision=='accent'
    worker.close()
