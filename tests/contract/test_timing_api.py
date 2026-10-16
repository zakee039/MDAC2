"""Real local CPU worker. Missing optional runtime is an explicit test skip."""
from pathlib import Path
from dataclasses import replace
import json
import pytest
from fastapi.testclient import TestClient
from mdac2.server import build_app
from mdac2.infrastructure.timing_models import TimingModels
from mdac2.application.export_runtime import ExportRuntime
from mdac2.adapters.audio_io import AudioDecoder
from mdac2.application.backup import backup_workspace,restore_workspace
from mdac2.application.manual_workflow import ManualWorkflow
from mdac2.infrastructure.artifact_store import ArtifactStore
from mdac2.infrastructure.project_repository import ProjectRepository


ROOT=Path(__file__).parents[2]


def setup(tmp_path):
    if not (ROOT/'tools/.deps/beat-this-py313/Scripts/python.exe').is_file():pytest.skip('Optional real CPU environment missing')
    source=ROOT/'tools/private/beat-this/final0-fixed_4_4.metrics.json'
    if not source.is_file():pytest.skip('Generate licensed synthetic experiment fixture first')
    runtime=tmp_path/'runtime'
    app=build_app(ROOT,'t'*40,runtime_root=runtime)
    api=TestClient(app,base_url='http://127.0.0.1:8000',client=('127.0.0.1',123),headers={'Authorization':'Bearer '+'t'*40})
    pid=api.post('/api/v1/projects',json={'title':'Real timing test'}).json()['project_id']
    url=f'/api/v1/projects/{pid}'
    h=json.loads(source.read_text())['audio_artifact_hash']
    wav=(ROOT/'tools/private/beat-this/sample-store/artifacts/audio'/h/'audio.wav').read_bytes()
    assert api.post(url+'/audio',content=wav,headers={'Content-Type':'audio/wav'}).status_code==201
    binary=ROOT/'tools/.deps/ffmpeg/extracted/ffmpeg-9.0.1-essentials_build/bin'
    worker=ExportRuntime(app.state.repository,app.state.store,AudioDecoder(binary/'ffmpeg.exe',binary/'ffprobe.exe'),runtime/'queue.sqlite',timing_models=TimingModels(ROOT))
    return api,url,worker


def request():return dict(expected_revision=None,model_id='final0',pulse_unit='1/4',first_q='-1',meter=[4,4])


def test_real_analysis_review_edit_export_and_backup(tmp_path):
    api,url,worker=setup(tmp_path)
    try:
        assert api.get('/api/v1/timing/models').json()['models'][0]['available']
        response=api.post(url+'/timing-analysis',json=request());assert response.status_code==202,response.text
        job=response.json();jid=job['job_id']
        assert api.post(url+'/timing-analysis',json=request()).json()['job_id']==jid
        assert api.get('/api/v1/jobs/'+jid+'/timing-analysis').status_code==422
        worker.tick()
        status=api.get('/api/v1/jobs/'+jid).json();assert status['state']=='succeeded',status
        result=api.get('/api/v1/jobs/'+jid+'/timing-analysis').json()
        assert result['fit']['accepted'] and result['timing']['review']=='pending'
        assert api.get(url+'/manual').json()['chart'] is None
        cid=result['candidate_id'];preview=api.post(url+'/timing-analysis/'+cid+'/preview',json={'position_policy':'keep_audio_time'}).json()
        adopted=api.post(url+'/timing-analysis/'+cid+'/adopt',json=dict(position_policy='keep_audio_time',preview_fingerprint=preview['fingerprint'],actor='tester',reason='synthetic fixture inspection',command_id='accept'))
        assert adopted.status_code==200,adopted.text
        state=adopted.json();assert state['timing']['review']=='accepted'
        chart=state['chart'];old=chart['revision_id'];chart['revision_id']='edited'
        # Use the authoritative event serializer, keeping this integration fixture independent of schema spelling.
        from mdac2.domain.chart import ChartEvent
        chart['events']=[ChartEvent('note',1,('ground',)).to_dict()]
        saved=api.post(url+'/manual',json=dict(expected_revision=old,command_id='edit',chart=chart,timing=state['timing'],position_policy=None,preview_fingerprint=None))
        assert saved.status_code==200,saved.text
        export=api.post(url+'/exports',json={'expected_revision':'edited'}).json();worker.tick()
        assert api.get('/api/v1/jobs/'+export['job_id']).json()['state']=='succeeded'
        out=api.get('/api/v1/jobs/'+export['job_id']+'/export').json()
        assert api.get('/api/v1/exports/'+out['export_id']+'/download').status_code==200
        assert api.get(url+'/timing-analysis').json()['jobs'][0]['job_id']==jid
        backup_workspace(worker.backend,worker.store,tmp_path/'backup')
        restored=restore_workspace(tmp_path/'backup',tmp_path/'restored')
        manual=ManualWorkflow(ProjectRepository(restored/'state.sqlite'),ArtifactStore(restored/'store'))
        assert manual.load(url.rsplit('/',1)[1])[1].review=='accepted'
        assert ArtifactStore(restored/'store').get('timing_candidate',cid).payload['fit']['accepted']
    finally:worker.close()


def test_cancel_missing_model_bad_request_and_stale_base(tmp_path):
    api,url,worker=setup(tmp_path)
    try:
        assert api.post(url+'/timing-analysis',json={**request(),'first_q':'1/0'}).status_code==422
        assert api.post(url+'/timing-analysis',json={**request(),'expected_revision':'old'}).status_code==409
        assert api.post(url+'/timing-analysis',json={**request(),'model_id':'not-a-model'}).status_code==422
        assert api.post(url+'/timing-analysis',json=request(),headers={'Authorization':''}).status_code==401
        job=api.post(url+'/timing-analysis',json=request()).json()
        assert api.post('/api/v1/jobs/'+job['job_id']+'/cancel',json={'run_id':job['run_id']}).status_code==200
        worker.tick();assert worker.backend.get_job(job['job_id']).state=='cancelled'
        assert api.get('/api/v1/jobs/'+job['job_id']+'/timing-analysis').status_code==422
    finally:worker.close()
