from pathlib import Path
from dataclasses import replace
from fastapi.testclient import TestClient
from test_onset_workflow import prepared
from mdac2.adapters.audio_io import AudioDecoder
from mdac2.api.projects import create_app
from mdac2.api.jobs import attach_jobs
from mdac2.api.manual import attach_manual
from mdac2.api.onsets import attach_onsets
from mdac2.api.export import attach_export
from mdac2.application.onset_workflow import OnsetWorkflow
from mdac2.application.manual_workflow import ManualWorkflow
from mdac2.application.export_runtime import ExportRuntime
from mdac2.application.backup import backup_workspace, restore_workspace
from mdac2.infrastructure.artifact_store import ArtifactStore
from mdac2.infrastructure.job_backend import JobBackend


def setup(tmp_path):
    repo, store, project, flow, chart, timing = prepared(tmp_path/'source')
    backend = JobBackend(repo.database)
    flow = OnsetWorkflow(ManualWorkflow(backend, store))
    app = create_app(backend, store, token='t'*40)
    attach_jobs(app, backend); attach_onsets(app, flow, backend); attach_manual(app, flow.manual)
    binary = Path(__file__).parents[2]/'tools/.deps/ffmpeg/extracted/ffmpeg-9.0.1-essentials_build/bin'
    decoder = AudioDecoder(binary/'ffmpeg.exe', binary/'ffprobe.exe')
    attach_export(app, flow.manual, backend, store, decoder)
    worker = ExportRuntime(backend, store, decoder, tmp_path/'queue.sqlite')
    api = TestClient(app, base_url='http://127.0.0.1:8000', client=('127.0.0.1',123), headers={'Authorization':'Bearer '+'t'*40})
    return api, f'/api/v1/projects/{project}', worker, flow, chart, timing


def test_real_child_candidate_adopt_export_and_backup_closure(tmp_path):
    api, url, worker, flow, chart, timing = setup(tmp_path)
    try:
        body = {'expected_revision':chart.revision_id, 'difficulty':'normal'}
        response = api.post(url+'/onset-analysis', json=body)
        assert response.status_code == 202, response.text
        job = response.json(); jid = job['job_id']
        assert api.post(url+'/onset-analysis', json=body).json()['job_id'] == jid
        worker.tick()
        assert worker.backend.get_job(jid).state == 'succeeded'
        result = api.get('/api/v1/jobs/'+jid+'/onset-analysis').json()
        cid = result['candidate_id']; target = url+'/onset-analysis/'+cid
        assert len(result['onsets']['candidates']) >= 5
        assert api.get(url+'/manual').json()['chart']['revision_id'] == chart.revision_id
        preview = api.post(target+'/preview', json={'selected_ids':None}).json()
        adopted = api.post(target+'/adopt', json={'selected_ids':preview['selected_ids'],
            'preview_fingerprint':preview['fingerprint'], 'actor':'fixture','reason':'synthetic fixture diff','command_id':'adopt'})
        assert adopted.status_code == 200, adopted.text
        state = adopted.json(); revision = state['chart']['revision_id']
        export = api.post(url+'/exports', json={'expected_revision':revision}).json()
        worker.tick()
        assert worker.backend.get_job(export['job_id']).state == 'succeeded'
        out = api.get('/api/v1/jobs/'+export['job_id']+'/export').json()
        assert api.get('/api/v1/exports/'+out['export_id']+'/download').status_code == 200
        assert api.get(url+'/onset-analysis').json()['jobs'][0]['job_id'] == jid
        assert api.post(target+'/preview', json={'selected_ids':None}).status_code == 409
        assert api.get('/api/v1/jobs/'+jid+'/onset-analysis').status_code == 200
        backup_workspace(worker.backend, worker.store, tmp_path/'backup')
        restored = restore_workspace(tmp_path/'backup', tmp_path/'restored')
        recovered = ArtifactStore(restored/'store')
        assert recovered.get('onset_candidate',cid).payload['base_revision'] == chart.revision_id
        assert recovered.get('onset_review',state['chart']['onset_revision']).payload['fingerprint'] == preview['fingerprint']
    finally: worker.close()


def test_invalid_cancel_and_stale_queued_analysis_preserve_head(tmp_path):
    api, url, worker, flow, chart, timing = setup(tmp_path)
    try:
        body = {'expected_revision':chart.revision_id,'difficulty':'normal'}
        assert api.post(url+'/onset-analysis',json=body,headers={'Authorization':''}).status_code == 401
        assert api.post(url+'/onset-analysis',json={**body,'difficulty':[]}).status_code == 422
        assert api.post(url+'/onset-analysis',json={**body,'difficulty':'unknown'}).status_code == 422
        job = api.post(url+'/onset-analysis',json=body).json()
        assert api.post('/api/v1/jobs/'+job['job_id']+'/cancel',json={'run_id':job['run_id']}).status_code == 200
        worker.tick(); assert worker.backend.get_job(job['job_id']).state == 'cancelled'
        assert api.get('/api/v1/jobs/'+job['job_id']+'/onset-analysis').status_code == 422
        next_job = api.post(url+'/onset-analysis',json={**body,'difficulty':'hard'}).json()
        project = url.rsplit('/',1)[1]
        changed = replace(chart,revision_id='concurrent-edit')
        flow.manual.save(project,expected_revision=chart.revision_id,command_id='edit',chart=changed,timing=timing)
        worker.tick(); assert worker.backend.get_job(next_job['job_id']).state == 'failed'
        assert flow.manual.load(project) == (changed,timing)
    finally: worker.close()
