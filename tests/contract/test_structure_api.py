from pathlib import Path
from fastapi.testclient import TestClient
from test_structure_workflow import ready
from mdac2.adapters.audio_io import AudioDecoder
from mdac2.api.projects import create_app
from mdac2.api.jobs import attach_jobs
from mdac2.api.structure import attach_structure
from mdac2.api.export import attach_export
from mdac2.application.structure_workflow import StructureWorkflow
from mdac2.application.manual_workflow import ManualWorkflow
from mdac2.application.export_runtime import ExportRuntime
from mdac2.application.backup import backup_workspace,restore_workspace
from mdac2.infrastructure.artifact_store import ArtifactStore
from mdac2.infrastructure.job_backend import JobBackend


def setup(tmp_path):
    repo,store,project,oldflow,chart,timing,_,_=ready(tmp_path/'source')
    backend=JobBackend(repo.database);flow=StructureWorkflow(ManualWorkflow(backend,store))
    app=create_app(backend,store,token='t'*40);attach_jobs(app,backend);attach_structure(app,flow,backend)
    binary=Path(__file__).parents[2]/'tools/.deps/ffmpeg/extracted/ffmpeg-9.0.1-essentials_build/bin'
    decoder=AudioDecoder(binary/'ffmpeg.exe',binary/'ffprobe.exe');attach_export(app,flow.manual,backend,store,decoder)
    worker=ExportRuntime(backend,store,decoder,tmp_path/'queue.sqlite')
    api=TestClient(app,base_url='http://127.0.0.1:8000',client=('127.0.0.1',123),headers={'Authorization':'Bearer '+'t'*40})
    return api,f'/api/v1/projects/{project}',worker,flow,chart,timing


def test_real_worker_confirm_reuse_export_reopen_undo_and_backup(tmp_path):
    api,url,worker,flow,chart,timing=setup(tmp_path)
    try:
        assert api.get('/api/v1/structure/capabilities').json()['available']
        body={'expected_revision':chart.revision_id}
        response=api.post(url+'/structure-analysis',json=body);assert response.status_code==202,response.text
        job=response.json();jid=job['job_id']
        assert api.post(url+'/structure-analysis',json=body).json()['job_id']==jid
        worker.tick();assert worker.backend.get_job(jid).state=='succeeded'
        result=api.get('/api/v1/jobs/'+jid+'/structure-analysis').json();p=result['proposals'][0]
        assert p['alignment']['status']=='accepted'
        review=api.post(url+'/structure-analysis/'+result['candidate_id']+'/confirm',json={
            'primary':p['source'],'targets':[{'proposal_id':p['proposal_id'],'span':p['target'],'relation':'exact_repeat',
                'allowed_modes':['arrangement'],'confirmed':True}],'actor':'fixture','reason':'synthetic review'}).json()
        options={'target_ids':[p['proposal_id']],'mode':'arrangement','source_hold':'reject','target_hold':'reject','locked':'reject'}
        target=url+'/structure-reviews/'+review['review_id']
        preview=api.post(target+'/preview',json=options).json()
        assert preview['changes'] and not preview['conflicts']
        applied=api.post(target+'/apply',json={'options':options,'preview_fingerprint':preview['fingerprint'],'command_id':'apply'})
        assert applied.status_code==200,applied.text
        value=applied.json();operation=value['operation_id'];revision=value['chart']['revision_id']
        assert api.get(url+'/structure-operation').json()['operation_id']==operation
        assert api.get(url+'/structure-analysis').json()['jobs'][0]['job_id']==jid
        assert api.get('/api/v1/jobs/'+jid+'/structure-analysis').status_code==200
        assert api.post(target+'/preview',json=options).status_code==409
        export=api.post(url+'/exports',json={'expected_revision':revision}).json();worker.tick()
        assert worker.backend.get_job(export['job_id']).state=='succeeded'
        undone=api.post(url+'/structure-operations/'+operation+'/undo',json={'expected_revision':revision,'command_id':'undo'})
        assert undone.status_code==200,undone.text
        assert undone.json()['chart']['events']==chart.to_dict()['events']
        assert api.get(url+'/structure-operation').json()['operation_id'] is None
        backup_workspace(worker.backend,worker.store,tmp_path/'backup')
        recovered=ArtifactStore(restore_workspace(tmp_path/'backup',tmp_path/'restored')/'store')
        assert recovered.get('structure_candidate',result['candidate_id']).payload['features']['artifact_type']=='similarity_features'
        assert recovered.get('structure_operation',operation).payload['applied_chart']==value['chart']
    finally:worker.close()


def test_auth_bad_request_and_cancel_no_head_change(tmp_path):
    api,url,worker,flow,chart,timing=setup(tmp_path)
    try:
        assert api.post(url+'/structure-analysis',json={'expected_revision':chart.revision_id},headers={'Authorization':''}).status_code==401
        assert api.post(url+'/structure-analysis',json={'expected_revision':'wrong'}).status_code==409
        job=api.post(url+'/structure-analysis',json={'expected_revision':chart.revision_id}).json()
        assert api.post('/api/v1/jobs/'+job['job_id']+'/cancel',json={'run_id':job['run_id']}).status_code==200
        worker.tick();assert worker.backend.get_job(job['job_id']).state=='cancelled'
        assert api.get('/api/v1/jobs/'+job['job_id']+'/structure-analysis').status_code==422
        assert flow.manual.load(url.rsplit('/',1)[1])==(chart,timing)
    finally:worker.close()
