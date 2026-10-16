from importlib.metadata import PackageNotFoundError
from fastapi import Request
from starlette.concurrency import run_in_threadpool
from mdac2.api.security import fields, identifier, json_body
from mdac2.application.jobs_contract import JobRequest, public_run
from mdac2.application.manual_workflow import timing_data
from mdac2.application.structure_workflow import producer, request_options, envelope, proposal


def reuse_options(body):
    fields(body,{'target_ids','mode','source_hold','target_hold','locked'})
    ids=body['target_ids']
    if type(ids) is not list or not 1<=len(ids)<=20 or any(type(i) is not str or len(i)>128 for i in ids):
        raise ValueError('Bounded explicit targets required')
    return {**body,'target_ids':tuple(ids)}


def attach_structure(app, flow, backend):
    @app.get('/api/v1/structure/capabilities')
    def capabilities():
        try:return {'available':True,'producer':producer(),'mode':'mix_only'}
        except (PackageNotFoundError,ValueError):return {'available':False,'reason':'VERIFIED_ANALYSIS_ENVIRONMENT_MISSING','mode':'mix_only'}

    @app.post('/api/v1/projects/{project_id}/structure-analysis',status_code=202)
    async def analyze(project_id:str,request:Request):
        identifier(project_id); body=await json_body(request); fields(body,{'expected_revision'}); identifier(body['expected_revision'])
        def enqueue():
            _,inputs,_,_=flow.context(project_id,body['expected_revision'])
            job=backend.create_job(JobRequest.create(project_id=project_id,project_revision=body['expected_revision'],operation='structure_candidate',
                inputs=inputs,producer=producer(),options=request_options(project_id,body['expected_revision'])))
            return public_run(job.runs[-1])
        return await run_in_threadpool(enqueue)

    @app.get('/api/v1/projects/{project_id}/structure-analysis')
    def history(project_id:str):
        identifier(project_id); backend.get_project(project_id)
        with backend._connection() as c:
            ids=[r[0] for r in c.execute("SELECT job_id FROM jobs WHERE project_id=? AND json_extract(request,'$.operation')='structure_candidate' ORDER BY rowid DESC LIMIT 100",(project_id,))]
        return {'jobs':[public_run(backend.get_job(i).runs[-1]) for i in ids]}

    @app.get('/api/v1/jobs/{job_id}/structure-analysis')
    def output(job_id:str):
        job=backend.get_job(identifier(job_id)); request=job.request.to_dict()
        if job.state!='succeeded' or request['operation']!='structure_candidate': raise ValueError('No successful structure analysis')
        with backend._connection() as c: row=c.execute('SELECT artifact_hash FROM run_outputs WHERE run_id=?',(job.runs[-1].run_id,)).fetchone()
        if row is None:raise KeyError(job_id)
        candidate=flow.load(request['project_id'],row[0],current=False); data=candidate.payload
        proposals=[]
        for raw in envelope(data['search']).payload['proposals']:
            p=proposal(raw); alignment=envelope(data['alignments'][p.proposal_id]).payload
            proposals.append({**raw,'fingerprint':p.fingerprint,'alignment':alignment})
        return {'candidate_id':candidate.artifact_hash,**{key:data[key] for key in ('project_id','base_revision','timing_fingerprint','warnings')},'proposals':proposals}

    @app.post('/api/v1/projects/{project_id}/structure-analysis/{candidate_id}/confirm')
    async def confirm(project_id:str,candidate_id:str,request:Request):
        identifier(project_id); identifier(candidate_id); body=await json_body(request)
        fields(body,{'primary','targets','actor','reason'})
        if type(body['targets']) is not list:raise ValueError('Explicit targets required')
        for target in body['targets']:
            if type(target) is not dict or type(target.get('allowed_modes')) is not list or any(type(v) is not str for v in target['allowed_modes']):
                raise ValueError('Explicit reuse modes required')
        review=await run_in_threadpool(flow.confirm,project_id,candidate_id,**body)
        return {'review_id':review.artifact_hash,**review.payload}

    @app.post('/api/v1/projects/{project_id}/structure-reviews/{review_id}/preview')
    async def preview(project_id:str,review_id:str,request:Request):
        identifier(project_id); identifier(review_id)
        return await run_in_threadpool(flow.preview,project_id,review_id,**reuse_options(await json_body(request)))

    @app.post('/api/v1/projects/{project_id}/structure-reviews/{review_id}/apply')
    async def apply(project_id:str,review_id:str,request:Request):
        identifier(project_id); identifier(review_id); body=await json_body(request)
        fields(body,{'options','preview_fingerprint','command_id'}); identifier(body['command_id'])
        chart,timing,operation=await run_in_threadpool(flow.apply,project_id,review_id,preview_fingerprint=body['preview_fingerprint'],
            command_id=body['command_id'],**reuse_options(body['options']))
        return {'chart':chart.to_dict(),'timing':timing_data(timing),'operation_id':operation,'invalidated':['preview','validation','export']}

    @app.get('/api/v1/projects/{project_id}/structure-operation')
    def current_operation(project_id:str):
        identifier(project_id); project=backend.get_project(project_id)
        if project.head_revision is None:return {'operation_id':None}
        artifact=flow.store.get('chart',backend.get_revision(project_id,project.head_revision).artifact_hash)
        for identity in artifact.to_dict()['inputs']:
            try: operation=flow.store.get('structure_operation',identity)
            except FileNotFoundError:continue
            if operation.payload['project_id']==project_id and operation.payload['applied_chart']==artifact.payload:
                return {'operation_id':identity,'revision':project.head_revision}
        return {'operation_id':None}

    @app.post('/api/v1/projects/{project_id}/structure-operations/{operation_id}/undo')
    async def undo(project_id:str,operation_id:str,request:Request):
        identifier(project_id); identifier(operation_id); body=await json_body(request)
        fields(body,{'expected_revision','command_id'}); identifier(body['expected_revision']); identifier(body['command_id'])
        chart,timing=await run_in_threadpool(flow.undo,project_id,operation_id,**body)
        return {'chart':chart.to_dict(),'timing':timing_data(timing),'invalidated':['preview','validation','export']}
