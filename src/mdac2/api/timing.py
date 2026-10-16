from fractions import Fraction
from fastapi import Request
from starlette.concurrency import run_in_threadpool
from mdac2.api.security import fields,identifier,json_body
from mdac2.application.jobs_contract import JobRequest,public_run
from mdac2.application.manual_workflow import timing_data
from mdac2.analysis.timing_fit import FitPolicy


def attach_timing(app,flow,backend,models):
    @app.get('/api/v1/timing/models')
    def capabilities():return {'models':models.capabilities()}

    @app.post('/api/v1/projects/{project_id}/timing-analysis',status_code=202)
    async def analyze(project_id:str,request:Request):
        identifier(project_id);body=await json_body(request)
        fields(body,{'expected_revision','model_id','pulse_unit','first_q','meter'})
        if body['expected_revision'] is not None:identifier(body['expected_revision'])
        if body['pulse_unit'] not in ('1/4','1/8','3/8'):raise ValueError('Unsupported pulse interpretation')
        if type(body['first_q']) is not str or len(body['first_q'])>32:raise ValueError('Bounded rational first beat required')
        try:first=Fraction(body['first_q'])
        except (ValueError,ZeroDivisionError) as exc:raise ValueError('Invalid first beat') from exc
        if str(first)!=body['first_q'] or abs(first)>32:raise ValueError('Invalid first beat')
        if body['meter'] not in ([4,4],[3,4],[6,8]) or any(type(n) is not int for n in body['meter']):raise ValueError('Unsupported meter')
        def enqueue():
            _,inputs=flow.context(project_id,body['expected_revision'])
            _,_,producer=models.configuration(body['model_id'])
            options={k:body[k] for k in ('model_id','pulse_unit','first_q','meter')}
            options.update(project_id=project_id,base_revision=body['expected_revision'],fit_policy=FitPolicy().fingerprint,
                device='cpu',precision='float32',postprocessor='minimal')
            job=backend.create_job(JobRequest.create(project_id=project_id,project_revision=body['expected_revision'],
                operation='timing_candidate',inputs=inputs,producer=producer,options=options))
            return public_run(job.runs[-1])
        return await run_in_threadpool(enqueue)

    @app.get('/api/v1/projects/{project_id}/timing-analysis')
    def history(project_id:str):
        identifier(project_id);backend.get_project(project_id)
        with backend._connection() as c:
            ids=[r[0] for r in c.execute('SELECT job_id FROM jobs WHERE project_id=? ORDER BY rowid DESC LIMIT 100',(project_id,))]
        return {'jobs':[public_run(j.runs[-1]) for j in (backend.get_job(i) for i in ids) if j.request.to_dict()['operation']=='timing_candidate']}

    @app.get('/api/v1/jobs/{job_id}/timing-analysis')
    def output(job_id:str):
        job=backend.get_job(identifier(job_id))
        if job.state!='succeeded' or job.request.to_dict()['operation']!='timing_candidate':raise ValueError('No successful analysis')
        with backend._connection() as c:row=c.execute('SELECT artifact_hash FROM run_outputs WHERE run_id=?',(job.runs[-1].run_id,)).fetchone()
        if row is None:raise KeyError(job_id)
        candidate=flow.load(job.request.to_dict()['project_id'],row[0],current=False)
        return {'candidate_id':candidate.artifact_hash,**candidate.payload}

    @app.post('/api/v1/projects/{project_id}/timing-analysis/{candidate_id}/preview')
    async def preview(project_id:str,candidate_id:str,request:Request):
        identifier(project_id);identifier(candidate_id);body=await json_body(request);fields(body,{'position_policy'})
        return await run_in_threadpool(flow.preview,project_id,candidate_id,body['position_policy'])

    @app.post('/api/v1/projects/{project_id}/timing-analysis/{candidate_id}/adopt')
    async def adopt(project_id:str,candidate_id:str,request:Request):
        identifier(project_id);identifier(candidate_id);body=await json_body(request)
        fields(body,{'position_policy','preview_fingerprint','actor','reason','command_id'});identifier(body['command_id'])
        chart,timing=await run_in_threadpool(flow.adopt,project_id,candidate_id,**body)
        return {'chart':chart.to_dict(),'timing':timing_data(timing),'invalidated':['preview','validation','export']}
