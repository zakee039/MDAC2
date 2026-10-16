from fastapi import Request
from starlette.concurrency import run_in_threadpool
from mdac2.api.security import fields, identifier, json_body
from mdac2.application.jobs_contract import JobRequest, public_run
from mdac2.application.manual_workflow import timing_data
from mdac2.application.onset_workflow import PRODUCER, request_options


def selection(value):
    if value is None: return None
    if type(value) is not list or len(value) > 2048 or any(type(i) is not str or len(i) != 64 for i in value):
        raise ValueError('Bounded candidate ID list required')
    return tuple(value)


def attach_onsets(app, flow, backend):
    @app.post('/api/v1/projects/{project_id}/onset-analysis', status_code=202)
    async def analyze(project_id: str, request: Request):
        identifier(project_id); body = await json_body(request)
        fields(body, {'expected_revision', 'difficulty'}); identifier(body['expected_revision'])
        options = request_options(project_id, body['expected_revision'], body['difficulty'])
        def enqueue():
            _, inputs, _, _ = flow.context(project_id, body['expected_revision'])
            job = backend.create_job(JobRequest.create(project_id=project_id, project_revision=body['expected_revision'],
                operation='onset_candidate', inputs=inputs, producer=PRODUCER, options=options))
            return public_run(job.runs[-1])
        return await run_in_threadpool(enqueue)

    @app.get('/api/v1/projects/{project_id}/onset-analysis')
    def history(project_id: str):
        identifier(project_id); backend.get_project(project_id)
        with backend._connection() as c:
            ids = [r[0] for r in c.execute("SELECT job_id FROM jobs WHERE project_id=? AND json_extract(request,'$.operation')='onset_candidate' ORDER BY rowid DESC LIMIT 100", (project_id,))]
        matches = []
        for job_id in ids:
            job = backend.get_job(job_id)
            if job.request.to_dict()['operation'] == 'onset_candidate': matches.append(public_run(job.runs[-1]))
            if len(matches) == 100: break
        return {'jobs': matches}

    @app.get('/api/v1/jobs/{job_id}/onset-analysis')
    def output(job_id: str):
        job = backend.get_job(identifier(job_id))
        if job.state != 'succeeded' or job.request.to_dict()['operation'] != 'onset_candidate':
            raise ValueError('No successful onset analysis')
        with backend._connection() as c:
            row = c.execute('SELECT artifact_hash FROM run_outputs WHERE run_id=?', (job.runs[-1].run_id,)).fetchone()
        if row is None: raise KeyError(job_id)
        candidate = flow.load(job.request.to_dict()['project_id'], row[0], current=False)
        return {'candidate_id': candidate.artifact_hash, **candidate.payload}

    @app.post('/api/v1/projects/{project_id}/onset-analysis/{candidate_id}/preview')
    async def preview(project_id: str, candidate_id: str, request: Request):
        identifier(project_id); identifier(candidate_id); body = await json_body(request)
        fields(body, {'selected_ids'})
        return await run_in_threadpool(flow.preview, project_id, candidate_id, selected_ids=selection(body['selected_ids']))

    @app.post('/api/v1/projects/{project_id}/onset-analysis/{candidate_id}/adopt')
    async def adopt(project_id: str, candidate_id: str, request: Request):
        identifier(project_id); identifier(candidate_id); body = await json_body(request)
        fields(body, {'selected_ids', 'preview_fingerprint', 'actor', 'reason', 'command_id'})
        identifier(body['command_id']); body['selected_ids'] = selection(body['selected_ids'])
        if body['selected_ids'] is None: raise ValueError('Explicit reviewed selection required')
        chart, timing = await run_in_threadpool(flow.adopt, project_id, candidate_id, **body)
        return {'chart': chart.to_dict(), 'timing': timing_data(timing), 'invalidated': ['preview', 'validation', 'export']}
