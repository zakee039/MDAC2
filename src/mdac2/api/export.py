import re
from fastapi import Request
from starlette.concurrency import run_in_threadpool
from starlette.responses import FileResponse
from mdac2.api.security import fields,identifier,json_body
from mdac2.application.jobs_contract import JobRequest,public_run
from mdac2.schemas.game_profiles import BASIC


def attach_export(app,workflow,backend,store,decoder):
    @app.post('/api/v1/projects/{project_id}/exports',status_code=202)
    async def create(project_id:str,request:Request):
        identifier(project_id);body=await json_body(request);fields(body,{'expected_revision'})
        identifier(body['expected_revision'])
        def enqueue():
            chart,timing=workflow.load(project_id,body['expected_revision'])
            ref=backend.get_revision(project_id,chart.revision_id)
            envelope=store.get('chart',ref.artifact_hash)
            # Chart lineage can contain review/prior-chart evidence. Export has
            # three direct inputs; transitive provenance stays attached to chart.
            with backend._connection() as c:
                audio=c.execute('SELECT artifact_hash FROM project_audio WHERE project_id=?',(project_id,)).fetchone()
                timing_ref=c.execute('SELECT artifact_hash FROM revision_timing WHERE revision_id=?',(chart.revision_id,)).fetchone()
            if audio is None or timing_ref is None:raise ValueError('Export source bindings missing')
            store.get('audio',audio[0]);store.get('timing',timing_ref[0])
            inputs=[envelope.artifact_hash,audio[0],timing_ref[0]]
            producer={'name':'mdac2.export','version':'1','model_hash':None,'dependencies':decoder.producer['dependencies']}
            options={'profile':BASIC.profile_id,'title':backend.get_project(project_id).title,'artist':'MDAC2','historical':False}
            job=backend.create_job(JobRequest.create(project_id=project_id,project_revision=chart.revision_id,operation='export',
                inputs=inputs,producer=producer,options=options))
            return {'job_id':job.job_id,**public_run(job.runs[-1])}
        return await run_in_threadpool(enqueue)

    @app.get('/api/v1/jobs/{job_id}/export')
    def output(job_id:str):
        job=backend.get_job(identifier(job_id))
        if job.state!='succeeded':raise ValueError('Job has no successful output')
        with backend._connection() as c:
            row=c.execute('SELECT artifact_hash FROM run_outputs WHERE run_id=?',(job.runs[-1].run_id,)).fetchone()
        if row is None:raise KeyError(job_id)
        artifact=store.get('export',row[0])
        return {'export_id':row[0],'manifest':artifact.payload}

    @app.get('/api/v1/exports/{export_id}/download')
    def download(export_id:str):
        if re.fullmatch('[0-9a-f]{64}',export_id) is None:raise ValueError('Invalid export ID')
        with backend._connection() as c:
            rows=c.execute('SELECT runs.snapshot FROM run_outputs JOIN runs USING(run_id) WHERE artifact_hash=?',(export_id,)).fetchall()
        from mdac2.domain.artifact import strict_json
        if not any(strict_json(r[0])['state']=='succeeded' for r in rows):raise KeyError(export_id)
        artifact=store.get('export',export_id)
        return FileResponse(store._destination('export',export_id)/'chart.mdm',filename='chart.mdm',media_type='application/zip')
