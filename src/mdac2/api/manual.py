from dataclasses import replace
from uuid import uuid4
from fastapi import Request
from starlette.concurrency import run_in_threadpool
from mdac2.api.security import identifier,fields,json_body
from mdac2.domain.artifact import canonical_json
from mdac2.domain.chart import ChartRevision
from mdac2.application.manual_workflow import timing_data,timing_from


def attach_manual(app,workflow):
    app.state.manual_workflow=workflow
    @app.get('/api/v1/projects/{project_id}/manual')
    def get(project_id:str):
        result=workflow.load(identifier(project_id))
        return {"chart":result[0].to_dict(),"timing":timing_data(result[1])} if result else {"chart":None,"timing":None}

    @app.post('/api/v1/projects/{project_id}/manual/preview')
    async def preview(project_id:str,request:Request):
        body=await json_body(request);fields(body,{"expected_revision","timing","position_policy"})
        new=timing_from(body["timing"])
        preview=await run_in_threadpool(workflow.preview,identifier(project_id),body["expected_revision"],new,body["position_policy"])
        old,_=workflow.load(project_id,body["expected_revision"])
        chart=replace(old,revision_id=str(uuid4()),timing_revision=new.revision_id,events=preview.proposed_events)
        return {"fingerprint":preview.fingerprint,"diffs":[d.to_dict() for d in preview.diffs],
            "conflicts":[{"code":c.code,"event_id":c.event_id} for c in preview.conflicts],"chart":chart.to_dict()}

    @app.post('/api/v1/projects/{project_id}/manual')
    async def save(project_id:str,request:Request):
        body=await json_body(request);fields(body,{"expected_revision","command_id","chart","timing","position_policy","preview_fingerprint"})
        chart=ChartRevision.from_json(canonical_json(body["chart"]));timing=timing_from(body["timing"])
        identifier(chart.revision_id);identifier(timing.revision_id);identifier(body["command_id"])
        chart,timing=await run_in_threadpool(workflow.save,identifier(project_id),expected_revision=body["expected_revision"],
            command_id=body["command_id"],chart=chart,timing=timing,position_policy=body["position_policy"],preview_fingerprint=body["preview_fingerprint"])
        return {"chart":chart.to_dict(),"timing":timing_data(timing),"invalidated":["preview","validation","export"]}
