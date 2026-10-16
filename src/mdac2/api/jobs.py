"""Persistent job status and resumable, bounded SSE sessions."""
import asyncio
from dataclasses import dataclass
import time

from fastapi import Request
from starlette.concurrency import run_in_threadpool
from starlette.responses import StreamingResponse

from mdac2.api.security import error, fields, identifier, json_body
from mdac2.application.jobs_contract import JobRequest, Resources, public_run
from mdac2.domain.artifact import canonical_json


@dataclass(frozen=True)
class JobCapability:
    producer: dict
    resources: Resources = Resources()


def attach_jobs(app, backend, *, capabilities=None, stream_seconds=15):
    capabilities = dict(capabilities or {})
    if not 0 < stream_seconds <= 60:
        raise ValueError("Bounded SSE session required")

    @app.post("/api/v1/projects/{project_id}/jobs", status_code=202)
    async def create(project_id: str, request: Request):
        identifier(project_id)
        body = await json_body(request)
        fields(body, {"type", "expected_revision", "inputs", "options"})
        if type(body["type"]) is not str:
            raise ValueError("Invalid job type")
        capability = capabilities.get(body["type"])
        if capability is None:
            return error(409, "CAPABILITY_MISSING", "Requested worker capability is not available")
        job_request = JobRequest.create(project_id=project_id, project_revision=body["expected_revision"],
            operation=body["type"], inputs=body["inputs"], options=body["options"],
            producer=capability.producer, resources=capability.resources)
        job = await run_in_threadpool(backend.create_job, job_request)
        return {"job_id": job.job_id, "request_key": job.request.request_key, **public_run(job.runs[-1])}

    @app.get("/api/v1/jobs/{job_id}")
    def get(job_id: str):
        job = backend.get_job(identifier(job_id))
        return public_run(job.runs[-1])

    @app.post("/api/v1/jobs/{job_id}/cancel")
    async def cancel(job_id: str, request: Request):
        body = await json_body(request)
        fields(body, {"run_id"})
        run = await run_in_threadpool(backend.cancel, identifier(job_id), identifier(body["run_id"]))
        return public_run(run)

    @app.get("/api/v1/jobs/{job_id}/events")
    async def events(job_id: str, request: Request):
        identifier(job_id)
        cursor = request.headers.get("last-event-id", "0")
        if not cursor.isascii() or not cursor.isdigit() or len(cursor) > 18:
            raise ValueError("Invalid event cursor")
        after = int(cursor)
        initial = await run_in_threadpool(backend.event_batch, job_id, after)
        async def stream():
            nonlocal after
            batch, terminal = initial
            deadline = time.monotonic() + stream_seconds
            while True:
                for event_id, snapshot in batch:
                    yield f"id: {event_id}\nevent: state\ndata: {canonical_json(snapshot).decode()}\n\n"
                    after = event_id
                if terminal or time.monotonic() >= deadline or await request.is_disconnected():
                    return
                if not batch:
                    yield ": keepalive\n\n"
                    await asyncio.sleep(min(.1, stream_seconds))
                batch, terminal = await run_in_threadpool(backend.event_batch, job_id, after)
        return StreamingResponse(stream(), media_type="text/event-stream", headers={"X-Accel-Buffering": "no"})
