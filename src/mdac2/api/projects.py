"""Project endpoints with immutable chart snapshots and expected-head CAS."""
from dataclasses import asdict
import sqlite3

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException

from mdac2.api.security import LocalBoundary, error, fields, identifier, json_body
from mdac2.domain.artifact import ArtifactEnvelope, canonical_json
from mdac2.domain.chart import ChartRevision
from mdac2.infrastructure.project_repository import RevisionConflict


def create_app(repository, store, *, token, hosts=("127.0.0.1:8000", "localhost:8000", "[::1]:8000"), public_static=False):
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    app.add_middleware(LocalBoundary, token=token, hosts=hosts, public_static=public_static)
    app.state.repository = repository
    app.state.store = store

    @app.exception_handler(RevisionConflict)
    async def conflict(request, exc):
        return error(409, "REVISION_CONFLICT", "Project changed; reload before editing")

    @app.exception_handler(KeyError)
    async def missing(request, exc):
        return error(404, "NOT_FOUND", "Resource not found")

    @app.exception_handler(ValueError)
    @app.exception_handler(RequestValidationError)
    async def invalid(request, exc):
        return error(422, "INVALID_CONTRACT", "Invalid request contract")

    @app.exception_handler(HTTPException)
    async def http_error(request, exc):
        return error(exc.status_code, "HTTP_ERROR", "Request cannot be served")

    @app.exception_handler(sqlite3.Error)
    @app.exception_handler(OSError)
    @app.exception_handler(Exception)
    async def internal(request, exc):
        return error(500, "INTERNAL_ERROR", "Operation failed", retryable=True)

    @app.post("/api/v1/projects", status_code=201)
    async def create(request: Request):
        body = await json_body(request)
        fields(body, {"title"})
        if type(body["title"]) is not str or not body["title"].strip() or len(body["title"]) > 256:
            raise ValueError("Invalid title")
        return asdict(await run_in_threadpool(repository.create_project, body["title"]))

    @app.get("/api/v1/projects/{project_id}")
    def get(project_id: str):
        project = repository.get_project(identifier(project_id))
        chart = None
        if project.head_revision is not None:
            revision = repository.get_revision(project_id, project.head_revision)
            chart = store.get("chart", revision.artifact_hash).payload
        return {**asdict(project), "chart": chart, "capabilities": {"game_export": False, "automatic_analysis": False}}

    @app.post("/api/v1/projects/{project_id}/commands")
    async def command(project_id: str, request: Request):
        identifier(project_id)
        body = await json_body(request)
        fields(body, {"command_id", "expected_revision", "type", "payload"})
        identifier(body["command_id"])
        if body["expected_revision"] is not None:
            identifier(body["expected_revision"])
        if body["type"] != "replace_chart":
            raise ValueError("Unsupported command")
        chart = ChartRevision.from_json(canonical_json(body["payload"]))
        identifier(chart.revision_id)
        envelope = ArtifactEnvelope.create(artifact_type="chart", producer={"name": "manual", "version": "1",
            "model_hash": None, "dependencies": {}}, options={}, payload=chart.to_dict())
        def save():
            workflow = getattr(app.state, "manual_workflow", None)
            if workflow is not None:
                current = workflow.load(project_id)
                if current is None:
                    raise ValueError("Initialize authoritative timing through manual workflow")
                workflow.save(project_id, expected_revision=body["expected_revision"], command_id=body["command_id"], chart=chart, timing=current[1])
                return repository.get_revision(project_id, chart.revision_id)
            # A losing CAS may leave a valid orphan, never a dangling revision.
            repository.get_project(project_id)
            store.publish(envelope)
            return repository.commit_revision(project_id, expected_revision=body["expected_revision"], artifact=envelope,
                command_id=body["command_id"], command_type=body["type"], command_payload={"chart_hash": envelope.artifact_hash})
        revision = await run_in_threadpool(save)
        return {"revision_id": revision.revision_id, "artifact_hash": revision.artifact_hash,
                "invalidated": ["validation", "preview", "export"]}

    return app
