import json
import time

from fastapi.testclient import TestClient
import pytest

from mdac2.api.jobs import JobCapability, attach_jobs
from mdac2.api.projects import create_app
from mdac2.infrastructure.artifact_store import ArtifactStore
from mdac2.infrastructure.job_backend import JobBackend


def setup(tmp_path):
    backend = JobBackend(tmp_path / "state.sqlite")
    token = "t" * 40
    app = create_app(backend, ArtifactStore(tmp_path / "runtime"), token=token)
    attach_jobs(app, backend, capabilities={"test": JobCapability({"name": "test", "version": "1", "model_hash": None, "dependencies": {}})}, stream_seconds=.02)
    client = TestClient(app, base_url="http://127.0.0.1:8000", client=("127.0.0.1", 123), headers={"Authorization": "Bearer " + token})
    project = client.post("/api/v1/projects", json={"title": "p"}).json()["project_id"]
    return backend, client, project


def create(api, project, **overrides):
    payload = {"type": "test", "expected_revision": None, "inputs": [], "options": {}}
    payload.update(overrides)
    return api.post(f"/api/v1/projects/{project}/jobs", json=payload)


def parse(response):
    ids = [int(line[4:]) for line in response.text.splitlines() if line.startswith("id: ")]
    data = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]
    return ids, data


def test_202_idempotency_status_cancel(tmp_path):
    backend, api, project = setup(tmp_path)
    with api:
        response = create(api, project)
        assert response.status_code == 202
        job = response.json()
        assert create(api, project).json()["job_id"] == job["job_id"]
        assert api.get(f"/api/v1/jobs/{job['job_id']}").json()["state"] == "queued"
        assert api.post(f"/api/v1/jobs/{job['job_id']}/cancel", json={"run_id": job["run_id"]}).json()["state"] == "cancelled"
        assert "lease" not in api.get(f"/api/v1/jobs/{job['job_id']}").text


def test_sse_reconnect_persistent_no_missing_state(tmp_path):
    backend, api, project = setup(tmp_path)
    with api:
        job = create(api, project).json()
        url = f"/api/v1/jobs/{job['job_id']}/events"
        first_ids, first = parse(api.get(url))
        assert [item["state"] for item in first] == ["queued"]
        run = backend.claim(job["job_id"], job["run_id"], "worker", now=time.time())
        backend.finish(run.job_id, run.run_id, "failed", run.lease_token, now=time.time(), exit_confirmed=True, error_code="C:/secret token=password")
        ids, data = parse(api.get(url, headers={"Last-Event-ID": str(first_ids[-1])}))
        assert [item["state"] for item in data] == ["running", "failed"]
        assert ids[0] > first_ids[-1]
        assert data[-1]["error_code"] == "WORKER_ERROR"
        reopened = JobBackend(tmp_path / "state.sqlite")
        assert len(reopened.event_batch(job["job_id"])[0]) == 3
        assert parse(api.get(url, headers={"Last-Event-ID": str(ids[-1])}))[0] == []


@pytest.mark.parametrize("cursor", ["-1", "1.0", "999999999", "../secret"])
def test_invalid_cursor(tmp_path, cursor):
    _, api, project = setup(tmp_path)
    with api:
        job = create(api, project).json()
        assert api.get(f"/api/v1/jobs/{job['job_id']}/events", headers={"Last-Event-ID": cursor}).status_code == 422


def test_missing_capability_and_stale_revision(tmp_path):
    _, api, project = setup(tmp_path)
    with api:
        assert create(api, project, type="missing").json()["code"] == "CAPABILITY_MISSING"
        assert create(api, project, expected_revision="old").status_code == 409


def test_event_rollback_with_state(tmp_path, monkeypatch):
    backend, api, project = setup(tmp_path)
    with api:
        job = create(api, project).json()
        original = backend._event
        def fail(*args):
            original(*args)
            raise OSError("failed transaction")
        monkeypatch.setattr(backend, "_event", fail)
        with pytest.raises(OSError):
            backend.claim(job["job_id"], job["run_id"], "w", now=time.time())
        assert backend.get_job(job["job_id"]).state == "queued"
        assert len(backend.event_batch(job["job_id"])[0]) == 1
