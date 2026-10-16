from fastapi.testclient import TestClient
import pytest

from mdac2.api.projects import create_app
from mdac2.domain.chart import ChartRevision
from mdac2.infrastructure.artifact_store import ArtifactStore
from mdac2.infrastructure.project_repository import ProjectRepository


TOKEN = "test-token-" + "a" * 32


def client(tmp_path, **kwargs):
    app = create_app(ProjectRepository(tmp_path / "state.sqlite"), ArtifactStore(tmp_path / "runtime"), token=TOKEN)
    return TestClient(app, base_url="http://127.0.0.1:8000", client=("127.0.0.1", 1234),
        headers={"Authorization": "Bearer " + TOKEN}, **kwargs)


def test_create_save_read_and_conflict(tmp_path):
    with client(tmp_path) as api:
        response = api.post("/api/v1/projects", json={"title": "测试"})
        assert response.status_code == 201
        project = response.json()["project_id"]
        command = {"command_id": "cmd1", "expected_revision": None, "type": "replace_chart",
                   "payload": ChartRevision("r1", "timing", ()).to_dict()}
        saved = api.post(f"/api/v1/projects/{project}/commands", json=command)
        assert saved.status_code == 200
        read = api.get(f"/api/v1/projects/{project}")
        assert read.json()["chart"]["revision_id"] == "r1"
        assert read.headers["cache-control"] == "no-store"
        assert api.post(f"/api/v1/projects/{project}/commands", json=command).status_code == 409


@pytest.mark.parametrize("headers,status", [({"Authorization": ""}, 401), ({"Authorization": "Bearer wrong"}, 401),
    ({"Host": "evil.test:8000"}, 403), ({"Origin": "https://evil.test"}, 403), ({"Origin": "null"}, 403),
    ({"Origin": "http://localhost:9999"}, 403), ({"Host": "127.0.0.1.evil:8000"}, 403)])
def test_security_boundary(tmp_path, headers, status):
    with client(tmp_path) as api:
        response = api.post("/api/v1/projects", json={"title": "p"}, headers=headers)
        assert response.status_code == status
        assert TOKEN not in response.text


def test_remote_client_even_with_valid_token_denied(tmp_path):
    app = create_app(ProjectRepository(tmp_path / "state.sqlite"), ArtifactStore(tmp_path / "runtime"), token=TOKEN)
    with TestClient(app, base_url="http://127.0.0.1:8000", client=("192.168.1.2", 1234)) as api:
        assert api.get("/api/v1/projects/p", headers={"Authorization": "Bearer " + TOKEN}).status_code == 403


@pytest.mark.parametrize("body", [b'{"title":"a","title":"b"}', b'{"title":"p","path":"C:/secret"}',
    b'{"title":NaN}', b'{"title":1}', b'{"title":""}'])
def test_strict_body(tmp_path, body):
    with client(tmp_path) as api:
        assert api.post("/api/v1/projects", content=body, headers={"Content-Type": "application/json"}).status_code == 422


def test_traversal_and_error_redaction(tmp_path, monkeypatch):
    with client(tmp_path, raise_server_exceptions=False) as api:
        for path in ("..%5Csecret", "C:%5Csecret", "%252e%252e", "unknown"):
            assert api.get("/api/v1/projects/" + path).status_code in (404, 422)
        def fail(*args):
            raise OSError("C:/secret password=private")
        monkeypatch.setattr(api.app.state.repository, "create_project", fail)
        response = api.post("/api/v1/projects", json={"title": "p"})
        assert response.status_code == 500
        assert "secret" not in response.text and "private" not in response.text
