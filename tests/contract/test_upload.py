import io
from pathlib import Path
import wave

from fastapi.testclient import TestClient
import pytest

from mdac2.adapters.audio_io import AudioDecoder, AudioLimits
from mdac2.api.assets import attach_assets
from mdac2.api.projects import create_app
from mdac2.infrastructure.artifact_store import ArtifactStore
from mdac2.infrastructure.project_repository import ProjectRepository


def setup(tmp_path, limits=AudioLimits()):
    repository = ProjectRepository(tmp_path / "state.sqlite")
    store = ArtifactStore(tmp_path / "runtime")
    binaries = Path(__file__).parents[2] / "tools/.deps/ffmpeg/extracted/ffmpeg-9.0.1-essentials_build/bin"
    decoder = AudioDecoder(binaries / "ffmpeg.exe", binaries / "ffprobe.exe", limits=limits)
    app = create_app(repository, store, token="t" * 40)
    attach_assets(app, repository, store, decoder)
    client = TestClient(app, base_url="http://127.0.0.1:8000", client=("127.0.0.1", 123), headers={"Authorization": "Bearer " + "t" * 40})
    project = repository.create_project("p").project_id
    return repository, store, client, project


def audio(frames=480):
    stream = io.BytesIO()
    with wave.open(stream, "wb") as writer:
        writer.setparams((2, 2, 48000, 0, "NONE", "none"))
        writer.writeframes(bytes(frames * 4))
    return stream.getvalue()


def test_stream_upload_and_range(tmp_path):
    repository, store, api, project = setup(tmp_path)
    data = audio()
    with api:
        response = api.post(f"/api/v1/projects/{project}/audio", content=iter([data[:32], data[32:]]), headers={"Content-Type": "audio/wav"})
        assert response.status_code == 201
        result = response.json()
        assert result["canonical"]["sample_count"] == 480
        whole = api.get(result["audio_url"])
        assert whole.status_code == 200 and whole.content[:4] == b"RIFF"
        part = api.get(result["audio_url"], headers={"Range": "bytes=0-15"})
        assert part.status_code == 206 and part.content == whole.content[:16]
        assert part.headers["content-range"].startswith("bytes 0-15/")
        assert api.get(result["audio_url"], headers={"Range": "bytes=-12"}).content == whole.content[-12:]
        assert api.get(result["audio_url"], headers={"Range": "bytes=999999-"}).status_code == 416
        assert api.get(result["audio_url"], headers={"Authorization": ""}).status_code == 401
        assert api.get(f"/api/v1/projects/{project}/audio").json()["asset_id"] == result["asset_id"]
        assert api.post(f"/api/v1/projects/{project}/audio", content=data, headers={"Content-Type": "audio/wav"}).status_code == 409


@pytest.mark.parametrize("body", [b"bad audio", b""])
def test_bad_upload_not_registered_then_retry(tmp_path, body):
    repository, store, api, project = setup(tmp_path)
    with api:
        url = f"/api/v1/projects/{project}/audio"
        assert api.post(url, content=body, headers={"Content-Type": "audio/wav"}).status_code == 422
        assert api.get(url).status_code == 404
        assert api.post(url, content=audio(), headers={"Content-Type": "audio/wav"}).status_code == 201


def test_stream_bytes_and_decoded_duration_limit(tmp_path):
    _, _, api, project = setup(tmp_path, AudioLimits(max_bytes=100))
    with api:
        response = api.post(f"/api/v1/projects/{project}/audio", content=iter([b"x" * 60, b"y" * 60]), headers={"Content-Type": "audio/wav"})
        assert response.status_code == 413
    other = tmp_path / "other"
    _, _, api, project = setup(other, AudioLimits(max_seconds=1))
    with api:
        assert api.post(f"/api/v1/projects/{project}/audio", content=audio(96000), headers={"Content-Type": "audio/wav"}).status_code == 422


def test_unregistered_artifact_and_path_not_served(tmp_path):
    _, _, api, project = setup(tmp_path)
    with api:
        assert api.get("/api/v1/assets/" + "a" * 64 + "/audio").status_code == 404
        assert api.get("/api/v1/assets/C:%5Cprivate/audio").status_code == 422
