from concurrent.futures import ThreadPoolExecutor
import errno
import hashlib
import os

import pytest

from mdac2.domain.artifact import ArtifactEnvelope, ArtifactError
from mdac2.infrastructure.artifact_store import ArtifactStore, StoreError


def envelope(data=b"audio"):
    return ArtifactEnvelope.create(artifact_type="test", producer={"name": "fixture", "version": "1",
        "model_hash": None, "dependencies": {}}, options={}, payload={"files": [
        {"name": "audio.bin", "sha256": hashlib.sha256(data).hexdigest(), "byte_count": len(data)}]})


def test_publish_and_reuse_deep_validation(tmp_path):
    store, item = ArtifactStore(tmp_path), envelope()
    path = store.publish(item, {"audio.bin": b"audio"})
    assert store.get("test", item.artifact_hash) == item
    assert store.publish(item, {"audio.bin": b"audio"}) == path
    (path / "audio.bin").write_bytes(b"wrong")  # same-size corruption
    with pytest.raises(StoreError):
        store.publish(item, {"audio.bin": b"audio"})
    assert (path / "audio.bin").read_bytes() == b"wrong"


@pytest.mark.parametrize("phase", ["write", "flush", "rename"])
def test_io_failure_never_exposes_partial_artifact(tmp_path, monkeypatch, phase):
    store, item = ArtifactStore(tmp_path), envelope()
    def fail(*args):
        raise OSError(errno.ENOSPC, "injected disk full")
    if phase == "write":
        monkeypatch.setattr(store, "_write", fail)
    else:
        monkeypatch.setattr(os, "fsync" if phase == "flush" else "rename", fail)
    with pytest.raises(OSError):
        store.publish(item, {"audio.bin": b"audio"})
    assert not (store.artifacts / "test" / item.artifact_hash).exists()
    with pytest.raises(FileNotFoundError):
        store.get("test", item.artifact_hash)
    assert list(store.staging.iterdir())  # diagnostics retained, not successful output


def test_stage_corruption_refused_and_successful_previous_preserved(tmp_path, monkeypatch):
    store, old = ArtifactStore(tmp_path), envelope()
    original = store.publish(old, {"audio.bin": b"audio"})
    write = store._write
    def corrupt(path, data):
        write(path, b"wrong" if path.name == "audio.bin" else data)
    monkeypatch.setattr(store, "_write", corrupt)
    newer = envelope(b"newer")
    with pytest.raises(StoreError):
        store.publish(newer, {"audio.bin": b"newer"})
    assert store.get("test", old.artifact_hash) == old
    assert (original / "audio.bin").read_bytes() == b"audio"


def test_concurrent_publish_one_immutable_result(tmp_path):
    store, item = ArtifactStore(tmp_path), envelope()
    with ThreadPoolExecutor(max_workers=4) as pool:
        paths = list(pool.map(lambda _: store.publish(item, {"audio.bin": b"audio"}), range(8)))
    assert len(set(paths)) == 1
    assert store.get("test", item.artifact_hash) == item


@pytest.mark.parametrize("name", ["../bad", "C:/bad", "con", "test.", "AUDIO", "x:y"])
def test_unsafe_components(tmp_path, name):
    with pytest.raises(StoreError):
        ArtifactStore(tmp_path).get(name, "a" * 64)


def test_invalid_input_and_incomplete_existing_id(tmp_path):
    store, item = ArtifactStore(tmp_path), envelope()
    with pytest.raises(StoreError):
        store.publish(item, {"audio.bin": b"wrong"})
    destination = store.artifacts / "test" / item.artifact_hash
    destination.mkdir(parents=True)
    with pytest.raises(FileNotFoundError):
        store.publish(item, {"audio.bin": b"audio"})
    assert list(destination.iterdir()) == []


def test_manifest_tamper_and_extra_file(tmp_path):
    store, item = ArtifactStore(tmp_path), envelope()
    path = store.publish(item, {"audio.bin": b"audio"})
    (path / "extra").write_bytes(b"x")
    with pytest.raises(StoreError):
        store.get("test", item.artifact_hash)
    (path / "manifest.json").write_bytes(b"{}")
    with pytest.raises(ArtifactError):
        store.get("test", item.artifact_hash)
