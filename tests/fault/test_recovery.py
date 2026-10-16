import sqlite3
import sys
import time

import pytest

from mdac2.application.jobs_contract import JobError, JobRequest
from mdac2.application.reconciliation import Reconciler
from mdac2.domain.artifact import ArtifactEnvelope
from mdac2.domain.chart import ChartRevision
from mdac2.infrastructure.artifact_store import ArtifactStore
from mdac2.infrastructure.job_backend import JobBackend
from mdac2.infrastructure.process_runner import ProcessRunner


PRODUCER = {"name": "test", "version": "1", "model_hash": None, "dependencies": {}}


def setup(tmp_path):
    backend = JobBackend(tmp_path / "state.sqlite")
    project = backend.create_project("test")
    store = ArtifactStore(tmp_path / "runtime")
    reconciler = Reconciler(backend, store)
    request = JobRequest.create(project_id=project.project_id, project_revision=None, operation="chart",
        inputs=[], producer=PRODUCER, options={})
    job = backend.create_job(request)
    run = backend.claim(job.job_id, job.runs[-1].run_id, "test", now=time.time())
    envelope = ArtifactEnvelope.create(artifact_type="chart", producer=PRODUCER, options={},
        payload=ChartRevision("generated", "timing", ()).to_dict())
    return backend, project, store, reconciler, run, envelope


def execute(backend, reconciler, run, envelope):
    return ProcessRunner(backend).execute(run.job_id, run.run_id, [sys.executable, "-c", "pass"], timeout=3,
        prepare_success=lambda: reconciler.prepare(envelope))


def test_publish_register_and_success_share_transaction(tmp_path):
    backend, project, store, reconciler, run, envelope = setup(tmp_path)
    assert execute(backend, reconciler, run, envelope).state == "succeeded"
    assert backend.get_project(project.project_id).head_revision == "generated"
    assert store.get("chart", envelope.artifact_hash) == envelope
    with backend._connection() as connection:
        assert connection.execute("SELECT head_advanced FROM run_outputs").fetchone()[0] == 1


@pytest.mark.parametrize("after_head", [False, True])
def test_db_failure_keeps_orphan_not_success_or_head(tmp_path, monkeypatch, after_head):
    backend, project, store, reconciler, run, envelope = setup(tmp_path)
    original = backend._advance_head
    def fail(*args):
        if after_head:
            original(*args)
        raise sqlite3.OperationalError("injected DB failure")
    monkeypatch.setattr(backend, "_advance_head", fail)
    with pytest.raises(sqlite3.OperationalError):
        execute(backend, reconciler, run, envelope)
    assert backend.get_project(project.project_id).head_revision is None
    assert backend.get_job(run.job_id).state == "interrupted"
    assert store.get("chart", envelope.artifact_hash) == envelope
    with backend._connection() as connection:
        for table in ("artifacts", "run_outputs", "revisions"):
            assert connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0
    report = reconciler.recover_artifacts()
    assert report.registered == (envelope.artifact_hash,)
    assert backend.get_project(project.project_id).head_revision is None
    assert backend.get_job(run.job_id).state == "interrupted"
    assert reconciler.recover_artifacts() == report


def test_old_project_output_only_enters_history(tmp_path):
    backend, project, store, reconciler, run, envelope = setup(tmp_path)
    manual = ArtifactEnvelope.create(artifact_type="chart", producer=PRODUCER, options={},
        payload=ChartRevision("manual", "timing", ()).to_dict())
    backend.commit_revision(project.project_id, expected_revision=None, artifact=manual,
        command_id="edit", command_type="manual", command_payload={})
    assert execute(backend, reconciler, run, envelope).state == "succeeded"
    assert backend.get_project(project.project_id).head_revision == "manual"
    with backend._connection() as connection:
        assert connection.execute("SELECT head_advanced FROM run_outputs").fetchone()[0] == 0


def test_bad_request_output_not_registered(tmp_path):
    backend, project, store, reconciler, run, _ = setup(tmp_path)
    envelope = ArtifactEnvelope.create(artifact_type="chart", producer=PRODUCER, options={"wrong": True},
        payload=ChartRevision("wrong", "timing", ()).to_dict())
    with pytest.raises(JobError):
        execute(backend, reconciler, run, envelope)
    assert backend.get_job(run.job_id).state == "interrupted"
    assert backend.get_project(project.project_id).head_revision is None


def test_staging_corrupt_and_missing_dependency_never_head(tmp_path):
    backend, project, store, reconciler, run, envelope = setup(tmp_path)
    stage = store.staging / "unfinished"
    stage.mkdir()
    (stage / "manifest.json").write_bytes(envelope.to_json())
    broken = ArtifactEnvelope.create(artifact_type="test", producer=PRODUCER, options={}, payload={})
    target = store.publish(broken)
    (target / "manifest.json").write_bytes(b"invalid")
    missing = ArtifactEnvelope.create(artifact_type="test", producer=PRODUCER, options={}, inputs=["a" * 64], payload={})
    store.publish(missing)
    report = reconciler.recover_artifacts()
    assert report.staging_count == 1 and len(report.unresolved) == 2
    assert report.registered == () and stage.exists()
    assert backend.get_project(project.project_id).head_revision is None


def test_dependency_order_and_queued_repair(tmp_path):
    backend, project, store, reconciler, run, envelope = setup(tmp_path)
    parent = ArtifactEnvelope.create(artifact_type="z", producer=PRODUCER, options={}, payload={})
    child = ArtifactEnvelope.create(artifact_type="a", producer=PRODUCER, options={}, inputs=[parent.artifact_hash], payload={})
    store.publish(child)
    store.publish(parent)
    assert len(reconciler.recover_artifacts().registered) == 2
    backend.interrupt(run.job_id, run.run_id, now=time.time(), exit_confirmed=True)
    new = backend.retry(run.job_id, run.run_id)
    class LostDelivery:
        def send(self, *args):
            pass
    backend.dispatch(LostDelivery())
    assert reconciler.recover_queued_delivery() == 1
    with backend._connection() as connection:
        assert connection.execute("SELECT delivered FROM job_outbox WHERE run_id=?", (new.run_id,)).fetchone()[0] == 0
