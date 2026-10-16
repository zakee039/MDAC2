from concurrent.futures import ThreadPoolExecutor
import sqlite3

import pytest

from mdac2.application.jobs_contract import JobError, JobRequest, Resources
from mdac2.infrastructure.job_backend import HueyTransport, JobBackend
from mdac2.infrastructure.project_repository import RepositoryError, RevisionConflict


def setup(tmp_path, **capacity):
    backend = JobBackend(tmp_path / "state.sqlite", **capacity)
    project = backend.create_project("test")
    return backend, project.project_id


def request(project, **overrides):
    values = dict(project_id=project, project_revision=None, operation="test", inputs=[],
        producer={"name": "test", "version": "1", "model_hash": None, "dependencies": {}}, options={})
    values.update(overrides)
    return JobRequest.create(**values)


def test_double_click_and_duplicate_real_huey_delivery(tmp_path):
    backend, project = setup(tmp_path)
    with ThreadPoolExecutor(max_workers=4) as pool:
        jobs = list(pool.map(lambda _: backend.create_job(request(project)), range(4)))
    assert len({job.job_id for job in jobs}) == 1
    claimed = []
    def consume(j, r):
        run = backend.claim(j, r, "worker", now=0)
        if run:
            claimed.append(run)
    transport = HueyTransport(tmp_path / "queue.sqlite", consume)
    def fail():
        raise OSError("crash before ACK")
    try:
        with pytest.raises(OSError):
            backend.dispatch(transport, after_send=fail)
        assert backend.dispatch(transport) == 1
        assert transport.huey.pending_count() == 2
        while task := transport.huey.dequeue():
            transport.huey.execute(task)
        assert len(claimed) == 1
        assert backend.dispatch(transport) == 0
        assert backend.get_project(project).head_revision is None
    finally:
        transport.close()


def test_create_and_outbox_atomic(tmp_path, monkeypatch):
    backend, project = setup(tmp_path)
    def fail(*args):
        raise sqlite3.OperationalError("disk full")
    monkeypatch.setattr(backend, "_outbox", fail)
    with pytest.raises(sqlite3.OperationalError):
        backend.create_job(request(project))
    with backend._connection() as connection:
        for table in ("jobs", "runs", "job_outbox"):
            assert connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0


def test_resource_refusal_cannot_be_erased_by_late_dispatch_ack(tmp_path):
    backend, project = setup(tmp_path)
    first = backend.create_job(request(project))
    one = backend.claim(first.job_id, first.runs[-1].run_id, "one", now=0)
    second = backend.create_job(request(project, options={"other": 1}))
    class ImmediateTransport:
        def send(self, j, r):
            backend.claim(j, r, "two", now=1)
    backend.dispatch(ImmediateTransport())
    with backend._connection() as connection:
        assert connection.execute("SELECT delivered FROM job_outbox WHERE run_id=?", (second.runs[-1].run_id,)).fetchone()[0] == 0
    backend.finish(one.job_id, one.run_id, "succeeded", one.lease_token, now=2, exit_confirmed=True)
    backend.dispatch(ImmediateTransport())
    assert backend.get_job(second.job_id).state == "running"


def test_expired_run_reserves_resources_until_exit_confirmed(tmp_path):
    backend, project = setup(tmp_path)
    first = backend.create_job(request(project))
    one = backend.claim(first.job_id, first.runs[-1].run_id, "one", now=0, lease_seconds=1)
    backend.interrupt(one.job_id, one.run_id, now=2)
    second = backend.create_job(request(project, options={"other": 1}))
    assert backend.claim(second.job_id, second.runs[-1].run_id, "two", now=2) is None
    with pytest.raises(JobError):
        backend.retry(one.job_id, one.run_id)
    backend.confirm_exit(one.job_id, one.run_id)
    assert backend.claim(second.job_id, second.runs[-1].run_id, "two", now=3)


def test_retry_lineage_stale_completion_and_reopen(tmp_path):
    backend, project = setup(tmp_path)
    job = backend.create_job(request(project))
    one = backend.claim(job.job_id, job.runs[-1].run_id, "one", now=0)
    backend.finish(one.job_id, one.run_id, "failed", one.lease_token, now=1, exit_confirmed=True, error_code="TEST")
    two = backend.retry(one.job_id, one.run_id)
    assert two.run_id != one.run_id and two.attempt == 2
    with pytest.raises(JobError):
        backend.finish(one.job_id, one.run_id, "succeeded", one.lease_token, now=2, exit_confirmed=True)
    assert backend.claim(one.job_id, one.run_id, "stale", now=2) is None
    reopened = JobBackend(tmp_path / "state.sqlite")
    assert reopened.get_job(job.job_id).runs[-1] == two


def test_cancel_before_delivery_and_foreign_lease(tmp_path):
    backend, project = setup(tmp_path)
    job = backend.create_job(request(project))
    backend.cancel(job.job_id, job.runs[-1].run_id)
    assert backend.claim(job.job_id, job.runs[-1].run_id, "w", now=0) is None
    other = backend.create_job(request(project))
    run = backend.claim(other.job_id, other.runs[-1].run_id, "w", now=0)
    with pytest.raises(JobError):
        backend.heartbeat(other.job_id, run.run_id, "foreign", now=1, lease_seconds=30, stage="test")


def test_gpu_exclusive_and_unavailable(tmp_path):
    backend, project = setup(tmp_path, cpu_slots=4, gpu_memory_mb={0: 1024})
    with pytest.raises(JobError):
        backend.create_job(request(project, resources=Resources(1, 0, 2048)))
    jobs = [backend.create_job(request(project, resources=Resources(1, 0, 256), options={"n": n})) for n in range(2)]
    assert backend.claim(jobs[0].job_id, jobs[0].runs[-1].run_id, "a", now=0)
    assert backend.claim(jobs[1].job_id, jobs[1].runs[-1].run_id, "b", now=0) is None


def test_stale_project_and_unknown_job_schema(tmp_path):
    backend, project = setup(tmp_path)
    with pytest.raises(RevisionConflict):
        backend.create_job(request(project, project_revision="stale"))
    with backend._connection() as connection:
        connection.execute("DROP TABLE job_schema")
    with pytest.raises(RepositoryError):
        JobBackend(tmp_path / "state.sqlite")
