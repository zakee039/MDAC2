from dataclasses import replace

import pytest

from mdac2.application.jobs_contract import Job, JobError, JobRequest, Resources, Run
from mdac2.domain.artifact import canonical_json


def request(**overrides):
    values = dict(project_id="p", project_revision="r", operation="timing", inputs=[],
                  producer={"name": "manual", "version": "1", "model_hash": None, "dependencies": {}}, options={})
    values.update(overrides)
    return JobRequest.create(**values)


def running():
    return Run("j", "r").start("worker", now=100, lease_seconds=10)


def test_identity_and_snapshot():
    options = {"a": [1]}
    first = request(options=options)
    options["a"].append(2)
    assert first.to_dict()["options"] == {"a": [1]}
    assert first.request_key == request(options={"a": [1]}).request_key
    for change in ({"project_revision": "other"}, {"project_id": "other"}, {"resources": Resources(2)}, {"inputs": ["a" * 64]}):
        assert first.request_key != request(options={"a": [1]}, **change).request_key
    assert Job.create(first).job_id != Job.create(first).job_id


@pytest.mark.parametrize("kwargs", [{"cpu_slots": True}, {"cpu_slots": 0}, {"gpu_device": 0},
    {"gpu_memory_mb": 1}, {"gpu_device": -1}, {"gpu_device": 0, "gpu_memory_mb": -1}])
def test_resources_reject_invalid(kwargs):
    with pytest.raises(JobError):
        Resources(**kwargs)


def test_request_unknown_schema_and_fields():
    for key, value in (("schema_version", True), ("schema_version", 99), ("extra", 1)):
        data = request().to_dict()
        data[key] = value
        with pytest.raises(JobError):
            JobRequest(canonical_json(data))


def test_lease_progress_finish():
    run = running()
    renewed = run.heartbeat(run.lease_token, now=105, lease_seconds=20, stage="decode", completed_units=2, total_units=4)
    assert renewed.lease_until == 125
    assert run.stage == "starting"
    success = renewed.finish("succeeded", run.lease_token, now=106, exit_confirmed=True)
    assert success.state == "succeeded"
    with pytest.raises(JobError):
        success.retry()


@pytest.mark.parametrize("token,now", [("foreign", 101), (None, 101), ("valid", 110), ("valid", float("nan"))])
def test_expired_and_foreign_lease(token, now):
    run = running()
    with pytest.raises(JobError):
        run.finish("succeeded", run.lease_token if token == "valid" else token, now=now, exit_confirmed=True)


def test_cancel_requires_exit_and_blocks_success():
    queued = Run("j", "r").cancel()
    assert queued.state == "cancelled" and queued.exit_confirmed
    run = running().cancel()
    for state, exited in (("succeeded", True), ("cancelled", False)):
        with pytest.raises(JobError):
            run.finish(state, run.lease_token, now=101, exit_confirmed=exited)
    assert run.finish("cancelled", run.lease_token, now=101, exit_confirmed=True).state == "cancelled"


def test_interrupted_retry_needs_confirmed_exit_and_new_run():
    run = running()
    with pytest.raises(JobError):
        run.interrupt(now=101)
    interrupted = run.interrupt(now=110)
    with pytest.raises(JobError):
        interrupted.retry()
    exited = interrupted.confirm_exit()
    retry = exited.retry()
    assert retry.run_id != run.run_id and retry.attempt == 2 and retry.state == "queued"
    assert Job("j", request(), (exited, retry)).state == "queued"
    with pytest.raises(JobError):
        Job("j", request(), (run, retry))


@pytest.mark.parametrize("change", [{"state": "bogus"}, {"attempt": True}, {"completed_units": -1},
    {"completed_units": 3, "total_units": 2}, {"exit_confirmed": True}, {"lease_until": float("inf")}])
def test_invalid_run_snapshots(change):
    with pytest.raises(JobError):
        replace(running(), **change)


def test_illegal_start_finish_retry():
    run = running()
    with pytest.raises(JobError):
        run.start("other", now=101, lease_seconds=5)
    with pytest.raises(JobError):
        run.finish("cancelled", run.lease_token, now=101, exit_confirmed=True)
    with pytest.raises(JobError):
        run.retry()
