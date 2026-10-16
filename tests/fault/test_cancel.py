from concurrent.futures import ThreadPoolExecutor
import ctypes as c
from ctypes import wintypes as w
import json
from pathlib import Path
import subprocess
import sys
import time

import pytest

from mdac2.application.jobs_contract import JobError, JobRequest
from mdac2.infrastructure.job_backend import JobBackend
from mdac2.infrastructure.process_runner import ProcessRunner, tree_name
from mdac2.infrastructure.windows_process import WindowsTree, bind, close, recover_tree, wait


def test_job_accounting_exit_lag(monkeypatch):
    tree = WindowsTree.__new__(WindowsTree)
    counts = iter([1, 1, 0])
    monkeypatch.setattr(tree, "active", lambda: next(counts))
    assert tree.wait_empty()
    monkeypatch.setattr(tree, "active", lambda: 1)
    assert not tree.wait_empty(.01)


def setup(tmp_path, *, lease=5):
    backend = JobBackend(tmp_path / "state.sqlite")
    project = backend.create_project("test")
    request = JobRequest.create(project_id=project.project_id, project_revision=None, operation="test", inputs=[],
        producer={"name": "test", "version": "1", "model_hash": None, "dependencies": {}}, options={})
    job = backend.create_job(request)
    run = backend.claim(job.job_id, job.runs[-1].run_id, "test", now=time.time(), lease_seconds=lease)
    return backend, run, ProcessRunner(backend, lease_seconds=lease)


def wait_for(predicate):
    deadline = time.monotonic() + 8
    while not predicate():
        assert time.monotonic() < deadline, "deadline exceeded"
        time.sleep(.02)


def dead(pid):
    open_process = bind("OpenProcess", w.HANDLE, w.DWORD, w.BOOL, w.DWORD)
    handle = open_process(0x100000, False, pid)
    if not handle:
        assert c.get_last_error() == 87  # PID no longer exists, not access denied
        return True
    try:
        return wait(handle, 0) == 0
    finally:
        close(handle)


@pytest.mark.parametrize("status", [0, 7])
def test_real_process_exit(tmp_path, status):
    backend, run, runner = setup(tmp_path)
    result = runner.execute(run.job_id, run.run_id, [sys.executable, "-c", f"raise SystemExit({status})"], timeout=3)
    assert result.state == ("succeeded" if status == 0 else "failed")
    assert result.exit_code == status
    assert backend.get_job(run.job_id).runs[-1].exit_confirmed


@pytest.mark.parametrize("cancel", [False, True])
def test_cancel_timeout_kills_descendants(tmp_path, cancel):
    backend, run, runner = setup(tmp_path)
    marker = tmp_path / "pids.json"
    argv = [sys.executable, str(Path(__file__).with_name("process_fixture.py")), str(marker)]
    with ThreadPoolExecutor(max_workers=1) as pool:
        result = pool.submit(runner.execute, run.job_id, run.run_id, argv, timeout=1 if not cancel else 5)
        wait_for(marker.exists)
        pids = json.loads(marker.read_text())
        if cancel:
            backend.cancel(run.job_id, run.run_id)
        finished = result.result(timeout=8)
    assert finished.state == ("cancelled" if cancel else "failed")
    assert finished.error_code == (None if cancel else "TIMEOUT")
    wait_for(lambda: all(dead(pid) for pid in pids))
    assert backend.get_job(run.job_id).runs[-1].exit_confirmed


def test_duplicate_supervisor_does_not_interrupt_owner(tmp_path):
    backend, run, runner = setup(tmp_path)
    marker = tmp_path / "pids.json"
    argv = [sys.executable, str(Path(__file__).with_name("process_fixture.py")), str(marker)]
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(runner.execute, run.job_id, run.run_id, argv, timeout=5)
        wait_for(marker.exists)
        with pytest.raises(JobError):
            runner.execute(run.job_id, run.run_id, argv, timeout=1)
        assert backend.get_job(run.job_id).state == "running"
        backend.cancel(run.job_id, run.run_id)
        assert future.result(timeout=8).state == "cancelled"


def test_controller_crash_and_lease_recovery(tmp_path):
    backend, run, runner = setup(tmp_path, lease=.6)
    marker = tmp_path / "pids.json"
    argv = [sys.executable, str(Path(__file__).with_name("process_fixture.py")), str(marker)]
    code = ("from mdac2.infrastructure.job_backend import JobBackend; "
            "from mdac2.infrastructure.process_runner import ProcessRunner; "
            f"ProcessRunner(JobBackend({str(backend.database)!r}),lease_seconds=.6).execute({run.job_id!r},{run.run_id!r},{argv!r},timeout=10)")
    controller = subprocess.Popen([sys.executable, "-c", code], creationflags=subprocess.CREATE_NO_WINDOW)
    try:
        wait_for(marker.exists)
        pids = json.loads(marker.read_text())
        controller.kill()
        controller.wait(timeout=3)
        wait_for(lambda: all(dead(pid) for pid in pids))
        until = backend.get_job(run.job_id).runs[-1].lease_until
        wait_for(lambda: time.time() >= until)
        recovered = runner.recover(run.job_id, run.run_id)
        assert recovered.state == "interrupted" and recovered.exit_confirmed
        assert backend.retry(run.job_id, run.run_id).run_id != run.run_id
    finally:
        if controller.poll() is None:
            controller.kill()
            controller.wait(timeout=3)
        recover_tree(tree_name(run))


def test_recover_live_lease_refused_and_bad_executable_exits(tmp_path):
    backend, run, runner = setup(tmp_path)
    with pytest.raises(JobError):
        runner.recover(run.job_id, run.run_id)
    with pytest.raises(OSError):
        runner.execute(run.job_id, run.run_id, [str(tmp_path / "missing.exe")], timeout=1)
    snapshot = backend.get_job(run.job_id).runs[-1]
    assert snapshot.state == "interrupted" and snapshot.exit_confirmed
