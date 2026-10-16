"""Trusted executable supervision; Windows trees and lease-fenced state writes.

No model download, shell command API, or filesystem cleanup. Process return code
is execution success only; artifact validation/publication is a separate step.
"""
from dataclasses import dataclass
import math
import time

from mdac2.application.jobs_contract import JobError


def tree_name(run):
    return "Local\\mdac2-" + run.lease_token


@dataclass(frozen=True, slots=True)
class ExecutionResult:
    state: str
    exit_code: int | None
    error_code: str | None


class ProcessRunner:
    def __init__(self, backend, *, lease_seconds=5, poll_seconds=.02):
        for value in (lease_seconds, poll_seconds):
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError("Positive finite intervals required")
        if poll_seconds * 3 >= lease_seconds:
            raise ValueError("Polling must be comfortably inside the lease")
        self.backend = backend
        self.lease_seconds = lease_seconds
        self.poll_seconds = poll_seconds

    def execute(self, job_id, run_id, argv, *, timeout, cwd=None, prepare_success=None):
        from mdac2.infrastructure.windows_process import WindowsTree
        if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("Positive finite timeout required")
        tree = None
        run = None
        code = None
        error = None
        try:
            # Serialize launch against recovery and duplicate execution. The
            # persistent stage prevents two supervisors owning the same attempt.
            with self.backend._connection() as connection:
                connection.execute("BEGIN IMMEDIATE")
                job = self.backend._load(connection, job_id)
                current = job.runs[-1]
                if current.run_id != run_id or current.state != "running" or current.stage != "starting":
                    raise JobError("A claimed current run is required")
                run = current
                run._fence(run.lease_token, time.time())
                if run.cancel_requested:
                    cancelled = run.finish("cancelled", run.lease_token, now=time.time(), exit_confirmed=True)
                    self.backend._save(connection, cancelled)
                    connection.commit()
                    return ExecutionResult("cancelled", None, None)
                tree = WindowsTree(argv, name=tree_name(run), cwd=cwd)
                owned = run.heartbeat(run.lease_token, now=time.time(), lease_seconds=self.lease_seconds,
                                      stage="process_owned")
                self.backend._save(connection, owned)
                tree.start()
                connection.commit()
            started = time.monotonic()
            renewed = time.monotonic()
            while True:
                current = self.backend.get_job(job_id).runs[-1]
                if current.run_id != run_id or current.state != "running":
                    raise JobError("Run was superseded or interrupted")
                if current.cancel_requested:
                    error = None
                    break
                code = tree.poll()
                if code is not None:
                    error = "PROCESS_EXIT" if code else (None if tree.wait_empty() else "CHILD_PROCESS_LEFT_RUNNING")
                    break
                if time.monotonic() - started >= timeout:
                    error = "TIMEOUT"
                    break
                if time.monotonic() - renewed >= self.lease_seconds / 3:
                    self.backend.heartbeat(job_id, run_id, run.lease_token, now=time.time(),
                        lease_seconds=self.lease_seconds, stage="computing")
                    renewed = time.monotonic()
                time.sleep(self.poll_seconds)
            tree.terminate()  # also joins descendants before releasing resources
            # Publish/validate files outside SQLite. The returned registration
            # callback shares the terminal-state transaction; any error rolls
            # both back and leaves immutable files for reconciliation.
            commit_success = None
            if error is None and code == 0 and not current.cancel_requested and prepare_success is not None:
                commit_success = prepare_success()
            with self.backend._connection() as connection:
                connection.execute("BEGIN IMMEDIATE")
                current = self.backend._load(connection, job_id).runs[-1]
                state = "cancelled" if current.cancel_requested else ("failed" if error else "succeeded")
                finished = current.finish(state, run.lease_token, now=time.time(), exit_confirmed=True, error_code=error)
                if state == "succeeded" and commit_success is not None:
                    commit_success(connection, current)
                self.backend._save(connection, finished)
                connection.commit()
            return ExecutionResult(state, code, error)
        except BaseException:
            # Preserve an interrupted reservation if termination/DB access fails.
            if tree is not None:
                tree.terminate()
            if run is not None:
                try:
                    self.backend.interrupt(job_id, run_id, now=time.time(), exit_confirmed=True)
                except JobError:
                    pass  # another recovery already fenced the attempt
            raise
        finally:
            if tree is not None:
                tree.close()

    def recover(self, job_id, run_id, *, now=None):
        from mdac2.infrastructure.windows_process import recover_tree
        now = time.time() if now is None else now
        with self.backend._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            run = self.backend._load(connection, job_id).runs[-1]
            if run.run_id != run_id:
                raise JobError("Stale recovery")
            if run.state == "running":
                run = run.interrupt(now=now)
            elif run.state != "interrupted":
                raise JobError("Only expired/interrupted runs can recover")
            recover_tree(tree_name(run))
            run = run.confirm_exit()
            self.backend._save(connection, run)
            connection.commit()
            return run
