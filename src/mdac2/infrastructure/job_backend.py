"""SQLite business authority + transactional outbox; Huey is delivery only.

Job tables have an independent schema gate within the project database. No
automatic migration, queue-result authority, or project-head update occurs here.
"""
from dataclasses import asdict
import sqlite3

from mdac2.application.jobs_contract import Job, JobError, JobRequest, Resources, Run, TERMINAL, public_run
from mdac2.domain.artifact import canonical_json, strict_json
from mdac2.infrastructure.project_repository import ProjectRepository, RepositoryError, RevisionConflict


_SCHEMA = """
CREATE TABLE job_schema(version INTEGER NOT NULL CHECK(version=1));
INSERT INTO job_schema VALUES(1);
CREATE TABLE jobs(
 job_id TEXT PRIMARY KEY, project_id TEXT NOT NULL REFERENCES projects(project_id),
 request_key TEXT NOT NULL, request BLOB NOT NULL, current_run TEXT NOT NULL,
 active INTEGER NOT NULL CHECK(active IN (0,1))
);
CREATE UNIQUE INDEX jobs_inflight ON jobs(request_key) WHERE active=1;
CREATE TABLE runs(
 run_id TEXT PRIMARY KEY, job_id TEXT NOT NULL REFERENCES jobs(job_id),
 attempt INTEGER NOT NULL, snapshot BLOB NOT NULL, UNIQUE(job_id,attempt)
);
CREATE TABLE job_outbox(
 run_id TEXT PRIMARY KEY REFERENCES runs(run_id), job_id TEXT NOT NULL REFERENCES jobs(job_id),
 delivered INTEGER NOT NULL DEFAULT 0 CHECK(delivered IN (0,1)),
 generation INTEGER NOT NULL DEFAULT 0
);
"""


class JobBackend(ProjectRepository):
    def __init__(self, database, *, cpu_slots=1, gpu_memory_mb=None):
        Resources(cpu_slots)
        self.cpu_slots = cpu_slots
        self.gpu_memory_mb = dict(gpu_memory_mb or {})
        for device, memory in self.gpu_memory_mb.items():
            Resources(1, device, memory)
        super().__init__(database)
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            exists = connection.execute("SELECT 1 FROM sqlite_master WHERE name='job_schema'").fetchone()
            if exists:
                if [tuple(row) for row in connection.execute("SELECT version FROM job_schema")] != [(1,)]:
                    raise RepositoryError("Unknown job schema; snapshot and migration required")
            else:
                if connection.execute("SELECT 1 FROM sqlite_master WHERE name IN ('jobs','runs','job_outbox')").fetchone():
                    raise RepositoryError("Unversioned job tables; explicit migration required")
                for statement in _SCHEMA.split(";"):
                    if statement.strip():
                        connection.execute(statement)
            connection.commit()

        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            if not connection.execute("SELECT 1 FROM sqlite_master WHERE name='job_event_schema'").fetchone():
                if connection.execute("SELECT 1 FROM sqlite_master WHERE name='job_events'").fetchone():
                    raise RepositoryError("Unversioned event table")
                connection.execute("CREATE TABLE job_event_schema(version INTEGER NOT NULL CHECK(version=1))")
                connection.execute("INSERT INTO job_event_schema VALUES(1)")
                connection.execute("CREATE TABLE job_events(event_id INTEGER PRIMARY KEY AUTOINCREMENT, job_id TEXT NOT NULL REFERENCES jobs(job_id), run_id TEXT NOT NULL REFERENCES runs(run_id), snapshot BLOB NOT NULL)")
                connection.execute("CREATE INDEX job_events_cursor ON job_events(job_id,event_id)")
                # Existing databases get a current snapshot, not invented history.
                for row in connection.execute("SELECT snapshot FROM runs ORDER BY job_id,attempt").fetchall():
                    self._event(connection, Run(**strict_json(row[0])))
            elif [tuple(row) for row in connection.execute("SELECT version FROM job_event_schema")] != [(1,)]:
                raise RepositoryError("Unknown event schema; migration required")
            connection.commit()

    def _load(self, connection, job_id):
        row = connection.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
        if row is None:
            raise KeyError(job_id)
        runs = tuple(Run(**strict_json(item[0])) for item in connection.execute(
            "SELECT snapshot FROM runs WHERE job_id=? ORDER BY attempt", (job_id,)))
        job = Job(job_id, JobRequest(row["request"]), runs)
        if job.runs[-1].run_id != row["current_run"]:
            raise RepositoryError("Inconsistent current run")
        return job

    def get_job(self, job_id):
        with self._connection() as connection:
            connection.execute("BEGIN")
            return self._load(connection, job_id)

    def _outbox(self, connection, run):
        connection.execute("INSERT INTO job_outbox(run_id,job_id) VALUES (?,?)", (run.run_id, run.job_id))
        self._event(connection, run)

    def _event(self, connection, run):
        connection.execute("INSERT INTO job_events(job_id,run_id,snapshot) VALUES (?,?,?)", (run.job_id, run.run_id, canonical_json(public_run(run))))

    def event_batch(self, job_id, after=0, *, limit=500):
        if type(after) is not int or after < 0 or type(limit) is not int or not 1 <= limit <= 500:
            raise JobError("Invalid event cursor/limit")
        with self._connection() as connection:
            connection.execute("BEGIN")
            job = self._load(connection, job_id)
            if after and not connection.execute("SELECT 1 FROM job_events WHERE job_id=? AND event_id=?", (job_id, after)).fetchone():
                raise JobError("Cursor does not belong to this job")
            rows = connection.execute("SELECT event_id,snapshot FROM job_events WHERE job_id=? AND event_id>? ORDER BY event_id LIMIT ?", (job_id, after, limit + 1)).fetchall()
            return [(row[0], strict_json(row[1])) for row in rows[:limit]], job.state in TERMINAL and len(rows) <= limit

    def create_job(self, request: JobRequest):
        if not isinstance(request, JobRequest):
            raise JobError("JobRequest required")
        data = request.to_dict()
        wanted = Resources(**data["resources"])
        if wanted.cpu_slots > self.cpu_slots or (wanted.gpu_device is not None and
                wanted.gpu_memory_mb > self.gpu_memory_mb.get(wanted.gpu_device, 0)):
            raise JobError("Requested resources unavailable")
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            project = connection.execute("SELECT head_revision,archived FROM projects WHERE project_id=?", (data["project_id"],)).fetchone()
            if project is None:
                raise KeyError(data["project_id"])
            if project["head_revision"] != data["project_revision"] or project["archived"]:
                raise RevisionConflict("Job expected revision is stale or archived")
            existing = connection.execute("SELECT job_id FROM jobs WHERE request_key=? AND active=1", (request.request_key,)).fetchone()
            if existing:
                return self._load(connection, existing[0])
            job = Job.create(request)
            run = job.runs[-1]
            connection.execute("INSERT INTO jobs VALUES (?,?,?,?,?,1)",
                (job.job_id, data["project_id"], request.request_key, request.encoded, run.run_id))
            connection.execute("INSERT INTO runs VALUES (?,?,?,?)", (run.run_id, job.job_id, run.attempt, canonical_json(asdict(run))))
            self._outbox(connection, run)
            connection.commit()
            return job

    def dispatch(self, transport, *, limit=100, after_send=None):
        if type(limit) is not int or limit < 1:
            raise JobError("Positive dispatch limit required")
        with self._connection() as connection:
            rows = connection.execute("SELECT job_id,run_id,generation FROM job_outbox WHERE delivered=0 ORDER BY rowid LIMIT ?", (limit,)).fetchall()
        for row in rows:
            transport.send(row["job_id"], row["run_id"])
            if after_send is not None:
                after_send()  # fault seam: delivery succeeded but ACK may fail
            with self._connection() as connection:
                connection.execute("UPDATE job_outbox SET delivered=1 WHERE run_id=? AND generation=?", (row["run_id"], row["generation"]))
        return len(rows)

    def _save(self, connection, run):
        connection.execute("UPDATE runs SET snapshot=? WHERE run_id=?", (canonical_json(asdict(run)), run.run_id))
        active = run.state in {"queued", "running"} or not run.exit_confirmed
        connection.execute("UPDATE jobs SET active=? WHERE job_id=?", (int(active), run.job_id))
        self._event(connection, run)

    def claim(self, job_id, run_id, worker_id, *, now, lease_seconds=30):
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            job = self._load(connection, job_id)
            run = job.runs[-1]
            if run.run_id != run_id or run.state != "queued":
                return None
            desired = Resources(**job.request.to_dict()["resources"])
            occupied_cpu = 0
            occupied_gpus = set()
            for row in connection.execute("SELECT request,snapshot FROM jobs JOIN runs ON jobs.current_run=runs.run_id WHERE jobs.active=1"):
                other = Run(**strict_json(row["snapshot"]))
                if other.state == "running" or (other.state == "interrupted" and not other.exit_confirmed):
                    resources = Resources(**JobRequest(row["request"]).to_dict()["resources"])
                    occupied_cpu += resources.cpu_slots
                    if resources.gpu_device is not None:
                        occupied_gpus.add(resources.gpu_device)
            if occupied_cpu + desired.cpu_slots > self.cpu_slots or desired.gpu_device in occupied_gpus:
                # Huey already dequeued this message: persist redelivery intent.
                connection.execute("UPDATE job_outbox SET delivered=0,generation=generation+1 WHERE run_id=?", (run_id,))
                connection.commit()
                return None
            started = run.start(worker_id, now=now, lease_seconds=lease_seconds)
            self._save(connection, started)
            connection.commit()
            return started

    def _change(self, job_id, run_id, change):
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            job = self._load(connection, job_id)
            if job.runs[-1].run_id != run_id:
                raise JobError("Stale run")
            run = change(job.runs[-1])
            self._save(connection, run)
            connection.commit()
            return run

    def heartbeat(self, job_id, run_id, token, **progress):
        return self._change(job_id, run_id, lambda run: run.heartbeat(token, **progress))

    def finish(self, job_id, run_id, state, token, **result):
        return self._change(job_id, run_id, lambda run: run.finish(state, token, **result))

    def cancel(self, job_id, run_id):
        return self._change(job_id, run_id, lambda run: run.cancel())

    def interrupt(self, job_id, run_id, **evidence):
        return self._change(job_id, run_id, lambda run: run.interrupt(**evidence))

    def confirm_exit(self, job_id, run_id):
        return self._change(job_id, run_id, lambda run: run.confirm_exit())

    def retry(self, job_id, run_id):
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            job = self._load(connection, job_id)
            if job.runs[-1].run_id != run_id:
                raise JobError("Stale retry")
            run = job.runs[-1].retry()
            # Preserve original inputs/revision; a late result may only enter history.
            try:
                connection.execute("UPDATE jobs SET current_run=?,active=1 WHERE job_id=?", (run.run_id, job_id))
            except sqlite3.IntegrityError as exc:
                raise JobError("An identical request is already active") from exc
            connection.execute("INSERT INTO runs VALUES (?,?,?,?)", (run.run_id, job_id, run.attempt, canonical_json(asdict(run))))
            self._outbox(connection, run)
            connection.commit()
            return run


class HueyTransport:
    def __init__(self, queue_database, on_message=None):
        from huey import SqliteHuey
        self.huey = SqliteHuey("mdac2", filename=str(queue_database), results=False)

        @self.huey.task(name="mdac2.delivery", retries=0)
        def deliver(job_id, run_id):
            if on_message is None:
                raise JobError("Consumer callback is not configured")
            return on_message(job_id, run_id)
        self.deliver = deliver

    def send(self, job_id, run_id):
        self.deliver(job_id, run_id)

    def close(self):
        self.huey.storage.close()
