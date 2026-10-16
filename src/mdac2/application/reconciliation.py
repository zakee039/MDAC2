"""Immutable file publication and transactional registration, never staging export."""
from dataclasses import dataclass

from mdac2.application.jobs_contract import JobError
from mdac2.domain.artifact import ArtifactEnvelope, canonical_json
from mdac2.domain.chart import ChartRevision
from mdac2.infrastructure.artifact_store import StoreError, _plain
from mdac2.infrastructure.project_repository import RepositoryError


_SCHEMA = """
CREATE TABLE publication_schema(version INTEGER NOT NULL CHECK(version=1));
INSERT INTO publication_schema VALUES(1);
CREATE TABLE artifacts(artifact_hash TEXT PRIMARY KEY, artifact_type TEXT NOT NULL, relative_path TEXT NOT NULL, verified INTEGER NOT NULL CHECK(verified=1));
CREATE TABLE artifact_inputs(artifact_hash TEXT NOT NULL REFERENCES artifacts(artifact_hash), input_hash TEXT NOT NULL REFERENCES artifacts(artifact_hash), position INTEGER NOT NULL, PRIMARY KEY(artifact_hash,position));
CREATE TABLE run_outputs(run_id TEXT PRIMARY KEY REFERENCES runs(run_id), artifact_hash TEXT NOT NULL REFERENCES artifacts(artifact_hash), head_advanced INTEGER NOT NULL CHECK(head_advanced IN (0,1)));
"""


@dataclass(frozen=True, slots=True)
class RecoveryReport:
    registered: tuple[str, ...]
    unresolved: tuple[str, ...]
    staging_count: int


class Reconciler:
    def __init__(self, backend, store):
        self.backend, self.store = backend, store
        with backend._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            if connection.execute("SELECT 1 FROM sqlite_master WHERE name='publication_schema'").fetchone():
                if [tuple(row) for row in connection.execute("SELECT version FROM publication_schema")] != [(1,)]:
                    raise RepositoryError("Unknown publication schema; migration required")
            else:
                if connection.execute("SELECT 1 FROM sqlite_master WHERE name IN ('artifacts','artifact_inputs','run_outputs')").fetchone():
                    raise RepositoryError("Unversioned publication tables")
                for statement in _SCHEMA.split(";"):
                    if statement.strip():
                        connection.execute(statement)
            connection.commit()

    def _register(self, connection, envelope):
        data = envelope.to_dict()
        kind, identity = data["artifact_type"], envelope.artifact_hash
        self.store.get(kind, identity)  # includes same-size sidecar tampering
        for dependency in data["inputs"]:
            row = connection.execute("SELECT artifact_type FROM artifacts WHERE artifact_hash=?", (dependency,)).fetchone()
            if row is None:
                raise RepositoryError("Artifact dependency is not registered")
            self.store.get(row[0], dependency)
        relative = f"artifacts/{kind}/{identity}"
        existing = connection.execute("SELECT artifact_type,relative_path FROM artifacts WHERE artifact_hash=?", (identity,)).fetchone()
        if existing is not None and tuple(existing) != (kind, relative):
            raise RepositoryError("Conflicting artifact registration")
        connection.execute("INSERT OR IGNORE INTO artifacts VALUES (?,?,?,1)", (identity, kind, relative))
        for position, dependency in enumerate(data["inputs"]):
            connection.execute("INSERT OR IGNORE INTO artifact_inputs VALUES (?,?,?)", (identity, dependency, position))

    def _commit_output(self, connection, run, envelope):
        job = self.backend._load(connection, run.job_id)
        if job.runs[-1].run_id != run.run_id or job.runs[-1].lease_token != run.lease_token:
            raise JobError("Stale output attempt")
        request = job.request.to_dict()
        expected = ArtifactEnvelope.create(artifact_type=request["operation"], producer=request["producer"],
            options=request["options"], inputs=request["inputs"], payload=None)
        if envelope.request_key != expected.request_key:
            raise JobError("Output does not match effective request identity")
        self._register(connection, envelope)
        head = connection.execute("SELECT head_revision,archived FROM projects WHERE project_id=?", (request["project_id"],)).fetchone()
        advanced = False
        if envelope.to_dict()["artifact_type"] == "chart" and head[0] == request["project_revision"] and not head[1]:
            chart = ChartRevision.from_json(canonical_json(envelope.payload))
            connection.execute("INSERT INTO revisions VALUES (?,?,?,?,?,?,?,?)", (
                chart.revision_id, request["project_id"], request["project_revision"], envelope.artifact_hash,
                run.run_id, canonical_json({"type": "generated", "payload": {"job_id": run.job_id}}).decode(),
                canonical_json(envelope.to_dict()["inputs"]).decode(), canonical_json(list(chart.user_patch_ids)).decode()))
            self.backend._advance_head(connection, request["project_id"], request["project_revision"], chart.revision_id)
            advanced = True
        connection.execute("INSERT INTO run_outputs VALUES (?,?,?)", (run.run_id, envelope.artifact_hash, int(advanced)))

    def prepare(self, envelope, files=None):
        """Called by runner after process exit, before its success transaction."""
        self.store.publish(envelope, files)
        def register(connection, run):
            self._commit_output(connection, run, envelope)
        return register

    def recover_artifacts(self):
        """Re-register verified orphans in dependency order; never advance head.

        Invalid entries and staging remain untouched for diagnosis. Startup may
        repeat this safely. No pending job is declared successful from files alone.
        """
        candidates = {}
        unresolved = []
        _plain(self.store.artifacts)
        _plain(self.store.staging)
        for kind in self.store.artifacts.iterdir():
            try:
                _plain(kind)
                if not kind.is_dir():
                    raise StoreError("Unexpected entry")
                for directory in kind.iterdir():
                    try:
                        envelope = self.store.get(kind.name, directory.name)
                        candidates[envelope.artifact_hash] = envelope
                    except (ValueError, OSError):
                        unresolved.append(f"{kind.name}/{directory.name}")
            except (ValueError, OSError):
                unresolved.append(kind.name)
        registered = []
        while candidates:
            progress = False
            for identity, envelope in tuple(candidates.items()):
                try:
                    with self.backend._connection() as connection:
                        connection.execute("BEGIN IMMEDIATE")
                        self._register(connection, envelope)
                        connection.commit()
                except (ValueError, OSError):
                    continue
                registered.append(identity)
                del candidates[identity]
                progress = True
            if not progress:
                unresolved.extend(candidates)
                break
        return RecoveryReport(tuple(sorted(registered)), tuple(sorted(unresolved)), sum(1 for _ in self.store.staging.iterdir()))

    def recover_queued_delivery(self):
        """Startup repair for messages consumed before a worker could claim them."""
        with self.backend._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            ids = [row[0] for row in connection.execute("SELECT job_id FROM jobs WHERE active=1")]
            count = 0
            for job_id in ids:
                run = self.backend._load(connection, job_id).runs[-1]
                if run.state == "queued":
                    connection.execute("UPDATE job_outbox SET delivered=0,generation=generation+1 WHERE run_id=?", (run.run_id,))
                    count += 1
            connection.commit()
            return count
