"""SQLite schema 1 revision index. Chart payloads remain in artifacts, not rows.

This repository records references, not proof that files have been published;
filesystem/DB reconciliation belongs to the application publication workflow.
Each command uses one short IMMEDIATE transaction and expected-head CAS.
"""

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
import sqlite3
from uuid import uuid4

from mdac2.domain.artifact import ArtifactEnvelope, canonical_json
from mdac2.domain.chart import ChartRevision


class RepositoryError(ValueError):
    pass


class RevisionConflict(RepositoryError):
    pass


def _text(value):
    if type(value) is not str or not value.strip():
        raise RepositoryError("Non-empty text required")


@dataclass(frozen=True, slots=True)
class Project:
    project_id: str
    title: str
    head_revision: str | None
    archived: bool


@dataclass(frozen=True, slots=True)
class RevisionReference:
    revision_id: str
    project_id: str
    parent_revision: str | None
    artifact_hash: str
    command_id: str
    command_json: str
    inputs_json: str
    patches_json: str


_SCHEMA = """
CREATE TABLE projects (
 project_id TEXT PRIMARY KEY, title TEXT NOT NULL,
 head_revision TEXT, archived INTEGER NOT NULL DEFAULT 0 CHECK(archived IN (0,1)),
 FOREIGN KEY(project_id, head_revision) REFERENCES revisions(project_id, revision_id)
 DEFERRABLE INITIALLY DEFERRED
);
CREATE TABLE revisions (
 revision_id TEXT PRIMARY KEY, project_id TEXT NOT NULL REFERENCES projects(project_id),
 parent_revision TEXT, artifact_hash TEXT NOT NULL,
 command_id TEXT NOT NULL, command_json TEXT NOT NULL,
 inputs_json TEXT NOT NULL, patches_json TEXT NOT NULL,
 UNIQUE(project_id, revision_id), UNIQUE(project_id, command_id),
 FOREIGN KEY(project_id, parent_revision) REFERENCES revisions(project_id, revision_id)
);
PRAGMA user_version=1;
"""


class ProjectRepository:
    def __init__(self, database):
        self.database = Path(database)
        self.database.parent.mkdir(parents=True, exist_ok=True)
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            version = connection.execute("PRAGMA user_version").fetchone()[0]
            if version == 0:
                if connection.execute("SELECT 1 FROM sqlite_master WHERE name NOT LIKE 'sqlite_%'").fetchone():
                    raise RepositoryError("Unversioned nonempty database; explicit migration required")
                # Execute statements individually: executescript would implicitly commit.
                for statement in _SCHEMA.split(";"):
                    if statement.strip():
                        connection.execute(statement)
            elif version != 1:
                raise RepositoryError("Unknown database schema; migration and snapshot required")
            else:
                connection.execute("SELECT project_id, title, head_revision, archived FROM projects LIMIT 0")
                connection.execute("SELECT revision_id, project_id, parent_revision, artifact_hash, command_id, command_json, inputs_json, patches_json FROM revisions LIMIT 0")
            connection.commit()

    @contextmanager
    def _connection(self):
        connection = sqlite3.connect(self.database, timeout=5, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA synchronous=FULL")
        try:
            yield connection
        except BaseException:
            if connection.in_transaction:
                connection.rollback()
            raise
        finally:
            connection.close()

    def create_project(self, title):
        _text(title)
        project = Project(str(uuid4()), title, None, False)
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute("INSERT INTO projects(project_id,title) VALUES (?,?)", (project.project_id, title))
            connection.commit()
        return project

    def get_project(self, project_id):
        with self._connection() as connection:
            row = connection.execute("SELECT * FROM projects WHERE project_id=?", (project_id,)).fetchone()
        if row is None:
            raise KeyError(project_id)
        return Project(row["project_id"], row["title"], row["head_revision"], bool(row["archived"]))

    def get_revision(self, project_id, revision_id):
        with self._connection() as connection:
            row = connection.execute("SELECT * FROM revisions WHERE project_id=? AND revision_id=?",
                                     (project_id, revision_id)).fetchone()
        if row is None:
            raise KeyError(revision_id)
        return RevisionReference(**dict(row))

    def _advance_head(self, connection, project_id, expected_revision, revision_id):
        changed = connection.execute(
            "UPDATE projects SET head_revision=? WHERE project_id=? AND head_revision IS ? AND archived=0",
            (revision_id, project_id, expected_revision)).rowcount
        if changed != 1:
            raise RevisionConflict("Project head changed or project is archived")

    def commit_revision(self, project_id, *, expected_revision, artifact: ArtifactEnvelope,
                        command_id, command_type, command_payload, on_commit=None):
        _text(project_id)
        _text(command_id)
        _text(command_type)
        if expected_revision is not None:
            _text(expected_revision)
        if artifact.to_dict()["artifact_type"] != "chart":
            raise RepositoryError("Expected chart artifact")
        chart = ChartRevision.from_json(canonical_json(artifact.payload))
        command_json = canonical_json({"type": command_type, "payload": command_payload}).decode("utf-8")
        inputs = canonical_json(artifact.to_dict()["inputs"]).decode("utf-8")
        patches = canonical_json(list(chart.user_patch_ids)).decode("utf-8")
        try:
            with self._connection() as connection:
                connection.execute("BEGIN IMMEDIATE")
                row = connection.execute("SELECT head_revision,archived FROM projects WHERE project_id=?", (project_id,)).fetchone()
                if row is None:
                    raise KeyError(project_id)
                if row["head_revision"] != expected_revision or row["archived"]:
                    raise RevisionConflict("Project head changed or project is archived")
                connection.execute("INSERT INTO revisions VALUES (?,?,?,?,?,?,?,?)", (
                    chart.revision_id, project_id, expected_revision, artifact.artifact_hash,
                    command_id, command_json, inputs, patches))
                self._advance_head(connection, project_id, expected_revision, chart.revision_id)
                if on_commit is not None:
                    on_commit(connection, chart.revision_id)
                connection.commit()
        except sqlite3.IntegrityError as exc:
            raise RevisionConflict("Revision/command ID already exists or references are invalid") from exc
        return self.get_revision(project_id, chart.revision_id)
