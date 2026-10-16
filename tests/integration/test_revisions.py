from concurrent.futures import ThreadPoolExecutor
import sqlite3

import pytest

from mdac2.domain.artifact import ArtifactEnvelope
from mdac2.domain.chart import ChartEvent, ChartRevision
from mdac2.infrastructure.project_repository import ProjectRepository, RepositoryError, RevisionConflict


def artifact(revision):
    chart = ChartRevision(revision, "timing", (ChartEvent("event", 0, ("ground",)),))
    return ArtifactEnvelope.create(artifact_type="chart", producer={"name": "manual", "version": "1",
        "model_hash": None, "dependencies": {}}, options={}, payload=chart.to_dict())


def commit(repo, project, revision, expected=None, command=None):
    return repo.commit_revision(project, expected_revision=expected, artifact=artifact(revision),
                                command_id=command or revision, command_type="edit", command_payload={"reason": "test"})


def test_revisions_are_references_and_survive_reopen(tmp_path):
    path = tmp_path / "state.sqlite"
    repo = ProjectRepository(path)
    project = repo.create_project("项目 ' SQL text")
    first = commit(repo, project.project_id, "r1")
    second = commit(repo, project.project_id, "r2", "r1")
    assert first.artifact_hash == artifact("r1").artifact_hash
    assert second.parent_revision == "r1"
    reopened = ProjectRepository(path)
    assert reopened.get_project(project.project_id).head_revision == "r2"
    assert reopened.get_revision(project.project_id, "r1") == first
    assert "events" not in first.command_json


def test_concurrent_old_head_only_one_winner(tmp_path):
    path = tmp_path / "state.sqlite"
    repo = ProjectRepository(path)
    project = repo.create_project("p").project_id
    commit(repo, project, "base")
    def run(index):
        try:
            commit(ProjectRepository(path), project, f"r{index}", "base")
            return "ok"
        except RevisionConflict:
            return "conflict"
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(run, range(4)))
    assert results.count("ok") == 1
    assert results.count("conflict") == 3
    with sqlite3.connect(path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM revisions").fetchone()[0] == 2


@pytest.mark.parametrize("after_update", [False, True])
def test_transaction_failure_rolls_back_revision_and_head(tmp_path, monkeypatch, after_update):
    repo = ProjectRepository(tmp_path / "state.sqlite")
    project = repo.create_project("p").project_id
    commit(repo, project, "base")
    advance = repo._advance_head
    def fail(*args):
        if after_update:
            advance(*args)
        raise sqlite3.OperationalError("injected failure")
    monkeypatch.setattr(repo, "_advance_head", fail)
    with pytest.raises(sqlite3.OperationalError):
        commit(repo, project, "new", "base")
    assert repo.get_project(project).head_revision == "base"
    with pytest.raises(KeyError):
        repo.get_revision(project, "new")


def test_duplicate_id_commands_and_cross_project_are_refused(tmp_path):
    repo = ProjectRepository(tmp_path / "state.sqlite")
    one, two = repo.create_project("one").project_id, repo.create_project("two").project_id
    commit(repo, one, "r1", command="cmd1")
    for revision, expected, command in [("r2", None, "cmd2"), ("r1", "r1", "cmd2"), ("r2", "r1", "cmd1")]:
        with pytest.raises(RevisionConflict):
            commit(repo, one, revision, expected, command)
    with pytest.raises(RevisionConflict):
        commit(repo, two, "r3", "r1")
    with pytest.raises(KeyError):
        repo.get_revision(two, "r1")
    assert repo.get_project(one).head_revision == "r1"
    assert repo.get_project(two).head_revision is None


@pytest.mark.parametrize("version", [0, 99])
def test_unknown_database_not_migrated(tmp_path, version):
    path = tmp_path / "unknown.sqlite"
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE keep_me(value TEXT)")
        connection.execute(f"PRAGMA user_version={version}")
    with pytest.raises(RepositoryError):
        ProjectRepository(path)
    with sqlite3.connect(path) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == version
        assert connection.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall() == [("keep_me",)]
