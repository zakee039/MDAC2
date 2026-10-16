"""One authoritative timing artifact per chart revision; atomic reference commit."""
from dataclasses import replace
from fractions import Fraction

from mdac2.domain.artifact import ArtifactEnvelope, canonical_json
from mdac2.domain.chart import ChartRevision
from mdac2.domain.timing import TimingMap
from mdac2.domain.timing_edits import TimingRevision
from mdac2.domain.validation import validate_chart
from mdac2.application.timing_migration import preview_migration, apply_migration
from mdac2.infrastructure.project_repository import RevisionConflict, RepositoryError


def timing_data(revision):
    return {"revision_id":revision.revision_id,"mapping":revision.mapping.to_dict(),
        "downbeat_q":[str(q) for q in revision.downbeat_q],"review":revision.review,
        "parent_fingerprint":revision.parent_fingerprint}


def timing_from(data):
    if type(data) is not dict or set(data)!={"revision_id","mapping","downbeat_q","review","parent_fingerprint"}:
        raise ValueError("Invalid timing revision fields")
    if type(data["downbeat_q"]) is not list:
        raise ValueError("Invalid downbeats")
    beats=tuple(Fraction(q) for q in data["downbeat_q"])
    if [str(q) for q in beats]!=data["downbeat_q"]:
        raise ValueError("Noncanonical downbeats")
    return TimingRevision(data["revision_id"],TimingMap.from_dict(data["mapping"]),beats,data["review"],data["parent_fingerprint"])


class ManualWorkflow:
    def __init__(self,repository,store):
        self.repository,self.store=repository,store
        with repository._connection() as c:
            c.execute("BEGIN IMMEDIATE")
            if not c.execute("SELECT 1 FROM sqlite_master WHERE name='manual_schema'").fetchone():
                if c.execute("SELECT 1 FROM sqlite_master WHERE name='revision_timing'").fetchone():raise RepositoryError("Unknown manual schema")
                c.execute("CREATE TABLE manual_schema(version INTEGER NOT NULL CHECK(version=1))")
                c.execute("INSERT INTO manual_schema VALUES(1)")
                c.execute("CREATE TABLE revision_timing(revision_id TEXT PRIMARY KEY REFERENCES revisions(revision_id),artifact_hash TEXT NOT NULL)")
            elif [tuple(r) for r in c.execute("SELECT version FROM manual_schema")] != [(1,)]:raise RepositoryError("Unknown manual schema")
            c.commit()

    def load(self,project_id,revision_id=None):
        project=self.repository.get_project(project_id)
        revision_id=project.head_revision if revision_id is None else revision_id
        if revision_id is None:return None
        ref=self.repository.get_revision(project_id,revision_id)
        chart=ChartRevision.from_json(canonical_json(self.store.get("chart",ref.artifact_hash).payload))
        with self.repository._connection() as c:
            row=c.execute("SELECT artifact_hash FROM revision_timing WHERE revision_id=?",(revision_id,)).fetchone()
        if row is None:raise ValueError("Legacy chart has no authoritative timing; explicit import required")
        timing=timing_from(self.store.get("timing",row[0]).payload)
        if chart.timing_revision!=timing.revision_id:raise ValueError("Inconsistent timing reference")
        return chart,timing

    def preview(self,project_id,expected_revision,new,policy):
        if self.repository.get_project(project_id).head_revision!=expected_revision:raise RevisionConflict("Stale timing base")
        chart,old=self.load(project_id,expected_revision)
        return preview_migration(chart,old,new,position_policy=policy)

    def save(self,project_id,*,expected_revision,command_id,chart,timing,position_policy=None,preview_fingerprint=None,reviewed=None,evidence=()):
        current=self.repository.get_project(project_id)
        if current.head_revision!=expected_revision:raise RevisionConflict("Stale manual save")
        with self.repository._connection() as c:
            row=c.execute("SELECT artifact_hash FROM project_audio WHERE project_id=?",(project_id,)).fetchone()
        if row is None:raise ValueError("Import audio before timing")
        audio=self.store.get("audio",row[0]); canonical=audio.payload["canonical"]
        if timing.mapping.audio_duration!=Fraction(canonical["sample_count"],canonical["sample_rate"]):raise ValueError("Timing duration mismatches audio")
        if expected_revision is not None:
            previous,old=self.load(project_id,expected_revision)
            if old.fingerprint!=timing.fingerprint:
                if timing.revision_id==old.revision_id:raise ValueError("Timing ID cannot change content")
                if timing.review!='pending' and not (reviewed and reviewed[1].is_current(timing,reviewed[0])):raise ValueError("Edited timing invalidates review")
                if previous.events:
                    migrated=apply_migration(previous,old,timing,position_policy=position_policy,expected_preview_fingerprint=preview_fingerprint)
                    if migrated.events!=chart.events:raise ValueError("Save migration before further chart edits")
        report=validate_chart(chart,timing.mapping,timing_revision=timing.revision_id,allow_empty=True)
        if report.has_errors:raise ValueError("Invalid manual chart")
        producer={"name":"manual","version":"1","model_hash":None,"dependencies":{}}
        if timing.review!='pending' and expected_revision is None and not (reviewed and reviewed[1].is_current(timing,reviewed[0])):
            raise ValueError("Initial review requires verified evidence")
        for item in evidence:self.store.get(item.to_dict()['artifact_type'],item.artifact_hash)
        timing_artifact=ArtifactEnvelope.create(artifact_type="timing",producer=producer,options={},inputs=[audio.artifact_hash,*[e.artifact_hash for e in evidence]],payload=timing_data(timing))
        if expected_revision is not None and old.fingerprint==timing.fingerprint:
            with self.repository._connection() as c:
                identity=c.execute("SELECT artifact_hash FROM revision_timing WHERE revision_id=?",(expected_revision,)).fetchone()[0]
            timing_artifact=self.store.get('timing',identity)
        # Chart-only analysis must retain its evidence even when timing is reused.
        chart_inputs=[audio.artifact_hash,timing_artifact.artifact_hash,*[e.artifact_hash for e in evidence]]
        if expected_revision is not None:chart_inputs.append(self.repository.get_revision(project_id,expected_revision).artifact_hash)
        chart_artifact=ArtifactEnvelope.create(artifact_type="chart",producer=producer,options={},inputs=list(dict.fromkeys(chart_inputs)),payload=chart.to_dict())
        self.store.publish(timing_artifact);self.store.publish(chart_artifact)
        def register(c,revision):c.execute("INSERT INTO revision_timing VALUES (?,?)",(revision,timing_artifact.artifact_hash))
        self.repository.commit_revision(project_id,expected_revision=expected_revision,artifact=chart_artifact,command_id=command_id,
            command_type="manual_save",command_payload={"position_policy":position_policy,"preview_fingerprint":preview_fingerprint},on_commit=register)
        return chart,timing
