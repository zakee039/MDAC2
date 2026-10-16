"""Preview/confirm timing migrations; no persistence or silent conflict repair."""

from dataclasses import dataclass, replace
from fractions import Fraction
from uuid import uuid4

from mdac2.domain.artifact import content_hash
from mdac2.domain.chart import ChartEvent, ChartRevision
from mdac2.domain.timing import TimingError
from mdac2.domain.timing_edits import TimingRevision
from mdac2.domain.validation import validate_chart


class MigrationError(ValueError):
    """Invalid policy, stale confirmation, or unresolved migration conflicts."""


@dataclass(frozen=True, slots=True)
class EventDiff:
    event_id: str
    old_q: Fraction
    old_end_q: Fraction
    new_q: Fraction
    new_end_q: Fraction
    old_seconds: Fraction
    old_end_seconds: Fraction
    new_seconds: Fraction
    new_end_seconds: Fraction

    def to_dict(self):
        from dataclasses import fields
        return {field.name: str(getattr(self, field.name)) for field in fields(self)}


@dataclass(frozen=True, slots=True)
class Conflict:
    code: str
    event_id: str | None


@dataclass(frozen=True, slots=True)
class MigrationPreview:
    base_chart_hash: str
    old_timing_fingerprint: str
    new_timing_fingerprint: str
    position_policy: str
    diffs: tuple[EventDiff, ...]
    conflicts: tuple[Conflict, ...]
    proposed_events: tuple[ChartEvent, ...]

    @property
    def fingerprint(self):
        return content_hash({
            "base_chart_hash": self.base_chart_hash, "old_timing": self.old_timing_fingerprint,
            "new_timing": self.new_timing_fingerprint, "policy": self.position_policy,
            "diffs": [diff.to_dict() for diff in self.diffs],
            "conflicts": [{"code": c.code, "event_id": c.event_id} for c in self.conflicts],
            "proposed_events": [event.to_dict() for event in self.proposed_events],
        })


def preview_migration(chart: ChartRevision, old: TimingRevision, new: TimingRevision,
                      *, position_policy: str) -> MigrationPreview:
    if position_policy not in ("keep_audio_time", "keep_musical_position"):
        raise MigrationError("Explicit supported position_policy required")
    if chart.timing_revision != old.revision_id:
        raise MigrationError("Chart does not reference supplied old TimingRevision")
    if old.revision_id == new.revision_id and old.fingerprint != new.fingerprint:
        raise MigrationError("Different timing content cannot reuse a revision ID")
    diffs, conflicts, proposed = [], [], []
    original_report = validate_chart(chart, old.mapping, timing_revision=old.revision_id)
    for issue in original_report.issues:
        if issue.severity == "error":
            conflicts.append(Conflict("SOURCE_" + issue.code, issue.event_id))
    for event in chart.events:
        try:
            old_head, old_tail = old.mapping.t(event.q), old.mapping.t(event.end_q)
            if position_policy == "keep_audio_time":
                new_q, new_end = new.mapping.q(old_head), new.mapping.q(old_tail)
            else:
                new_q, new_end = event.q, event.end_q
            new_head, new_tail = new.mapping.t(new_q), new.mapping.t(new_end)
        except TimingError:
            conflicts.append(Conflict("TIME_OUT_OF_RANGE", event.event_id))
            continue
        diff = EventDiff(event.event_id, event.q, event.end_q, new_q, new_end,
                         old_head, old_tail, new_head, new_tail)
        diffs.append(diff)
        if event.locked and (event.q, event.end_q, old_head, old_tail) != (new_q, new_end, new_head, new_tail):
            conflicts.append(Conflict("LOCKED_EVENT_CHANGED", event.event_id))
        proposed.append(replace(event, q=new_q, duration_q=new_end - new_q))
    if len(proposed) == len(chart.events):
        candidate = replace(chart, timing_revision=new.revision_id, events=tuple(proposed))
        for issue in validate_chart(candidate, new.mapping, timing_revision=new.revision_id).issues:
            if issue.severity == "error":
                conflicts.append(Conflict("TARGET_" + issue.code, issue.event_id))
    return MigrationPreview(content_hash(chart.to_dict()), old.fingerprint, new.fingerprint,
                            position_policy, tuple(diffs), tuple(conflicts), tuple(proposed))


def apply_migration(chart: ChartRevision, old: TimingRevision, new: TimingRevision, *,
                    position_policy: str, expected_preview_fingerprint: str) -> ChartRevision:
    # Recompute from current immutable inputs, never trust a caller-modified preview.
    preview = preview_migration(chart, old, new, position_policy=position_policy)
    if preview.fingerprint != expected_preview_fingerprint:
        raise MigrationError("Preview is stale; review a new diff before confirmation")
    if preview.conflicts:
        raise MigrationError("Unresolved migration conflicts; nothing was applied")
    return replace(chart, revision_id=str(uuid4()), timing_revision=new.revision_id,
                   events=preview.proposed_events)
