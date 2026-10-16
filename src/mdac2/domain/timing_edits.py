"""Pure manual timing-anchor edits, not raw tracker observation mutation.

Editable beats in this first contract are explicit mapping anchors. Downbeats
reference those anchors and must agree with existing meter/bar boundaries.
Changing meter/bar interpretation is a separate operation, not guessed here.
"""

from dataclasses import dataclass, replace
from fractions import Fraction
from uuid import uuid4

from .artifact import content_hash
from .timing import Anchor, TimingError, TimingMap, rational


class EditConflict(TimingError):
    """The operation targets an obsolete base revision."""


@dataclass(frozen=True, slots=True)
class TimingRevision:
    revision_id: str
    mapping: TimingMap
    downbeat_q: tuple[Fraction, ...] = ()
    review: str = "pending"
    parent_fingerprint: str | None = None

    def __post_init__(self):
        if type(self.revision_id) is not str or not self.revision_id.strip():
            raise TimingError("Revision identity required")
        if type(self.mapping) is not TimingMap or type(self.downbeat_q) is not tuple:
            raise TimingError("Immutable mapping/downbeats required")
        if self.review not in ("pending", "accepted", "rejected"):
            raise TimingError("Invalid review state")
        if self.parent_fingerprint is not None:
            import re
            if type(self.parent_fingerprint) is not str or re.fullmatch(r"[0-9a-f]{64}", self.parent_fingerprint) is None:
                raise TimingError("Invalid parent fingerprint")
        beats = tuple(rational(q) for q in self.downbeat_q)
        object.__setattr__(self, "downbeat_q", beats)
        if any(a >= b for a, b in zip(beats, beats[1:])):
            raise TimingError("Downbeats must be strictly increasing")
        anchor_q = {anchor.q for anchor in self.mapping.anchors}
        for q in beats:
            meter = self.mapping.meter_at(q)
            if q not in anchor_q or ((q - meter.start_q) / meter.bar_quarters).denominator != 1:
                raise TimingError("Downbeat must reference an anchor on a meter bar boundary")

    @property
    def fingerprint(self):
        return content_hash({"revision_id": self.revision_id, "mapping": self.mapping.to_dict(),
                             "downbeat_q": [str(q) for q in self.downbeat_q], "review": self.review,
                             "parent_fingerprint": self.parent_fingerprint})


@dataclass(frozen=True, slots=True)
class EditRecord:
    operation_id: str
    base_fingerprint: str
    result_fingerprint: str
    operation: str
    parameters: tuple[tuple[str, str], ...]
    actor: str
    reason: str
    timestamp: str


@dataclass(frozen=True, slots=True)
class EditResult:
    before: TimingRevision
    after: TimingRevision
    record: EditRecord


def _base(base, expected):
    if type(base) is not TimingRevision:
        raise TimingError("Expected TimingRevision")
    if base.fingerprint != expected:
        raise EditConflict("Timing base fingerprint changed")


def _finish(base, mapping, downbeats, operation, parameters, actor, reason, timestamp):
    for text in (actor, reason, timestamp):
        if type(text) is not str or not text.strip():
            raise TimingError("Actor/reason/timestamp must be explicit non-empty text")
    after = TimingRevision(str(uuid4()), mapping, downbeats, "pending", base.fingerprint)
    record = EditRecord(str(uuid4()), base.fingerprint, after.fingerprint, operation,
                        tuple(sorted((key, str(value)) for key, value in parameters.items())),
                        actor, reason, timestamp)
    return EditResult(base, after, record)


def edit_timing(base: TimingRevision, *, expected_fingerprint: str, operation: str,
                q, t=None, enabled=None, actor: str, reason: str, timestamp: str) -> EditResult:
    _base(base, expected_fingerprint)
    q = rational(q)
    anchors = list(base.mapping.anchors)
    index = next((i for i, anchor in enumerate(anchors) if anchor.q == q), None)
    downbeats = base.downbeat_q
    if operation in ("insert_beat", "drag_beat"):
        if enabled is not None:
            raise TimingError("Unexpected enabled parameter")
        target = Anchor(q, rational(t))
        if operation == "insert_beat":
            if index is not None:
                raise TimingError("Beat already exists")
            anchors.append(target)
            anchors.sort(key=lambda anchor: anchor.q)
        else:
            if index is None:
                raise TimingError("Beat does not exist")
            anchors[index] = target
    elif operation == "delete_beat":
        if t is not None or enabled is not None:
            raise TimingError("Unexpected delete parameters")
        if index is None:
            raise TimingError("Beat does not exist")
        if q in downbeats:
            raise TimingError("Unmark downbeat explicitly before deleting its anchor")
        del anchors[index]
    elif operation == "set_downbeat":
        if t is not None or type(enabled) is not bool or index is None:
            raise TimingError("Downbeat operation requires existing beat and explicit bool")
        values = set(downbeats)
        if enabled:
            values.add(q)
        else:
            values.discard(q)
        downbeats = tuple(sorted(values))
    else:
        raise TimingError("Unknown timing operation")
    mapping = replace(base.mapping, anchors=tuple(anchors))
    return _finish(base, mapping, downbeats, operation, {"q": q, "t": t, "enabled": enabled},
                   actor, reason, timestamp)


def undo_timing(current: TimingRevision, edit: EditResult, *, expected_fingerprint: str,
                actor: str, reason: str, timestamp: str) -> EditResult:
    _base(current, expected_fingerprint)
    if (current.fingerprint != edit.after.fingerprint or
            edit.record.result_fingerprint != edit.after.fingerprint or
            edit.record.base_fingerprint != edit.before.fingerprint):
        raise EditConflict("Undo is not for current revision or edit record was altered")
    return _finish(current, edit.before.mapping, edit.before.downbeat_q, "undo",
                   {"operation_id": edit.record.operation_id}, actor, reason, timestamp)
