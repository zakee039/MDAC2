"""ChartIR schema 1; logical events are not game entities.

Shape validation happens here; musical legality (lane masks, Hold length and
audio bounds) is reported separately by the validator, without editing input.
Event ordering is retained, never keyed/deduplicated by musical position.
"""

from dataclasses import dataclass
from fractions import Fraction
import re
from uuid import uuid4

from .artifact import canonical_json, strict_json
from .timing import rational


class ChartError(ValueError):
    """Invalid ChartIR structural contract."""


def new_event_id():
    return str(uuid4())


def _text(value):
    if type(value) is not str or not value.strip():
        raise ChartError("Expected non-empty identity string")


def _fields(value, names):
    if type(value) is not dict or set(value) != set(names):
        raise ChartError("Missing/unknown ChartIR fields; migration required")


def _fraction(value):
    if type(value) is not str:
        raise ChartError("Serialized musical positions must be fraction strings")
    try:
        result = Fraction(value)
    except (ValueError, ZeroDivisionError) as exc:
        raise ChartError("Invalid musical position") from exc
    if str(result) != value:
        raise ChartError("Noncanonical fraction string")
    return result


@dataclass(frozen=True, slots=True)
class ChartEvent:
    event_id: str
    q: Fraction
    lane_mask: tuple[str, ...]
    semantic: str = "tap"
    accent: bool = False
    duration_q: Fraction = Fraction(0)
    source_kind: str = "manual"
    source_id: str | None = None
    locked: bool = False

    def __post_init__(self):
        _text(self.event_id)
        object.__setattr__(self, "q", rational(self.q))
        object.__setattr__(self, "duration_q", rational(self.duration_q))
        if type(self.lane_mask) is not tuple or any(type(lane) is not str for lane in self.lane_mask):
            raise ChartError("lane_mask must be immutable tuple of strings")
        if self.semantic not in ("tap", "hold", "gemini"):
            raise ChartError("Unsupported semantic; explicit schema extension required")
        if type(self.accent) is not bool or type(self.locked) is not bool:
            raise ChartError("accent/locked must be bool")
        if self.source_kind not in ("manual", "event", "candidate"):
            raise ChartError("Invalid source kind")
        if self.source_kind == "manual":
            if self.source_id is not None:
                raise ChartError("Manual source must have null source_id")
        else:
            _text(self.source_id)

    @property
    def end_q(self):
        return self.q + self.duration_q

    def to_dict(self):
        return {"event_id": self.event_id, "q": str(self.q), "lane_mask": list(self.lane_mask),
                "semantic": self.semantic, "accent": self.accent, "duration_q": str(self.duration_q),
                "source_kind": self.source_kind, "source_id": self.source_id, "locked": self.locked}

    @classmethod
    def from_dict(cls, data):
        _fields(data, {"event_id", "q", "lane_mask", "semantic", "accent", "duration_q",
                       "source_kind", "source_id", "locked"})
        if type(data["lane_mask"]) is not list:
            raise ChartError("Serialized lane mask must be a list")
        return cls(**{**data, "q": _fraction(data["q"]), "duration_q": _fraction(data["duration_q"]),
                      "lane_mask": tuple(data["lane_mask"])})


@dataclass(frozen=True, slots=True)
class ChartRevision:
    revision_id: str
    timing_revision: str
    events: tuple[ChartEvent, ...]
    onset_revision: str | None = None
    arrangement_policy_hash: str | None = None
    semantics_policy_hash: str | None = None
    user_patch_ids: tuple[str, ...] = ()
    schema_version: int = 1

    def __post_init__(self):
        _text(self.revision_id)
        _text(self.timing_revision)
        if self.onset_revision is not None:
            _text(self.onset_revision)
        for value in (self.arrangement_policy_hash, self.semantics_policy_hash):
            if value is not None and (type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None):
                raise ChartError("Policy identity must be SHA-256 or null for manual baseline")
        if type(self.events) is not tuple or any(type(event) is not ChartEvent for event in self.events):
            raise ChartError("events must be immutable ChartEvent tuple")
        if len({event.event_id for event in self.events}) != len(self.events):
            raise ChartError("Duplicate event ID; simultaneous distinct events are allowed")
        if type(self.user_patch_ids) is not tuple:
            raise ChartError("user_patch_ids must be immutable tuple")
        for value in self.user_patch_ids:
            _text(value)
        if len(set(self.user_patch_ids)) != len(self.user_patch_ids):
            raise ChartError("Duplicate patch ID")
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ChartError("Unsupported ChartIR schema")

    def to_dict(self):
        return {"revision_id": self.revision_id, "timing_revision": self.timing_revision,
                "events": [event.to_dict() for event in self.events],
                "onset_revision": self.onset_revision, "arrangement_policy_hash": self.arrangement_policy_hash,
                "semantics_policy_hash": self.semantics_policy_hash, "user_patch_ids": list(self.user_patch_ids),
                "schema_version": self.schema_version}

    def to_json(self):
        return canonical_json(self.to_dict())

    @classmethod
    def from_json(cls, data):
        document = strict_json(data)
        _fields(document, {"revision_id", "timing_revision", "events", "onset_revision",
                           "arrangement_policy_hash", "semantics_policy_hash", "user_patch_ids", "schema_version"})
        if type(document["events"]) is not list or type(document["user_patch_ids"]) is not list:
            raise ChartError("Serialized events/patches must be lists")
        return cls(**{**document, "events": tuple(ChartEvent.from_dict(event) for event in document["events"]),
                      "user_patch_ids": tuple(document["user_patch_ids"])})
