"""Six-class training events; independent of the legacy game-facing ChartIR.

Times are canonical audio seconds. Source identities and masks never enter the
audio model. Validation reports conflicts without editing the user's reference.
"""
from dataclasses import asdict, dataclass
import math

CLASSES = ('light', 'heavy', 'extra_heavy', 'gemini', 'hold', 'gear')
LANES = ('ground', 'air')
POLICY_ID = 'community-six-v1'


@dataclass(frozen=True)
class TrainingEvent:
    event_id: str
    kind: str
    lanes: tuple[str, ...]
    start: float
    end: float
    source_ids: tuple[str, ...]
    source_code: str
    source_role: str
    exclusive: bool = False
    group_id: str | None = None

    def __post_init__(self):
        if self.kind not in CLASSES or not self.event_id or not self.source_ids:
            raise ValueError('Invalid training event identity/type')
        if not self.lanes or len(set(self.lanes)) != len(self.lanes) or not set(self.lanes) <= set(LANES):
            raise ValueError('Invalid lanes')
        if not all(math.isfinite(t) for t in (self.start, self.end)) or self.start < 0 or self.end < self.start:
            raise ValueError('Invalid event time')
        if (self.kind == 'hold') != (self.end > self.start):
            raise ValueError('Only holds have positive duration')
        if self.kind == 'gemini' and set(self.lanes) != set(LANES):
            raise ValueError('Gemini requires both lanes')
        if self.kind not in ('hold','gemini') and len(self.lanes) != 1:
            raise ValueError('Single attacks/obstacles retain one actual input lane')
        if self.exclusive != (self.kind == 'extra_heavy'):
            raise ValueError('Special heavy must be exclusive')

    def to_dict(self):
        return {**asdict(self), 'lanes': list(self.lanes), 'source_ids': list(self.source_ids)}

    @classmethod
    def from_dict(cls, value):
        return cls(**{**value, 'lanes': tuple(value['lanes']), 'source_ids': tuple(value['source_ids'])})


def validate_events(events, duration):
    """Half-open hold occupancy; instantaneous exclusivity, no invented gap."""
    issues = []
    ids = set()
    for e in events:
        if e.event_id in ids:
            raise ValueError('Duplicate event identity')
        ids.add(e.event_id)
        if e.end > duration + 1e-6:
            issues.append({'code': 'outside_audio', 'event_ids': [e.event_id], 'start': e.start, 'end': e.end})
    ordered = sorted(events, key=lambda e: (e.start, e.event_id))
    active = []
    for e in ordered:
        active = [a for a in active if a.end > e.start or a.start == e.start]
        for other in active:
            same_start = e.start == other.start
            intersects = same_start or (other.kind == 'hold' and other.start <= e.start < other.end)
            if intersects and (e.exclusive or other.exclusive):
                issues.append({'code': 'exclusive_overlap', 'event_ids': [other.event_id, e.event_id],
                               'start': e.start, 'end': max(e.start, min(e.end, other.end))})
            elif intersects and set(e.lanes) & set(other.lanes):
                if e.kind == other.kind == 'hold':
                    code = 'overlapping_holds'
                elif same_start and e.kind == other.kind and e.lanes == other.lanes:
                    code = 'duplicate_target'
                elif 'gear' in (e.kind, other.kind):
                    code = 'gear_overlap'
                else:
                    # Hold + strike is an explicitly retained user-requested form.
                    continue
                issues.append({'code': code, 'event_ids': [other.event_id, e.event_id],
                               'start': e.start, 'end': max(e.start, min(e.end, other.end))})
        active.append(e)
    return issues
