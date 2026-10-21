"""Probability reroute: opposite-lane notes inside a hold may move onto it.

A sustained lane used to blank its whole time window. It no longer does: the
opposite lane keeps its notes in place and only spacing limits them. This
module adds the optional interior hits the workbench asks for, on top of that
baseline:

* only holds at least 500 ms long take part;
* candidates are single-lane attacks on the opposite lane, ordered by time;
* the first candidate reroutes with probability 50 %;
* if it does not, the second reroutes unconditionally;
* afterwards the hold is settled and no further rerouting happens.

Rerouting keeps the candidate time and changes only its lane, so the note lands
inside the sustained lane. A reroute that would break the doubled interior
spacing floor is skipped instead of forcing a later drop.
"""
from dataclasses import replace

from .v5_spacing import HOLD_SPACING_FACTOR

MINIMUM_HOLD_SECONDS = .5
FIRST_PROBABILITY = .5
ATTACK_KINDS = ('light', 'heavy', 'extra_heavy')


def _candidate(event, hold, opposite):
    return (event.kind in ATTACK_KINDS and len(event.lanes) == 1 and
            event.lanes == (opposite,) and hold.start + 1e-8 < event.start < hold.end - 1e-8)


def _fits(event, lane, hold, others, floor):
    """The rerouted note must clear the hold head and every same-lane neighbour."""
    for other in others:
        if other.event_id == event.event_id or lane not in other.lanes:continue
        if abs(other.start - event.start) >= floor - 1e-8:continue
        if other.kind in ATTACK_KINDS or other.kind in ('gemini', 'hold', 'masher'):return False
    return abs(event.start - hold.start) >= floor - 1e-8


def reroute(events, spacing, rng, minimum_hold_seconds=MINIMUM_HOLD_SECONDS):
    """Return the chart with at most one rerouted interior hit per settled hold."""
    floor = spacing * HOLD_SPACING_FACTOR
    holds = sorted((e for e in events if e.kind == 'hold' and
                    e.end - e.start >= minimum_hold_seconds - 1e-8), key=lambda e:(e.start, e.event_id))
    out = list(events)
    records, changed = [], []
    for hold in holds:
        lane = hold.lanes[0];opposite = 'air' if lane == 'ground' else 'ground'
        candidates = sorted((e for e in out if _candidate(e, hold, opposite)), key=lambda e:(e.start, e.event_id))
        if not candidates:continue
        record = dict(event_id=hold.event_id, lane=lane, candidates=[e.event_id for e in candidates], rolls=[],
                      rerouted=None, reason='')
        for order, candidate in enumerate(candidates[:2], 1):
            roll = True if order == 2 else rng.random() < FIRST_PROBABILITY
            record['rolls'].append(dict(event_id=candidate.event_id, order=order, rerouted=roll))
            if not roll:
                record['reason'] = 'first_roll_kept_original' if order == 1 else 'first_roll_kept_original_then_blocked'
                continue
            others = [e for e in out if e.event_id != candidate.event_id]
            if not _fits(candidate, lane, hold, others, floor):
                record['reason'] = 'interior_spacing_blocked';continue
            moved = replace(candidate, lanes=(lane,), group_id=None)
            out = [moved if e.event_id == candidate.event_id else e for e in out]
            record.update(rerouted=candidate.event_id, reason='rerouted')
            changed.append(dict(event_id=candidate.event_id, from_lane=opposite, to_lane=lane,
                                time=candidate.start, order=order))
            break
        records.append(record)
    return sorted(out, key=lambda e:(e.start, e.event_id)), dict(
        minimal_hold_seconds=minimum_hold_seconds, first_probability=FIRST_PROBABILITY,
        interior_spacing_ms=round(floor * 1000), rerouted=changed, holds=records,
        rerouted_count=len(changed))
