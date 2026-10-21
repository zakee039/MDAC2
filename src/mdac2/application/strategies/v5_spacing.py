"""V5 same-lane spacing: a configurable floor, doubled inside one hold.

V4's spacing resolver is bound byte-for-byte by `models/v4/manifest.json`, so
this module reimplements the same resolution loop against a V5-local conflict
predicate. The ordinary floor comes from the workbench slider (75-175 ms);
notes that share one hold interior use twice that floor. Holds no longer blank
their interior: a note inside a sustained lane is legal and only spacing-bound.

Hold endpoints are ordinary notes too: the fixed 500 ms tail guard is gone and
a hold head/tail is now just a position that obeys the same-lane floor. That is
what stops hold exits from piling every note onto the opposite lane.
"""
from collections import defaultdict
from dataclasses import replace
from mdac2.application.studio_timing_v4 import (
    ATTACKS, LANES, _exact_time, _is_interior, _priority, structural_issues)
from mdac2.data_preparation.training_chart import validate_events

DEFAULT_SPACING_SECONDS = .125
MINIMUM_SPACING_SECONDS = .075
MAXIMUM_SPACING_SECONDS = .175
HOLD_SPACING_FACTOR = 2


def spacing_seconds(value=None):
    """Clamp the workbench slider value into the documented 75-175 ms range."""
    if value is None:return DEFAULT_SPACING_SECONDS
    seconds=round(float(value)/1000,3) if float(value)>1 else round(float(value),3)
    return min(MAXIMUM_SPACING_SECONDS,max(MINIMUM_SPACING_SECONDS,seconds))


def _inside_same_hold(left, right, holds):
    """True when both notes live inside one hold, or when one of them is it."""
    if any(_is_interior(left, h) and _is_interior(right, h) for h in holds):return True
    if left.kind == 'hold' and any(_is_interior(right, h) and left.event_id == h.event_id for h in holds):
        return True
    if right.kind == 'hold' and any(_is_interior(left, h) and right.event_id == h.event_id for h in holds):
        return True
    return False


def hold_edge_conflict(event, lane, holds, spacing=DEFAULT_SPACING_SECONDS):
    """A hold endpoint behaves like an ordinary note on its own lane.

    Replaces V4's fixed 500 ms tail guard. Only the ordinary floor applies, so
    notes may stay on the sustained lane right after the release instead of
    being pushed onto the opposite lane and read as a pile driver.
    """
    if lane not in event.lanes:return None
    for h in holds:
        if h.event_id == event.event_id or lane not in h.lanes:continue
        if abs(event.start - h.end) < spacing - 1e-8:return h
    return None


def conflict(a, b, lane, holds, config, all_events=None, four=None, spacing=DEFAULT_SPACING_SECONDS):
    if lane not in a.lanes or lane not in b.lanes or a.event_id == b.event_id:return False
    if abs(a.start - b.start) > 2.001:return False
    floor = spacing * (HOLD_SPACING_FACTOR if _inside_same_hold(a, b, holds) else 1)
    gap = abs(_exact_time(a.start, config) - _exact_time(b.start, config))
    return gap < floor - 1e-8


def violations(events, duration, config, spacing=DEFAULT_SPACING_SECONDS):
    result = structural_issues(events, duration)
    holds = [e for e in events if e.kind == 'hold']
    ordinary = [e for e in events if e.kind in ATTACKS | {'hold'}]
    for e in ordinary:
        for lane in e.lanes:
            if hold_edge_conflict(e, lane, holds, spacing):result.append(dict(code='hold_edge_gap', event_ids=[e.event_id], lane=lane))
    for lane in LANES:
        items = sorted((e for e in ordinary if lane in e.lanes), key=lambda e:(e.start, e.event_id))
        for i, a in enumerate(items):
            for b in items[i+1:]:
                if b.start - a.start > 2.001:break
                if conflict(a, b, lane, holds, config, ordinary, None, spacing):
                    result.append(dict(code='same_lane_gap', event_ids=[a.event_id, b.event_id], lane=lane))
    return result


def resolve(events, duration, config, spacing=DEFAULT_SPACING_SECONDS):
    out = list(events)
    edits, dropped, flipped = [], [], set()
    def change(event, blocked, reason):
        nonlocal out
        others = [e for e in out if e.event_id != event.event_id]
        def legal(candidate):
            proposal = others + [candidate]
            holds = [e for e in proposal if e.kind == 'hold']
            if any(hold_edge_conflict(candidate, lane, holds, spacing) for lane in candidate.lanes):return False
            if structural_issues([candidate], duration):return False
            if any(validate_events([candidate, e], duration) for e in others):return False
            return not any(conflict(candidate, e, lane, holds, config, proposal, None, spacing)
                           for e in others if e.kind in ATTACKS | {'hold'} and abs(e.start - candidate.start) <= 2.001
                           for lane in candidate.lanes)
        candidate = None
        if event.kind == 'gemini':
            remaining = tuple(lane for lane in LANES if lane not in blocked)
            if len(remaining) == 1:candidate = replace(event, kind='heavy', lanes=remaining, group_id=None)
        elif len(event.lanes) == 1 and event.event_id not in flipped:
            candidate = replace(event, lanes=('air' if event.lanes[0] == 'ground' else 'ground',), group_id=None)
        if candidate is not None and legal(candidate):
            out = others + [candidate];flipped.add(event.event_id)
            edits.append(dict(event_id=event.event_id, from_kind=event.kind, to_kind=candidate.kind,
                              from_lane=list(event.lanes), to_lane=list(candidate.lanes), reason=reason))
        else:
            out = others;dropped.append(dict(event_id=event.event_id, reason=reason + '_blocked'))
    # Each event can move once and then be removed; termination is monotone.
    for _ in range(len(out) * 2 + 1):
        holds = [e for e in out if e.kind == 'hold']
        ordinary = [e for e in out if e.kind in ATTACKS | {'hold'}]
        tail = next(((e, {lane for lane in e.lanes if hold_edge_conflict(e, lane, holds, spacing)})
                     for e in sorted(ordinary, key=lambda e:(-_priority(e), e.start, e.event_id))
                     if any(hold_edge_conflict(e, lane, holds, spacing) for lane in e.lanes)), None)
        if tail:
            change(tail[0], tail[1], 'hold_edge');continue
        # Also resolve structural conflicts introduced by snapping.
        ordered = sorted(out, key=lambda e:(e.start, e.event_id));pair = None
        for i, a in enumerate(ordered):
            for b in ordered[i+1:]:
                if b.start > max(a.end, a.start + 2.001):break
                blocked = {lane for lane in LANES if a.kind in ATTACKS | {'hold'} and
                           b.kind in ATTACKS | {'hold'} and conflict(a, b, lane, holds, config, ordinary, None, spacing)}
                structural = validate_events([a, b], duration)
                if blocked or structural:
                    pair = (a, b, blocked or set(a.lanes) & set(b.lanes) or set(LANES));break
            if pair:break
        if not pair:break
        a, b, blocked = pair
        loser = min((a, b), key=lambda e:(_priority(e), -e.start, e.event_id))
        change(loser, blocked, 'spacing')
    groups = defaultdict(list)
    for e in out:
        if e.group_id:groups[e.group_id].append(e)
    valid = {key for key, es in groups.items() if len(es) == 2 and
             {e.lanes for e in es} == {('ground',), ('air',)} and len({e.start for e in es}) == 1}
    out = [replace(e, group_id=None) if e.group_id and e.group_id not in valid else e for e in out]
    remaining = violations(out, duration, config, spacing)
    if remaining:raise ValueError(f'V5 unresolved rules: {remaining[:3]}')
    return sorted(out, key=lambda e:(e.start, e.event_id)), dict(
        edits=edits, dropped=dropped, same_lane_spacing_ms=round(spacing * 1000),
        hold_interior_spacing_ms=round(spacing * HOLD_SPACING_FACTOR * 1000))
