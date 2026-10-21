"""V4 BPM and no-BPM spacing adapter.

This adapter is deliberately separate from the V3 timing adapter.  BPM mode
uniformizes ordinary notes within each beat; no-BPM mode uses absolute gaps.
"""
from collections import defaultdict
from dataclasses import replace
from fractions import Fraction
import math
from functools import lru_cache

from .studio_timing import settings
from mdac2.data_preparation.global_postprocess_v4 import TAIL_GUARD_SECONDS
from mdac2.data_preparation.training_chart import validate_events

LANES = ('ground', 'air')
ATTACKS = frozenset(('light', 'heavy', 'extra_heavy', 'gemini'))


def _segments(config):
    return [dict(start_ms=-float('inf'), **{k: config[k] for k in ('bpm', 'offset_ms')})] + config.get('segments', [])


def _segment_for(t, segments):
    selected = 0
    for index, item in enumerate(segments[1:], 1):
        if t * 1000 >= item['start_ms']:
            selected = index
        else:
            break
    return selected


def _beat_data(t, config):
    segments = _segments(config)
    index = _segment_for(t, segments)
    segment = segments[index]
    return _beat_cached(t, index, segment['bpm'], segment['offset_ms'])


@lru_cache(maxsize=65536)
def _beat_cached(t, index, bpm, offset):
    step = Fraction(5) / Fraction(str(bpm))
    beat = step * 12
    origin = Fraction(str(offset)) / 1000
    tick = round((Fraction(str(t)) - origin) / step)
    beat_index, slot = divmod(tick, 12)
    return index, beat_index, slot, step, beat, origin



def _build_grid(duration, config):
    segments = _segments(config)
    grid = []
    for index, segment in enumerate(segments):
        lower = max(0, segment['start_ms'] / 1000)
        upper = min(duration, segments[index + 1]['start_ms'] / 1000) if index + 1 < len(segments) else duration
        if upper < lower:
            continue
        step = Fraction(60, 12) / Fraction(str(segment['bpm']))
        origin = Fraction(str(segment['offset_ms'])) / 1000
        low = math.ceil((Fraction(str(lower)) - origin) / step)
        high = math.floor((Fraction(str(upper)) - origin) / step)
        for slot in range(low, high + 1):
            exact = origin + slot * step
            t = round(float(exact) * 1000) / 1000
            if lower <= t <= duration and (t < upper or upper == duration):
                grid.append(t)
    return sorted(set(grid))


def _nearest(grid, t):
    if not grid:
        raise ValueError('BPM网格为空')
    return min(grid, key=lambda x: (abs(x - t), x))


def _is_interior(event, hold):
    return hold.kind == 'hold' and hold.start + 1e-8 < event.start < hold.end - 1e-8 and set(event.lanes) & set(hold.lanes)


def _tail_blocked(event, lane, holds):
    return any(h.event_id != event.event_id and lane in h.lanes and
               abs(event.start - h.end) <= TAIL_GUARD_SECONDS + 1e-8 for h in holds)


def _priority(event):
    if event.kind == 'gemini':
        return 6
    if event.kind == 'extra_heavy':
        return 5
    if event.kind == 'heavy':
        return 2
    if event.kind == 'hold':
        return 3
    return 1


def _group_slots(count):
    return {1: [0], 2: [0, 6], 3: [0, 4, 8], 4: [0, 3, 6, 9]}.get(count)


def _uniformize(events, duration, config):
    groups = defaultdict(list)
    details = {}
    for event in events:
        if event.kind not in ATTACKS | {'hold'}:
            continue
        segment, beat, slot, step, beat_step, origin = _beat_data(event.start, config)
        details[event.event_id] = (segment, beat, slot, step, beat_step, origin)
        for lane in event.lanes:
            groups[segment, beat, lane].append(event)
    choices = {}
    costs = {}
    dropped = set()
    reports = []
    for key, values in groups.items():
        unique = {event.event_id: event for event in values}
        ordered = sorted(unique.values(), key=lambda e: (-_priority(e), e.start, e.event_id))
        if len(ordered) > 4:
            keep = ordered[:4]
            for event in ordered[4:]:
                if event.kind != 'gemini' or sum(e.kind == 'gemini' for e in keep) >= 4:
                    dropped.add(event.event_id)
            ordered = [e for e in ordered if e.event_id not in dropped]
        slots = _group_slots(len(ordered))
        if slots is None:
            reports.append(dict(segment=key[0], beat=key[1], lane=key[2], action='drop_group'))
            dropped.update(e.event_id for e in ordered)
            continue
        reports.append(dict(segment=key[0], beat=key[1], lane=key[2], count=len(ordered), slots=slots))
        # Restore temporal order when assigning the evenly spaced slots.
        temporal = sorted(ordered, key=lambda e: (e.start, e.event_id))
        for event, slot in zip(temporal, slots):
            segment, beat, _, step, beat_step, origin = details[event.event_id]
            target = round(float(origin + (beat * beat_step) + slot * step) * 1000) / 1000
            cost = abs(target - event.start)
            if cost < costs.get(event.event_id, float('inf')):
                choices[event.event_id] = target
                costs[event.event_id] = cost
    out = []
    moved = 0
    grid = set(_build_grid(duration, config))
    for event in events:
        if event.event_id in dropped:
            continue
        target = choices.get(event.event_id, event.start)
        # Keep partial beats within the song and their own tempo segment.
        if target not in grid or _segment_for(target, _segments(config)) != details.get(event.event_id, (None,))[0]:
            target = event.start
        if event.kind == 'hold' and event.end - target < .03 - 1e-8:
            dropped.add(event.event_id)
            continue
        if abs(target - event.start) > 1e-9:
            moved += 1
        out.append(replace(event, start=target,
                           end=target if event.kind != 'hold' else event.end))
    return out, dict(enabled=True, positions_per_beat=12, moved=moved,
                     dropped=sorted(dropped), groups=reports)


def _reduce_four_chains(events, threshold):
    """Only over-limit four-chains; retain endpoints and one middle identity."""
    out = list(events)
    edits, used = [], set()
    for lane in LANES:
        candidates = sorted((e for e in out if e.kind in ATTACKS | {'hold'} and lane in e.lanes),
                            key=lambda e: (e.start, e.event_id))
        for index in range(len(candidates) - 3):
            chain = candidates[index:index + 4]
            if any(e.event_id in used for e in chain):continue
            gaps = [b.start-a.start for a,b in zip(chain,chain[1:])]
            if max(gaps) > threshold + 1e-8 or min(gaps) >= threshold - 1e-8:continue
            first, second, third, last = chain
            midpoint = round((first.start+last.start)*500)/1000
            middle = max((second, third), key=lambda e: (_priority(e), -e.start))
            removed = third if middle == second else second
            if middle.kind == 'hold' and middle.end-midpoint < .03-1e-8:continue
            out = [replace(e, start=midpoint, end=e.end if e.kind=='hold' else midpoint)
                   if e.event_id == middle.event_id else e for e in out if e.event_id != removed.event_id]
            used.update(e.event_id for e in chain)
            edits.append(dict(lane=lane, removed=[removed.event_id], moved=middle.event_id,
                              first=first.start, middle=midpoint, last=last.start))
    return out, edits


def _pair_threshold(left, right, holds, config):
    if config['enabled']:
        base = 4 * float(_beat_data(max(left.start, right.start), config)[3])
        interior = float(_beat_data(max(left.start, right.start), config)[4])
    else:
        base, interior = .15, .5
    if any(_is_interior(left, h) and _is_interior(right, h) for h in holds):
        return interior
    if left.kind == 'hold' and any(_is_interior(right, h) and left.event_id == h.event_id for h in holds):
        return interior
    if right.kind == 'hold' and any(_is_interior(left, h) and right.event_id == h.event_id for h in holds):
        return interior
    return base


def _exact_time(t, config):
    if not config['enabled']:return t
    _, beat, slot, step, _, origin = _beat_data(t, config)
    exact = float(origin+(beat*12+slot)*step)
    return exact if abs(t-exact) <= .000501 else t


def _four_groups(events, config):
    groups = defaultdict(list)
    if config['enabled']:
        for e in events:
            if e.kind in ATTACKS | {'hold'}:
                for lane in e.lanes:groups[(*_beat_data(e.start, config)[:2], lane)].append(e)
    return {key for key, members in groups.items() if len(members)==4 and
            {_beat_data(e.start,config)[2] for e in members}=={0,3,6,9}}


def _conflict(a, b, lane, holds, config, all_events=None, four=None):
    if lane not in a.lanes or lane not in b.lanes or a.event_id == b.event_id:return False
    if abs(a.start-b.start)>2.001:return False
    threshold = _pair_threshold(a, b, holds, config)
    gap = abs(_exact_time(a.start,config)-_exact_time(b.start,config))
    if config['enabled']:
        base = 4*float(_beat_data(max(a.start,b.start),config)[3])
        # Four-chains never relax the stricter hold-interior rule.
        if threshold <= base+1e-8:
            four = _four_groups(all_events or [a,b],config) if four is None else four
            left,right = sorted((a,b),key=lambda e:e.start)
            ad,bd = _beat_data(left.start,config),_beat_data(right.start,config)
            if (*ad[:2],lane) in four and (ad[:2]==bd[:2] or
                    ad[0]==bd[0] and bd[1]==ad[1]+1 and ad[2]==9 and bd[2]==0):
                threshold = 3*float(ad[3])
    return gap < threshold-1e-8


def violations(events, duration, config):
    """Final invariant shared by resolution and the read-only gate."""
    result = structural_issues(events,duration)
    holds = [e for e in events if e.kind=='hold']
    ordinary = [e for e in events if e.kind in ATTACKS | {'hold'}]
    four = _four_groups(ordinary,config)
    for e in ordinary:
        for lane in e.lanes:
            if _tail_blocked(e,lane,holds):result.append(dict(code='hold_tail_guard',event_ids=[e.event_id],lane=lane))
    for lane in LANES:
        items = sorted((e for e in ordinary if lane in e.lanes),key=lambda e:(e.start,e.event_id))
        for i,a in enumerate(items):
            for b in items[i+1:]:
                if b.start-a.start>2.001:break
                if _conflict(a,b,lane,holds,config,ordinary,four):
                    result.append(dict(code='same_lane_gap',event_ids=[a.event_id,b.event_id],lane=lane))
    return result


def _resolve(events, duration, config):
    out = list(events)
    edits, dropped, flipped = [], [], set()
    def change(event, blocked, reason):
        nonlocal out
        others = [e for e in out if e.event_id!=event.event_id]
        def legal(candidate):
            proposal = others+[candidate]
            holds = [e for e in proposal if e.kind=='hold']
            if any(_tail_blocked(candidate,lane,holds) for lane in candidate.lanes):return False
            if structural_issues([candidate],duration):return False
            if any(validate_events([candidate,e],duration) for e in others):return False
            four = _four_groups(proposal,config)
            return not any(_conflict(candidate,e,lane,holds,config,proposal,four)
                           for e in others if e.kind in ATTACKS | {'hold'} and abs(e.start-candidate.start)<=2.001
                           for lane in candidate.lanes)
        candidate = None
        if event.kind=='gemini':
            remaining = tuple(lane for lane in LANES if lane not in blocked)
            if len(remaining)==1:candidate = replace(event,kind='heavy',lanes=remaining,group_id=None)
        elif len(event.lanes)==1 and event.event_id not in flipped:
            candidate = replace(event,lanes=('air' if event.lanes[0]=='ground' else 'ground',),group_id=None)
        if candidate is not None and legal(candidate):
            out = others+[candidate];flipped.add(event.event_id)
            edits.append(dict(event_id=event.event_id,from_kind=event.kind,to_kind=candidate.kind,
                              from_lane=list(event.lanes),to_lane=list(candidate.lanes),reason=reason))
        else:
            out = others;dropped.append(dict(event_id=event.event_id,reason=reason+'_blocked'))
    # Each event can move once and then be removed; termination is monotone.
    for _ in range(len(out)*2+1):
        holds = [e for e in out if e.kind=='hold']
        ordinary = [e for e in out if e.kind in ATTACKS | {'hold'}]
        tail = next(((e,{lane for lane in e.lanes if _tail_blocked(e,lane,holds)})
                     for e in sorted(ordinary,key=lambda e:(-_priority(e),e.start,e.event_id))
                     if any(_tail_blocked(e,lane,holds) for lane in e.lanes)),None)
        if tail:
            change(tail[0],tail[1],'tail_guard');continue
        # Also resolve structural conflicts introduced by snapping.
        four = _four_groups(ordinary,config)
        ordered = sorted(out,key=lambda e:(e.start,e.event_id));pair = None
        for i,a in enumerate(ordered):
            for b in ordered[i+1:]:
                if b.start>max(a.end,a.start+2.001):break
                blocked = {lane for lane in LANES if a.kind in ATTACKS | {'hold'} and
                           b.kind in ATTACKS | {'hold'} and _conflict(a,b,lane,holds,config,ordinary,four)}
                structural = validate_events([a,b],duration)
                if blocked or structural:
                    pair=(a,b,blocked or set(a.lanes)&set(b.lanes) or set(LANES));break
            if pair:break
        if not pair:break
        a,b,blocked=pair
        loser = min((a,b),key=lambda e:(_priority(e),-e.start,e.event_id))
        change(loser,blocked,'spacing')
    groups=defaultdict(list)
    for e in out:
        if e.group_id:groups[e.group_id].append(e)
    valid={key for key,es in groups.items() if len(es)==2 and
           {e.lanes for e in es}=={('ground',),('air',)} and len({e.start for e in es})==1}
    out=[replace(e,group_id=None) if e.group_id and e.group_id not in valid else e for e in out]
    remaining=violations(out,duration,config)
    if remaining:raise ValueError(f'V4 unresolved rules: {remaining[:3]}')
    return sorted(out,key=lambda e:(e.start,e.event_id)),dict(edits=edits,dropped=dropped)


def structural_issues(events, duration):
    return validate_events(events, duration) + [
        dict(code='hold_under_30ms', event_ids=[e.event_id])
        for e in events if e.kind == 'hold' and e.end-e.start < .03-1e-8
    ]


def apply(events, duration, value=None):
    config = settings(value)
    out = [replace(e,start=round(e.start*1000)/1000,end=round(e.end*1000)/1000)
           for e in events if e.kind!='hold' or round(e.end*1000)-round(e.start*1000)>=30]
    grid_report = dict(enabled=config['enabled'], policy='v4-no-bpm-v1', settings=config,
                       storage_ms=1, dropped=[], moved=0, uniformization=None,
                       conflict_resolution=None)
    if config['enabled']:
        grid = _build_grid(duration, config)
        if not grid:
            raise ValueError('音频范围内没有 BPM 网格点')
        snapped = []
        for event in out:
            start, end = _nearest(grid, event.start), _nearest(grid, event.end)
            if event.kind == 'hold' and end-start < .03-1e-8:
                grid_report['dropped'].append(dict(event_id=event.event_id, reason='collapsed_hold'))
                continue
            snapped.append(replace(event, start=start, end=end))
        out, uniform = _uniformize(snapped, duration, config)
        grid_report.update(policy='v4-bpm-12th-v1', uniformization=uniform)
    else:
        out, reduction = _reduce_four_chains(out, .15)
        grid_report['four_chain_reduction'] = reduction
    out, conflicts = _resolve(out, duration, config)
    grid_report['conflict_resolution'] = conflicts
    grid_report['after'] = len(out)
    grid_report['stage_order'] = ['bpm_snap' if config['enabled'] else 'absolute_ms',
                                  'same_beat_uniformization' if config['enabled'] else 'four_chain_reduction',
                                  'ordinary_spacing', 'hold_interior_spacing',
                                  'tail_guard', 'validation']
    if structural_issues(out, duration):
        raise ValueError('V4 timing rules left structural conflicts')
    return out, grid_report
