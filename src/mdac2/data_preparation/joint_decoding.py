"""Stitch supervised centers first, then decode full-song events and rules."""
from bisect import bisect_left, bisect_right, insort
import math
import numpy as np

from .candidate_rules import select_candidates
from .training_chart import CLASSES, LANES, TrainingEvent, validate_events
from .training_tensors import CONTEXT_FRAMES, CORE_FRAMES


DEFAULT_DECODER = dict(thresholds=[.5] * 6, tail_threshold=.5, occupancy_threshold=.35,
                       nms_frames=3, pair_tolerance_seconds=.02, precision_seconds=.001)


def select_song_candidates(candidates, duration):
    """Same pairwise rules and score ordering, with time-indexed conflicts."""
    if len({e.event_id for e,_ in candidates}) != len(candidates):
        raise ValueError('Duplicate candidate id')
    if any(not math.isfinite(s) or not 0 <= s <= 1 for _,s in candidates):
        raise ValueError('Finite probability scores required')
    instants = {}
    restrictive_times = []
    restrictive = {}
    holds = []
    accepted, rejected = [], []
    for event,score in sorted(candidates,key=lambda pair:(-pair[1],pair[0].event_id)):
        possible = {e.event_id:e for e in instants.get(event.start,[])}
        for other in holds:
            intersects = (other.start <= event.start < other.end or
                          (event.kind == 'hold' and event.start <= other.start < event.end))
            if intersects:
                possible[other.event_id] = other
        if event.kind == 'hold':
            first = bisect_left(restrictive_times,event.start)
            last = bisect_left(restrictive_times,event.end)
            for t in restrictive_times[first:last]:
                for other in restrictive[t]:
                    possible[other.event_id] = other
        conflicts = validate_events([event],duration)
        for other in possible.values():
            conflicts.extend(validate_events([other,event],duration))
        if conflicts:
            rejected.append(dict(event_id=event.event_id,score=score,issues=conflicts))
            continue
        accepted.append(event)
        if event.kind == 'hold':
            holds.append(event)
        else:
            instants.setdefault(event.start,[]).append(event)
            if event.exclusive or event.kind == 'gear':
                if event.start not in restrictive:
                    insort(restrictive_times,event.start)
                    restrictive[event.start] = []
                restrictive[event.start].append(event)
    return sorted(accepted,key=lambda e:(e.start,e.event_id)),rejected


def place_core(destination, coverage, start, values):
    """One and only one owner per frame; neither average nor crop hold events."""
    first = max(0, start + CONTEXT_FRAMES)
    last = min(len(coverage), start + CONTEXT_FRAMES + CORE_FRAMES)
    if np.any(coverage[first:last]):
        raise ValueError('Duplicate supervised center ownership')
    for name in destination:
        destination[name][first:last] = values[name][first-start:last-start]
    coverage[first:last] += 1


def peaks(probability, threshold, radius=3):
    p = np.asarray(probability)
    candidates = np.flatnonzero((p >= threshold) &
                               (p >= np.r_[-1., p[:-1]]) & (p >= np.r_[p[1:], -1.]))
    blocked = np.zeros(len(p), bool)
    selected = []
    for index in sorted(candidates, key=lambda i: (-float(p[i]), int(i))):
        if not blocked[index]:
            selected.append(int(index))
            blocked[max(0, index-radius):index+radius+1] = True
    return sorted(selected)


def event_time(frame, offset, duration):
    # Explicit versioned 1ms export quantization. The network's unrounded
    # offsets remain in the prediction cache for error analysis.
    raw = (frame + float(np.clip(offset, 0, .999999))) * .01
    return max(0., min(np.floor(duration * 1000) / 1000, round(raw, 3)))


def pair_points(left, right, tolerance):
    """Monotonic maximum-cardinality matching within the timing tolerance."""
    pairs = []
    l = r = 0
    while l < len(left) and r < len(right):
        delta = left[l][0] - right[r][0]
        if abs(delta) <= tolerance + 1e-9:
            pairs.append((l, r))
            l += 1
            r += 1
        elif delta < 0:
            l += 1
        else:
            r += 1
    return pairs


def decode_song(prediction, duration, config=None):
    cfg = {**DEFAULT_DECODER, **(config or {})}
    if len(cfg['thresholds']) != 6 or any(not 0 < v < 1 for v in cfg['thresholds']):
        raise ValueError('Six probability thresholds required')
    if cfg['precision_seconds'] != .001:
        raise ValueError('Experimental BMS requires explicit millisecond precision')
    for name, value in prediction.items():
        if not np.isfinite(value).all():
            raise ValueError(f'Nonfinite predictions: {name}')
    onsets = prediction['onsets']
    offsets = prediction['offsets']
    ends = prediction['hold_ends']
    end_offsets = prediction['hold_end_offsets']
    occupancy = prediction['occupancy']
    points = {}
    rejected = []
    for lane in range(2):
        for kind, threshold in enumerate(cfg['thresholds']):
            points[lane, kind] = [(event_time(f, offsets[f, lane, kind], duration), float(onsets[f, lane, kind]), f)
                                 for f in peaks(onsets[:, lane, kind], threshold, cfg['nms_frames'])]
    candidates = []
    serial = 0
    def add(kind, lanes, start, end, score, group=None):
        nonlocal serial
        serial += 1
        identifier = f'pred-{serial:07d}'
        event = TrainingEvent(identifier, kind, tuple(lanes), start, end,
                              (identifier,), 'model', 'joint-tcn-v1', kind == 'extra_heavy', group)
        candidates.append((event, float(np.clip(score, 0, 1))))
    # Gemini is one paired logical event. A one-sided Gemini cannot be
    # reinterpreted as a heavy attack merely to make export succeed.
    gemini_kind = CLASSES.index('gemini')
    left, right = points[0, gemini_kind], points[1, gemini_kind]
    pairs = pair_points(left, right, cfg['pair_tolerance_seconds'])
    used_left, used_right = set(), set()
    for l, r in pairs:
        used_left.add(l)
        used_right.add(r)
        start = round((left[l][0] + right[r][0]) / 2, 3)
        add('gemini', LANES, start, start, min(left[l][1], right[r][1]))
    for lane, group, used in ((0, left, used_left), (1, right, used_right)):
        for i, point in enumerate(group):
            if i not in used:
                rejected.append(dict(code='unpaired_gemini', lane=LANES[lane], start=point[0], score=point[1]))
    # Both-lane heavy attacks are TWO actual hits sharing a group. Their kind
    # remains heavy, and is never promoted to Gemini.
    heavy_kind = CLASSES.index('heavy')
    left, right = points[0, heavy_kind], points[1, heavy_kind]
    pairs = pair_points(left, right, cfg['pair_tolerance_seconds'])
    heavy_paired = {}
    for number, (l, r) in enumerate(pairs):
        start = round((left[l][0] + right[r][0]) / 2, 3)
        for lane, index in ((0, l), (1, r)):
            heavy_paired[lane, index] = (start, f'double-heavy-{number}')
    for kind, name in enumerate(CLASSES):
        if name in ('gemini', 'hold'):
            continue
        for lane in range(2):
            for index, (start, score, frame) in enumerate(points[lane, kind]):
                group = None
                if name == 'heavy' and (lane, index) in heavy_paired:
                    start, group = heavy_paired[lane, index]
                add(name, (LANES[lane],), start, start, score, group)
    # Whole-song pairing has no window boundary or maximum-duration cutoff.
    # Ends and occupancy must agree. No fabricated tail from the window edge.
    holds = [[], []]
    for lane in range(2):
        tails = [(event_time(f, end_offsets[f, lane], duration), float(ends[f, lane]), f)
                 for f in peaks(ends[:, lane], cfg['tail_threshold'], cfg['nms_frames'])]
        used = set()
        starts = points[lane, CLASSES.index('hold')]
        for i, (start, score, head_frame) in enumerate(starts):
            next_head = starts[i+1][0] if i+1 < len(starts) else duration + .001
            tail_index = bisect_left([t[0] for t in tails], start)
            chosen = None
            for ti in range(tail_index, len(tails)):
                end, tail_score, tail_frame = tails[ti]
                if end > next_head:
                    break
                if ti in used or end <= start:
                    continue
                first = int(np.ceil(start * 100 - 1e-7))
                last = int(np.ceil(end * 100 - 1e-7))
                support = float(occupancy[first:last, lane].mean()) if last > first else 1.
                if support >= cfg['occupancy_threshold']:
                    chosen = (end, min(score, tail_score), ti, support)
                    break
            if chosen is None:
                rejected.append(dict(code='unpaired_hold_head', lane=LANES[lane], start=start, score=score))
            else:
                end, joint_score, ti, support = chosen
                used.add(ti)
                holds[lane].append((start, end, joint_score))
        for i, (end, score, _) in enumerate(tails):
            if i not in used:
                rejected.append(dict(code='orphan_hold_tail', lane=LANES[lane], end=end, score=score))
    # Preserve a both-lane sustained object when head AND tail agree. Retain
    # independent lane holds when they don't, without hiding this difference.
    pairs = pair_points(holds[0], holds[1], cfg['pair_tolerance_seconds'])
    used = [set(), set()]
    for l, r in pairs:
        left, right = holds[0][l], holds[1][r]
        if abs(left[1] - right[1]) > cfg['pair_tolerance_seconds']:
            continue
        used[0].add(l)
        used[1].add(r)
        start, end = round((left[0]+right[0])/2, 3), round((left[1]+right[1])/2, 3)
        if end > start:
            add('hold', LANES, start, end, min(left[2], right[2]), f'dual-hold-{l}-{r}')
    for lane in range(2):
        for i, (start, end, score) in enumerate(holds[lane]):
            if i not in used[lane]:
                add('hold', (LANES[lane],), start, end, score)
    accepted, rule_rejected = select_song_candidates(candidates, duration)
    rejected.extend(dict(code='rule_conflict', **r) for r in rule_rejected)
    return accepted, dict(candidate_count=len(candidates), accepted_count=len(accepted),
                          rejected=rejected, rule_rejection_count=len(rule_rejected),
                          cross_window_holds=sum(int(e.start//16 != e.end//16) for e in accepted if e.kind == 'hold'),
                          decoder_config=cfg, time_quantization_seconds=.001)
