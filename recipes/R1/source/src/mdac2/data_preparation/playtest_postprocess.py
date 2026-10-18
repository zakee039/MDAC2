"""Versioned playtest editing, separate from frozen model weights and training.

Absolute seconds throughout. Stage order is density, phrasing, tail extension,
then matched repeat arrangement. No beat grid, reference-chart input at inference,
global lane quota, automatic gear, or unconditional gap filling.
"""
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, replace
import math

import numpy as np
from scipy.ndimage import uniform_filter1d

from .training_chart import TrainingEvent
from .generated_constraints import validate_generated_events


@dataclass(frozen=True)
class Policy:
    version: str = 'playtest-postprocess-r1.1'
    phrase_gap: float = .65
    prior_weight: float = 1.
    flip_cost: float = .7
    alternation_cost: float = .65
    tail_extension: float = .25
    climax_min_seconds: float = 6.
    candidate_threshold: float = .20
    candidate_spacing: float = .10
    added_fraction: float = .15
    similarity_threshold: float = .82
    repeat_match_seconds: float = .06
    repeat_coverage: float = .8
    max_seconds: float = 600.


def checked(events, duration):
    issues = validate_generated_events(events, duration)
    if issues:
        raise ValueError(f'Invalid generated chart: {issues[:3]}')


def phrases(events, policy=Policy()):
    """Only ungrouped, isolated single clicks; hold passages are protected."""
    cells = defaultdict(list)
    holds = [e for e in events if e.kind == 'hold']
    for e in events:
        cells[e.start].append(e)
    result, current = [], []
    for t, items in sorted(cells.items()):
        e = items[0]
        eligible = (len(items) == 1 and e.kind in ('light', 'heavy') and
                    len(e.lanes) == 1 and not e.group_id and
                    not any(h.start <= t <= h.end + .02 for h in holds))
        if not eligible or (current and t - current[-1].start > policy.phrase_gap):
            if current:
                result.append(current)
            current = []
        if eligible:
            current.append(e)
    if current:
        result.append(current)
    return result


def run_lengths(sequence):
    runs = []
    for e in sequence:
        if runs and runs[-1][0] == e.lanes[0]:
            runs[-1][1] += 1
        else:
            runs.append([e.lanes[0], 1])
    return [r[1] for r in runs]


def aligned_tail_ids(events):
    attack_lanes = defaultdict(set)
    for e in events:
        if e.kind in ('light', 'heavy', 'gemini'):
            attack_lanes[e.start].update(e.lanes)
    return {h.event_id for h in events if h.kind == 'hold' and
            any(abs(t-h.end) < .0011 and set(h.lanes) <= lanes
                for t, lanes in attack_lanes.items())}


def structure_stats(events, policy=Policy()):
    groups = phrases(events, policy)
    runs = Counter(n for seq in groups for n in run_lengths(seq))
    pairs = switches = long_alternations = 0
    for seq in groups:
        streak = 0
        for a, b in zip(seq, seq[1:]):
            pairs += 1
            switch = a.lanes != b.lanes
            switches += switch
            streak = streak + 1 if switch else 0
            long_alternations += streak >= 4
    tails = [e for e in events if e.kind == 'hold']
    aligned = len(aligned_tail_ids(events))
    return dict(eligible_single_clicks=sum(map(len, groups)), adjacent_pairs=pairs,
                switches=switches, switch_fraction=switches/max(1, pairs),
                alternating_four_or_more_transitions=long_alternations,
                same_lane_run_histogram=dict(sorted(runs.items())),
                three_runs=runs[3], four_runs=runs[4], longest_run=max(runs, default=0),
                holds=len(tails), aligned_hold_tails=aligned,
                logical_events=len(events), classes=dict(Counter(e.kind for e in events)))


def fit_run_prior(charts, policy=Policy()):
    """Empirical stay/switch hazards, trained only on explicitly supplied train rows.

    Ends at rests/special objects are censored, not treated as a switch. Gap bins
    are elapsed seconds, not inferred tempo. Symmetric lanes share statistics.
    """
    counts = np.zeros((4, 16, 2), np.int64)
    histogram = Counter()
    for events in charts:
        for seq in phrases(events, policy):
            histogram.update(run_lengths(seq))
            length = 1
            for a, b in zip(seq, seq[1:]):
                gap_bin = int(np.searchsorted([.15, .30, .45], b.start-a.start))
                switch = a.lanes != b.lanes
                counts[gap_bin, min(length, 16)-1, int(switch)] += 1
                length = 1 if switch else length+1
    # Shrink sparse gap/length cells to a global empirical hazard; Jeffreys prior.
    global_counts = counts.sum(0)
    global_hazard = (global_counts[:, 1]+.5)/(global_counts.sum(1)+1)
    hazard = (counts[:, :, 1]+8*global_hazard)/(counts.sum(2)+8)
    return dict(version='training-run-hazard-1', counts=counts.tolist(),
                switch_hazard=np.clip(hazard, .05, .95).tolist(),
                run_histogram=dict(sorted(histogram.items())),
                observations=int(counts.sum()), policy=asdict(policy))


def arrange_phrases(events, duration, prior, policy=Policy()):
    """Viterbi over side, current run length, and consecutive switching history."""
    hazards = np.asarray(prior['switch_hazard'])
    if hazards.shape != (4, 16) or not np.isfinite(hazards).all() or np.any((hazards <= 0)|(hazards >= 1)):
        raise ValueError('Invalid training run prior')
    changes = {}
    for seq in phrases(events, policy):
        if len(seq) < 3:
            continue
        # Preserve already-present triples/quads and their boundaries. Repairing
        # an alternating passage must not dismantle a useful existing phrase.
        protected = set()
        run_start = 0
        for j in range(1, len(seq)+1):
            if j == len(seq) or seq[j].lanes != seq[run_start].lanes:
                if j-run_start in (3, 4):
                    protected.update(range(max(0, run_start-1), min(len(seq), j+1)))
                run_start = j
        # State graph is bounded (2 * 16 * 9); no fixed three/four-note template.
        first_side = int(seq[0].lanes[0] == 'air')
        costs = {(side, 1, 0): policy.flip_cost*(side != first_side) for side in (0, 1)
                 if 0 not in protected or side == first_side}
        back = []
        for i in range(1, len(seq)):
            gap_bin = int(np.searchsorted([.15, .30, .45], seq[i].start-seq[i-1].start))
            original = int(seq[i].lanes[0] == 'air')
            next_costs, parents = {}, {}
            for state, cost in costs.items():
                side, run, alternate = state
                hazard = float(hazards[gap_bin, run-1])
                for switch in (False, True):
                    target = 1-side if switch else side
                    if i in protected and target != original:
                        continue
                    streak = min(8, alternate+1) if switch else 0
                    new = (target, 1 if switch else min(16, run+1), streak)
                    transition = -math.log(hazard if switch else 1-hazard)
                    fatigue = policy.alternation_cost*max(0, streak-3) if switch else 0.
                    # Bound the overflow state without dictating the length of ordinary runs.
                    endurance = .15*max(0, run-7) if not switch else 0.
                    value = cost+policy.prior_weight*transition+fatigue+endurance+policy.flip_cost*(target != original)
                    if value < next_costs.get(new, math.inf):
                        next_costs[new], parents[new] = value, state
            costs = next_costs
            back.append(parents)
        state = min(costs, key=costs.get)
        sides = [state[0]]
        for parents in reversed(back):
            state = parents[state]
            sides.append(state[0])
        for e, side in zip(seq, reversed(sides)):
            lane = ('ground', 'air')[side]
            if e.lanes != (lane,):
                changes[e.event_id] = replace(e, lanes=(lane,))
    out = [changes.get(e.event_id, e) for e in events]
    # An opposite-lane close pair may become a same-lane collision. Reject that
    # phrase's edits rather than deleting/retiming any click or changing its type.
    issues = validate_generated_events(out, duration)
    if issues:
        bad = {eid for issue in issues for eid in issue['event_ids']}
        for seq in phrases(events, policy):
            if any(e.event_id in bad for e in seq):
                for e in seq:
                    changes.pop(e.event_id, None)
        out = [changes.get(e.event_id, e) for e in events]
    checked(out, duration)
    return out, dict(changed=len(changes), edits=[dict(event_id=e.event_id, before=list(e.lanes),
                after=list(changes[e.event_id].lanes)) for e in events if e.event_id in changes],
                method='training-only run hazard + bounded alternation fatigue; no lane quota')


def extend_tails(events, duration, policy=Policy()):
    out = list(events)
    edits, rejected = [], []
    attacks = [e for e in events if e.kind in ('light', 'heavy', 'gemini')]
    already_aligned = aligned_tail_ids(events)
    processed = set()
    for i, h in enumerate(out):
        if h.kind != 'hold':
            continue
        if h.event_id in processed:
            continue
        group = [(j, e) for j, e in enumerate(out) if e.kind == 'hold' and
                 ((h.group_id and e.group_id == h.group_id and e.start == h.start and e.end == h.end)
                  or e.event_id == h.event_id)]
        processed.update(e.event_id for _, e in group)
        if any(e.event_id in already_aligned for _, e in group):
            continue
        lanes = {lane for _, e in group for lane in e.lanes}
        attack_lanes = defaultdict(set)
        for a in attacks:
            attack_lanes[a.start].update(a.lanes)
        candidates = sorted({t for t, alanes in attack_lanes.items() if lanes <= alanes
                             and .001 < t-h.end <= policy.tail_extension+1e-9})
        if not candidates:
            continue
        target = candidates[0]
        # Only the nearby final gap can be closed. Never extend through another
        # hold/exclusive object, or move a click/start, or shorten an existing hold.
        proposal = list(out)
        for j, member in group:
            proposal[j] = replace(member, end=target)
        if validate_generated_events(proposal, duration):
            rejected.append(dict(event_id=h.event_id, target=target, reason='occupancy'))
            continue
        out = proposal
        edits.extend(dict(event_id=member.event_id, before=member.end, after=target,
                          extension=target-member.end) for _, member in group)
    checked(out, duration)
    return out, dict(extended=len(edits), edits=edits, rejected=rejected,
                     maximum_extension_seconds=policy.tail_extension)


def robust_unit(x):
    lo, hi = np.quantile(x, [.1, .9])
    return np.clip((x-lo)/max(1e-6, hi-lo), 0, 1)


def analyze_audio(features, duration, policy=Policy()):
    """Heuristic high-intensity sections, not a trained semantic chorus detector."""
    if features.ndim != 3 or features.shape[:2] != (3, 96) or not 0 < duration <= policy.max_seconds:
        raise ValueError('Bounded 10ms three-stream features required')
    n = features.shape[-1]
    if abs(n-duration*100) > 2:
        raise ValueError('Feature duration mismatch')
    mix = np.asarray(features[0], np.float32)
    if not np.isfinite(mix).all():
        raise ValueError('Nonfinite audio features')
    energy = np.log1p(np.maximum(0, np.expm1(np.clip(mix, 0, 30))).mean(0))
    rise = np.maximum(0, np.diff(mix, axis=1, prepend=mix[:, :1])).mean(0)
    # Sustained intensity + change activity; constant loud tones cannot qualify.
    smooth_energy = uniform_filter1d(energy, 400, mode='nearest')
    smooth_rise = uniform_filter1d(rise, 400, mode='nearest')
    score = .55*robust_unit(smooth_energy)+.45*robust_unit(smooth_rise)
    floor = max(float(np.max(energy))*.08, .03)
    active = (smooth_energy > floor)&(smooth_rise > .008)
    high = (score >= .72)&active
    # A relative rank alone must not call every second of uniform audio a climax.
    spans, start = [], None
    for i, flag in enumerate(np.r_[high, False]):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            if (i-start)*.01 >= policy.climax_min_seconds:
                spans.append(dict(start=round(start*.01, 3), end=min(duration, round(i*.01, 3)),
                                  score=float(np.mean(score[start:i]))))
            start = None
    # Only a bounded high-intensity subset can receive new candidate notes.
    chosen = []
    for s in sorted(spans, key=lambda s: -s['score']):
        if sum(t['end']-t['start'] for t in chosen)+s['end']-s['start'] <= duration*.4:
            chosen.append(s)
    return dict(sections=sorted(chosen, key=lambda s: s['start']), rise=rise,
                intensity=score, energy=energy,
                method='mix sustained energy + positive spectral change; heuristic, not semantic ground truth')


def add_climax_candidates(events, prediction, analysis, duration, policy=Policy()):
    from .joint_decoding import peaks
    probability = np.asarray(prediction['probability'])
    offsets = np.asarray(prediction['offsets'])
    if probability.shape != analysis['rise'].shape or offsets.shape != probability.shape:
        raise ValueError('Onset/cache frame identity mismatch')
    if not np.isfinite(probability).all() or not np.isfinite(offsets).all():
        raise ValueError('Nonfinite onset candidates')
    points = [(round(float((i+offsets[i])*.01), 3), i) for i in peaks(probability, policy.candidate_threshold, 3)]
    out, edits, rejected = list(events), [], []
    original_times = sorted(set(e.start for e in events))
    for section in analysis['sections']:
        left, right = section['start'], section['end']
        baseline_count = sum(left <= t < right for t in original_times)
        budget = min(int(baseline_count*policy.added_fraction), int((right-left)/2))
        local_rise = analysis['rise'][int(left*100):int(right*100)]
        floor = max(.025, float(np.quantile(local_rise, .75)))
        candidates = [(float(probability[i]), t, i) for t, i in points
                      if left <= t < right and 0 <= t < duration and analysis['rise'][i] >= floor]
        used = 0
        for probability_value, t, i in sorted(candidates, key=lambda x: (-x[0], x[1])):
            if used >= budget:
                break
            if any(abs(e.start-t) < policy.candidate_spacing for e in out):
                continue
            if any(abs(e['seconds']-t) < 1. for e in edits):
                continue
            if sum(abs(e.start-t) <= .5 for e in out) >= 8:
                continue
            if any(e.kind == 'hold' and e.start <= t <= e.end+.02 for e in out):
                continue
            lane = min(('ground', 'air'), key=lambda lane: sum(lane in e.lanes and abs(e.start-t) < 1 for e in out))
            event = TrainingEvent(f'climax-{i}', 'light', (lane,), t, t, (f'onset-frame-{i}',),
                                  policy.version, 'climax-candidate')
            if validate_generated_events(out+[event], duration):
                rejected.append(dict(seconds=t, reason='occupancy'))
                continue
            out.append(event)
            used += 1
            edits.append(dict(event_id=event.event_id, seconds=t, frame=i, probability=probability_value,
                              spectral_rise=float(analysis['rise'][i]), section=[left, right]))
    out.sort(key=lambda e: (e.start, e.event_id))
    checked(out, duration)
    return out, dict(added=len(edits), edits=edits, rejected=rejected, sections=analysis['sections'],
                     detector=analysis['method'], model_threshold_unchanged=.3,
                     postprocess_candidate_threshold=policy.candidate_threshold,
                     no_candidate_no_addition=True)


def repeat_candidates(features, duration, policy=Policy()):
    """Bounded absolute-time diagonal recurrence; no timing/beat prerequisite.

    Eight/twelve/sixteen-second analysis windows are search scales, never note
    grids. Only near-unwarped repeats are accepted in this first implementation.
    """
    hop = .25
    n = features.shape[-1]//25
    if n < 64 or duration > policy.max_seconds:
        return []
    matrices, silent_streams = [], []
    for stream in (0, 1):
        f = np.asarray(features[stream, :, :n*25], np.float32).reshape(96, n, 25).mean(2).T
        silent_streams.append(np.max(f, axis=1) < .05)
        f = (f-f.mean(0))/np.maximum(.5, f.std(0))
        norm = np.linalg.norm(f, axis=1)
        unit = f/np.maximum(norm[:, None], 1e-6)
        sim = unit@unit.T
        sim[norm < 1., :] = -1
        sim[:, norm < 1.] = -1
        matrices.append(sim)
    # Original music drives matching; vocal disagreement rejects similar backing
    # with changed lyrics/phrasing. Inactive vocals are handled conservatively.
    combined = matrices[0]*.8+matrices[1]*.2
    both_no_vocals = silent_streams[1][:, None]&silent_streams[1][None, :]
    combined = np.where(both_no_vocals, matrices[0], combined)
    proposals = []
    for length in (64, 48, 32):
        for offset in range(length+8, n-length):
            diagonal = np.diagonal(combined, offset=offset)
            if len(diagonal) < length:
                continue
            cs = np.r_[0., np.cumsum(diagonal, dtype=np.float64)]
            scores = (cs[length:]-cs[:-length])/length
            for i in np.flatnonzero(scores >= policy.similarity_threshold)[::4]:
                j = i+offset
                mix_score = float(np.mean(matrices[0][np.arange(i, i+length), np.arange(j, j+length)]))
                if mix_score < policy.similarity_threshold:
                    continue
                proposals.append(dict(source=[round(i*hop, 3), round((i+length)*hop, 3)],
                                      target=[round(j*hop, 3), round((j+length)*hop, 3)],
                                      score=float(scores[i]), mix_score=mix_score, offset=offset*hop))
    proposals.sort(key=lambda p: (-(p['source'][1]-p['source'][0]), -p['score'], p['source'][0]))
    kept = []
    overlap = lambda a, b: max(a[0], b[0]) < min(a[1], b[1])
    for p in proposals:
        if any(overlap(p['target'], q['target']) or overlap(p['source'], q['target']) or
               overlap(p['target'], q['source']) for q in kept):
            continue
        kept.append(p)
        if len(kept) == 12:
            break
    return kept


def apply_repeats(events, proposals, duration, policy=Policy()):
    """Last musical edit: matched same-type events inherit mirrored source lanes.

    Base snapshot is fixed. Target times, types, hold duration, double-heavy
    groups and unmatched events are preserved. Whole proposal is rejected if
    endpoints/groups/occupancy or onset-coverage checks fail.
    """
    snapshot, out, reports = list(events), list(events), []
    for p in proposals:
        source = [e for e in snapshot if p['source'][0] <= e.start < p['source'][1]]
        target = [e for e in snapshot if p['target'][0] <= e.start < p['target'][1]]
        report = dict(**p, applied=False, reason='insufficient_matching')
        reports.append(report)
        if min(len(source), len(target)) < 8:
            continue
        crossed = any(e.kind == 'hold' and e.start < bound < e.end
                      for e in snapshot for bound in [*p['source'], *p['target']])
        if crossed:
            report['reason'] = 'hold_crosses_boundary'
            continue
        # Refine a constant offset using onset correspondence, bounded to 150ms.
        st = np.array(sorted(set(e.start for e in source)))
        tt = np.array(sorted(set(e.start for e in target)))
        options = []
        for offset in p['offset']+np.arange(-.15, .15001, .01):
            distances = np.abs(st[:, None]+offset-tt[None, :])
            nearest = distances.min(1)
            options.append((int(np.sum(nearest <= policy.repeat_match_seconds)),
                            -float(np.minimum(nearest, .2).mean()), float(offset)))
        count, _, offset = max(options)
        if count/max(len(st), len(tt)) < policy.repeat_coverage:
            continue
        edges = sorted((abs(a.start+offset-b.start), a.event_id, b.event_id, a, b)
                       for a in source for b in target
                       if a.kind == b.kind and len(a.lanes) == len(b.lanes)
                       and bool(a.group_id) == bool(b.group_id)
                       and abs(a.start+offset-b.start) <= policy.repeat_match_seconds)
        matched, used = {}, set()
        for _, aid, bid, a, b in edges:
            if aid in used or bid in matched:
                continue
            # Hold endpoints must align too. A matching head alone is insufficient.
            if a.kind == 'hold' and abs(a.end+offset-b.end) > policy.repeat_match_seconds:
                continue
            used.add(aid)
            matched[bid] = a
        if len(matched)/max(len(source), len(target)) < policy.repeat_coverage:
            continue
        edits = {}
        for b in target:
            a = matched.get(b.event_id)
            if a is None or b.kind in ('extra_heavy', 'gear') or b.group_id:
                continue
            lanes = tuple('air' if lane == 'ground' else 'ground' for lane in a.lanes)
            if len(lanes) == 2:
                lanes = ('ground', 'air')
            if set(lanes) != set(b.lanes):
                edits[b.event_id] = replace(b, lanes=lanes)
        proposal = [edits.get(e.event_id, e) for e in out]
        if validate_generated_events(proposal, duration):
            report['reason'] = 'occupancy'
            continue
        # Do not undo aligned tail+click endpoints or introduce more fatigue.
        before, after = structure_stats(out, policy), structure_stats(proposal, policy)
        if not aligned_tail_ids(out) <= aligned_tail_ids(proposal):
            report['reason'] = 'tail_alignment_would_break'
            continue
        if after['alternating_four_or_more_transitions'] > before['alternating_four_or_more_transitions']:
            report['reason'] = 'alternation_would_increase'
            continue
        out = proposal
        report.update(applied=True, reason='matched_mirror', refined_offset=offset,
                      matched=len(matched), changed=len(edits),
                      edits=[dict(event_id=e.event_id, lanes=list(e.lanes)) for e in edits.values()])
    checked(out, duration)
    return out, dict(proposals=reports, applied=sum(p['applied'] for p in reports),
                     mode='compatible one-to-one mirrored arrangement on fixed existing onsets',
                     last_musical_edit=True)


def process(events, features, prediction, duration, prior, policy=Policy()):
    checked(events, duration)
    before = structure_stats(events, policy)
    analysis = analyze_audio(features, duration, policy)
    out, density = add_climax_candidates(events, prediction, analysis, duration, policy)
    out, phrasing = arrange_phrases(out, duration, prior, policy)
    out, tails = extend_tails(out, duration, policy)
    out, repeats = apply_repeats(out, repeat_candidates(features, duration, policy), duration, policy)
    original = {e.event_id: e for e in events}
    for e in out:
        if e.event_id in original:
            old = original[e.event_id]
            if (e.start, e.kind, e.group_id) != (old.start, old.kind, old.group_id) or e.end < old.end:
                raise AssertionError('Existing onset/type/group changed or hold shortened')
    if not set(original) <= {e.event_id for e in out}:
        raise AssertionError('Existing event deleted')
    checked(out, duration)
    return out, dict(policy=asdict(policy), stage_order=['climax_candidates', 'single_lane_phrases',
                    'hold_tail_extension', 'similar_passage_mirroring', 'validation_only'],
                    before=before, after=structure_stats(out, policy), density=density,
                    phrasing=phrasing, tails=tails, repeats=repeats,
                    frozen_model='v1', model_retrained=False, human_playtest='pending')
