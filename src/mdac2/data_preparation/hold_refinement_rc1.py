"""R1 hold-feedback candidate: conservative acoustic continuation and click spacing.

No learned weights change. No hold-chain merging or cross-lane hold repair.
"""
from dataclasses import dataclass, asdict, replace
import numpy as np

from .fixed_grid_r2 import snap_events, align_holds, issues, grid_tick, on_grid
from .playtest_postprocess import (Policy, analyze_audio, add_climax_candidates,
    arrange_phrases, repeat_candidates, apply_repeats, structure_stats)


@dataclass(frozen=True)
class HoldPolicy:
    version: str = 'hold-feedback-rc2'
    minimum_hold_seconds: float = 1.0
    chain_gap_seconds: float = .1
    template_seconds: float = .6
    spectrum_cosine: float = .97
    stable_energy_range_db: float = 4.0
    continuation_energy_db: float = -6.0
    boundary_confirmation_seconds: float = .12
    minimum_extension_seconds: float = .15
    maximum_extension_seconds: float = 6.0
    minimum_click_gap_seconds: float = .505


def dense_pairs(events, policy=HoldPolicy()):
    pairs = []
    for hold in events:
        if hold.kind != 'hold':
            continue
        for lane in hold.lanes:
            clicks = sorted((e for e in events if e.kind != 'hold' and lane in e.lanes
                and hold.start < e.start < hold.end), key=lambda e: (e.start, e.event_id))
            for a, b in zip(clicks, clicks[1:]):
                if b.start-a.start < policy.minimum_click_gap_seconds-1e-8:
                    pairs.append((hold.event_id, lane, a.event_id, b.event_id))
    return pairs


def extend_sustained_tails(events, features, duration, policy=HoldPolicy()):
    """A fixed tail spectrum must continue to a confirmed acoustic boundary.

    Trailing 50ms power averages suppress frame noise; the reference never
    follows a changing sound. A missing boundary or occupied future rejects
    extension, rather than inventing a tail at a scan/occupancy limit.
    """
    if features.ndim != 3 or features.shape[:2] != (3, 96) or not np.isfinite(features).all():
        raise ValueError('Finite 3-stream, 96-bin features required')
    if issues(events, duration):
        raise ValueError('Invalid input chart')
    power = np.expm1(np.asarray(features, dtype=np.float32))/10000
    cs = np.pad(np.cumsum(power, axis=2, dtype=np.float64), ((0, 0), (0, 0), (1, 0)))
    frames = np.arange(power.shape[2])
    smooth = (cs[:, :, frames+1]-cs[:, :, np.maximum(0, frames-4)]) / np.minimum(frames+1, 5)
    out, records = list(events), []
    for idx, event in enumerate(events):
        if event.kind != 'hold' or event.end-event.start < policy.minimum_hold_seconds:
            continue
        report = dict(event_id=event.event_id, before=event.end, applied=False)
        records.append(report)
        if any(other.kind == 'hold' and other.event_id != event.event_id
               and set(other.lanes) & set(event.lanes)
               and 0 <= event.start-other.end <= policy.chain_gap_seconds+1e-8 for other in events):
            report['reason'] = 'hold_chain_deferred'; continue
        end = int(round(event.end*100))
        begin = max(int(np.ceil(event.start*100))+5, end-int(policy.template_seconds*100))
        if end-begin < 50 or end >= power.shape[2]:
            report['reason'] = 'insufficient_tail_context'; continue
        source = 1+int(np.argmax(power[1:, :, begin:end].mean((1, 2))))
        ref = power[source, :, begin:end].mean(1)
        ref_energy = float(ref.sum())
        mix_energy = float(power[0, :, begin:end].mean(1).sum())
        if ref_energy < 1e-6 or ref_energy < mix_energy*.001:
            report['reason'] = 'inaudible_source'; continue
        template = np.sqrt(ref)
        template /= max(np.linalg.norm(template), 1e-12)
        vectors = np.sqrt(np.maximum(smooth[source], 0))
        similarity = template @ vectors / np.maximum(np.linalg.norm(vectors, axis=0), 1e-12)
        db = 10*np.log10(np.maximum(smooth[source].sum(0), 1e-12)/ref_energy)
        report.update(source=('vocals' if source == 1 else 'accompaniment'),
            reference=[begin/100, end/100], stable_cosine_p10=float(np.quantile(similarity[begin:end], .1)),
            stable_energy_range_db=float(np.ptp(db[begin:end])))
        if report['stable_cosine_p10'] < policy.spectrum_cosine or report['stable_energy_range_db'] > policy.stable_energy_range_db:
            report['reason'] = 'tail_not_stable'; continue
        limit = min(power.shape[2], int((event.end+policy.maximum_extension_seconds)*100))
        good = (similarity >= policy.spectrum_cosine) & (db >= policy.continuation_energy_db)
        confirmation = int(round(policy.boundary_confirmation_seconds*100))
        boundary = None
        for frame in range(end, limit-confirmation+1):
            if not good[frame:frame+confirmation].any():
                boundary = frame; break
        if boundary is None:
            report['reason'] = 'no_confirmed_boundary_within_limit'; continue
        # Every tolerated dip before the boundary must be shorter than 120ms.
        target = grid_tick(boundary/100)/200
        report.update(acoustic_boundary=boundary/100, proposed_end=target,
            supported_fraction=float(good[end:boundary].mean()) if boundary > end else 0.)
        if target-event.end < policy.minimum_extension_seconds or report['supported_fraction'] < .9:
            report['reason'] = 'no_supported_continuation'; continue
        proposal = list(out); proposal[idx] = replace(event, end=target)
        if target > duration or issues(proposal, duration):
            report['reason'] = 'occupied_or_outside_audio'; continue
        out = proposal
        report.update(applied=True, reason='stable_source_to_confirmed_acoustic_boundary', after=target)
    return out, dict(policy=asdict(policy), changes=[r for r in records if r['applied']], candidates=records,
        source_identity_verified=False, limitation='Stable spectral continuation is evidence, not labeled sound-source truth.')


def space_hold_clicks(events, duration, policy=HoldPolicy()):
    out, edits, rejected = list(events), [], []
    before = dense_pairs(out, policy)
    attempted = set()
    while True:
        pairs = dense_pairs(out, policy)
        pending = [p for p in pairs if p not in attempted]
        if not pending:
            break
        pair = pending[0]; attempted.add(pair)
        event = next(e for e in out if e.event_id == pair[3])
        if event.kind not in ('light', 'heavy') or event.group_id or len(event.lanes) != 1:
            rejected.append(dict(pair=pair, reason='protected_atomic_combination')); continue
        target = 'air' if event.lanes == ('ground',) else 'ground'
        neighbours = [e for e in out if e.event_id != event.event_id and e.kind != 'hold'
            and target in e.lanes and abs(e.start-event.start) <= .02000001]
        merge = None
        if neighbours:
            if len(neighbours) == 1:
                other = neighbours[0]
                if other.kind == event.kind and not other.group_id and len(other.lanes) == 1 and abs(other.start-event.start) < 1e-8:
                    merge = other
            if merge is None:
                rejected.append(dict(pair=pair, reason='incompatible_target_click')); continue
        if merge:
            replacement = replace(merge, source_ids=tuple(dict.fromkeys((*merge.source_ids, *event.source_ids))))
            proposal = [replacement if e.event_id == merge.event_id else e for e in out if e.event_id != event.event_id]
        else:
            proposal = [replace(e, lanes=(target,)) if e.event_id == event.event_id else e for e in out]
        new_pairs = dense_pairs(proposal, policy)
        if issues(proposal, duration) or not set(new_pairs) < set(pairs):
            rejected.append(dict(pair=pair, reason='occupancy_or_new_density')); continue
        out = proposal
        edits.append(dict(event_id=event.event_id, time=event.start, from_lane=event.lanes[0], to_lane=target,
            operation='merge_same_type_same_time' if merge else 'move_lane', merged_into=merge.event_id if merge else None))
    return out, dict(before=before, after=dense_pairs(out, policy), edits=edits, rejected=rejected,
        minimum_gap_seconds=policy.minimum_click_gap_seconds, attack_times_moved=False)


def process(events, features, prediction, duration, prior, policy=Policy(), hold_policy=HoldPolicy()):
    analysis = analyze_audio(features, duration, policy)
    out, density = add_climax_candidates(events, prediction, analysis, duration, policy)
    out, phrasing = arrange_phrases(out, duration, prior, policy)
    out, snapping = snap_events(out, duration)
    out, tails = extend_sustained_tails(out, features, duration, hold_policy)
    out, alignment = align_holds(out, duration)
    out, spacing = space_hold_clicks(out, duration, hold_policy)
    times = {e.event_id: (e.start, e.end) for e in out}
    # Keep the R1 repeat algorithm and fixed snapshot semantics unchanged.
    # Reject the whole optional mirror stage if it introduces dense pairs.
    proposal, repeats = apply_repeats(out, repeat_candidates(features, duration, policy), duration, policy)
    if not set(dense_pairs(proposal, hold_policy)) <= set(dense_pairs(out, hold_policy)):
        repeats.update(applied=0, stage_rejected_by_hold_spacing=True)
    else:
        out = proposal
    if issues(out, duration) or any(not on_grid(t) for e in out for t in (e.start, e.end)):
        raise AssertionError('Final legality/grid failed')
    if any(times[e.event_id] != (e.start, e.end) for e in out):
        raise AssertionError('Similarity changed event timing')
    return out, dict(policy=asdict(hold_policy), before=structure_stats(events), after=structure_stats(out),
        density=density, phrasing=phrasing, snapping=snapping, tails=tails, alignment=alignment,
        spacing=spacing, repeats=repeats, final_dense_pairs=dense_pairs(out, hold_policy),
        stage_order=['gated_climax','phrases','5ms_snap','acoustic_tail','100ms_alignment',
                     'hold_click_spacing','similarity_with_spacing_gate','validation_only'],
        model_retrained=False, figure_2_hold_chain_repair=False, human_playtest='pending')
