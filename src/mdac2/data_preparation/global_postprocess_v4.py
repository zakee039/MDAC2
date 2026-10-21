"""V4 postprocessing.

V4 keeps the V3 model and acoustic stages but owns a separate final timing
policy.  Hold interiors are playable track space; only hold tails create a
500 ms lane guard.  The module is intentionally separate from the frozen V3
postprocessor.
"""
from .global_postprocess import (STORAGE, align_holds, normalize_ms,
    prioritize_exclusive)
from .hold_refinement_rc1 import extend_sustained_tails, HoldPolicy, dense_pairs
from .playtest_postprocess import (Policy, analyze_audio, add_climax_candidates,
    arrange_phrases, repeat_candidates, apply_repeats, structure_stats)
from .training_chart import validate_events

POLICY_ID = 'global-postprocess-1ms-v4'
TAIL_GUARD_SECONDS = .5


def prioritize_v4_exclusive(events, duration):
    # The frozen helper prioritizes specials over Gemini. V4 explicitly reverses
    # that decision before the helper can delete the original Gemini identity.
    from .generated_constraints import NEARBY_SECONDS
    gemini = [e for e in events if e.kind == 'gemini']
    dropped = [e for e in events if e.kind == 'extra_heavy' and
               any(abs(e.start-g.start) <= NEARBY_SECONDS+1e-9 for g in gemini)]
    ids = {e.event_id for e in dropped}
    out, report = prioritize_exclusive([e for e in events if e.event_id not in ids], duration)
    report['dropped'].extend(dict(event_id=e.event_id, time=e.start, reason='gemini_priority') for e in dropped)
    return out, report


def structural_issues(events, duration):
    return validate_events(events, duration) + [
        dict(code='hold_under_30ms', event_ids=[e.event_id])
        for e in events if e.kind == 'hold' and e.end-e.start < .03-1e-8
    ]


def process(events, features, prediction, duration, prior):
    """Run shared acoustic stages without the frozen V3 hold-click guard."""
    before = structure_stats(events)
    out, exclusive = prioritize_v4_exclusive(events, duration)
    out = normalize_ms(out, duration)
    analysis = analyze_audio(features, duration, Policy())
    out, density = add_climax_candidates(out, prediction, analysis, duration, Policy())
    out, phrasing = arrange_phrases(out, duration, prior, Policy())
    out = normalize_ms(out, duration)
    out, tails = extend_sustained_tails(out, features, duration, HoldPolicy())
    out, alignment = align_holds(out, duration)
    if structural_issues(out, duration):
        raise ValueError('V4 structural postprocess conflict')
    times = {e.event_id: (e.start, e.end) for e in out}
    proposal, repeats = apply_repeats(
        out, repeat_candidates(features, duration, Policy()), duration, Policy())
    # Repetition is optional; never let it introduce a V4 spacing or tail issue.
    if (not set(dense_pairs(proposal)) <= set(dense_pairs(out)) or
            structural_issues(proposal, duration)):
        repeats.update(applied=0, stage_rejected_by_v4_rules=True)
    else:
        out = proposal
    if structural_issues(out, duration):
        raise AssertionError('V4 structural gate failed')
    if any(times[e.event_id] != (e.start, e.end) for e in out):
        raise AssertionError('V4 repetition changed timing')
    return out, dict(
        policy=dict(
            version=POLICY_ID, storage=STORAGE,
            hold_interior='ordinary_track_with_strict_spacing',
            hold_tail_guard_seconds=TAIL_GUARD_SECONDS,
            gemini='ordinary_note_dual_lane_with_priority',
            blocked_transfer='discard_or_gemini_single_heavy',
            model_retrained=False,
        ), before=before, after=structure_stats(out), density=density,
        phrasing=phrasing, tails=tails, alignment=alignment,
        exclusive=exclusive, repeats=repeats,
        stage_order=['exclusive_priority', '1ms', 'density', 'phrases',
                     '1ms', 'acoustic_tail', '100ms_alignment',
                     'v4_timing_and_lane_rules', 'validation'],
        human_playtest='pending')
