from dataclasses import replace
import numpy as np
import pytest

from mdac2.data_preparation.training_chart import TrainingEvent
from mdac2.data_preparation.generated_constraints import validate_generated_events
from mdac2.data_preparation.playtest_postprocess import (
    Policy, fit_run_prior, arrange_phrases, extend_tails, analyze_audio,
    add_climax_candidates, apply_repeats, structure_stats, repeat_candidates,
)


def click(i, t, lane='ground', kind='light', end=None, group=None):
    return TrainingEvent(str(i), kind, (lane,) if kind != 'gemini' else ('ground', 'air'),
                         t, t if end is None else end, (str(i),), 'test', 'test',
                         kind == 'extra_heavy', group)


def test_prior_censored_and_phrasing_preserves_onsets_and_types():
    train = [[click(i, i*.2, 'ground' if i//n % 2 == 0 else 'air') for i in range(48)]
             for n in (2, 3, 3, 4, 4, 5)]
    prior = fit_run_prior(train)
    events = [click(i, i*.2, ('ground', 'air')[i % 2]) for i in range(48)]
    out, report = arrange_phrases(events, 10., prior)
    assert report['changed'] > 0
    assert [(e.event_id, e.start, e.end, e.kind) for e in events] == [(e.event_id, e.start, e.end, e.kind) for e in out]
    assert structure_stats(out)['alternating_four_or_more_transitions'] < structure_stats(events)['alternating_four_or_more_transitions']
    assert structure_stats(out)['three_runs']+structure_stats(out)['four_runs'] > 0
    assert not validate_generated_events(out, 10.)


def test_tail_only_extends_to_compatible_near_click_and_respects_exclusive():
    h = click('h', 0., end=1., kind='hold')
    end = click('a', 1.12)
    out, report = extend_tails([h, end], 2.)
    assert out[0].end == end.start and out[1] == end
    assert report['extended'] == 1
    assert extend_tails([h, replace(end, lanes=('air',))], 2.)[0][0] == h
    assert extend_tails([h, replace(end, start=1.4, end=1.4)], 2.)[0][0] == h
    blocked = [h, click('x', 1.05, kind='extra_heavy'), end]
    assert extend_tails(blocked, 2.)[0][0] == h


def test_existing_triples_quads_and_boundaries_survive_rephrasing():
    lanes = ['ground']*3+['air']*4+['ground', 'air']*12
    events = [click(i, i*.2, lane) for i, lane in enumerate(lanes)]
    prior = fit_run_prior([[click(i, i*.2, 'ground' if i//3 % 2 else 'air') for i in range(60)]])
    out, _ = arrange_phrases(events, 8., prior)
    assert out[:8] == events[:8]
    stats = structure_stats(out)
    assert stats['three_runs'] >= 1 and stats['four_runs'] >= 1


def test_special_double_heavy_and_hold_click_are_preserved():
    events = [click('h', 0., end=1., kind='hold'), click('c', .5),
              click('d0', 2., kind='heavy', group='double'),
              click('d1', 2., 'air', kind='heavy', group='double'),
              click('g', 3., kind='gemini'), click('x', 4., kind='extra_heavy')]
    out, report = arrange_phrases(events, 5., fit_run_prior([events]))
    assert out == events and report['changed'] == 0


def test_double_hold_tail_extension_remains_atomic():
    holds = [click('h0', 0., kind='hold', end=1., group='dualhold'),
             click('h1', 0., 'air', kind='hold', end=1., group='dualhold')]
    assert extend_tails(holds+[click('a', 1.1)], 2.)[0][:2] == holds
    out, report = extend_tails(holds+[click('g', 1.1, kind='gemini')], 2.)
    assert out[0].end == out[1].end == 1.1
    assert report['extended'] == 2


def test_already_aligned_tail_never_moves_to_next_click():
    hold = click('h', 0., kind='hold', end=1.)
    events = [hold, click('aligned', 1.), click('nearby', 1.2)]
    out, report = extend_tails(events, 2.)
    assert out == events and report['extended'] == 0


def test_mirroring_cannot_trade_one_existing_tail_alignment_for_another():
    # Equal total aligned-tail counts can hide destruction of one specific pair.
    source = [click('s'+str(i), 1.+i*.4, 'air') for i in range(12)]
    target = [click('t'+str(i), 11.+i*.4, 'air') for i in range(12)]
    holds = [click('sh0', .5, 'air', 'hold', 1.4, 'protect-s0'),
             click('sh1', 2., 'ground', 'hold', 2.6, 'protect-s1'),
             click('th0', 10.5, 'air', 'hold', 11.4, 'protect-t0'),
             click('th1', 12., 'ground', 'hold', 12.6, 'protect-t1')]
    events = source+target+holds
    proposals = [dict(source=[0., 8.], target=[10., 18.], offset=10., score=.99)]
    out, report = apply_repeats(events, proposals, 20.)
    assert out == events
    assert report['proposals'][0]['reason'] == 'tail_alignment_would_break'


def test_silence_or_uniform_tone_has_no_climax_or_repeats():
    for features in (np.zeros((3, 96, 4000), np.float32), np.ones((3, 96, 4000), np.float32)):
        analysis = analyze_audio(features, 40.)
        assert analysis['sections'] == []
        assert repeat_candidates(features, 40.) == []
        out, report = add_climax_candidates([], dict(probability=np.ones(4000)*.29,
                            offsets=np.zeros(4000)), analysis, 40.)
        assert out == [] and report['added'] == 0


def test_density_requires_joint_evidence_spacing_and_budget():
    events = [click(i, i*.4) for i in range(40)]
    p = np.zeros(1600)
    rise = np.zeros(1600)
    for t in np.arange(.2, 16., .4):
        p[round(t*100)] = .25
        rise[round(t*100)] = .2
    analysis = dict(rise=rise, sections=[dict(start=0., end=16., score=.9)], method='test')
    out, report = add_climax_candidates(events, dict(probability=p, offsets=np.zeros(1600)), analysis, 16.)
    assert 0 < report['added'] <= int(len(events)*.15)
    assert {e.event_id for e in events} <= {e.event_id for e in out}
    assert not validate_generated_events(out, 16.)
    no_rise = {**analysis, 'rise': np.zeros_like(rise)}
    assert add_climax_candidates(events, dict(probability=p, offsets=np.zeros(1600)), no_rise, 16.)[0] == events


def test_repeats_use_timed_semantic_matching_not_indices():
    source = [click(i, 1.+i*.4, 'ground' if i < 4 else 'air') for i in range(12)]
    target = [click('b'+str(i), 11.+i*.4, 'ground' if i < 4 else 'air') for i in range(12)]
    extra = click('unmatched', 16.5)
    proposals = [dict(source=[0., 8.], target=[10., 18.], offset=10., score=.99)]
    out, report = apply_repeats(source+target+[extra], proposals, 20.)
    assert report['applied'] == 1
    assert out[:12] == source and out[-1] == extra
    assert all(a.start == b.start and a.kind == b.kind and a.lanes != b.lanes for a, b in zip(target, out[12:24]))
    shifted = [replace(e, start=e.start+.12*(i % 3), end=e.end+.12*(i % 3)) for i, e in enumerate(target)]
    assert apply_repeats(source+shifted, proposals, 20.)[1]['applied'] == 0


def test_repeat_detector_accepts_true_repeat_rejects_changed_order():
    rng = np.random.default_rng(42)
    block = rng.random((3, 96, 64)).astype(np.float32)*3
    f = np.repeat(np.concatenate([block, rng.random((3, 96, 40)), block], axis=2), 25, axis=2)
    candidates = repeat_candidates(f, f.shape[-1]*.01)
    assert candidates
    assert any(abs(p['offset']-26.) < .01 for p in candidates)
    instrumental = f.copy()
    instrumental[1] = 0
    assert repeat_candidates(instrumental, instrumental.shape[-1]*.01)
    shuffled = f.copy()
    shuffled[:, :, 2600:] = shuffled[:, :, 2600:][:, :, ::-1]
    assert not repeat_candidates(shuffled, shuffled.shape[-1]*.01)


def test_repeats_reject_cross_boundary_holds():
    source = [click(i, 1.+i*.4) for i in range(12)]
    target = [click('b'+str(i), 11.+i*.4) for i in range(12)]
    hold = click('hold', 7.5, kind='hold', end=8.5)
    events = source+target+[hold]
    out, report = apply_repeats(events, [dict(source=[0., 8.], target=[10., 18.], offset=10., score=.99)], 20.)
    assert out == events
    assert report['proposals'][0]['reason'] == 'hold_crosses_boundary'
