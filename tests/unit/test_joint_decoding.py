import numpy as np
import pytest
from mdac2.data_preparation.joint_decoding import decode_song, place_core, select_song_candidates
from mdac2.data_preparation.candidate_rules import select_candidates
from mdac2.data_preparation.joint_evaluation import counts, EventMetrics
from mdac2.data_preparation.training_chart import TrainingEvent, validate_events


def empty_prediction(n=4000):
    return dict(onsets=np.zeros((n,2,6)), offsets=np.zeros((n,2,6)),
                hold_ends=np.zeros((n,2)), hold_end_offsets=np.zeros((n,2)), occupancy=np.zeros((n,2)))


def test_centers_cover_once_and_context_never_overwrites():
    destination = dict(x=np.zeros(1700))
    coverage = np.zeros(1700, np.uint8)
    place_core(destination, coverage, -800, dict(x=np.full(3200,1)))
    place_core(destination, coverage, 800, dict(x=np.full(3200,2)))
    assert np.all(coverage == 1)
    assert np.all(destination['x'][:1600] == 1)
    assert np.all(destination['x'][1600:] == 2)
    with pytest.raises(ValueError, match='Duplicate'):
        place_core(destination, coverage, -800, dict(x=np.zeros(3200)))


def test_cross_window_dual_hold_gemini_and_double_heavy_remain_distinct():
    p = empty_prediction()
    p['onsets'][1500,:,4] = .99
    p['hold_ends'][3300,:] = .99
    p['occupancy'][1500:3300,:] = .99
    p['onsets'][500,:,3] = .99
    p['onsets'][700,:,1] = .99
    events, audit = decode_song(p,40)
    hold = next(e for e in events if e.kind == 'hold')
    assert (hold.start, hold.end, hold.lanes) == (15,33,('ground','air'))
    assert audit['cross_window_holds'] == 1
    gemini = [e for e in events if e.kind == 'gemini']
    heavy = [e for e in events if e.kind == 'heavy']
    assert len(gemini) == 1 and len(heavy) == 2
    assert heavy[0].group_id == heavy[1].group_id
    assert not validate_events(events,40)


def test_no_fabricated_hold_tail_or_one_sided_gemini():
    p = empty_prediction()
    p['onsets'][1500,0,4] = .99
    p['onsets'][200,0,3] = .99
    p['occupancy'][1500:,:] = .99
    events, audit = decode_song(p,40)
    assert events == []
    assert {r['code'] for r in audit['rejected']} == {'unpaired_hold_head','unpaired_gemini'}


def test_tail_and_occupancy_both_required_and_1ms_hold_supported():
    p = empty_prediction(1000)
    p['onsets'][100,0,4] = .99
    p['hold_ends'][300,0] = .99
    events, audit = decode_song(p,10)
    assert events == []
    p['hold_ends'][:] = 0
    p['hold_ends'][100,0] = .99
    p['hold_end_offsets'][100,0] = .1
    p['occupancy'][100,0] = .99
    events, _ = decode_song(p,10)
    assert len(events) == 1 and events[0].end == 1.001


def test_extra_heavy_rule_exclusion_retains_reason():
    p = empty_prediction(1000)
    p['onsets'][100,0,2] = .99
    p['onsets'][100,1,0] = .8
    events, audit = decode_song(p,10)
    assert len(events) == 1 and events[0].kind == 'extra_heavy'
    assert audit['rule_rejection_count'] == 1
    assert audit['rejected'][0]['issues'][0]['code'] == 'exclusive_overlap'


def test_matching_one_to_one_and_masks_exclude_unlearned_cells():
    matched, errors = counts([1.01,1.02,2], [1,2], .05)
    assert matched.tolist() == [2,1,0]
    event = TrainingEvent('a','light',('ground',),1,1,('a',),'x','x')
    metrics = EventMetrics()
    mask = np.ones((1000,2,6),bool)
    mask[100,0,0] = False
    metrics.add([event],[],10,dict(rejected=[],candidate_count=1,rule_rejection_count=0),mask)
    assert metrics.report()['six_classes']['light']['fp'] == 0


def test_indexed_selection_matches_original_rules_under_random_overlap():
    rng = np.random.default_rng(77)
    for repeat in range(10):
        candidates = []
        for i in range(80):
            kind = rng.choice(['hold','light','heavy','extra_heavy','gear','gemini'])
            lanes = ('ground','air') if kind=='gemini' else (rng.choice(['ground','air']),)
            start = float(rng.integers(0,40))/10
            end = start + float(rng.integers(1,10))/10 if kind=='hold' else start
            event = TrainingEvent(str(i),kind,lanes,start,end,(str(i),),'x','x',kind=='extra_heavy')
            candidates.append((event,float(rng.random())))
        original, rejected = select_candidates(candidates,5)
        indexed, faster_rejected = select_song_candidates(candidates,5)
        assert indexed == original
        assert {r['event_id'] for r in rejected} == {r['event_id'] for r in faster_rejected}


def test_sampling_match_does_not_hide_wrong_lane_or_class_and_tail_errors():
    reference = [TrainingEvent('r','hold',('ground',),1,3,('r',),'x','x')]
    predicted = [TrainingEvent('p','heavy',('air',),1,1,('p',),'x','x')]
    audit = dict(rejected=[],candidate_count=1,rule_rejection_count=0)
    metrics = EventMetrics()
    metrics.add(predicted,reference,5,audit)
    report = metrics.report()
    assert report['sampling']['f1'] == 1
    assert report['lanes']['ground']['fn'] == 1
    assert report['lanes']['air']['fp'] == 1
    assert report['six_classes']['hold']['fn'] == 1
    assert report['six_classes']['heavy']['fp'] == 1
    wrong_tail = [TrainingEvent('p','hold',('ground',),1,4,('p',),'x','x')]
    metrics = EventMetrics()
    metrics.add(wrong_tail,reference,5,audit)
    assert metrics.report()['holds']['hold_head']['f1'] == 1
    assert metrics.report()['holds']['hold_joint']['f1'] == 0
