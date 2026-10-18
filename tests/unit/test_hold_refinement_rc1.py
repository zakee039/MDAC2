import numpy as np
from dataclasses import replace
from mdac2.data_preparation.training_chart import TrainingEvent
from mdac2.data_preparation.hold_refinement_rc1 import extend_sustained_tails, space_hold_clicks, dense_pairs


def e(key, kind, start, end=None, lane='ground', group=None):
    return TrainingEvent(key, kind, (lane,) if kind != 'gemini' else ('ground','air'),
        start, start if end is None else end, (key,), 'test', 'test', kind == 'extra_heavy', group)


def sound(stop=4., change=False):
    p = np.zeros((3,96,600),np.float32)
    p[0,10:20,:int(stop*100)] = .1
    p[2] = p[0]
    if change:
        p[:,10:20,300:400] = 0
        p[0,60:70,300:400] = .1
        p[2] = p[0]
    return np.log1p(p*10000)


def test_extend_from_audio_without_any_next_note():
    out, report = extend_sustained_tails([e('h','hold',.5,2)],sound(),6)
    assert 4 <= out[0].end <= 4.05
    assert report['changes'] and out[0].start == .5


def test_equal_loudness_different_spectrum_is_boundary():
    out,_ = extend_sustained_tails([e('h','hold',.5,2)],sound(change=True),6)
    assert 3 <= out[0].end <= 3.05


def test_no_continuation_does_not_extend_to_next_note():
    es=[e('h','hold',.5,2),e('n','light',5)]
    out,report=extend_sustained_tails(es,sound(stop=2),6)
    assert out == es and not report['changes']


def test_missing_boundary_rejects_scan_limit():
    es=[e('h','hold',.5,2)]
    out,report=extend_sustained_tails(es,sound(stop=6),6)
    assert out==es and report['candidates'][0]['reason']=='no_confirmed_boundary_within_limit'


def test_future_hold_or_exclusive_blocks_extension():
    for blocker in [e('h2','hold',3,5), e('x','extra_heavy',3)]:
        es=[e('h','hold',.5,2),blocker]
        out,_=extend_sustained_tails(es,sound(),6)
        assert out[0]==es[0]


def test_short_hold_chains_not_extended():
    es=[e('h','hold',1,1.7)]
    out,_=extend_sustained_tails(es,sound(),6)
    assert out==es


def test_long_last_member_of_chain_is_also_deferred():
    es=[e('a','hold',0,.45),e('b','hold',.5,2)]
    out,report=extend_sustained_tails(es,sound(),6)
    assert out==es and report['candidates'][0]['reason']=='hold_chain_deferred'


def test_dense_second_click_moves_without_retiming():
    es=[e('h','hold',0,3),e('a','light',1),e('b','light',1.5)]
    out,r=space_hold_clicks(es,6)
    assert out[-1].lanes==('air',) and out[-1].start==1.5 and not dense_pairs(out)
    assert r['edits'][0]['operation']=='move_lane'


def test_505ms_is_allowed_and_boundaries_excluded():
    es=[e('h','hold',0,3),e('a','light',1),e('b','light',1.505),e('c','light',3)]
    out,r=space_hold_clicks(es,6)
    assert out==es and not r['before']


def test_same_type_same_time_merge_preserves_sources():
    es=[e('h','hold',0,3),e('a','light',1),e('b','light',1.4),e('c','light',1.4,lane='air')]
    out,r=space_hold_clicks(es,6)
    assert len(out)==3 and set(out[-1].source_ids)=={'b','c'} and not dense_pairs(out)


def test_no_new_density_on_other_held_lane():
    es=[e('h','hold',0,3),e('j','hold',0,3,lane='air'),e('a','light',1),e('b','light',1.4),e('c','light',1.3,lane='air')]
    out,r=space_hold_clicks(es,6)
    assert out==es and r['rejected']


def test_atomic_groups_and_different_types_not_merged():
    for b,c in [(e('b','heavy',1.4,group='double'),e('c','heavy',1.4,lane='air',group='double')),
                (e('b','light',1.4),e('c','heavy',1.4,lane='air')),
                (e('b','gemini',1.4),e('c','light',2.5,lane='air'))]:
        es=[e('h','hold',0,3),e('a','light',1),b,c]
        out,r=space_hold_clicks(es,6)
        assert out==es and r['rejected']
