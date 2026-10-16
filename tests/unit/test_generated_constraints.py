import numpy as np
import pytest
from mdac2.data_preparation.generated_constraints import (
    constrain_decoded_events,select_generated_candidates,validate_generated_events,
)
from mdac2.data_preparation.joint_decoding import DEFAULT_DECODER,decode_song
from mdac2.data_preparation.six_export import write_bms,read_bms,semantics
from mdac2.data_preparation.training_chart import TrainingEvent


def event(key,kind,t=1,end=None,lanes=('ground',),group=None):
    return TrainingEvent(key,kind,lanes,t,t if end is None else end,(key,),'test','test',kind=='extra_heavy',group)


@pytest.mark.parametrize('left,right',[
    ('light','heavy'),('light','light'),('light','gear'),('heavy','gear'),('gear','gear'),
])
def test_one_nonhold_per_position_regardless_of_class(left,right):
    assert validate_generated_events([event('a',left),event('b',right)],5)


def test_hold_plus_exactly_one_click_is_allowed_and_second_click_rejected():
    h=event('hold','hold',end=4)
    tap=event('tap','light')
    assert not validate_generated_events([h,tap],5)
    assert not validate_generated_events([h,event('inside','heavy',t=2)],5)
    assert validate_generated_events([h,tap,event('extra','heavy')],5)
    assert validate_generated_events([h,event('gear','gear',t=2)],5)
    assert validate_generated_events([h,event('exclusive','extra_heavy',t=2,lanes=('air',))],5)


def test_overlapping_holds_are_rejected_but_half_open_tail_head_can_touch():
    h=event('h1','hold',end=3)
    assert validate_generated_events([h,event('h2','hold',t=2,end=4)],5)
    assert not validate_generated_events([h,event('h2','hold',t=3,end=4)],5)


def test_gemini_occupies_both_lanes_and_is_never_cut_into_single_lane():
    g=event('g','gemini',lanes=('ground','air'))
    ground=event('tap','light')
    accepted,audit=select_generated_candidates([(g,.8),(ground,.9)],5)
    assert accepted==[ground]
    assert audit['rejected'][0]['event_ids']==['g']
    accepted,_=select_generated_candidates([(g,.95),(ground,.9)],5)
    assert accepted==[g]


def test_explicit_double_heavy_is_atomic_and_keeps_its_type():
    pair=[event('h1','heavy',group='double'),event('h2','heavy',lanes=('air',),group='double')]
    g=event('g','gemini',lanes=('ground','air'))
    accepted,_=select_generated_candidates([(pair[0],.95),(pair[1],.94),(g,.9)],5)
    assert accepted==pair
    accepted,_=select_generated_candidates([(pair[0],.8),(pair[1],.7),(event('tap','light'),.9)],5)
    assert [e.event_id for e in accepted]==['tap']
    with pytest.raises(ValueError,match='exactly two'):
        select_generated_candidates([(pair[0],.9)],5)


def test_nearby_fractional_outputs_do_not_evade_occupancy_and_times_never_move():
    tap=event('tap','light')
    heavy=event('heavy','heavy',t=1.001)
    accepted,_=select_generated_candidates([(tap,.8),(heavy,.9)],5)
    assert accepted==[heavy] and accepted[0].start==1.001
    # Distinct lanes at one time remain legal.
    assert not validate_generated_events([tap,event('air','heavy',lanes=('air',))],5)


def test_real_decoder_proposals_use_scores_and_keep_hold_click_exception():
    n=500
    p=dict(onsets=np.zeros((n,2,6)),offsets=np.full((n,2,6),.5),
           hold_ends=np.zeros((n,2)),hold_end_offsets=np.full((n,2),.5),occupancy=np.zeros((n,2)))
    p['onsets'][100,:,0]=.75
    p['onsets'][100,:,1]=.8
    p['onsets'][100,:,3]=.9
    p['onsets'][100,:,4]=.95
    p['hold_ends'][300,:]=.9
    p['occupancy'][100:301,:]=.9
    proposals,_=decode_song(p,5)
    accepted,audit=constrain_decoded_events(proposals,p,5,DEFAULT_DECODER)
    assert {e.kind for e in accepted}=={'gemini','hold'}
    assert audit['rejected_count']==4
    assert not validate_generated_events(accepted,5)
    assert semantics(read_bms(write_bms(accepted,5),5))==semantics(accepted)


def test_candidate_order_does_not_change_tied_score_result():
    c=[(event('a','light'),.9),(event('b','heavy'),.9)]
    forward,_=select_generated_candidates(c,5)
    reverse,_=select_generated_candidates(list(reversed(c)),5)
    assert forward==reverse==[c[0][0]]
