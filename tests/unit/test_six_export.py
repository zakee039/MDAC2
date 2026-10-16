import pytest
from mdac2.data_preparation.training_chart import TrainingEvent
from mdac2.data_preparation.six_export import write_bms,read_bms,semantics
from mdac2.data_preparation.candidate_rules import select_candidates


def e(key,kind,t,end=None,lanes=('ground',)):
    return TrainingEvent(key,kind,lanes,t,t if end is None else end,(key,),'test','test',kind=='extra_heavy')


def test_six_classes_both_holds_and_adjacent_boundaries_round_trip():
    events=[e('l','light',1),e('h','heavy',2,lanes=('air',)),e('x','extra_heavy',3),
            e('g','gemini',4,lanes=('ground','air')),e('hold','hold',5,7,('ground','air')),
            e('tailhead','hold',7,8),e('gear','gear',9),e('withhold','light',6)]
    text=write_bms(events,12)
    assert semantics(read_bms(text,12))==semantics(events)


def test_exclusive_conflict_export_refuses_and_candidate_selection_is_auditable():
    special=e('special','extra_heavy',2)
    light=e('light','light',2,lanes=('air',))
    with pytest.raises(ValueError,match='conflicts'):write_bms([special,light],10)
    accepted,rejected=select_candidates([(light,.5),(special,.9)],10)
    assert accepted==[special] and rejected[0]['event_id']=='light'
    assert rejected[0]['issues'][0]['code']=='exclusive_overlap'
    assert light.start==2


def test_export_never_silently_quantizes_or_discards_event():
    with pytest.raises(ValueError,match='precision'):write_bms([e('x','light',1.0001)],10)
    with pytest.raises(ValueError,match='Duplicate'):select_candidates([(e('a','light',1),.9),(e('a','light',2),.8)],10)


def test_one_millisecond_hold_float_tail_is_not_false_precision_failure():
    events=[e('tiny','hold',1.2,1.2009999999999998)]
    assert semantics(read_bms(write_bms(events,10),10))==semantics(events)
