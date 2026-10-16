from mdac2.data_preparation.playability_audit import audit_playability
from mdac2.data_preparation.training_chart import TrainingEvent


def event(key,kind,start,end=None,lanes=('ground',)):
    return TrainingEvent(key,kind,lanes,start,start if end is None else end,(key,),'test','test',kind=='extra_heavy')


def test_gemini_and_two_lane_heavy_are_alternatives_not_three_stacked_pairs():
    result=audit_playability([event('g','gemini',1,lanes=('ground','air')),
        event('l','light',1),event('h','heavy',1,lanes=('air',))],5)
    assert not result['engineering_playtest_gate_passed']
    assert result['exact_attack_stack_lane_times']==2
    assert result['excess_same_lane_attacks']==2
    assert len(result['gemini_with_nearby_attack'])==1


def test_one_gemini_and_a_later_double_heavy_pair_are_allowed():
    result=audit_playability([event('g','gemini',1,lanes=('ground','air')),
        event('h1','heavy',2),event('h2','heavy',2,lanes=('air',))],5)
    assert result['engineering_playtest_gate_passed']


def test_sub_frame_offset_does_not_hide_a_gemini_light_stack():
    result=audit_playability([event('g','gemini',1,lanes=('ground','air')),
                             event('light','light',1.001)],5)
    assert not result['engineering_playtest_gate_passed']
    assert len(result['nearby_attack_stacks'])==1


def test_hold_plus_attack_is_retained_and_sustained_span_is_not_an_empty_gap():
    result=audit_playability([event('hold','hold',1,8),event('tap','light',1),event('after','light',10)],11)
    assert result['engineering_playtest_gate_passed']
    assert result['no_attack_or_hold_intervals']==[dict(start=8,end=10,seconds=2.)]


def test_valid_silence_is_reported_without_being_treated_as_a_rule_violation():
    result=audit_playability([event('a','light',1),event('b','light',7)],10)
    assert result['engineering_playtest_gate_passed']
    assert result['total_reported_gap_seconds']==6


def test_packaging_refuses_known_stacking_before_creating_any_package(tmp_path):
    import json
    from pathlib import Path
    import runpy
    from types import SimpleNamespace
    import pytest
    package=runpy.run_path(str(Path(__file__).resolve().parents[2]/'tools/generate_joint_playtest.py'))['package']
    bad=[event('g','gemini',1,lanes=('ground','air')),event('light','light',1)]
    (tmp_path/'chart.json').write_text(json.dumps({'events':[e.to_dict() for e in bad]}),encoding='utf-8')
    (tmp_path/'audio.json').write_text(json.dumps({'duration':5}),encoding='utf-8')
    with pytest.raises(ValueError,match='Playtest blocked'):
        package(SimpleNamespace(output=tmp_path))
    assert not (tmp_path/'package').exists()
    assert not list(tmp_path.glob('*.mdm'))
    audit=json.loads((tmp_path/'playability-audit.json').read_text(encoding='utf-8'))
    assert not audit['engineering_playtest_gate_passed']
