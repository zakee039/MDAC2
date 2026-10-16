from dataclasses import replace
import pytest
from test_dataset import song,freeze
from mdac2.evaluation.dataset_manifest import Reference
from mdac2.evaluation.report import freeze_plan,EvaluationRun,build_report


def fixture():
    a=song();onset=replace(a.references[0],kind='onsets',label_hash='e'*64)
    a=replace(a,references=(*a.references,onset));m=freeze((a,))
    p=freeze_plan(m,candidate_hash='1'*64,configuration_hash='2'*64,metric_profile_hash='3'*64)
    run=EvaluationRun(p.artifact_hash,a.song_id,'oracle_timing',onset.label_hash,a.references[0].label_hash,'succeeded',(('onset_f1',1.),),'4'*64)
    return m,p,run


def test_missing_auto_remains_in_slice_and_no_samples_not_zero_success():
    m,p,run=fixture();r=build_report(m,p,(run,)).payload
    auto=next(s for s in r['summaries'] if s['slice']=='all' and s['mode']=='auto_timing')
    assert auto['song_count']==auto['failed_count']==1 and auto['metrics']['onset_f1']['all_locked_zero_failure_mean']==0
    assert auto['metrics']['onset_f1']['successful_mean'] is None
    assert next(s for s in r['summaries'] if s['slice']=='6/8')['uncertainty']=='NO_SAMPLES'
    assert r['pretrained_overlap_counts']['unknown']==1 and r['paired_deltas'][0]['excluded_pair_count']==1


def test_paired_difference_and_frozen_identity():
    m,p,a=fixture();b=replace(a,mode='auto_timing',input_reference_hash=None,metrics=(('onset_f1',.6),))
    report=build_report(m,p,(a,b));assert report.payload['paired_deltas'][0]['auto_minus_oracle_mean']==pytest.approx(-.4)
    assert build_report(m,p,(a,b)).artifact_hash==report.artifact_hash
    with pytest.raises(ValueError,match='frozen'):build_report(m,p,(replace(a,plan_hash='5'*64),))
    with pytest.raises(ValueError,match='Duplicate'):build_report(m,p,(a,a))


def test_pseudo_human_and_different_reference_cannot_enter_gold_report():
    m,p,a=fixture();s=m.songs[0];refs=tuple(replace(r,origin='automatic_confirmed') if r.kind=='onsets' else r for r in s.references)
    m=freeze((replace(s,references=refs),))
    with pytest.raises(ValueError,match='gold'):freeze_plan(m,candidate_hash='1'*64,configuration_hash='2'*64,metric_profile_hash='3'*64)
    with pytest.raises(ValueError):EvaluationRun(p.artifact_hash,'a','auto_timing','e'*64,None,'failed',(('onset_f1',1.),),failure_code='OOM')


def test_uncertainty_resamples_families_not_correlated_covers():
    from mdac2.evaluation.report import interval
    assert interval([0.,1.],['one','one']) is None
    assert interval([0.,1.],['one','two'])==[0.,1.]
