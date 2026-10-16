from dataclasses import replace
from fractions import Fraction as F
import pytest
from mdac2.domain.review import QualityItem,TimingQuality,review_timing
from mdac2.domain.timing_edits import TimingRevision,edit_timing,EditConflict
from mdac2.analysis.timing_fit import fit_timing
from mdac2.ports.timing import TimingObservation,TimingInput
from mdac2.policies.timing import assess_fit,TimingReviewPolicy


def candidate():
    observation=TimingObservation.create(TimingInput('a'*64,100,1000,1),producer={'name':'fixture','version':'1','model_hash':None,'dependencies':{}},options={},beat_times=[1+i*.5 for i in range(16)],downbeat_times=[])
    fit=fit_timing(observation);revision=TimingRevision('t',fit.mapping)
    return revision,assess_fit(revision,fit),fit


def accept(revision,quality):
    return review_timing(revision,quality,expected_fingerprint=revision.fingerprint,decision='accepted',actor='tester',reason='listened to candidate',timestamp='2026-09-20T22:00:00+08:00')


def test_quality_does_not_auto_accept_and_review_does_not_change_metrics():
    revision,quality,_=candidate();before=quality.fingerprint
    assert revision.review=='pending'
    assert {i.name for i in quality.items}=={'residual_p95','residual_max','coverage','pulse_meter','exportability'}
    reviewed,decision=accept(revision,quality)
    assert reviewed.review=='accepted' and decision.is_current(reviewed,quality)
    assert quality.fingerprint==before and quality.items[3].status=='unknown'
    assert not hasattr(decision,'reference_eligibility')


def test_old_approval_does_not_apply_to_edit_or_new_revision():
    revision,quality,_=candidate();reviewed,decision=accept(revision,quality)
    edited=edit_timing(reviewed,expected_fingerprint=reviewed.fingerprint,operation='drag_beat',q=reviewed.mapping.anchors[-1].q,t=8.4,actor='tester',reason='adjust',timestamp='now').after
    assert edited.review=='pending' and not decision.is_current(edited,quality)
    assert not decision.is_current(replace(reviewed,revision_id='new'),quality)
    changed=replace(quality,items=quality.items+(QualityItem('extra','warn',None,'none'),))
    assert not decision.is_current(reviewed,changed)
    with pytest.raises(EditConflict):accept(edited,quality)


def test_failed_gate_cannot_be_accepted_but_can_be_rejected():
    revision,quality,_=candidate()
    quality=replace(quality,items=(QualityItem('alignment','fail',.5,'seconds'),))
    with pytest.raises(ValueError):accept(revision,quality)
    after,record=review_timing(revision,quality,expected_fingerprint=revision.fingerprint,decision='rejected',actor='tester',reason='bad alignment',timestamp='now')
    assert record.is_current(after,quality)


def test_stale_fit_policy_and_missing_actor_rejected():
    revision,quality,fit=candidate()
    assert assess_fit(revision,fit,policy=TimingReviewPolicy(minimum_coverage=.5)).fingerprint!=quality.fingerprint
    with pytest.raises(ValueError):review_timing(revision,quality,expected_fingerprint=revision.fingerprint,decision='accepted',actor='',reason='x',timestamp='now')
    with pytest.raises(ValueError):assess_fit(replace(revision,mapping=replace(revision.mapping,audio_duration=20)),fit)
