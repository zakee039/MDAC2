from dataclasses import replace
from fractions import Fraction as F
import pytest
from mdac2.ports.timing import TimingInput,TimingObservation
from mdac2.analysis.timing_fit import FitPolicy,fit_timing


def observation(times,duration=30):
    return TimingObservation.create(TimingInput('a'*64,1000,duration*1000,1),producer={'name':'fixture','version':'1','model_hash':None,'dependencies':{}},options={},beat_times=times,downbeat_times=[])


def test_fixed_priority_under_jitter_and_does_not_mutate_observations():
    times=[.2+i*.5+([-.01,.008,0,.006][i%4]) for i in range(40)]
    obs=observation(times);before=obs.encoded
    result=fit_timing(obs)
    assert result.accepted and result.model_type=='fixed' and len(result.mapping.anchors)==2
    assert result.max_seconds<.035 and len(result.residual_seconds)==40
    assert obs.encoded==before and result.coverage_seconds==(times[0],times[-1])
    assert result.mapping.extrapolation_seconds==0


def test_real_tempo_step_requires_two_segments():
    times=[.2+i*.5 if i<=12 else 6.2+(i-12)*.4 for i in range(30)]
    result=fit_timing(observation(times))
    assert result.accepted and result.model_type=='piecewise'
    assert len(result.mapping.anchors)==3 and result.max_seconds<1e-7
    assert not fit_timing(observation(times),policy=FitPolicy(max_segments=1)).accepted


def test_gradual_tempo_or_outlier_rejected_without_dense_fake_tempo():
    result=fit_timing(observation([.2+.45*i+.003*i*i for i in range(40)]))
    assert not result.accepted and result.mapping is None
    times=[.2+i*.5 for i in range(30)];times[15]+=.2
    result=fit_timing(observation(times))
    assert not result.accepted and result.max_seconds>.08


def test_compound_pulse_and_pickup_are_explicit_not_inferred():
    result=fit_timing(observation([.25+.75*i for i in range(12)]),pulse_unit=F(3,8),first_q=F(-3,2),meter=(6,8))
    assert result.accepted and result.mapping.anchors[0].q==F(-3,2)
    assert result.mapping.local_bpm(0)==120
    assert result.mapping.meter_at(0).bar_quarters==3
    other=fit_timing(observation([.25+.75*i for i in range(12)]))
    assert other.interpretation_hash!=result.interpretation_hash


def test_limits_and_policy_identity():
    assert fit_timing(observation([])).reason=='INSUFFICIENT_BEATS'
    assert fit_timing(observation([.2+i*.5 for i in range(10)]),policy=FitPolicy(max_observations=6)).reason=='OBSERVATION_LIMIT'
    assert FitPolicy().fingerprint!=FitPolicy(tolerance_seconds=.02).fingerprint
    with pytest.raises(ValueError):FitPolicy(complexity_penalty=float('nan'))
    with pytest.raises(ValueError):FitPolicy(max_segments=100)
    with pytest.raises(ValueError):fit_timing(observation([.2+i*.5 for i in range(10)]),pulse_unit=0)
