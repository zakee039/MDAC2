from dataclasses import replace
from fractions import Fraction as F
import struct
import pytest
from mdac2.analysis.onset_candidates import CandidatePolicy,detect_candidates,quantize
from mdac2.domain.timing import Anchor,TimingMap
from mdac2.domain.timing_edits import TimingRevision
from mdac2.ports.timing import TimingInput


def audio(pulses):
    values=[0]*8000
    for sec in pulses:
        start=round(sec*1000)
        for i in range(start,min(start+30,4000)):values[2*i]=15000;values[2*i+1]=-15000
    return TimingInput('a'*64,1000,4000,2),struct.pack('<8000h',*values)


def timing():return TimingRevision('t',TimingMap((Anchor(0,0),Anchor(4,2),Anchor(12,4)),4))


def test_silence_stereo_impulses_and_no_density_fill():
    a,pcm=audio([.5,1,2.5]);result=detect_candidates(a,pcm,timing())
    assert [c.seconds for c in result.candidates]==[F(1,2),F(1),F(5,2)]
    assert [c.quantized_q for c in result.candidates]==[1,2,6]
    assert all(c.score>0 and c.resolution_seconds==F(1,100) for c in result.candidates)
    assert not detect_candidates(a,bytes(len(pcm)),timing()).candidates
    assert result==detect_candidates(a,pcm,timing())


def test_local_quantization_limits_and_unknown_coverage():
    p=CandidatePolicy();t=timing()
    assert quantize(F('2.53'),t,p)[1:3]==(F(6),F('-.03'))
    assert quantize(F('.55'),t,p)[1] is None
    assert quantize(F('.55'),t,p)[3]==('QUANTIZATION_LIMIT',)
    partial=replace(t,mapping=TimingMap((Anchor(1,.5),Anchor(7,3)),4))
    assert quantize(F('.1'),partial,p)[3]==('OUTSIDE_TIMING_COVERAGE',)
    a,pcm=audio([.1,.55]);result=detect_candidates(a,pcm,partial)
    assert len(result.candidates)==2 and result.candidates[0].quantized_q is None


def test_identity_and_bad_inputs():
    a,pcm=audio([1]);one=detect_candidates(a,pcm,timing())
    two=detect_candidates(a,pcm,replace(timing(),revision_id='new'))
    assert one.fingerprint!=two.fingerprint and one.candidates[0].candidate_id!=two.candidates[0].candidate_id
    with pytest.raises(ValueError):detect_candidates(a,pcm[:-1],timing())
    with pytest.raises(ValueError):detect_candidates(a,pcm,timing(),policy=CandidatePolicy(max_seconds=1))
    with pytest.raises(ValueError):CandidatePolicy(hop_seconds=float('nan'))
