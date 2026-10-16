from dataclasses import replace
from fractions import Fraction as F
import hashlib
import struct
import pytest
from mdac2.analysis.onset_candidates import OnsetSet,OnsetCandidate
from mdac2.analysis.stem_evidence import fuse_evidence,EvidencePolicy
from mdac2.domain.stems import OutputProfile,OutputRole,StemTrack,StemBundle


def fixture(amplitude=1000):
    pcm=struct.pack('<1000h',*([0]*500+[amplitude]*50+[0]*450))
    h=hashlib.sha256(pcm).hexdigest()
    bundle=StemBundle('a'*64,OutputProfile('one',(OutputRole('vocals',('vocals',)),)),
        (StemTrack('vocals','b'*64,h,1000,1000,('M',),amplitude/32768*(.05**.5),amplitude/32768,'model_output'),))
    candidate=OnsetCandidate('c'*64,F(1,2),F(1),F(1),F(0),.6,.1,F(1,100))
    onsets=OnsetSet('a'*64,'d'*64,'e'*64,(candidate,))
    return onsets,bundle,{'vocals':pcm}


def test_missing_optional_visible_and_no_candidate_movement():
    o,b,pcm=fixture();before=o.fingerprint;result=fuse_evidence(o,b,pcm)
    assert result.missing_optional==('drums',)
    assert result.candidates[0].candidate_id==o.candidates[0].candidate_id
    assert o.fingerprint==before and len(result.candidates)==len(o.candidates)
    assert 'MISSING_OPTIONAL_DRUMS' in result.candidates[0].reasons
    with pytest.raises(ValueError,match='REQUIRED'):fuse_evidence(o,b,pcm,policy=EvidencePolicy(required=('drums',)))


def test_gray_zone_is_continuous_and_identity_tracks_pcm():
    a=fuse_evidence(*fixture(1000));b=fuse_evidence(*fixture(1001))
    assert 0<b.candidates[0].score-a.candidates[0].score<.001
    assert a.fingerprint!=b.fingerprint
    o,bundle,pcm=fixture(0);result=fuse_evidence(o,bundle,pcm)
    assert result.candidates[0].support==( ('vocals',0), )


def test_wrong_audio_tamper_and_offset_fail():
    o,b,pcm=fixture()
    with pytest.raises(ValueError):fuse_evidence(replace(o,audio_hash='f'*64),b,pcm)
    with pytest.raises(ValueError):fuse_evidence(o,b,{'vocals':bytes(2000)})
    with pytest.raises(ValueError):fuse_evidence(o,replace(b,tracks=(replace(b.tracks[0],offset_samples=1),)),pcm)
