from dataclasses import replace
import pytest
from mdac2.domain.stems import StemTrack,StemBundle,OutputRole,OutputProfile
from mdac2.ports.stems import PCMStem


def bundle():
    roles=OutputProfile('native-two-v1',(OutputRole('vocals',('vocals',)),OutputRole('accompaniment',('accompaniment',))))
    tracks=tuple(StemTrack(r.role,c*64,d*64,48000,96000,('L','R'),.1,.5,'model_output') for r,c,d in zip(roles.roles,'bc','de'))
    return StemBundle('a'*64,roles,tracks)


def test_two_tracks_capabilities_and_roundtrip():
    b=bundle();assert StemBundle.from_dict(b.to_dict())==b
    report=b.require(('vocals',),('drums','bass'))
    assert report.available and report.missing_optional==('drums','bass')
    assert b.require(('drums',)).missing_required==('drums',)
    assert len(b.tracks)==2
    copy=b.to_dict();copy['tracks'][0]['rms']=.2
    assert StemBundle.from_dict(copy).fingerprint!=b.fingerprint


def test_derived_sources_and_other_is_not_accompaniment():
    b=bundle();derived=replace(b.tracks[1],provenance='derived',source_hashes=('e'*64,'f'*64),derivation='sum-pcm-v1')
    assert replace(b,tracks=(b.tracks[0],derived)).tracks[1].provenance=='derived'
    other=StemBundle('a'*64,OutputProfile('other-v1',(OutputRole('other',('other',)),)),(replace(b.tracks[0],role='other'),))
    assert not other.require(('accompaniment',)).available
    with pytest.raises(ValueError):replace(derived,source_hashes=())
    with pytest.raises(ValueError):replace(b.tracks[0],source_hashes=('f'*64,))


@pytest.mark.parametrize('change',[{'sample_count':0},{'sample_rate':True},{'rms':float('nan')},{'peak':.01},{'offset_samples':.5},{'provenance':'unknown'},{'role':'../vocals'}])
def test_invalid_tracks(change):
    with pytest.raises(ValueError):replace(bundle().tracks[0],**change)


def test_bundle_shape_time_and_pcm_rejected():
    b=bundle()
    with pytest.raises(ValueError):replace(b,tracks=b.tracks[:1])
    with pytest.raises(ValueError):replace(b,tracks=tuple(reversed(b.tracks)))
    with pytest.raises(ValueError):replace(b,tracks=(b.tracks[0],replace(b.tracks[1],sample_count=95999)))
    with pytest.raises(ValueError):OutputProfile('bad',(OutputRole('a',('vocals',)),OutputRole('b',('vocals',))))
    with pytest.raises(ValueError):StemBundle.from_dict({**b.to_dict(),'extra':True})
    with pytest.raises(ValueError):PCMStem('vocals',b'\x00',48000,1,2)
    assert PCMStem('vocals',bytes(4),48000,1,2).sample_count==1  # real silence may be valid
