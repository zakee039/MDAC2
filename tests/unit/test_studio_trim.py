import json
import numpy as np
import pytest
import soundfile as sf
from mdac2.application.studio_trim import settings,prepare,waveform,suggest
from mdac2.data_preparation.audio_features import sha


def test_invalid_intervals_and_legacy_short_whole_song():
    assert settings(settings(None,.1),.1)==dict(start_ms=0,end_ms=100)
    for value in [dict(start_ms=-1,end_ms=2000),dict(start_ms=0,end_ms=500),dict(start_ms=0,end_ms=5001),dict(start_ms=float('nan'),end_ms=2000),dict(start_ms=.5,end_ms=2000)]:
        with pytest.raises(ValueError):settings(value,5)


def test_sample_exact_crop_preserves_original_and_checks_cache(tmp_path):
    source=tmp_path/'original';source.mkdir();rate=44100
    x=np.random.default_rng(7).normal(0,.1,(rate*4,2)).astype('float32')
    sf.write(source/'mix.wav',x,rate,subtype='FLOAT')
    original_hash=sha(source/'mix.wav')
    info=dict(source_sha256='original-file',wav_sha256=original_hash,duration=4,sample_rate=rate)
    (source/'audio.json').write_text(json.dumps(info))
    prepared=dict(path=str(source),record_sha256=sha(source/'audio.json'))
    interval=dict(start_ms=123,end_ms=3123);folder=tmp_path/'cropped'
    record=prepare(prepared,folder,interval);y,sr=sf.read(folder/'mix.wav',dtype='float32',always_2d=True)
    assert sr==rate and np.array_equal(y,x[round(.123*rate):round(3.123*rate)])
    assert record['source_sha256']=='original-file' and record['duration']==3
    assert sha(source/'mix.wav')==original_hash
    assert prepare(prepared,folder,interval)==record
    with pytest.raises(ValueError):prepare(prepared,folder,dict(start_ms=124,end_ms=3123))
    wave=waveform(source/'mix.wav',.123,.623,500)
    assert wave['start']==.123 and wave['end']==.623 and len(wave['peaks'])==500


def test_structure_analysis_abstains_on_silence(tmp_path):
    path=tmp_path/'silent.wav';sf.write(path,np.zeros((44100*45,2),np.float32),44100)
    result=suggest(path)
    assert result['candidate_ms'] is None and result['method']=='structure-candidate-v1'
