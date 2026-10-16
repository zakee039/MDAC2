from dataclasses import replace
import io
import random
import struct
import wave
import pytest
from mdac2.adapters.audio_io import AudioDecoder
from mdac2.application.stem_pipeline import publish_stems,load_stems,AlignmentPolicy
from mdac2.domain.stems import OutputProfile,OutputRole
from mdac2.infrastructure.artifact_store import ArtifactStore,StoreError
from mdac2.ports.stems import PCMStem
from pathlib import Path


def setup(tmp_path):
    root=Path(__file__).parents[2];binary=root/'tools/.deps/ffmpeg/extracted/ffmpeg-9.0.1-essentials_build/bin'
    store=ArtifactStore(tmp_path/'store');decoder=AudioDecoder(binary/'ffmpeg.exe',binary/'ffprobe.exe')
    rng=random.Random(1);values=[rng.randint(-10000,10000) for _ in range(1000)]
    pcm=struct.pack('<'+'h'*2000,*[v for v in values for _ in range(2)])
    stream=io.BytesIO()
    with wave.open(stream,'wb') as wav:wav.setparams((2,2,48000,0,'NONE','none'));wav.writeframes(pcm)
    audio=decoder.import_audio(stream.getvalue(),store)
    profile=OutputProfile('fixture-two',(OutputRole('vocals',('vocals',)),OutputRole('accompaniment',('accompaniment',))))
    outputs=(PCMStem('vocals',pcm,48000,1000,2),PCMStem('accompaniment',bytes(len(pcm)),48000,1000,2))
    kwargs=dict(producer={'name':'fake-stems-contract','version':'1','model_hash':None,'dependencies':{}},options={'mode':'fixture'})
    return store,audio,profile,outputs,kwargs


def test_atomic_multifile_and_same_size_tampering(tmp_path):
    store,audio,profile,outputs,kwargs=setup(tmp_path)
    result=publish_stems(store,audio,profile,outputs,**kwargs)
    bundle,report=load_stems(store,result.artifact_hash)
    assert len(bundle.tracks)==2 and report['shift_samples']==0
    assert bundle.tracks[1].rms==0
    path=store._destination('stems',result.artifact_hash)/'vocals.wav'
    data=bytearray(path.read_bytes());data[-1]^=1;path.write_bytes(data)
    with pytest.raises(StoreError,match='content mismatch'):load_stems(store,result.artifact_hash)


def test_disk_full_no_partial_bundle_and_retry(tmp_path,monkeypatch):
    store,audio,profile,outputs,kwargs=setup(tmp_path);original=store._write;count=0
    def failing(path,data):
        nonlocal count
        count+=1
        if count==2:raise OSError('disk full')
        original(path,data)
    monkeypatch.setattr(store,'_write',failing)
    with pytest.raises(OSError):publish_stems(store,audio,profile,outputs,**kwargs)
    assert not list((store.artifacts/'stems').iterdir())
    monkeypatch.setattr(store,'_write',original)
    result=publish_stems(store,audio,profile,outputs,**kwargs)
    assert load_stems(store,result.artifact_hash)[0].require(('vocals',)).available


def test_length_delay_and_reconstruction_fail_without_publication(tmp_path):
    store,audio,profile,outputs,kwargs=setup(tmp_path)
    bad=replace(outputs[0],sample_count=999,pcm=outputs[0].pcm[:-4])
    with pytest.raises(ValueError,match='implicit'):publish_stems(store,audio,profile,(bad,outputs[1]),**kwargs)
    delayed=replace(outputs[0],pcm=bytes(20)+outputs[0].pcm[:-20])
    with pytest.raises(ValueError,match='offset'):publish_stems(store,audio,profile,(delayed,outputs[1]),policy=AlignmentPolicy(max_reconstruction_relative_rms=2),**kwargs)
    with pytest.raises(ValueError,match='Reconstruction'):publish_stems(store,audio,profile,(replace(outputs[0],pcm=bytes(4000)),outputs[1]),**kwargs)
    assert not (store.artifacts/'stems').exists()


def test_residual_provenance_verified_in_samples_and_on_reload(tmp_path):
    store,audio,profile,outputs,kwargs=setup(tmp_path)
    kwargs['options']['backend_policy']={'derivations':{'accompaniment':{'operation':'mix_minus_vocals_pcm16','sources':['mix','vocals']}}}
    result=publish_stems(store,audio,profile,outputs,**kwargs)
    bundle,report=load_stems(store,result.artifact_hash)
    derived=bundle.tracks[1]
    assert derived.provenance=='derived' and derived.source_hashes==(audio.artifact_hash,bundle.tracks[0].audio_hash)
    assert 'RESIDUAL_RECONSTRUCTION_IS_AN_IDENTITY_NOT_SEPARATION_QUALITY' in report['warnings']
    bad=replace(outputs[1],pcm=b'\x01\x00'+outputs[1].pcm[2:])
    with pytest.raises(ValueError,match='exact residual'):publish_stems(store,audio,profile,(outputs[0],bad),**kwargs)
