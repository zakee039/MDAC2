from pathlib import Path
import pytest
from mdac2.adapters.separation.demucs import SeparationOptions
from mdac2.adapters.separation.demucs_local import LocalDemucsBackend
from mdac2.ports.timing import TimingInput
from dataclasses import asdict


def options():
    return {**asdict(SeparationOptions()),'allow_download':False,'output_format':'pcm_s16le','backend_policy':dict(LocalDemucsBackend.policy)}


def test_local_backend_rejects_unverified_checkpoint_before_deserialization(tmp_path):
    checkpoint=tmp_path/'untrusted.th';checkpoint.write_bytes(b'not trusted model bytes')
    with pytest.raises(ValueError,match='Unrecognized'):
        LocalDemucsBackend().infer(checkpoint=checkpoint,audio=TimingInput('a'*64,48000,100,2),pcm=bytes(400),options=options())


@pytest.mark.parametrize('change',[{'device':'cuda:0'},{'allow_download':True},{'segment_seconds':8},{'backend_policy':{}}])
def test_unsupported_runtime_never_loads_weights(change):
    with pytest.raises(ValueError):
        LocalDemucsBackend().infer(checkpoint=Path('must-not-open'),audio=TimingInput('a'*64,48000,100,2),pcm=bytes(400),options={**options(),**change})


@pytest.mark.parametrize('change',[{'overlap':.999999},{'segment_seconds':.1,'overlap':.9,'shifts':10}])
def test_excessive_chunk_work_rejected_before_loading(change):
    with pytest.raises(ValueError,match='chunk evaluation limit'):
        LocalDemucsBackend().infer(checkpoint=Path('must-not-open'),audio=TimingInput('a'*64,48000,480000,2),pcm=bytes(1920000),options={**options(),**change})
