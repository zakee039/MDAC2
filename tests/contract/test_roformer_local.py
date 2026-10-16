from dataclasses import asdict
from pathlib import Path
import pytest
from mdac2.adapters.separation.demucs import SeparationOptions
from mdac2.adapters.separation.roformer_local import LocalRoformerBackend
from mdac2.ports.timing import TimingInput


def options():return {**asdict(SeparationOptions()),'allow_download':False,'output_format':'pcm_s16le','backend_policy':LocalRoformerBackend.policy}


@pytest.mark.parametrize('change',[{'allow_download':True},{'shifts':1},{'device':'cuda:0'},{'segment_seconds':9},{'backend_policy':{}}])
def test_unsupported_roformer_options_fail_before_opening_weights(change):
    with pytest.raises(ValueError):LocalRoformerBackend('no-config').infer(checkpoint=Path('no-weights'),audio=TimingInput('a'*64,48000,100,2),pcm=bytes(400),options={**options(),**change})


def test_config_tampering_fails_before_deserialization(tmp_path):
    path=tmp_path/'config.yaml';path.write_text('model: unverified')
    with pytest.raises(ValueError,match='config identity'):
        LocalRoformerBackend(path).infer(checkpoint=Path('no-weights'),audio=TimingInput('a'*64,48000,100,2),pcm=bytes(400),options=options())


def test_excessive_overlap_rejected_before_config_or_weights():
    with pytest.raises(ValueError,match='chunk limit'):
        LocalRoformerBackend('no-config').infer(checkpoint=Path('no-weights'),audio=TimingInput('a'*64,48000,480000,2),pcm=bytes(1920000),options={**options(),'overlap':.999999})
