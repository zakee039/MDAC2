import hashlib
import pytest
from mdac2.adapters.beat_this import BeatThisProvider,BeatThisOptions
from mdac2.ports.timing import TimingInput
from mdac2.infrastructure.model_registry import ModelRegistry,ModelSpec


class Fake:
    name='fake-contract'
    version='1'
    def __init__(self):self.calls=[];self.change=lambda result:result
    def infer(self,**kwargs):
        self.calls.append(kwargs)
        result={'beat_times':[.13,.63,1.13],'downbeat_times':[.13],
            'model_hash':hashlib.sha256(kwargs['checkpoint'].read_bytes()).hexdigest(),
            'effective_options':kwargs['options'],'warnings':['FAKE_BACKEND_NO_QUALITY_CLAIM'],
            'frame_logits':{'start_seconds':0,'step_seconds':.5,'beat':[-3,2,1],'downbeat':[1,-2,0]} if kwargs['include_logits'] else None}
        return self.change(result)


def setup(tmp_path):
    blob=b'fake test weights';(tmp_path/'fake.ckpt').write_bytes(blob)
    spec=ModelSpec('fake','fake.ckpt',hashlib.sha256(blob).hexdigest(),'fixture','fixture-only',('inference',))
    backend=Fake();registry=ModelRegistry(tmp_path,[spec])
    return BeatThisProvider(registry,'fake',backend),backend,TimingInput('a'*64,100,200,2),bytes(800)


def test_raw_time_logits_and_effective_identity(tmp_path):
    provider,fake,audio,pcm=setup(tmp_path)
    result=provider.observe(audio,pcm,include_logits=True)
    assert result.payload['beat_times']==[.13,.63,1.13]
    assert result.payload['frame_logits']['beat']==[-3,2,1]
    assert 'fake-contract' in result.artifact.to_dict()['producer']['name']
    assert fake.calls[0]['checkpoint'].is_absolute()
    without=provider.observe(audio,pcm)
    assert without.payload['frame_logits'] is None
    assert without.artifact.request_key!=result.artifact.request_key


def test_missing_weights_fails_before_backend(tmp_path):
    provider,fake,audio,pcm=setup(tmp_path)
    (tmp_path/'fake.ckpt').unlink()
    with pytest.raises(ValueError,match='WEIGHTS_MISSING'):provider.observe(audio,pcm)
    assert not fake.calls


@pytest.mark.parametrize('mutate',[
    lambda r:{**r,'effective_options':{**r['effective_options'],'device':'cuda:0'}},
    lambda r:{**r,'model_hash':'b'*64},
    lambda r:{**r,'beat_times':[1,.5]},
    lambda r:{**r,'frame_logits':{}},
    lambda r:{**r,'extra':True},
])
def test_backend_contract_and_silent_fallback_rejected(tmp_path,mutate):
    provider,fake,audio,pcm=setup(tmp_path);fake.change=mutate
    with pytest.raises(ValueError):provider.observe(audio,pcm)


def test_pcm_mismatch_and_options(tmp_path):
    provider,fake,audio,pcm=setup(tmp_path)
    with pytest.raises(ValueError):provider.observe(audio,pcm[:-1])
    assert not fake.calls
    with pytest.raises(ValueError):BeatThisOptions(precision='float16')
