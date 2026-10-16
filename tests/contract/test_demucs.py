import hashlib
from dataclasses import replace
import pytest
from mdac2.adapters.separation.demucs import DemucsProvider,SeparationOptions
from mdac2.infrastructure.model_registry import ModelSpec,ModelRegistry
from mdac2.ports.timing import TimingInput
from mdac2.ports.stems import PCMStem


class Backend:
    name='fake-demucs-contract';version='test';engine='fake-demucs';engine_version='1'
    def __init__(self):self.calls=[];self.change=lambda r:r
    def infer(self,**kwargs):
        self.calls.append(kwargs);a=kwargs['audio']
        return self.change({'stems':{r:PCMStem(r,kwargs['pcm'],a.sample_rate,a.sample_count,a.channels) for r in ('vocals','other','bass','drums')},
            'model_hash':hashlib.sha256(kwargs['checkpoint'].read_bytes()).hexdigest(),'effective_options':kwargs['options'],
            'engine':self.engine,'engine_version':self.engine_version})


def setup(tmp_path):
    p=tmp_path/'fake.ckpt';p.write_bytes(b'contract-only')
    spec=ModelSpec('fake',p.name,hashlib.sha256(p.read_bytes()).hexdigest(),'fixture','fixture-only',('inference',))
    backend=Backend();provider=DemucsProvider(ModelRegistry(tmp_path,[spec]),'fake',backend)
    return provider,backend,TimingInput('a'*64,48000,100,2),bytes(400)


def test_named_outputs_canonical_order_and_actual_identity(tmp_path):
    p,b,a,pcm=setup(tmp_path);result=p.run(a,pcm)
    assert tuple(s.role for s in result.outputs)==('drums','bass','other','vocals')
    assert result.producer['dependencies']['engine']=='fake-demucs'
    assert result.options['allow_download'] is False
    assert b.calls[0]['checkpoint'].is_absolute()


def test_missing_weights_and_dependency_fail_before_backend(tmp_path):
    p,b,a,pcm=setup(tmp_path);(tmp_path/'fake.ckpt').unlink()
    with pytest.raises(ValueError,match='WEIGHTS_MISSING'):p.run(a,pcm)
    assert not b.calls
    (tmp_path/'fake.ckpt').write_bytes(b'contract-only')
    p.registry.specs['fake']=replace(p.registry.specs['fake'],dependencies=(('mdac-no-such-dependency','1'),))
    with pytest.raises(ValueError,match='DEPENDENCY_MISSING'):p.run(a,pcm)
    assert not b.calls


@pytest.mark.parametrize('change',[
    lambda r:{**r,'model_hash':'a'*64},
    lambda r:{**r,'effective_options':{**r['effective_options'],'device':'cuda:0'}},
    lambda r:{**r,'stems':{k:v for k,v in r['stems'].items() if k!='other'}},
    lambda r:{**r,'engine':'silent-fallback'},
    lambda r:{**r,'stems':{**r['stems'],'vocals':replace(r['stems']['vocals'],offset_samples=1)}}])
def test_invalid_backend_results(tmp_path,change):
    p,b,a,pcm=setup(tmp_path);b.change=change
    with pytest.raises(ValueError):p.run(a,pcm)


def test_invalid_options_and_pcm(tmp_path):
    p,b,a,pcm=setup(tmp_path)
    with pytest.raises(ValueError):p.run(a,pcm[:-1])
    with pytest.raises(ValueError):SeparationOptions(device='cpu',precision='float16')
    with pytest.raises(ValueError):SeparationOptions(overlap=1)
