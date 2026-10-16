from dataclasses import replace
import hashlib
import pytest
from mdac2.ports.timing import TimingInput,TimingObservation
from mdac2.infrastructure.model_registry import ModelRegistry,ModelSpec

PRODUCER={'name':'fake.contract','version':'1','model_hash':None,'dependencies':{}}
AUDIO=TimingInput('a'*64,48000,96000,2)


def test_observation_keeps_raw_time_and_detached_payload():
    obs=TimingObservation.create(AUDIO,producer=PRODUCER,options={'device':'cpu'},beat_times=[.103,.609,1.107],downbeat_times=[.105],warnings=['downbeat_offset'])
    assert obs.payload['beat_times']==[.103,.609,1.107]
    copy=obs.payload;copy['beat_times'][0]=99
    assert obs.payload['beat_times'][0]==.103
    assert TimingObservation(obs.encoded)==obs
    assert 'review' not in obs.payload and 'confidence' not in obs.payload


@pytest.mark.parametrize('beats',[[1,1],[1,.5],[-.1],[2.1],[True],[float('nan')]])
def test_invalid_times_rejected(beats):
    with pytest.raises(ValueError):TimingObservation.create(AUDIO,producer=PRODUCER,options={},beat_times=beats,downbeat_times=[])


def test_optional_logits_not_probability_and_silence():
    obs=TimingObservation.create(AUDIO,producer=PRODUCER,options={},beat_times=[],downbeat_times=[],warnings=['NO_BEATS'],frame_logits={'start_seconds':0,'step_seconds':.5,'beat':[-10,2,3],'downbeat':[5,-3,8]})
    assert obs.payload['frame_logits']['beat']==[-10,2,3]
    with pytest.raises(ValueError):TimingObservation.create(AUDIO,producer=PRODUCER,options={},beat_times=[],downbeat_times=[],frame_logits={'start_seconds':2,'step_seconds':1,'beat':[1,2],'downbeat':[1,2]})


def test_registry_only_checks_local_pinned_weights(tmp_path,monkeypatch):
    blob=b'fake checkpoint for interface test';digest=hashlib.sha256(blob).hexdigest()
    spec=ModelSpec('fake','fake.ckpt',digest,'fixture','fixture-only',('inference',))
    registry=ModelRegistry(tmp_path,[spec])
    assert registry.check('fake').code=='WEIGHTS_MISSING'
    assert not list(tmp_path.iterdir())
    (tmp_path/'fake.ckpt').write_bytes(blob)
    assert registry.require('fake')[1].available
    assert registry.check('fake',use='commercial').code=='USE_NOT_ALLOWED'
    (tmp_path/'fake.ckpt').write_bytes(b'x'*len(blob))
    assert registry.check('fake').code=='WEIGHTS_HASH_MISMATCH'
    (tmp_path/'fake.ckpt').write_bytes(blob)
    missing=ModelRegistry(tmp_path,[replace(spec,dependencies=(('mdac2-nonexistent-test-package','1'),))])
    assert missing.check('fake').code=='DEPENDENCY_MISSING'
    from mdac2.infrastructure import model_registry
    monkeypatch.setattr(model_registry.metadata,'version',lambda name:'2')
    assert missing.check('fake').code=='DEPENDENCY_VERSION_MISMATCH'


def test_registry_rejects_path_escape_and_duplicate_identity(tmp_path):
    with pytest.raises(ValueError):ModelSpec('f','../x','a'*64,'fixture','fixture',('inference',))
    spec=ModelSpec('f','x','a'*64,'fixture','fixture',('inference',))
    with pytest.raises(ValueError):ModelRegistry(tmp_path,[spec,spec])
