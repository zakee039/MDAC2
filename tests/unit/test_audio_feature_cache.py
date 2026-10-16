import json
from pathlib import Path
import numpy as np
import pytest
from mdac2.data_preparation import audio_features
from mdac2.data_preparation.audio_features import FEATURE_CONFIG,config_hash,sha,validate_cache


def cache(tmp_path):
    np.save(tmp_path/'features.npy',np.zeros((3,96,10),dtype=np.float16),allow_pickle=False)
    for name in ('vocals.wav','accompaniment.wav'):(tmp_path/name).write_bytes(b'fixture')
    data={'config_sha256':config_hash(),'config':FEATURE_CONFIG,
          'implementation_sha256':sha(Path(audio_features.__file__)),
          'source_pcm_sha256':'pcm','partial_excerpt':False,'feature_shape':[3,96,10],
          'files':{n:sha(tmp_path/n) for n in ('features.npy','vocals.wav','accompaniment.wav')}}
    (tmp_path/'metadata.json').write_text(json.dumps(data),encoding='utf-8')
    return {'pcm':{'sha256':'pcm'}}


def test_cache_validates_each_stem_and_feature_identity(tmp_path):
    row=cache(tmp_path)
    assert validate_cache(tmp_path,row)['feature_shape']==[3,96,10]
    (tmp_path/'accompaniment.wav').write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='checksum'):validate_cache(tmp_path,row)


@pytest.mark.parametrize('change',[{'partial_excerpt':True},{'config_sha256':'old'},
                                  {'implementation_sha256':'old'},{'source_pcm_sha256':'other'}])
def test_partial_or_stale_cache_never_reused(tmp_path,change):
    row=cache(tmp_path);path=tmp_path/'metadata.json'
    data=json.loads(path.read_text());data.update(change);path.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='Incompatible'):validate_cache(tmp_path,row)


def test_nonfinite_cache_is_rejected_even_when_checksum_matches(tmp_path):
    row=cache(tmp_path);a=np.zeros((3,96,10),dtype=np.float16);a[0,0,0]=np.nan
    np.save(tmp_path/'features.npy',a)
    path=tmp_path/'metadata.json';data=json.loads(path.read_text())
    data['files']['features.npy']=sha(tmp_path/'features.npy');path.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='Corrupt'):validate_cache(tmp_path,row)
