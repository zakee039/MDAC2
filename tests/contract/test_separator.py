from dataclasses import replace
import pytest
from test_demucs import setup
from mdac2.adapters.separation.audio_separator import AudioSeparatorProvider


def pair(tmp_path):
    p,b,a,pcm=setup(tmp_path)
    b.name='fake-audio-separator';b.engine='fake-melband-roformer'
    def change(r):
        return {**r,'stems':{'Vocals':r['stems']['vocals'],'Instrumental':replace(r['stems']['other'],role='accompaniment')}}
    b.change=change
    return AudioSeparatorProvider(p.registry,'fake',b,native_labels=('Vocals','Instrumental')),b,a,pcm


def test_two_track_mapping_keeps_wrapper_and_engine(tmp_path):
    p,b,a,pcm=pair(tmp_path);result=p.run(a,pcm)
    assert [o.role for o in result.outputs]==['vocals','accompaniment']
    assert result.producer['name'].endswith('fake-audio-separator')
    assert result.producer['dependencies']['engine']=='fake-melband-roformer'
    assert result.options['native_roles']==['Vocals','Instrumental']
    assert b.calls[0]['options']['allow_download'] is False


def test_missing_tracks_and_implicit_aliases_fail(tmp_path):
    p,b,a,pcm=pair(tmp_path)
    with pytest.raises(ValueError):AudioSeparatorProvider(p.registry,'fake',b,native_labels=('Vocals','other'))
    change=b.change;b.change=lambda r:{**change(r),'stems':{'Vocals':r['stems']['vocals']}}
    with pytest.raises(ValueError,match='native stems'):p.run(a,pcm)


def test_effective_precision_and_changed_checkpoint_fail(tmp_path):
    p,b,a,pcm=pair(tmp_path);change=b.change
    b.change=lambda r:{**change(r),'effective_options':{**r['effective_options'],'precision':'float16'}}
    with pytest.raises(ValueError,match='fallback'):p.run(a,pcm)
    def tamper(r):
        (tmp_path/'fake.ckpt').write_bytes(b'changed-by-fixture')
        return change(r)
    b.change=tamper
    with pytest.raises(ValueError,match='HASH_MISMATCH'):p.run(a,pcm)
