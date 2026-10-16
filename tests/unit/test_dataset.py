from dataclasses import replace
import json
import pytest
from mdac2.evaluation.dataset_manifest import Reference,DatasetSong,DatasetManifest


def song(identity='a',family='family',role='locked_test'):
    reference=Reference(identity*64,'v1','timing','human_annotation',('f' if identity!='f' else 'e')*64,'annotator','reviewed','reviewer')
    return DatasetSong(identity,identity*64,('c' if identity=='a' else 'd')*64,family,role,'self-created',('training','evaluation'),'normal',('fixed','4/4'),'unknown',(reference,))


def freeze(songs):return DatasetManifest.freeze(songs,version='1',family_policy='covers-remixes-together-v1',split_seed=42)


def test_family_pcm_and_label_leakage_rejected():
    a=song();b=song('b',role='train')
    with pytest.raises(ValueError,match='family leakage'):freeze((a,b))
    with pytest.raises(ValueError,match='normalized'):freeze((a,replace(b,work_family='other',pcm_hash=a.pcm_hash)))
    with pytest.raises(ValueError,match='lineage leakage'):freeze((a,replace(b,work_family='other')))


def test_frozen_identity_tamper_and_label_version_changes():
    a=song();m=freeze((a,));assert DatasetManifest.load(m.artifact.to_json(),expected_hash=m.fingerprint)==m
    document=json.loads(m.artifact.to_json());document['payload']['songs'][0]['work_family']='tampered'
    with pytest.raises(ValueError):DatasetManifest.load(json.dumps(document),expected_hash=m.fingerprint)
    changed=freeze((replace(a,references=(replace(a.references[0],version='v2'),)),))
    assert changed.fingerprint!=m.fingerprint
    with pytest.raises(ValueError):DatasetManifest.load(changed.artifact.to_json(),expected_hash=m.fingerprint)


def test_automatic_confirmation_is_not_gold_and_unknown_overlap_retained():
    a=song();assert a.references[0].gold_eligible
    assert not replace(a.references[0],origin='automatic_confirmed').gold_eligible
    assert not replace(a.references[0],reviewer='annotator').gold_eligible
    assert freeze((a,)).songs[0].pretrained_overlap=='unknown'
    with pytest.raises(ValueError):replace(a,usage_scope=('training',))
