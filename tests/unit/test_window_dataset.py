import json
from pathlib import Path
import numpy as np
import pytest
from mdac2.data_preparation import audio_features
from mdac2.data_preparation.audio_features import sha,config_hash,FEATURE_CONFIG
from mdac2.data_preparation.training_chart import CLASSES,POLICY_ID
from mdac2.data_preparation.window_dataset import WindowDataset


def fixture(tmp_path):
    label={'policy':POLICY_ID,'chart_sha256':'chart','duration':.1,'events':[],
           'excluded':[],'training_eligible':True,
           'timing':{'tempos':[{'seconds':0,'quarter':'0','bpm':'120'}]}}
    (tmp_path/'label.json').write_text(json.dumps(label))
    row={'chart_sha256':'chart','split':'train','training_eligible':True,'labels':'label.json',
         'labels_sha256':sha(tmp_path/'label.json'),'pcm':{'sha256':'pcm','seconds':.1},'rating':'9'}
    m={'schema_version':1,'policy':POLICY_ID,'classes':CLASSES,'charts':[row]}
    (tmp_path/'manifest.json').write_text(json.dumps(m))
    cache=tmp_path/'cache';directory=cache/'pcm';directory.mkdir(parents=True)
    np.save(directory/'features.npy',np.zeros((3,96,10),dtype=np.float16),allow_pickle=False)
    for name in ('vocals.wav','accompaniment.wav'):(directory/name).write_bytes(b'fixture')
    metadata={'config':FEATURE_CONFIG,'config_sha256':config_hash(),
        'implementation_sha256':sha(Path(audio_features.__file__)),
        'source_pcm_sha256':'pcm','partial_excerpt':False,'feature_shape':[3,96,10],
        'frame_seconds':.01,'feature_origin_seconds':0,
        'files':{name:sha(directory/name) for name in ('features.npy','vocals.wav','accompaniment.wav')}}
    (directory/'metadata.json').write_text(json.dumps(metadata))
    report={'status':'completed','manifest_sha256':sha(tmp_path/'manifest.json'),'errors':[],
            'complete':[{'pcm_sha256':'pcm','metadata_sha256':sha(directory/'metadata.json')}]}
    (cache/'report.json').write_text(json.dumps(report))
    return tmp_path/'manifest.json',cache


def test_dataset_has_only_expected_audio_and_timing_model_inputs(tmp_path):
    manifest,cache=fixture(tmp_path);dataset=WindowDataset(manifest,cache,'train')
    sample=dataset[0]
    assert sample['features'].shape==(3,96,3200)
    assert sample['features'].dtype==np.float32 and sample['loss_region'].sum()==10
    assert sample['onset_mask'].sum()==10*2*6
    assert not set(sample)&{'title','source_ids','source_code','chart_sha256'}
    assert len(WindowDataset(manifest,cache,'validation'))==0
    with pytest.raises(ValueError,match='Locked'):WindowDataset(manifest,cache,'locked_test')


def test_incomplete_feature_pipeline_blocks_training_loader(tmp_path):
    manifest,cache=fixture(tmp_path);path=cache/'report.json'
    report=json.loads(path.read_text());report['status']='running';path.write_text(json.dumps(report))
    with pytest.raises(ValueError,match='Complete'):WindowDataset(manifest,cache,'train')
    assert len(WindowDataset(manifest,cache,'train',allow_partial_cache=True))==1


def test_changed_label_after_construction_is_rejected(tmp_path):
    manifest,cache=fixture(tmp_path);dataset=WindowDataset(manifest,cache,'train')
    (tmp_path/'label.json').write_text('{}')
    with pytest.raises(ValueError,match='Labels changed'):dataset[0]


def test_changed_metadata_or_class_order_is_not_accepted(tmp_path):
    manifest,cache=fixture(tmp_path);dataset=WindowDataset(manifest,cache,'train')
    path=cache/'pcm/metadata.json';data=json.loads(path.read_text());data['feature_origin_seconds']=1
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='metadata identity'):dataset[0]
    m=json.loads(manifest.read_text());m['classes']=list(reversed(CLASSES));manifest.write_text(json.dumps(m))
    with pytest.raises(ValueError,match='class order'):WindowDataset(manifest,cache,'train')


@pytest.mark.parametrize('change', ['manifest', 'errors', 'missing'])
def test_completed_report_must_match_dataset_and_cover_audio(tmp_path,change):
    manifest,cache=fixture(tmp_path);path=cache/'report.json'
    report=json.loads(path.read_text())
    if change=='manifest':report['manifest_sha256']='stale'
    elif change=='errors':report['errors']=[{'error':'failed'}]
    else:report['complete']=[]
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError,match='Feature report'):WindowDataset(manifest,cache,'train')
