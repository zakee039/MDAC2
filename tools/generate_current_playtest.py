"""Current supervised generation: global rules + separately selected V2 onset bias.

Frozen historical scripts remain reproduction tools. This entry reads the shared
generation defaults and applies the same postprocessor to every current song.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse
from collections import Counter
from dataclasses import asdict
from pathlib import Path
import shutil
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import torch
from mdac2.data_preparation.audio_features import sha,validate_cache
from mdac2.data_preparation.arrangement_context_v6 import ContextArrangement,predict,decode
from mdac2.data_preparation.v2_vocal_bias import select,VocalBias
from mdac2.data_preparation.global_postprocess import process,STORAGE,POLICY_ID
from mdac2.data_preparation.playability_audit import audit_playability
from mdac2.data_preparation.training_chart import TrainingEvent
from mdac2.data_preparation.supervised_training import seed_all
from mdac2.data_preparation.six_export import write_bms,read_bms,semantics
from mdac2.data_preparation.joint_evaluation import EventMetrics
from generate_joint_playtest import read,save

PREP=ROOT/'tools/private/community/training-preparation-v3'
DATA_SHA='34add12c1c47fc99afc04ae1fa9f06e968d5c2c50db41cb32a047f033668eb71'


def es(chart):return [TrainingEvent.from_dict(e) for e in chart['events']]


def prepare(output):
    defaults=read(ROOT/'recipes/generation-defaults.json')
    if defaults['base_model']!='v3' or defaults['postprocess_profile']!=POLICY_ID or defaults['inference_profile']!='v2-vocal-soft-bias-v1':
        raise ValueError('Unsupported explicit generation profile; cannot silently substitute')
    assert sha(PREP/'manifest.json')==DATA_SHA
    manifest=read(ROOT/'models'/defaults['base_model']/'manifest.json')
    prior_path=ROOT/manifest['postprocessing']['prior_path']
    assert sha(prior_path)==manifest['postprocessing']['prior_sha256']
    for path,item in manifest['postprocessing']['sources'].items():assert sha(ROOT/path)==item['sha256']
    model=None
    seed_all(20260928);torch.set_num_threads(4)
    for stage,item in manifest['checkpoints'].items():
        path=ROOT/item['path'];assert sha(path)==item['sha256']
        saved=torch.load(path,map_location='cpu',weights_only=False)
        for name,digest in saved['identity']['code'].items():assert sha(ROOT/'src/mdac2/data_preparation'/name)==digest
        if stage=='arrangement':
            model=ContextArrangement().cuda().eval();model.load_state_dict(saved['model'])
    output.mkdir(parents=True,exist_ok=False)
    files=[Path(__file__),ROOT/'src/mdac2/data_preparation/global_postprocess.py',
           ROOT/'src/mdac2/data_preparation/v2_vocal_bias.py',ROOT/'recipes/generation-defaults.json',
           ROOT/'src/mdac2/data_preparation/hold_refinement_rc1.py',ROOT/'src/mdac2/data_preparation/vocal_priority_r2.py',
           ROOT/'src/mdac2/data_preparation/vocal_priority.py',ROOT/'src/mdac2/data_preparation/six_export.py',
           ROOT/'tests/unit/test_global_postprocess.py']
    (output/'source').mkdir()
    for p in files:shutil.copyfile(p,output/'source'/p.name)
    identity=dict(command=[sys.executable,*sys.argv],defaults=defaults,data_manifest_sha256=DATA_SHA,
        model_manifest_sha256=sha(ROOT/'models'/defaults['base_model']/'manifest.json'),checkpoints=manifest['checkpoints'],
        prior_sha256=sha(prior_path),vocal_bias=asdict(VocalBias()),sources={str(p.relative_to(ROOT)):sha(p) for p in files},
        weights_retrained=False,separation_repeated=False,locked_test_opened=False,
        tuning='single predeclared configuration; reused validation and user feedback, not independent test')
    save(output/'identity.json',identity)
    return model,read(prior_path),identity


def generate(model,prior,features,prediction,duration,out):
    times,boosted,routing=select(features,prediction,duration)
    save(out/'vocal-bias.json',routing)
    np.savez_compressed(out/'onset-selection.npz',times=np.asarray(times),**boosted)
    scores=predict(model,features,times,duration)
    np.savez_compressed(out/'arrangement-scores.npz',times=np.asarray(times),**scores)
    events,decoding=decode(scores['logits'],scores['log_lengths'],times,duration)
    save(out/'neural-chart.json',dict(duration=duration,events=[e.to_dict() for e in events],audit=decoding))
    after,report=process(events,features,boosted,duration,prior)
    audit=audit_playability(after,duration)
    if not audit['engineering_playtest_gate_passed']:raise AssertionError('Engineering gate failed')
    bms=write_bms(after,duration)
    if semantics(after)!=semantics(read_bms(bms,duration)):raise AssertionError('BMS roundtrip changed event semantics')
    save(out/'report.json',report);save(out/'playability-audit.json',audit)
    (out/'experimental.bms').write_text(bms,encoding='utf-8')
    return after,decoding,report,routing


def song(args):
    model,prior,identity=prepare(args.output)
    audio=read(args.cached/'audio.json');baseline=read(args.baseline/'chart.json')
    assert baseline['source_audio']==audio and sha(args.cached/'mix.wav')==audio['wav_sha256']
    assert sha(Path(audio['source']))==audio['source_sha256']
    validate_cache(args.cached/'features',dict(pcm=dict(sha256=audio['pcm_sha256'])))
    features=np.load(args.cached/'features/features.npy',mmap_mode='r',allow_pickle=False)
    with np.load(args.prediction,allow_pickle=False) as a:prediction={k:a[k] for k in ('probability','offsets')}
    save(args.output/'status.json',dict(status='running'))
    after,decoding,report,routing=generate(model,prior,features,prediction,audio['duration'],args.output)
    chart=dict(baseline,model_id=identity['defaults']['base_model'],parent_model='v2',recipe=POLICY_ID,
        events=[e.to_dict() for e in after],class_counts=dict(Counter(e.kind for e in after)),
        checkpoint_identity={k:dict(v,path=str(ROOT/v['path'])) for k,v in identity['checkpoints'].items()},
        model_manifest_sha256=identity['model_manifest_sha256'],postprocess_policy=report['policy'],
        postprocess_report_sha256=sha(args.output/'report.json'),base_decoding=decoding,
        decoding=dict(stage='global_postprocessing',rules_passed=True),storage=STORAGE,
        human_listening='pending',game_acceptance='pending',production_promoted=False)
    for key in ('export_grid','grid','bpm_used_for_storage','lane_balance'):chart.pop(key,None)
    chart.update(bpm_used=False,storage_step_seconds=.001)
    save(args.output/'chart.json',chart)
    save(args.output/'inputs.json',dict(audio_sha256=audio['source_sha256'],prediction_sha256=sha(args.prediction),
        baseline_chart_sha256=sha(args.baseline/'chart.json'),feature_metadata_sha256=sha(args.cached/'features/metadata.json')))
    save(args.output/'status.json',dict(status='completed',events=len(after),routing=routing,
        hold_guard_edits=len(report['avoidance']['edits']),unresolved_guard=len(report['final_avoidance_violations']),
        dense_pairs=len(report['final_dense_pairs'])))
    print(str(args.output),len(after),routing,flush=True)


def evaluate(args):
    model,prior,identity=prepare(args.output)
    rows=[r for r in read(PREP/'manifest.json')['charts'] if r['training_eligible'] and r['split']=='validation']
    assert len(rows)==22
    tensors={r['chart_sha256']:r for r in read(PREP/'tensors/report.json')['charts']}
    metrics={name:EventMetrics() for name in ('v2','current')};records=[]
    for i,row in enumerate(rows):
        if (args.output/'CANCEL').exists():save(args.output/'status.json',dict(status='cancelled'));return
        key=row['chart_sha256'];out=args.output/key;out.mkdir()
        save(args.output/'status.json',dict(status='running',chart=i+1,total=22))
        cache=ROOT/'tools/private/community/pretraining-feature-cache-v1'/row['pcm']['sha256']
        validate_cache(cache,row)
        features=np.load(cache/'features.npy',mmap_mode='r',allow_pickle=False)
        onset=ROOT/'tools/private/community/supervised-dual-v1-20260928/evaluation/predictions'/f'{key}.npz'
        with np.load(onset,allow_pickle=False) as a:prediction={k:a[k] for k in ('probability','offsets')}
        original=read(ROOT/'tools/private/community/supervised-dual-v6-20260930/evaluation/charts-predicted_onsets'/f'{key}.json')
        base=read(ROOT/'tools/private/community/playability-postprocess-r1-20261016/validation-r1.1'/f'{key}.json')
        after,decoding,report,routing=generate(model,prior,features,prediction,original['duration'],out)
        labels=PREP/row['labels'];assert sha(labels)==row['labels_sha256']
        truth=[e for e in es(read(labels)) if e.kind!='gear']
        tensor=tensors[key];tp=PREP/'tensors'/tensor['file'];assert sha(tp)==tensor['sha256']
        with np.load(tp,allow_pickle=False) as a:mask=a['onset_mask']
        for name,events in [('v2',es(base)),('current',after)]:metrics[name].add(events,truth,original['duration'],decoding if name=='current' else original['audit'],mask)
        save(out/'chart.json',dict(duration=original['duration'],events=[e.to_dict() for e in after]))
        records.append(dict(chart=key,onsets=routing,tail_changes=len(report['tails']['changes']),
            avoidance_edits=len(report['avoidance']['edits']),gemini_conversions=sum(e['gemini_to_opposite_light'] for e in report['avoidance']['edits']),
            unresolved_guard=len(report['final_avoidance_violations']),dense_pairs=len(report['final_dense_pairs'])))
        print(i+1,records[-1]['avoidance_edits'],flush=True)
    save(args.output/'validation.json',dict(metrics={k:m.report() for k,m in metrics.items()},charts=records,identity=identity))
    save(args.output/'status.json',dict(status='completed',charts=22))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['song','evaluate']);p.add_argument('--output',type=Path,required=True)
    for name in ('cached','baseline','prediction'):p.add_argument('--'+name,type=Path)
    a=p.parse_args();a.output=a.output.resolve()
    if a.mode=='song':
        if any(getattr(a,k) is None for k in ('cached','baseline','prediction')):p.error('song needs cached/baseline/prediction')
        song(a)
    else:evaluate(a)
