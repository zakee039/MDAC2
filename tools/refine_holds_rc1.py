"""Replay verified R1 caches into an independent hold-feedback candidate."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from mdac2.data_preparation.audio_features import sha, validate_cache
from mdac2.data_preparation.hold_refinement_rc1 import process, HoldPolicy
from mdac2.data_preparation.fixed_grid_r2 import process as baseline_process
from mdac2.data_preparation.training_chart import TrainingEvent
from mdac2.data_preparation.six_grid_export_r2 import write_bms, read_bms, semantics
from mdac2.data_preparation.playability_audit import audit_playability
from mdac2.data_preparation.joint_evaluation import EventMetrics

BASE = ROOT/'tools/private/community/vocal-grid-r2-20261017'
PREP = ROOT/'tools/private/community/training-preparation-v3'
MANIFEST_SHA = '34add12c1c47fc99afc04ae1fa9f06e968d5c2c50db41cb32a047f033668eb71'


def read(p): return json.loads(p.read_text(encoding='utf-8'))
def save(p, value): p.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
def events(chart): return [TrainingEvent.from_dict(e) for e in chart['events']]


def run_one(base, cache, onset, out, prior):
    out.mkdir(parents=True, exist_ok=False)
    chart, neural = read(base/'chart.json'), read(base/'neural-chart.json')
    metadata = read(cache/'metadata.json')
    # Hash checks cover actual arrays, not only the metadata text.
    if sha(cache/'features.npy') != metadata['files']['features.npy']:
        raise ValueError('Feature cache changed')
    features = np.load(cache/'features.npy', mmap_mode='r', allow_pickle=False)
    with np.load(base/'routing.npz', allow_pickle=False) as a:
        density = a['density_probability'].copy()
    with np.load(onset, allow_pickle=False) as a:
        offsets = a['offsets'].copy()
    prediction = dict(probability=density, offsets=offsets)
    replay, _ = baseline_process(events(neural), features, prediction, chart['duration'], prior)
    if [e.to_dict() for e in replay] != chart['events']:
        raise AssertionError('R1 cache replay differs: '+str(base))
    refined, report = process(events(neural), features, prediction, chart['duration'], prior)
    audit = audit_playability(refined, chart['duration'])
    if not audit['engineering_playtest_gate_passed']:
        raise AssertionError('Playability gate failed')
    bms = write_bms(refined, chart['duration'])
    if semantics(read_bms(bms, chart['duration'])) != semantics(refined):
        raise AssertionError('BMS roundtrip failed')
    report['identity'] = dict(base_chart_sha256=sha(base/'chart.json'), neural_chart_sha256=sha(base/'neural-chart.json'),
        routing_sha256=sha(base/'routing.npz'), onset_sha256=sha(onset), feature_sha256=sha(cache/'features.npy'),
        baseline_replay_exact=True)
    report['event_changes'] = [dict(before=old.to_dict(), after=next((e.to_dict() for e in refined if e.event_id==old.event_id),None))
        for old in replay if next((e for e in refined if e.event_id==old.event_id),None) != old]
    report['human_listening'] = 'pending'
    save(out/'report.json', report)
    updated = dict(chart, model_id='V2+R1-hold-feedback-rc1', parent_recipe='R1',
        events=[e.to_dict() for e in refined], human_listening='pending', game_acceptance='pending',
        postprocess_policy=report['policy'], postprocess_report_sha256=sha(out/'report.json'), production_promoted=False)
    from collections import Counter
    updated['class_counts'] = dict(Counter(e.kind for e in refined))
    save(out/'chart.json', updated)
    save(out/'playability-audit.json', audit)
    (out/'experimental.bms').write_text(bms, encoding='utf-8')
    return replay, refined, report


def main(out):
    if out.exists(): raise FileExistsError('Preserve completed/partial candidate')
    if sha(PREP/'manifest.json') != MANIFEST_SHA: raise ValueError('Data identity changed')
    recipe = read(ROOT/'recipes/R1/manifest.json')
    for item in recipe['checkpoints'].values():
        if sha(ROOT/item['path']) != item['sha256']: raise ValueError('Frozen weights changed')
    # All bound R1 working sources must still match their frozen snapshots.
    for name, item in recipe['sources'].items():
        if sha(ROOT/name) != item['sha256']: raise ValueError('Frozen source changed: '+name)
    prior_path = ROOT/'models/v2/weights/run-prior.json'
    prior = read(prior_path)
    out.mkdir(parents=True)
    (out/'source').mkdir()
    source_files = [Path(__file__), ROOT/'src/mdac2/data_preparation/hold_refinement_rc1.py',ROOT/'tests/unit/test_hold_refinement_rc1.py']
    for p in source_files: shutil.copyfile(p,out/'source'/p.name)
    identity = dict(command=[sys.executable,*sys.argv], manifest_sha256=MANIFEST_SHA, recipe_sha256=sha(ROOT/'recipes/R1/manifest.json'),
        registry_sha256=sha(ROOT/'models/registry.json'), checkpoints=recipe['checkpoints'], prior_sha256=sha(prior_path),
        sources={p.relative_to(ROOT).as_posix():sha(p) for p in source_files}, policy=asdict(HoldPolicy()),
        training_started=False, locked_test_opened=False, human_feedback_used=True,
        evaluation_scope='22 reused validation charts; not independent test; no threshold sweep')
    save(out/'identity.json',identity)
    started=time.monotonic()
    rows=[r for r in read(PREP/'manifest.json')['charts'] if r['training_eligible'] and r['split']=='validation']
    assert len(rows)==22
    tensors={r['chart_sha256']:r for r in read(PREP/'tensors/report.json')['charts']}
    metrics={k:EventMetrics() for k in ('R1','candidate')}
    summaries=[]
    for n,row in enumerate(rows):
        if (out/'CANCEL').exists():
            save(out/'status.json',dict(status='cancelled',charts=n));return
        key=row['chart_sha256']
        save(out/'status.json',dict(status='running',chart=n+1,total=24))
        before,after,report=run_one(BASE/'validation'/key,
            ROOT/'tools/private/community/pretraining-feature-cache-v1'/row['pcm']['sha256'],
            ROOT/'tools/private/community/supervised-dual-v1-20260928/evaluation/predictions'/f'{key}.npz',out/'validation'/key,prior)
        label_path=PREP/row['labels'];assert sha(label_path)==row['labels_sha256']
        truth=[e for e in events(read(label_path)) if e.kind!='gear']
        tensor=tensors[key];tp=PREP/'tensors'/tensor['file'];assert sha(tp)==tensor['sha256']
        with np.load(tp,allow_pickle=False) as a: mask=a['onset_mask'].copy()
        original=read(BASE/'validation'/key/'chart.json')
        for label,es in [('R1',before),('candidate',after)]:
            metrics[label].add(es,truth,original['duration'],original['audit'],mask)
        summaries.append(dict(chart=key,tail_changes=len(report['tails']['changes']),spacing_changes=len(report['spacing']['edits']),
            unresolved_dense_pairs=len(report['final_dense_pairs']),event_changes=len(report['event_changes'])))
        print(json.dumps(summaries[-1]),flush=True)
    save(out/'validation.json',dict(metrics={k:m.report() for k,m in metrics.items()},charts=summaries))
    songs=[('pipa','20260927-pipaxing-v1',ROOT/'tools/private/community/supervised-dual-v6-20260930/pipaxing-review-v6/prediction.npz'),
           ('snow','20261016-snow-song-show-v1',ROOT/'tools/private/playtests/20261016-snow-song-show-v1/prediction.npz')]
    for name,cache_name,onset in songs:
        before,after,report=run_one(BASE/name,ROOT/'tools/private/playtests'/cache_name/'features',onset,out/name,prior)
        if name=='snow':
            crop=lambda es:[e.to_dict() for e in es if e.start<179 and e.end>171]
            if crop(before)!=crop(after):raise AssertionError('Deferred figure 2 region changed')
        print(json.dumps(dict(song=name,tails=report['tails']['changes'],spacing=report['spacing']['edits']),ensure_ascii=False),flush=True)
    assert sha(ROOT/'models/registry.json')==identity['registry_sha256']
    save(out/'status.json',dict(status='completed',charts=24,seconds=time.monotonic()-started,
        figure_2_171_179_unchanged=True,human_listening='pending',production_promoted=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    main(a.output.resolve())
