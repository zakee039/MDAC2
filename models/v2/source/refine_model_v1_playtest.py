"""Fit training-only phrase prior, audit fixed validation, or create a new playtest."""
import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
import numpy as np
from mdac2.data_preparation.audio_features import sha, validate_cache
from mdac2.data_preparation.training_chart import TrainingEvent
from mdac2.data_preparation.playtest_postprocess import (
    Policy, fit_run_prior, process, structure_stats,
)
from mdac2.data_preparation.playability_audit import audit_playability
from mdac2.data_preparation.six_export import write_bms, read_bms, semantics

MANIFEST = ROOT/'tools/private/community/training-preparation-v3/manifest.json'
MANIFEST_SHA = '34add12c1c47fc99afc04ae1fa9f06e968d5c2c50db41cb32a047f033668eb71'
CACHE = ROOT/'tools/private/community/pretraining-feature-cache-v1'
V6 = ROOT/'tools/private/community/supervised-dual-v6-20260930'
ONSET_CACHE = ROOT/'tools/private/community/supervised-dual-v1-20260928/evaluation/predictions'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def provenance():
    if sha(MANIFEST) != MANIFEST_SHA:
        raise ValueError('Frozen data manifest changed')
    manifest = read(ROOT/'models/v1/manifest.json')
    for name, entry in manifest['checkpoints'].items():
        if sha(ROOT/entry['path']) != entry['sha256']:
            raise ValueError('Frozen release model changed: '+name)
    module = ROOT/'src/mdac2/data_preparation/playtest_postprocess.py'
    return dict(manifest_sha256=MANIFEST_SHA, model_manifest_sha256=sha(ROOT/'models/v1/manifest.json'),
                checkpoints=manifest['checkpoints'], source_sha256={str(module.relative_to(ROOT)): sha(module),
                str(Path(__file__).relative_to(ROOT)): sha(Path(__file__))}, command=[sys.executable, *sys.argv])


def fit(output):
    identity = provenance()
    rows = [r for r in read(MANIFEST)['charts'] if r['training_eligible'] and r['split'] == 'train']
    if len(rows) != 172:
        raise ValueError('Unexpected train role')
    charts, labels = [], {}
    for row in rows:
        path = MANIFEST.parent/row['labels']
        if sha(path) != row['labels_sha256']:
            raise ValueError('Training label identity changed')
        charts.append([TrainingEvent.from_dict(e) for e in read(path)['events'] if e['kind'] != 'gear'])
        labels[row['chart_sha256']] = row['labels_sha256']
    prior = fit_run_prior(charts)
    if prior['observations'] < 10000:
        raise ValueError('Too few training transitions; inspect event grouping')
    prior.update(role='train', charts=len(rows), labels_sha256=labels, identity=identity,
                 locked_test_opened=False, reference_chart_at_inference=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as f:
        json.dump(prior, f, ensure_ascii=False, indent=2)
    print(json.dumps(dict(prior=str(output), observations=prior['observations'], histogram=prior['run_histogram'])))


def load_prior(path):
    prior = read(path)
    if prior['role'] != 'train' or prior['charts'] != 172 or prior['identity']['manifest_sha256'] != MANIFEST_SHA:
        raise ValueError('Train-only prior required')
    return prior


def evaluate(prior_path, output):
    from mdac2.data_preparation.joint_evaluation import EventMetrics, result
    identity = provenance()
    prior = load_prior(prior_path)
    output.mkdir(parents=True, exist_ok=False)
    rows = [r for r in read(MANIFEST)['charts'] if r['training_eligible'] and r['split'] == 'validation']
    if len(rows) != 22:
        raise ValueError('Unexpected validation role')
    metrics = {'before': EventMetrics(), 'after': EventMetrics()}
    records = []
    tensor_root = MANIFEST.parent/'tensors'
    tensors = {r['chart_sha256']: r for r in read(tensor_root/'report.json')['charts']}
    for index, row in enumerate(rows):
        key = row['chart_sha256']
        path = MANIFEST.parent/row['labels']
        if sha(path) != row['labels_sha256']:
            raise ValueError('Validation label changed')
        truth = [TrainingEvent.from_dict(e) for e in read(path)['events'] if e['kind'] != 'gear']
        chart_path = V6/'evaluation/charts-predicted_onsets'/f'{key}.json'
        chart = read(chart_path)
        events = [TrainingEvent.from_dict(e) for e in chart['events']]
        cache = CACHE/row['pcm']['sha256']
        validate_cache(cache, row)
        features = np.load(cache/'features.npy', mmap_mode='r', allow_pickle=False)
        with np.load(ONSET_CACHE/f'{key}.npz', allow_pickle=False) as a:
            prediction = {k: a[k] for k in ('probability', 'offsets')}
        after, report = process(events, features, prediction, chart['duration'], prior)
        tensor = tensors[key]
        tensor_path = tensor_root/tensor['file']
        if sha(tensor_path) != tensor['sha256']:
            raise ValueError('Tensor identity changed')
        with np.load(tensor_path, allow_pickle=False) as a:
            mask = a['onset_mask']
        for name, es in [('before', events), ('after', after)]:
            metrics[name].add(es, truth, chart['duration'], chart['audit'], mask)
        if semantics(after) != semantics(read_bms(write_bms(after, chart['duration']), chart['duration'])):
            raise ValueError('Validation BMS roundtrip failed')
        report.update(chart_sha256=key, reference_structure=structure_stats(truth),
                      source_chart_sha256=sha(chart_path), onset_cache_sha256=sha(ONSET_CACHE/f'{key}.npz'),
                      feature_metadata_sha256=sha(cache/'metadata.json'))
        save(output/f'{key}.json', dict(events=[e.to_dict() for e in after], report=report))
        records.append(report)
        print(json.dumps(dict(chart=index+1, total=22, added=report['density']['added'],
              flips=report['phrasing']['changed'], tails=report['tails']['extended'],
              repeats=report['repeats']['applied'])), flush=True)
    reports = {name: m.report() for name, m in metrics.items()}
    for report in reports.values():
        classes = [v for k, v in report['six_classes'].items() if k != 'gear']
        report['macro_five_class_f1'] = sum(v['f1'] for v in classes)/5
        report['micro_five_class'] = result(np.sum([[v[k] for k in ('tp', 'fp', 'fn')] for v in classes], axis=0))
    result_report = dict(status='completed', identity=identity, prior_sha256=sha(prior_path),
                        policy=asdict(Policy()), metrics=reports, charts=records,
                        locked_test_opened=False, human_playtest='pending', production_promoted=False)
    save(output/'validation.json', result_report)
    print(json.dumps({name: dict(macro=r['macro_five_class_f1'], micro=r['micro_five_class']['f1'],
                    onset=r['sampling']['f1'], hold=r['holds']['hold_joint']['f1']) for name, r in reports.items()}))


def song(prior_path, cached, prediction_dir, output):
    identity = provenance()
    prior = load_prior(prior_path)
    chart = read(prediction_dir/'chart.json')
    audio = read(cached/'audio.json')
    if chart['source_audio'] != audio or sha(cached/'mix.wav') != audio['wav_sha256']:
        raise ValueError('Song cache/prediction audio mismatch')
    if sha(Path(audio['source'])) != audio['source_sha256']:
        raise ValueError('Original audio changed')
    expected = identity['checkpoints']
    if any(chart['checkpoint_identity'][s]['sha256'] != expected[s]['sha256'] for s in expected):
        raise ValueError('Song prediction is not model release v1')
    validate_cache(cached/'features', dict(pcm=dict(sha256=audio['pcm_sha256'])))
    features = np.load(cached/'features/features.npy', mmap_mode='r', allow_pickle=False)
    with np.load(prediction_dir/'prediction.npz', allow_pickle=False) as a:
        prediction = {k: a[k] for k in ('probability', 'offsets')}
    events = [TrainingEvent.from_dict(e) for e in chart['events']]
    started = time.monotonic()
    after, report = process(events, features, prediction, chart['duration'], prior)
    bms = write_bms(after, chart['duration'])
    if semantics(after) != semantics(read_bms(bms, chart['duration'])):
        raise ValueError('BMS roundtrip changed note semantics')
    output.mkdir(parents=True, exist_ok=False)
    report.update(identity=identity, prior_sha256=sha(prior_path),
                  base_chart_sha256=sha(prediction_dir/'chart.json'),
                  onset_cache_sha256=sha(prediction_dir/'prediction.npz'), seconds=time.monotonic()-started)
    save(output/'report.json', report)
    save(output/'base-chart.json', chart)
    base_decoding = chart.pop('decoding', None)
    chart.pop('lane_balance', None)
    chart.update(events=[e.to_dict() for e in after], class_counts=dict(Counter(e.kind for e in after)),
                 model_id='v1', postprocess_policy=asdict(Policy()),
                 base_decoding=base_decoding, decoding=dict(stage='postprocessing',
                     final_events_match_frozen_model_choices=False, rules_passed=True),
                 postprocess_report_sha256=sha(output/'report.json'), human_listening='pending',
                 game_acceptance='pending', production_promoted=False,
                 decoding_scope='frozen base model; additional postprocessing edits audited separately')
    save(output/'chart.json', chart)
    save(output/'playability-audit.json', audit_playability(after, chart['duration']))
    (output/'experimental.bms').write_text(bms, encoding='utf-8')
    print(json.dumps(dict(output=str(output), before=report['before'], after=report['after'],
                         added=report['density']['added'], mirrored=report['repeats']['applied'])))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['fit', 'evaluate', 'song'])
    p.add_argument('--prior', type=Path)
    p.add_argument('--cached', type=Path)
    p.add_argument('--prediction', type=Path)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.mode == 'fit':
        fit(a.output.resolve())
    elif a.mode == 'evaluate':
        evaluate(a.prior.resolve(), a.output.resolve())
    else:
        song(a.prior.resolve(), a.cached.resolve(), a.prediction.resolve(), a.output.resolve())
