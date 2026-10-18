"""Bounded, independently versioned vocal routing evaluation or cached-song trial."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
import numpy as np
import torch
from mdac2.data_preparation.audio_features import sha, validate_cache
from mdac2.data_preparation.arrangement_context_v6 import ContextArrangement, predict, decode
from mdac2.data_preparation.supervised_training import seed_all
from mdac2.data_preparation.vocal_priority import VocalPolicy, select_onsets
from mdac2.data_preparation.playtest_postprocess import process, Policy, structure_stats
from mdac2.data_preparation.training_chart import TrainingEvent
from mdac2.data_preparation.six_export import write_bms, read_bms, semantics
from mdac2.data_preparation.playability_audit import audit_playability
from mdac2.data_preparation.joint_evaluation import EventMetrics, counts, result

MANIFEST = ROOT/'tools/private/community/training-preparation-v3/manifest.json'
MANIFEST_SHA = '34add12c1c47fc99afc04ae1fa9f06e968d5c2c50db41cb32a047f033668eb71'
CACHE = ROOT/'tools/private/community/pretraining-feature-cache-v1'
ONSET = ROOT/'tools/private/community/supervised-dual-v1-20260928/evaluation/predictions'
V2 = ROOT/'tools/private/community/playability-postprocess-r1-20261016'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def save(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def load_npz(path):
    with np.load(path, allow_pickle=False) as a:
        return {k: a[k] for k in ('probability', 'offsets')}


def prepare(output):
    output.mkdir(parents=True, exist_ok=False)
    manifest = read(ROOT/'models/v2/manifest.json')
    if manifest['model_id'] != 'v2' or sha(MANIFEST) != MANIFEST_SHA:
        raise ValueError('Model or data identity changed')
    model = None
    seed_all(20260928)
    torch.set_num_threads(4)
    for stage, item in manifest['checkpoints'].items():
        if sha(ROOT/item['path']) != item['sha256']:
            raise ValueError('Frozen checkpoint changed')
        saved = torch.load(ROOT/item['path'], map_location='cpu', weights_only=False)
        for name, expected in saved['identity']['code'].items():
            if sha(ROOT/'src/mdac2/data_preparation'/name) != expected:
                raise ValueError('Bound model source changed: '+name)
        if stage == 'arrangement':
            model = ContextArrangement().cuda().eval()
            model.load_state_dict(saved['model'])
    post = manifest['postprocessing']
    for name, identity in post['sources'].items():
        if sha(ROOT/name) != identity['sha256']:
            raise ValueError('Selected v2 postprocessing changed')
    if sha(ROOT/post['prior_path']) != post['prior_sha256']:
        raise ValueError('v2 training prior changed')
    prior = read(ROOT/post['prior_path'])
    sources = [Path(__file__), ROOT/'src/mdac2/data_preparation/vocal_priority.py']
    (output/'source').mkdir()
    for path in sources:
        shutil.copyfile(path, output/'source'/path.name)
    identity = dict(baseline='v2', model_manifest_sha256=sha(ROOT/'models/v2/manifest.json'),
        manifest_sha256=MANIFEST_SHA, policy=asdict(VocalPolicy()),
        checkpoints=manifest['checkpoints'], postprocessing=post,
        code={p.relative_to(ROOT).as_posix(): sha(p) for p in sources},
        command=[sys.executable, *sys.argv], training_started=False, locked_test_opened=False,
        vocal_source_truth_available=False, validation_used_for_tuning=False)
    save(output/'identity.json', identity)
    return model, prior, identity


def generate(model, prior, features, prediction, duration, output):
    times, routing, report = select_onsets(features, prediction, duration)
    np.savez_compressed(output/'routing.npz', **routing)
    save(output/'routing.json', report)
    scores = predict(model, features, times, duration)
    np.savez_compressed(output/'arrangement-scores.npz', times=np.asarray(times), **scores)
    events, audit = decode(scores['logits'], scores['log_lengths'], times, duration)
    density = dict(probability=routing['density_probability'], offsets=prediction['offsets'])
    after, post = process(events, features, density, duration, prior)
    post['density']['vocal_gate_applied'] = True
    post['density']['ungated_instrument_candidates_while_vocal_mode'] = False
    playback = audit_playability(after, duration)
    if not playback['engineering_playtest_gate_passed']:
        raise ValueError('Final chart violates playability rules')
    bms = write_bms(after, duration)
    if semantics(read_bms(bms, duration)) != semantics(after):
        raise ValueError('BMS roundtrip mismatch')
    (output/'experimental.bms').write_text(bms, encoding='utf-8')
    save(output/'playability-audit.json', playback)
    save(output/'postprocessing.json', post)
    return after, audit, report, routing, post


def evaluate(output):
    model, prior, identity = prepare(output)
    rows = [r for r in read(MANIFEST)['charts'] if r['training_eligible'] and r['split'] == 'validation']
    if len(rows) != 22:
        raise ValueError('Validation split changed')
    tensors = {r['chart_sha256']: r for r in read(MANIFEST.parent/'tensors/report.json')['charts']}
    metrics = {key: EventMetrics() for key in ('v2', 'vocal_trial')}
    records = []
    started = time.monotonic()
    for i, row in enumerate(rows):
        if (output/'CANCEL').exists():
            save(output/'status.json', dict(status='cancelled', completed_charts=len(records)))
            return
        key = row['chart_sha256']
        out = output/key
        out.mkdir()
        save(output/'status.json', dict(status='running', chart=i+1, total=22, elapsed_seconds=time.monotonic()-started))
        labels = MANIFEST.parent/row['labels']
        if sha(labels) != row['labels_sha256']:
            raise ValueError('Validation reference changed')
        truth = [TrainingEvent.from_dict(e) for e in read(labels)['events'] if e['kind'] != 'gear']
        base_path = V2/'validation-r1.1'/f'{key}.json'
        base = read(base_path)
        baseline = [TrainingEvent.from_dict(e) for e in base['events']]
        metadata = validate_cache(CACHE/row['pcm']['sha256'], row)
        features = np.load(CACHE/row['pcm']['sha256']/'features.npy', mmap_mode='r', allow_pickle=False)
        # Cached chart duration is the same PCM duration used for v2.
        original = read(ROOT/'tools/private/community/supervised-dual-v6-20260930/evaluation/charts-predicted_onsets'/f'{key}.json')
        duration = original['duration']
        prediction = load_npz(ONSET/f'{key}.npz')
        after, audit, report, routing, post = generate(model, prior, features, prediction, duration, out)
        tensor = tensors[key]
        tp = MANIFEST.parent/'tensors'/tensor['file']
        if sha(tp) != tensor['sha256']:
            raise ValueError('Frozen tensor changed')
        with np.load(tp, allow_pickle=False) as data:
            mask = data['onset_mask']
        for name, es in [('v2', baseline), ('vocal_trial', after)]:
            metrics[name].add(es, truth, duration, original['audit'] if name == 'v2' else audit, mask)
        # Evidence-conditioned alignment is a proxy diagnostic, not vocal truth.
        proxy = {}
        for name, es in [('v2', baseline), ('vocal_trial', after)]:
            starts = sorted(set(e.start for e in es))
            frame = np.minimum((np.asarray(starts)*100).astype(int), len(routing['vocal_mode'])-1)
            vocal = routing['vocal_mode'][frame]
            proxy[name] = dict(starts_during_vocal_mode=int(vocal.sum()),
                supported_starts=int((vocal & (routing['support'][frame] >= VocalPolicy().novelty_threshold)).sum()))
        record = dict(chart_sha256=key, baseline_chart_sha256=sha(base_path),
            onset_cache_sha256=sha(ONSET/f'{key}.npz'), feature_metadata_sha256=sha(CACHE/row['pcm']['sha256']/'metadata.json'),
            routing=report, proxy=proxy, structure_before=structure_stats(baseline), structure_after=structure_stats(after))
        save(out/'chart.json', dict(duration=duration, events=[e.to_dict() for e in after], audit=audit))
        save(out/'report.json', record)
        records.append(record)
        print(json.dumps(dict(chart=i+1, before=report['baseline_candidates'], after=report['selected_candidates'],
                              voice_seconds=report['vocal_mode_seconds'])), flush=True)
    reports = {name: metric.report() for name, metric in metrics.items()}
    for report in reports.values():
        classes = [v for k, v in report['six_classes'].items() if k != 'gear']
        report['macro_five_class_f1'] = sum(v['f1'] for v in classes)/5
        report['micro_five_class'] = result(np.sum([[v[k] for k in ('tp','fp','fn')] for v in classes], axis=0))
    save(output/'validation.json', dict(status='completed', identity=identity, metrics=reports, charts=records,
        seconds=time.monotonic()-started, human_listening='pending', production_promoted=False,
        vocal_quality='unverified: references do not label voice origin', parameter_selection='one fixed exploratory configuration'))
    save(output/'status.json', dict(status='completed', charts=22, seconds=time.monotonic()-started))
    print(json.dumps({k: dict(onset=v['sampling'], macro=v['macro_five_class_f1']) for k,v in reports.items()}))


def song(cached, base_dir, neural_dir, output):
    model, prior, identity = prepare(output)
    audio = read(cached/'audio.json')
    base = read(base_dir/'chart.json')
    if base['source_audio'] != audio or sha(cached/'mix.wav') != audio['wav_sha256']:
        raise ValueError('Baseline audio identity mismatch')
    if sha(Path(audio['source'])) != audio['source_sha256']:
        raise ValueError('User source changed')
    validate_cache(cached/'features', dict(pcm=dict(sha256=audio['pcm_sha256'])))
    features = np.load(cached/'features/features.npy', mmap_mode='r', allow_pickle=False)
    pred = load_npz(neural_dir/'prediction.npz')
    save(output/'status.json', dict(status='running'))
    after, audit, routing, arrays, post = generate(model, prior, features, pred, audio['duration'], output)
    chart = dict(base)
    chart.update(model_id='v2+vocal-priority-experiment-r1', parent_model='v2',
        events=[e.to_dict() for e in after], class_counts=dict(Counter(e.kind for e in after)),
        checkpoint_identity={k: dict(v, path=(ROOT/v['path']).as_posix()) for k,v in identity['checkpoints'].items()},
        model_manifest_sha256=identity['model_manifest_sha256'], onset_routing=routing['policy'],
        base_decoding=audit, decoding=dict(stage='postprocessing', final_events_match_frozen_model_choices=False, rules_passed=True),
        human_listening='pending', game_acceptance='pending', production_promoted=False)
    chart.pop('lane_balance', None)
    report = dict(post, before=structure_stats([TrainingEvent.from_dict(e) for e in base['events']]),
        after=structure_stats(after), routing=routing, identity=identity,
        base_chart_sha256=sha(base_dir/'chart.json'), onset_cache_sha256=sha(neural_dir/'prediction.npz'),
        feature_metadata_sha256=sha(cached/'features/metadata.json'))
    save(output/'report.json', report)
    chart['postprocess_report_sha256'] = sha(output/'report.json')
    save(output/'chart.json', chart)
    save(output/'base-chart.json', base)
    save(output/'status.json', dict(status='completed', onsets=routing['selected_candidates'], events=len(after)))
    print(json.dumps(dict(status='completed', output=str(output), routing=routing)))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['evaluate', 'song'])
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--cached', type=Path)
    p.add_argument('--baseline', type=Path)
    p.add_argument('--neural', type=Path)
    a = p.parse_args()
    try:
        if a.mode == 'evaluate':
            evaluate(a.output.resolve())
        else:
            song(a.cached.resolve(), a.baseline.resolve(), a.neural.resolve(), a.output.resolve())
    except BaseException as error:
        if a.output.exists():
            # Keep completed/partial evidence; never retry by overwriting it.
            failure = a.output/'failure.json'
            if not failure.exists():
                save(failure, dict(error=repr(error), command=[sys.executable,*sys.argv]))
        raise
