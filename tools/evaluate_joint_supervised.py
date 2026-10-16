"""Validation-only threshold selection, freezing, and explicit locked evaluation."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from mdac2.data_preparation.audio_features import sha
from mdac2.data_preparation import joint_decoding, joint_evaluation
from mdac2.data_preparation.joint_decoding import decode_song, event_time, peaks, place_core
from mdac2.data_preparation.joint_evaluation import EventMetrics, counts, result
from mdac2.data_preparation.joint_model import JointChartModel
from mdac2.data_preparation.supervised_training import atomic_json, seed_all
from mdac2.data_preparation.training_chart import CLASSES, LANES, TrainingEvent, validate_events
from mdac2.data_preparation.window_dataset import WindowDataset


def infer(config, checkpoint_path, output_dir, role):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_hash = sha(checkpoint_path)
    saved = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if saved['identity']['manifest_sha256'] != sha(Path(config['manifest'])):
        raise ValueError('Model/data identity mismatch')
    seed_all(config['seed'])
    model = JointChartModel().to(config['device']).eval()
    model.load_state_dict(saved['model'])
    dataset = WindowDataset(config['manifest'], config['cache'], role, tensor_cache=config['tensors'],
                            allow_locked_test=role=='locked_test')
    prediction_index = []
    for ri, row in enumerate(dataset.rows):
        path = output_dir/(row['chart_sha256']+'.npz')
        metadata_path = path.with_suffix('.json')
        identity = dict(checkpoint_sha256=checkpoint_hash, labels_sha256=row['labels_sha256'],
                        manifest_sha256=sha(Path(config['manifest'])), role=role)
        if path.exists() and metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
            if any(metadata['identity'].get(k) != v for k,v in identity.items()) or metadata['sha256'] != sha(path) or metadata['center_coverage'] != 'exactly once':
                raise ValueError('Stale prediction cache cannot be reused')
        else:
            chart = json.loads((dataset.root/row['labels']).read_text(encoding='utf-8'))
            frame_count = int(np.ceil(chart['duration']*100-1e-8))
            arrays = dict(onsets=np.zeros((frame_count,2,6),np.float32),offsets=np.zeros((frame_count,2,6),np.float32),
                          hold_ends=np.zeros((frame_count,2),np.float32),hold_end_offsets=np.zeros((frame_count,2),np.float32),
                          occupancy=np.zeros((frame_count,2),np.float32),vocal_activity_prior=np.zeros(frame_count,np.float32))
            coverage = np.zeros(frame_count,np.uint8)
            indices = [i for i,(row_index,_) in enumerate(dataset.index) if row_index == ri]
            for first in range(0,len(indices),config['batch_size']):
                part = indices[first:first+config['batch_size']]
                samples = [dataset[i] for i in part]
                features = torch.from_numpy(np.stack([s['features'] for s in samples])).to(config['device'])
                timing = torch.from_numpy(np.stack([s['timing'] for s in samples])).to(config['device'])
                with torch.no_grad(), torch.autocast('cuda',dtype=torch.bfloat16,enabled=config['amp']):
                    out = model(features,timing)
                out = {k:v.float() for k,v in out.items()}
                values = dict(onsets=out['onset_logits'].sigmoid().cpu().numpy(),offsets=out['offsets'].cpu().numpy(),
                              hold_ends=out['hold_end_logits'].sigmoid().cpu().numpy(),hold_end_offsets=out['hold_end_offsets'].cpu().numpy(),
                              occupancy=out['occupancy_logits'].sigmoid().cpu().numpy(),vocal_activity_prior=out['vocal_activity_prior'].cpu().numpy())
                for b,index in enumerate(part):
                    place_core(arrays,coverage,dataset.index[index][1],{k:v[b] for k,v in values.items()})
            if not np.all(coverage == 1):
                raise ValueError('Incomplete full-song center stitching')
            temporary = path.with_suffix('.tmp.npz')
            np.savez_compressed(temporary,**arrays)
            os.replace(temporary,path)
            metadata = dict(identity=identity,sha256=sha(path),frames=frame_count,center_coverage='exactly once',windows=len(indices))
            atomic_json(metadata_path,metadata)
        prediction_index.append(dict(chart_sha256=row['chart_sha256'],prediction=str(path),**metadata))
        print(json.dumps(dict(event='prediction',role=role,song=ri+1,total=len(dataset.rows)),ensure_ascii=True),flush=True)
    atomic_json(output_dir/'index.json',dict(role=role,checkpoint_sha256=checkpoint_hash,charts=prediction_index))
    return dataset


def song_inputs(dataset, config, directory):
    for row in dataset.rows:
        reference = json.loads((dataset.root/row['labels']).read_text(encoding='utf-8'))
        tensor_record = dataset.target_index[row['chart_sha256']]
        with np.load(Path(config['tensors'])/tensor_record['file'],allow_pickle=False) as target:
            masks = {k:target[k] for k in ('onset_mask','hold_end_mask')}
        path = Path(directory)/(row['chart_sha256']+'.npz')
        with np.load(path,allow_pickle=False) as saved:
            prediction = {k:saved[k] for k in saved.files}
        yield row,reference,prediction,masks


def threshold_selection(dataset, config, directory):
    grid = [.01,.025,.05,.075,.1,.15,.2,.3,.4,.5,.6,.7,.8,.9]
    class_counts = np.zeros((6,len(grid),3),np.int64)
    tail_counts = np.zeros((len(grid),3),np.int64)
    for row,reference,prediction,masks in song_inputs(dataset,config,directory):
        duration = reference['duration']
        for k,kind in enumerate(CLASSES):
            for lane,side in enumerate(LANES):
                truth = [e['start'] for e in reference['events'] if e['kind']==kind and side in e['lanes']
                         and masks['onset_mask'][min(len(masks['onset_mask'])-1,int(e['start']*100+1e-7)),lane,k]]
                for ti,threshold in enumerate(grid):
                    times = [event_time(f,prediction['offsets'][f,lane,k],duration)
                             for f in peaks(prediction['onsets'][:,lane,k],threshold)]
                    times = [t for t in times if masks['onset_mask'][min(len(masks['onset_mask'])-1,int(t*100+1e-7)),lane,k]]
                    matched,_ = counts(times,truth,.05)
                    class_counts[k,ti] += matched
            # Gemini remains per-lane here; its strict pair requirement and all
            # candidate conflicts are assessed in the final decoded report.
        for lane,side in enumerate(LANES):
            truth = [e['end'] for e in reference['events'] if e['kind']=='hold' and side in e['lanes']
                     and e['end']*100 < len(masks['hold_end_mask'])
                     and masks['hold_end_mask'][int(e['end']*100+1e-7),lane]]
            for ti,threshold in enumerate(grid):
                times = [event_time(f,prediction['hold_end_offsets'][f,lane],duration)
                         for f in peaks(prediction['hold_ends'][:,lane],threshold)]
                times = [t for t in times if masks['hold_end_mask'][min(len(masks['hold_end_mask'])-1,int(t*100+1e-7)),lane]]
                matched,_ = counts(times,truth,.1)
                tail_counts[ti] += matched
    chosen = []
    report = {}
    for k,kind in enumerate(CLASSES):
        candidates = [result(c) for c in class_counts[k]]
        best = max(range(len(grid)),key=lambda i:(candidates[i]['f1'],candidates[i]['precision'],grid[i]))
        chosen.append(grid[best])
        report[kind] = dict(chosen=grid[best],grid=[dict(threshold=t,**s) for t,s in zip(grid,candidates)])
    tail_results = [result(c) for c in tail_counts]
    best_tail = max(range(len(grid)),key=lambda i:(tail_results[i]['f1'],tail_results[i]['precision'],grid[i]))
    return dict(thresholds=chosen,tail_threshold=grid[best_tail]),dict(role='validation',six_classes=report,
        tails=dict(chosen=grid[best_tail],grid=[dict(threshold=t,**s) for t,s in zip(grid,tail_results)]),
        objective='micro lane-entity F1 with 50ms head / 100ms tail tolerance; independent thresholds before pairing/rules',
        locked_test_used=False)


def decoded_report(dataset, config, directory, decoder, save_dir=None):
    metrics = EventMetrics(.05)
    strict = EventMetrics(.025)
    song_reports = []
    if save_dir:
        Path(save_dir).mkdir(parents=True,exist_ok=True)
    for row,reference,prediction,masks in song_inputs(dataset,config,directory):
        events,audit = decode_song(prediction,reference['duration'],decoder)
        if validate_events(events,reference['duration']):
            raise ValueError('Decoded event rules did not resolve conflicts')
        truth = [TrainingEvent.from_dict(e) for e in reference['events']]
        metrics.add(events,truth,reference['duration'],audit,masks['onset_mask'])
        strict.add(events,truth,reference['duration'],audit,masks['onset_mask'])
        song_reports.append(dict(chart_sha256=row['chart_sha256'],predicted_events=len(events),
                                 reference_events=len(truth),cross_window_holds=audit['cross_window_holds']))
        if save_dir:
            atomic_json(Path(save_dir)/(row['chart_sha256']+'.json'),dict(chart_sha256=row['chart_sha256'],
                title=row['title'],role=row['split'],duration=reference['duration'],events=[e.to_dict() for e in events],
                audit=audit,human_listening='pending',game_acceptance='pending'))
    return dict(metrics=metrics.report(),strict_25ms=strict.report(),songs=song_reports,decoder=decoder)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--locked-test',action='store_true')
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding='utf-8'))
    args.output.mkdir(parents=True,exist_ok=True)
    if args.locked_test:
        frozen = json.loads((args.output/'frozen.json').read_text(encoding='utf-8'))
        if frozen['checkpoint_sha256'] != sha(args.checkpoint) or frozen['config_sha256'] != sha(args.config):
            raise ValueError('Locked test requires unchanged frozen weights/config')
        if frozen['decoder_code_sha256'] != sha(Path(joint_decoding.__file__)) or frozen['metrics_code_sha256'] != sha(Path(joint_evaluation.__file__)):
            raise ValueError('Evaluation code changed after freeze')
        if frozen['evaluation_tool_sha256'] != sha(Path(__file__)):
            raise ValueError('Evaluation tool changed after freeze')
        completed = args.output/'locked-test.json'
        if completed.exists():
            raise ValueError('Locked test was already evaluated; do not repeat or retune')
        marker = args.output/'locked-test-started.json'
        if marker.exists():
            recorded = json.loads(marker.read_text(encoding='utf-8'))
            if recorded['frozen_sha256'] != sha(args.output/'frozen.json'):
                raise ValueError('Cannot change frozen experiment after opening test')
        else:
            atomic_json(marker,dict(started=time.strftime('%Y-%m-%d %H:%M:%S'),frozen_sha256=sha(args.output/'frozen.json'),
                                   purpose='one final frozen evaluation; never threshold selection'))
        dataset = infer(config,args.checkpoint,args.output/'predictions-locked-test','locked_test')
        report = decoded_report(dataset,config,args.output/'predictions-locked-test',frozen['decoder'])
        atomic_json(completed,dict(role='locked_test',frozen_sha256=sha(args.output/'frozen.json'),**report))
    else:
        if (args.output/'frozen.json').exists():
            raise ValueError('Experiment already frozen; do not retune')
        dataset = infer(config,args.checkpoint,args.output/'predictions-validation','validation')
        decoder,tuning = threshold_selection(dataset,config,args.output/'predictions-validation')
        atomic_json(args.output/'threshold-search.json',tuning)
        occupancy_trials = []
        for threshold in (.2,.35,.5,.65):
            candidate = {**joint_decoding.DEFAULT_DECODER,**decoder,'occupancy_threshold':threshold}
            report = decoded_report(dataset,config,args.output/'predictions-validation',candidate)
            occupancy_trials.append(report)
            print(json.dumps(dict(event='occupancy_trial',threshold=threshold,hold_f1=report['metrics']['holds']['hold_joint']['f1'])),flush=True)
        best = max(occupancy_trials,key=lambda r:(r['metrics']['holds']['hold_joint']['f1'],r['metrics']['macro_class_f1']))
        atomic_json(args.output/'occupancy-search.json',occupancy_trials)
        report = decoded_report(dataset,config,args.output/'predictions-validation',best['decoder'],args.output/'charts-validation')
        atomic_json(args.output/'validation.json',dict(role='validation',locked_test_opened=False,**report))
        frozen = dict(frozen_at=time.strftime('%Y-%m-%d %H:%M:%S'),checkpoint=str(args.checkpoint.resolve()),
            checkpoint_sha256=sha(args.checkpoint),config_sha256=sha(args.config),manifest_sha256=sha(Path(config['manifest'])),
            decoder=best['decoder'],decoder_code_sha256=sha(Path(joint_decoding.__file__)),
            metrics_code_sha256=sha(Path(joint_evaluation.__file__)),evaluation_tool_sha256=sha(Path(__file__)),
            validation_report_sha256=sha(args.output/'validation.json'),
            checkpoint_selection='minimum masked validation loss with min_delta, finite epochs and early stopping',
            threshold_selection='validation only; raw per-head F1 then paired hold F1 for occupancy',
            human_listening='pending',game_acceptance='pending',production_promoted=False)
        atomic_json(args.output/'frozen.json',frozen)
        print(json.dumps(dict(event='frozen',**report['metrics']),ensure_ascii=True),flush=True)


if __name__ == '__main__':
    main()
