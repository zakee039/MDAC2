"""Validation-only threshold selection, oracle-time and generated-time decoding."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from mdac2.data_preparation.dual_data import DualDataset,PATTERNS
from mdac2.data_preparation.dual_training import identity,make_model,evaluate,balance
from mdac2.data_preparation.dual_inference import acoustic_predictions,onset_times,arrange
from mdac2.data_preparation.joint_evaluation import EventMetrics,counts,result
from mdac2.data_preparation.training_chart import TrainingEvent
from mdac2.data_preparation.supervised_training import atomic_json,seed_all
from mdac2.data_preparation.audio_features import sha

def main(root,output):
    output.mkdir(parents=True,exist_ok=False);seed_all(20260928)
    configs={s:json.loads((root/f'{s}.json').read_text(encoding='utf-8')) for s in ('onset','arrangement')}
    models={};checkpoints={}
    for stage,c in configs.items():
        path=root/f'formal-{stage}'/'best.pt';saved=torch.load(path,map_location='cpu',weights_only=False)
        if saved['identity']!=identity(c) or saved['config']!=c:raise ValueError('Checkpoint identity mismatch')
        model=make_model(stage).cuda();model.load_state_dict(saved['model']);model.eval();models[stage]=model
        checkpoints[stage]=dict(path=path.as_posix(),sha256=sha(path),epoch=saved['state']['best_epoch'])
    data=DualDataset(configs['onset'],'validation');base=data.base
    first={}
    for i,(ri,_) in enumerate(base.index):first.setdefault(ri,i)
    thresholds=[.1,.2,.3,.4,.5,.6,.7,.8,.9];scores={t:{tol:np.zeros(3,np.int64) for tol in (.02,.05)} for t in thresholds}
    song_data=[];prediction_dir=output/'predictions';prediction_dir.mkdir()
    for ri,row in enumerate(data.rows):
        base[first[ri]];features,_,target=base.loaded[ri];lab=data.labels(ri)
        chart=json.loads((base.root/row['labels']).read_text(encoding='utf-8'))
        pred=acoustic_predictions(models['onset'],models['arrangement'],features,len(target['onsets']))
        np.savez_compressed(prediction_dir/(row['chart_sha256']+'.npz'),**pred)
        def eligible(t):return bool(lab['mask'][min(len(lab['mask'])-1,int(t*100+1e-7))])
        reference=[float(t) for t in lab['times'] if eligible(t)]
        for threshold in thresholds:
            times=[t for t in onset_times(pred,threshold,chart['duration']) if eligible(t)]
            for tolerance in (.02,.05):scores[threshold][tolerance]+=counts(times,reference,tolerance)[0]
        song_data.append((row,chart))
        print(json.dumps(dict(event='acoustic_prediction',chart=ri+1,total=len(data.rows))),flush=True)
    selected=max(thresholds,key=lambda t:(result(scores[t][.05])['f1'],result(scores[t][.02])['f1'],-t))
    selection=dict(role='validation',criterion='maximum onset F1 at 50ms; 20ms tie-break',threshold=selected,
                   grid={str(t):{str(tol):result(c) for tol,c in by_tol.items()} for t,by_tol in scores.items()},
                   locked_test_opened=False,checkpoints=checkpoints)
    atomic_json(output/'selection.json',selection)
    metrics={mode:EventMetrics() for mode in ('reference_onsets','predicted_onsets')};distribution={mode:[] for mode in metrics}
    combinations=np.zeros((len(PATTERNS),len(PATTERNS)+1),np.int64)
    reference_distribution=[]
    for mode in metrics:(output/f'charts-{mode}').mkdir()
    for ri,(row,chart) in enumerate(song_data):
        base[first[ri]];target=base.loaded[ri][2];lab=data.labels(ri)
        with np.load(prediction_dir/(row['chart_sha256']+'.npz')) as archive:pred={k:archive[k] for k in archive.files}
        truth=[TrainingEvent.from_dict(e) for e in chart['events']]
        reference_attacks={}
        for event in truth:
            if event.kind not in ('hold','gear'):reference_attacks.setdefault(round(event.start,3),set()).update(event.lanes)
        reference_distribution.append(dict(chart_sha256=row['chart_sha256'],attack_positions=len(reference_attacks),
                                           dual_positions=sum(len(lanes)==2 for lanes in reference_attacks.values())))
        for mode in metrics:
            times=lab['times'].tolist() if mode=='reference_onsets' else onset_times(pred,selected,chart['duration'])
            events,audit=arrange(models['arrangement'],pred['acoustic'],times,chart['duration'])
            if mode=='reference_onsets':
                decisions={round(d['start'],3):d['token'] for d in audit['decisions']}
                for t,token,valid in zip(lab['times'],lab['tokens'],lab['valid']):
                    if valid:combinations[token,decisions.get(round(float(t),3),len(PATTERNS))]+=1
            metrics[mode].add(events,truth,chart['duration'],audit,target['onset_mask'])
            atoms={}
            for e in events:
                if e.kind not in ('hold','gear'):atoms.setdefault(e.start,set()).update(e.lanes)
            entry=dict(chart_sha256=row['chart_sha256'],attack_positions=len(atoms),dual_positions=sum(len(l)==2 for l in atoms.values()),
                       gemini=sum(e.kind=='gemini' for e in events),hold_count=sum(e.kind=='hold' for e in events))
            distribution[mode].append(entry)
            atomic_json(output/f'charts-{mode}'/(row['chart_sha256']+'.json'),dict(chart_sha256=row['chart_sha256'],duration=chart['duration'],
                        split='validation',events=[e.to_dict() for e in events],audit=audit,mode=mode,review='pending human listening'))
        print(json.dumps(dict(event='decoded',chart=ri+1,total=len(song_data))),flush=True)
    teacher=evaluate(models['arrangement'],DualDataset(configs['arrangement'],'validation'),configs['arrangement'],balance(configs['arrangement']))
    reports={mode:metric.report() for mode,metric in metrics.items()}
    for mode in reports:
        positions=sum(r['attack_positions'] for r in distribution[mode]);dual=sum(r['dual_positions'] for r in distribution[mode])
        reports[mode].update(attack_positions=positions,dual_attack_positions=dual,dual_attack_fraction=dual/max(1,positions),per_chart_distribution=distribution[mode])
    oracle=reports['reference_onsets']
    oracle['pattern_accuracy']=float(np.trace(combinations[:,:-1])/max(1,combinations.sum()))
    oracle['pattern_confusion_including_rejected_column']=combinations.tolist()
    oracle['pattern_events']=int(combinations.sum())
    oracle['ambiguous_pattern_events_excluded']=sum(data.records[row['chart_sha256']]['ambiguous_events'] for row in data.rows)
    reference_positions=sum(r['attack_positions'] for r in reference_distribution)
    report=dict(status='completed',selection=selection,teacher_forced_arrangement=teacher,free_running=reports,
                reference_distribution=dict(per_chart=reference_distribution,attack_positions=reference_positions,
                     dual_attack_fraction=sum(r['dual_positions'] for r in reference_distribution)/max(1,reference_positions)),
                locked_test_opened=False,test_status='previously used by old joint baseline; no new independent test claim',
                human_listening='pending',game_acceptance='pending',production_promoted=False,
                vocal_preference='architectural gain only; no source truth or listening verification',
                source_sha256={p.name:sha(p) for p in [Path(__file__),ROOT/'src/mdac2/data_preparation/dual_inference.py']})
    atomic_json(output/'validation.json',report)
    atomic_json(output/'frozen.json',dict(checkpoints=checkpoints,threshold=selected,source_sha256=report['source_sha256'],
                 validation_sha256=sha(output/'validation.json'),manifest_sha256=sha(Path(configs['onset']['manifest'])),locked_test_used=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--experiment',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();main(a.experiment.resolve(),a.output.resolve())
