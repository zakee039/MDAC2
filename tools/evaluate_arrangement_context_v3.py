"""Compare a revised arranger on unchanged validation onset predictions."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from mdac2.data_preparation.arrangement_context_v3 import ContextDataset,predict,decode,collapse_stats
from mdac2.data_preparation.arrangement_context_training_v3 import identity,make_model
from mdac2.data_preparation.dual_inference import onset_times
from mdac2.data_preparation.joint_evaluation import EventMetrics
from mdac2.data_preparation.arrangement_balance_v3 import lane_stats,five_class_report
from mdac2.data_preparation.training_chart import TrainingEvent
from mdac2.data_preparation.supervised_training import atomic_json,seed_all
from mdac2.data_preparation.audio_features import sha

def main(root,output):
    output.mkdir(parents=True,exist_ok=False);seed_all(20260928)
    config=json.loads((root/'arrangement.json').read_text(encoding='utf-8'))
    reuse=json.loads((root/'reuse.json').read_text(encoding='utf-8'))
    original=Path(reuse['onset_experiment']);old_evaluation=original/'evaluation'
    frozen=json.loads((old_evaluation/'frozen.json').read_text(encoding='utf-8'))
    if sha(Path(frozen['checkpoints']['onset']['path']))!=frozen['checkpoints']['onset']['sha256']:raise ValueError('Onset checkpoint changed')
    path=root/'formal-arrangement/best.pt';saved=torch.load(path,map_location='cpu',weights_only=False)
    if saved['identity']!=identity(config) or saved['config']!=config:raise ValueError('Arrangement identity changed')
    model=make_model('arrangement').cuda();model.load_state_dict(saved['model']);model.eval()
    data=ContextDataset(config,'validation');first={}
    for i,(ri,_) in enumerate(data.index):first.setdefault(ri,i)
    metrics={mode:EventMetrics() for mode in ('reference_onsets','predicted_onsets')}
    distributions={mode:[] for mode in metrics};correct=total=0;cache_identity={};score_identity={}
    for mode in metrics:
        (output/f'charts-{mode}').mkdir()
        (output/f'scores-{mode}').mkdir()
    for ri,row in enumerate(data.rows):
        data.base[first[ri]];features,_,target=data.base.loaded[ri];lab=data.labels(ri)
        chart=json.loads((data.root/row['labels']).read_text(encoding='utf-8'));truth=[TrainingEvent.from_dict(e) for e in chart['events'] if e['kind']!='gear']
        cache=old_evaluation/'predictions'/(row['chart_sha256']+'.npz')
        cache_identity[row['chart_sha256']]=sha(cache)
        with np.load(cache) as a:cached={k:a[k] for k in ('probability','offsets')}
        for mode in metrics:
            times=lab['times'] if mode=='reference_onsets' else onset_times(cached,reuse['onset_threshold'],chart['duration'])
            out=predict(model,features,times,chart['duration'])
            np.savez_compressed(output/f'scores-{mode}'/(row['chart_sha256']+'.npz'),
                                times=np.asarray(times,np.float64),**out)
            score_path=output/f'scores-{mode}'/(row['chart_sha256']+'.npz')
            score_identity[score_path.relative_to(output).as_posix()]=sha(score_path)
            events,audit=decode(out['logits'],out['log_lengths'],times,chart['duration'])
            metrics[mode].add(events,truth,chart['duration'],audit,target['onset_mask'])
            decisions={round(d['start'],3):d['token'] for d in audit['decisions']}
            if mode=='reference_onsets':
                p=np.array([decisions.get(round(float(t),3),-1) for t in times]);valid=lab['valid']
                correct+=int(np.sum((p==lab['tokens'])&valid));total+=int(valid.sum())
            attacks={}
            for e in events:
                if e.kind not in ('hold','gear'):attacks.setdefault(e.start,set()).update(e.lanes)
            entry=dict(chart_sha256=row['chart_sha256'],attack_positions=len(attacks),dual_positions=sum(len(v)==2 for v in attacks.values()),
                       lane_balance=dict(prediction=lane_stats(events),reference=lane_stats(truth)),collapse=collapse_stats([d['token'] for d in audit['decisions']],[d['start'] for d in audit['decisions']]),
                       lane_entities={lane:sum(lane in e.lanes for e in events) for lane in ('ground','air')})
            distributions[mode].append(entry)
            atomic_json(output/f'charts-{mode}'/(row['chart_sha256']+'.json'),dict(chart_sha256=row['chart_sha256'],duration=chart['duration'],
                split='validation',events=[e.to_dict() for e in events],audit=audit,mode=mode,review='pending human listening'))
        print(json.dumps(dict(event='decoded',chart=ri+1,total=len(data.rows))),flush=True)
    reports={mode:five_class_report(m.report()) for mode,m in metrics.items()}
    for mode,report in reports.items():
        entries=distributions[mode];positions=sum(d['attack_positions'] for d in entries);dual=sum(d['dual_positions'] for d in entries)
        report.update(per_chart_distribution=entries,attack_positions=positions,dual_attack_positions=dual,dual_attack_fraction=dual/max(1,positions))
    reports['reference_onsets']['pattern_accuracy']=correct/max(1,total)
    checkpoints=dict(onset=frozen['checkpoints']['onset'],arrangement=dict(path=path.as_posix(),sha256=sha(path),epoch=saved['state']['best_epoch']))
    report=dict(status='completed',free_running=reports,checkpoints=checkpoints,threshold=reuse['onset_threshold'],
                onset_frozen=True,gear_only_onset_policy='legacy onset frozen; predicted gear-only positions may remain and count as false positives against gear-free reference',onset_cache_sha256=cache_identity,score_cache_sha256=score_identity,
                evaluation_tool_sha256=sha(Path(__file__)),selection='full-song reference-onset decoded validation score each epoch',
                reference_history_used=False,locked_test_opened=False,human_listening='pending',game_acceptance='pending',production_promoted=False)
    atomic_json(output/'validation.json',report)
    atomic_json(output/'frozen.json',dict(checkpoints=checkpoints,threshold=reuse['onset_threshold'],validation_sha256=sha(output/'validation.json'),
                manifest_sha256=sha(Path(config['manifest'])),locked_test_used=False,arrangement_identity=saved['identity']))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--experiment',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();main(a.experiment.resolve(),a.output.resolve())
