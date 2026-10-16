"""Rescore v2 and v3 with one gear-free validation definition, never locked test."""
import argparse,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from mdac2.data_preparation.arrangement_context_v4 import ContextDataset
from mdac2.data_preparation.arrangement_balance_v3 import lane_stats,balance_error,five_class_report
from mdac2.data_preparation.joint_evaluation import EventMetrics
from mdac2.data_preparation.training_chart import TrainingEvent
from mdac2.data_preparation.supervised_training import atomic_json
from mdac2.data_preparation.audio_features import sha

def main(root,old,output):
    if output.exists():raise ValueError('Preserve previous comparison')
    config=json.loads((root/'arrangement.json').read_text(encoding='utf-8'));data=ContextDataset(config,'validation')
    current=root/'evaluation'
    variant='v4b' if 'v4b' in config['revision'] else 'v4'
    paths={'v2':old,variant:current};meters={k:EventMetrics() for k in paths};rows=[];first={}
    for i,(ri,_) in enumerate(data.index):first.setdefault(ri,i)
    for ri,row in enumerate(data.rows):
        data.base[first[ri]];target=data.base.loaded[ri][2]
        chart=json.loads((data.root/row['labels']).read_text(encoding='utf-8'))
        truth=[TrainingEvent.from_dict(e) for e in chart['events'] if e['kind']!='gear']
        reference=lane_stats(truth);entry=dict(chart_sha256=row['chart_sha256'],title=row['title'],reference=reference)
        for name,path in paths.items():
            record=json.loads((path/'charts-predicted_onsets'/(row['chart_sha256']+'.json')).read_text(encoding='utf-8'))
            events=[TrainingEvent.from_dict(e) for e in record['events'] if e['kind']!='gear']
            meters[name].add(events,truth,chart['duration'],record['audit'],target['onset_mask'])
            stats=lane_stats(events);entry[name]=dict(**stats,balance_error=balance_error(stats,reference))
        rows.append(entry)
    summaries={}
    for name,meter in meters.items():
        report=five_class_report(meter.report());entries=[r[name] for r in rows]
        report['balance_summary']=dict(mean_reference_relative_error=float(np.mean([e['balance_error'] for e in entries])),
            within_45_55=sum(.45<=e['ground_share']<=.55 for e in entries),
            single_click_within_45_55=sum(.45<=e['single_ground_share']<=.55 for e in entries),
            mean_longest_single_lane_run=float(np.mean([e['longest_single_lane_run'] for e in entries])))
        summaries[name]=report
    atomic_json(output,dict(versions=summaries,per_chart=rows,charts=22,locked_test_opened=False,
        policy='filter gear from reference and predictions for BOTH versions; same frozen onset and masks; legacy gear-origin predictions can remain false positives',
        v2_evaluation=old.as_posix(),current_variant=variant,current_evaluation=paths[variant].as_posix(),source_sha256=sha(Path(__file__)),
        production_promoted=False,human_listening='pending',game_acceptance='pending'))
    print(json.dumps({k:v['balance_summary'] for k,v in summaries.items()}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--experiment',type=Path,required=True);p.add_argument('--old-evaluation',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();main(a.experiment.resolve(),a.old_evaluation.resolve(),a.output.resolve())
