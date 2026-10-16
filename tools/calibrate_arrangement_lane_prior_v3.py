"""Validation-only bounded, type-preserving whole-song orientation prior."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from mdac2.data_preparation.arrangement_context_v3 import ContextDataset,decode,collapse_stats
from mdac2.data_preparation.arrangement_context_training_v3 import balance
from mdac2.data_preparation.arrangement_data_v3 import PATTERNS
from mdac2.data_preparation.joint_evaluation import EventMetrics
from mdac2.data_preparation.arrangement_lane_prior_v3 import decode_with_lane_prior
from mdac2.data_preparation.arrangement_balance_v3 import lane_stats,balance_error,five_class_report
from mdac2.data_preparation.training_chart import TrainingEvent
from mdac2.data_preparation.supervised_training import atomic_json
from mdac2.data_preparation.audio_features import sha

def run(root,output):
    output.mkdir(parents=True,exist_ok=False)
    config=json.loads((root/'arrangement.json').read_text(encoding='utf-8'))
    raw=root/'evaluation'
    base=Path(json.loads((root/'calibration-v1/selection.json').read_text(encoding='utf-8'))['evaluation'])
    original=json.loads((base/'validation.json').read_text(encoding='utf-8'))
    class_alpha=original['logit_weight_correction']
    data=ContextDataset(config,'validation');first={}
    for i,(ri,_) in enumerate(data.index):first.setdefault(ri,i)
    weights=np.asarray(balance(config)['pattern_weights'],np.float64)
    # Fixed candidate strengths, preserving class/dual/hold at raw argmax.
    grid=[0.,.5,1.];reports={};cache={};reference_dual=reference_attacks=0
    for ri,row in enumerate(data.rows):
        data.base[first[ri]];mask=data.base.loaded[ri][2]['onset_mask'];lab=data.labels(ri)
        chart=json.loads((data.root/row['labels']).read_text(encoding='utf-8'))
        truth=[TrainingEvent.from_dict(e) for e in chart['events'] if e['kind']!='gear']
        scores={}
        for mode in ('reference_onsets','predicted_onsets'):
            path=raw/f'scores-{mode}'/(row['chart_sha256']+'.npz')
            if sha(path)!=original['score_cache_sha256'][path.relative_to(raw).as_posix()]:raise ValueError('Raw score identity changed')
            with np.load(path) as a:scores[mode]={k:a[k] for k in a.files}
        positions={}
        for e in truth:
            if e.kind not in ('hold','gear'):positions.setdefault(round(e.start,3),set()).update(e.lanes)
        reference_attacks+=len(positions);reference_dual+=sum(len(x)==2 for x in positions.values())
        cache[row['chart_sha256']]=(chart,truth,mask,lab,scores)
    reference_fraction=reference_dual/max(1,reference_attacks)
    for strength in grid:
        result={};outputs={};correct=total=0
        for mode in ('reference_onsets','predicted_onsets'):
            metric=EventMetrics();distribution=[];records={}
            for key,(chart,truth,mask,lab,scores) in cache.items():
                s=scores[mode]
                events,audit=decode_with_lane_prior(s['logits']-class_alpha*np.log(weights),s['log_lengths'],s['times'],chart['duration'],strength)
                metric.add(events,truth,chart['duration'],audit,mask)
                if mode=='reference_onsets':
                    decisions={round(d['start'],3):d['token'] for d in audit['decisions']}
                    selected=np.array([decisions.get(round(float(t),3),-1) for t in lab['times']]);valid=lab['valid']
                    correct+=int(((selected==lab['tokens'])&valid).sum());total+=int(valid.sum())
                attacks={}
                for e in events:
                    if e.kind not in ('hold','gear'):attacks.setdefault(e.start,set()).update(e.lanes)
                collapse=collapse_stats([d['token'] for d in audit['decisions']],[d['start'] for d in audit['decisions']])
                reference_collapse=collapse_stats(lab['tokens'][lab['valid']],lab['times'][lab['valid']])
                distribution.append(dict(chart_sha256=key,attack_positions=len(attacks),dual_positions=sum(len(x)==2 for x in attacks.values()),
                    lane_balance=dict(prediction=lane_stats(events),reference=lane_stats(truth)),collapse=collapse,excess_dominance=max(0.,collapse['dominant_pattern_fraction']-reference_collapse['dominant_pattern_fraction']-.15),
                    lane_entities={lane:sum(lane in e.lanes for e in events) for lane in ('ground','air')}))
                records[key]=dict(chart_sha256=key,duration=chart['duration'],split='validation',events=[e.to_dict() for e in events],audit=audit,
                                  mode=mode,logit_weight_correction=class_alpha,lane_prior_strength=strength,review='pending human listening')
            m=five_class_report(metric.report());positions=sum(x['attack_positions'] for x in distribution);dual=sum(x['dual_positions'] for x in distribution)
            m.update(per_chart_distribution=distribution,attack_positions=positions,dual_attack_positions=dual,dual_attack_fraction=dual/max(1,positions))
            result[mode]=m;outputs[mode]=records
        result['reference_onsets']['pattern_accuracy']=correct/max(1,total)
        f=result['predicted_onsets'];tp=sum(x['tp'] for x in f['six_classes'].values());fp=sum(x['fp'] for x in f['six_classes'].values());fn=sum(x['fn'] for x in f['six_classes'].values())
        micro=2*tp/max(1,2*tp+fp+fn)
        # Do not reward lane coverage obtained by outputting both lanes.
        excess=max(0.,f['dual_attack_fraction']-reference_fraction-.05)
        dominance=float(np.mean([d['excess_dominance'] for d in f['per_chart_distribution']]))
        imbalance=float(np.mean([balance_error(d['lane_balance']['prediction'],d['lane_balance']['reference']) for d in f['per_chart_distribution']]))
        score=.5*f['macro_five_class_f1']+.25*micro+.25*result['reference_onsets']['pattern_accuracy']-.5*imbalance-.25*dominance
        if strength==0:
            for mode in result:
                for kind,m in result[mode]['six_classes'].items():
                    for k in ('tp','fp','fn'):
                        if m[k]!=original['free_running'][mode]['six_classes'][kind][k]:raise AssertionError('Raw cache reproduction failed')
        reports[str(strength)]=dict(free_running=result,selection_score=score,balance_error=imbalance,micro_class_f1=micro,excess_dual_fraction=excess,excess_dominance=dominance)
        candidate=output/f'strength-{strength:g}';candidate.mkdir()
        atomic_json(candidate/'metrics.json',reports[str(strength)])
        for mode,records in outputs.items():
            (candidate/f'charts-{mode}').mkdir()
            for key,value in records.items():atomic_json(candidate/f'charts-{mode}'/(key+'.json'),value)
        print(json.dumps(dict(strength=strength,score=score,macro=f['macro_five_class_f1'],micro=micro,dual=f['dual_attack_fraction'])),flush=True)
    baseline=reports['0.0']
    eligible=[]
    for value in grid:
        c=reports[str(value)];f=c['free_running']['predicted_onsets'];b=baseline['free_running']['predicted_onsets']
        acceptable=(f['macro_five_class_f1']>=b['macro_five_class_f1']-.01 and c['micro_class_f1']>=baseline['micro_class_f1']-.01 and c['free_running']['reference_onsets']['pattern_accuracy']>=baseline['free_running']['reference_onsets']['pattern_accuracy']-.02)
        c['quality_guard_passed']=acceptable
        if acceptable:eligible.append(value)
    selected=max(eligible,key=lambda a:(reports[str(a)]['selection_score'],-a))
    chosen=output/f'strength-{selected:g}'
    report={**original,'free_running':reports[str(selected)]['free_running'],'logit_weight_correction':class_alpha,'lane_prior_strength':selected,
            'selection':'0.5 five-class macro + 0.25 class micro + 0.25 oracle combo accuracy - 0.5 per-song lane/single-click/dual error - 0.25 excess dominance; fixed bounded lane-prior strength grid after frozen class correction; same-kind single light/heavy orientations only',
            'calibration_grid':grid,'raw_evaluation_sha256':sha(base/'validation.json'),'reference_dual_fraction':reference_fraction}
    atomic_json(chosen/'validation.json',report)
    frozen=json.loads((raw/'frozen.json').read_text(encoding='utf-8'))
    frozen.update(logit_weight_correction=class_alpha,lane_prior_strength=selected,pattern_weights=weights.tolist(),validation_sha256=sha(chosen/'validation.json'),
                  calibration_tool_sha256=sha(Path(__file__)),lane_prior_implementation_sha256=sha(ROOT/'src/mdac2/data_preparation/arrangement_lane_prior_v3.py'))
    atomic_json(chosen/'frozen.json',frozen)
    atomic_json(output/'selection.json',dict(selected_strength=selected,evaluation=chosen.resolve().as_posix(),grid=reports,
                                           locked_test_opened=False,checkpoint_changed=False,threshold_changed=False,quality_guard='at most 0.01 absolute macro/micro F1 loss and 0.02 oracle combination accuracy loss vs strength0',eligible_strengths=eligible))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--experiment',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.experiment.resolve(),a.output.resolve())
