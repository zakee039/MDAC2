"""Fixed-grid sanity baselines and raw-versus-decoded failure attribution.

Read existing validation predictions. These diagnostics do not select settings.
"""
from collections import Counter
from fractions import Fraction
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from diagnose_joint_baseline import BASE,read,groups,operations,valid_attack_groups
from mdac2.data_preparation.audio_features import sha
from mdac2.data_preparation.joint_decoding import peaks,event_time,pair_points
from mdac2.data_preparation.joint_evaluation import counts,result
from mdac2.data_preparation.training_chart import TrainingEvent
from mdac2.data_preparation.community_labels import AIR,LIGHT


def grid_times(chart,step,first,last):
    tempos=chart['timing']['tempos'];out=[]
    for i,t in enumerate(tempos):
        left=max(first,t['seconds']);right=min(last,tempos[i+1]['seconds'] if i+1<len(tempos) else last)
        bpm=float(Fraction(t['bpm']));q=float(Fraction(t['quarter']))
        qleft=q+(left-t['seconds'])*bpm/60
        qright=q+(right-t['seconds'])*bpm/60
        if right<=left:continue
        for v in np.arange(np.ceil(qleft/step-1e-9),np.ceil(qright/step-1e-9))*step:
            out.append(float(t['seconds']+(v-q)*60/bpm))
    return sorted(set(out))


def main():
    output=Path(sys.argv[1]);output.mkdir(parents=True,exist_ok=False)
    config=read(BASE/'baseline-lr3e4.json');manifest=read(config['manifest']);root=Path(config['manifest']).parent
    frozen=read(BASE/'evaluation-lr3e4/frozen.json');thresholds=np.array(frozen['decoder']['thresholds'])
    tables={str(s):np.zeros(3,np.int64) for s in (1,.5,.25)};stages=Counter();rows=[];source_lane_checked=0
    score_groups={lane:[] for lane in range(2)}
    same_frame_multi=Counter()
    for row in manifest['charts']:
        if not row['training_eligible'] or row['split'] not in ('train','validation'):continue
        chart=read(root/row['labels']);sources={s['id']:s for s in chart['source_events']}
        for e in chart['events']:
            if e['kind']=='light' and e['source_code'] in LIGHT:
                assert len(e['source_ids'])==1
                cell=sources[e['source_ids'][0]]
                assert e['lanes']==(['air'] if cell['channel'] in AIR else ['ground'])
                assert abs(e['start']-cell['seconds'])<1e-9
                source_lane_checked+=1
        if row['split']!='validation':continue
        duration=chart['duration'];first,last=13.5,duration-5.11
        truth=[TrainingEvent.from_dict(e) for e in chart['events']]
        decoded=read(BASE/'evaluation-lr3e4/charts-validation'/(row['chart_sha256']+'.json'))
        events=[TrainingEvent.from_dict(e) for e in decoded['events']]
        with np.load(Path(config['tensors'])/(row['chart_sha256']+'.npz')) as data:
            mask=data['onset_mask'];target=data['onsets']
        path=BASE/'evaluation-lr3e4/predictions-validation'/(row['chart_sha256']+'.npz')
        assert sha(path)==read(path.with_suffix('.json'))['sha256']
        with np.load(path) as data:prediction={k:data[k] for k in data.files}
        rt=[g[0].start for g in valid_attack_groups(truth,mask,first,last)]
        pt=[g[0].start for g in valid_attack_groups(events,mask,first,last)]
        for step in (1,.5,.25):
            grid=grid_times(chart,step,first,last)
            grid=[t for t in grid if mask[min(len(mask)-1,int(t*100)),:,:4].all()]
            c,_=counts(grid,rt,.05);tables[str(step)]+=c
        matches=pair_points([(t,) for t in pt],[(t,) for t in rt],.05)
        used={r for p,r in matches};local=Counter(matched=len(matches),missed=len(rt)-len(matches))
        raw=[]
        for lane in range(2):
            for kind in range(4):
                raw.extend(event_time(f,prediction['offsets'][f,lane,kind],duration) for f in peaks(prediction['onsets'][:,lane,kind],thresholds[kind],3))
        raw=np.array(raw)
        for i,t in enumerate(rt):
            if i in used:continue
            left=max(0,int((t-.05)*100));right=min(len(mask),int((t+.05)*100)+1)
            if not (prediction['onsets'][left:right,:,:4]>=thresholds[:4]).any():local['no_raw_head_above_threshold_near_reference']+=1
            elif len(raw) and (np.abs(raw-t)<=.05+1e-9).any():local['raw_peak_exists_but_not_matched_after_decode']+=1
            else:local['raw_score_but_no_peak_within_50ms']+=1
        stages.update(local)
        light=target[:,:,0]>.5;valid=mask[:,:,0].all(1)&(light.sum(1)==1)
        for lane in range(2):score_groups[lane].extend(prediction['onsets'][valid&light[:,lane],:,0].tolist())
        attack=target[:,:,:4].any((1,2))&mask[:,:,:4].all((1,2))
        active=(prediction['onsets'][attack,:,:4]>=thresholds[:4])
        same_frame_multi['reference_attack_frames']+=int(attack.sum())
        same_frame_multi['both_lanes_above_threshold']+=int(active.any(2).all(1).sum())
        same_frame_multi['multiple_types_above_threshold']+=int((active.any(1).sum(1)>1).sum())
        rows.append(dict(chart_sha256=row['chart_sha256'],stage_counts=dict(local)))
    stage_report=dict(locked_test_opened=False,thresholds_changed=False,source_lane_checked=source_lane_checked,
        ordinary_light_lane_and_seconds_preserved_from_source_cells=True,
        source_auditory_alignment='not established by these code/label comparisons',
        grid_baselines={k:result(v) for k,v in tables.items()},
        grid_definition='one unconditional attack per 1, 1/2, or 1/4 quarter note; source timing only, no audio; identical 50ms tolerance/interior crop; illustrative fixed comparators, no selection',
        miss_stages=dict(stages),per_chart=rows,same_frame_multi=dict(same_frame_multi),
        mean_light_scores_at_true_single_lane={str(k):dict(count=len(v),ground=float(np.mean(np.array(v)[:,0])),air=float(np.mean(np.array(v)[:,1]))) for k,v in score_groups.items()},
        metric_warning='Onset matching does not establish musical quality, pattern choice, or vocal-priority alignment.',
        source_sha256=sha(Path(__file__)))
    (output/'report.json').write_text(json.dumps(stage_report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in stage_report.items() if k!='per_chart'}),flush=True)


if __name__=='__main__':main()
