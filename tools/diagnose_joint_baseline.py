"""Read-only failure investigation of the frozen baseline; no fitting or tuning.

Only train/validation are opened. Perturbations are counterfactual diagnostics,
not alternative production inputs, models, or decoder recommendations.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from mdac2.data_preparation.audio_features import sha
from mdac2.data_preparation.joint_model import JointChartModel
from mdac2.data_preparation.joint_decoding import decode_song, place_core, pair_points
from mdac2.data_preparation.joint_evaluation import counts, result
from mdac2.data_preparation.training_chart import TrainingEvent
from mdac2.data_preparation.training_tensors import frame_targets, timing_features, window_starts, slice_window
from mdac2.data_preparation.window_dataset import WindowDataset
from mdac2.data_preparation.supervised_training import atomic_json, seed_all, data_identity

BASE = ROOT/'tools/private/community/supervised-joint-v1-20260927'
VARIANTS = ('audio_shift_8.13s', 'audio_silent', 'beat_phase_plus_quarter', 'beat_phase_removed')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def groups(events, first=0., last=float('inf')):
    out = []
    for e in sorted((e for e in events if e.kind not in ('hold','gear') and first <= e.start < last), key=lambda e:e.start):
        if not out or e.start-out[-1][0].start > .02:
            out.append([e])
        else:
            out[-1].append(e)
    return out


def operations(events, first=0., last=float('inf')):
    grouped = groups(events, first, last)
    both = sum(len({l for e in g for l in e.lanes}) == 2 for g in grouped)
    return dict(positions=len(grouped), both=both, both_fraction=both/max(1,len(grouped)),
                gemini=sum(any(e.kind == 'gemini' for e in g) for g in grouped))


def infer(model, features, timing):
    count = features.shape[-1]
    arrays = dict(onsets=np.zeros((count,2,6),np.float32),offsets=np.zeros((count,2,6),np.float32),
                  hold_ends=np.zeros((count,2),np.float32),hold_end_offsets=np.zeros((count,2),np.float32),
                  occupancy=np.zeros((count,2),np.float32),vocal_activity_prior=np.zeros(count,np.float32))
    coverage = np.zeros(count,np.uint8)
    starts = window_starts(count)
    for first in range(0,len(starts),8):
        part = starts[first:first+8]
        audio = torch.from_numpy(np.stack([slice_window(features,s,time_axis=2)[0] for s in part]).astype(np.float32)).cuda()
        beats = torch.from_numpy(np.stack([slice_window(timing,s)[0] for s in part])).cuda()
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            out = {k:v.float() for k,v in model(audio,beats).items()}
        values = dict(onsets=out['onset_logits'].sigmoid().cpu().numpy(),offsets=out['offsets'].cpu().numpy(),
            hold_ends=out['hold_end_logits'].sigmoid().cpu().numpy(),hold_end_offsets=out['hold_end_offsets'].cpu().numpy(),
            occupancy=out['occupancy_logits'].sigmoid().cpu().numpy(),vocal_activity_prior=out['vocal_activity_prior'].cpu().numpy())
        for i,start in enumerate(part):
            place_core(arrays,coverage,start,{k:v[i] for k,v in values.items()})
    assert (coverage == 1).all()
    return arrays


def correlation(a,b):
    a,b = np.asarray(a,dtype=np.float64).ravel(),np.asarray(b,dtype=np.float64).ravel()
    return float(np.corrcoef(a,b)[0,1]) if a.std()>1e-9 and b.std()>1e-9 else None


def valid_attack_groups(events, mask, first, last):
    filtered = []
    from mdac2.data_preparation.training_chart import CLASSES,LANES
    for e in events:
        f = min(len(mask)-1,int(e.start*100+1e-7))
        if all(mask[f,LANES.index(l),CLASSES.index(e.kind)] for l in e.lanes):
            filtered.append(e)
    return groups(filtered,first,last)


def compare(prediction, baseline, truth, duration, decoder, mask):
    # The shift joins at 8.13 s. Exclude that seam's entire 5.10 s receptive
    # radius, plus the ending padded context, identically for every variant.
    first,last = 13.5,duration-5.11
    events,audit = decode_song(prediction,duration,decoder)
    pg = valid_attack_groups(events,mask,first,last)
    rg = valid_attack_groups(truth,mask,first,last)
    pt,rt = [g[0].start for g in pg],[g[0].start for g in rg]
    matched,errors = counts(pt,rt,.05)
    pairs = pair_points([(t,) for t in pt],[(t,) for t in rt],.05)
    exact_lane = sum({l for e in pg[p] for l in e.lanes} == {l for e in rg[r] for l in e.lanes} for p,r in pairs)
    frames = slice(int(np.ceil(first*100)),int(last*100))
    original = baseline['onsets'][frames].max((1,2))
    altered = prediction['onsets'][frames].max((1,2))
    return dict(attack_counts=matched.tolist(),attack_50ms=result(matched),
        lane_set_correct_at_matched_attacks=exact_lane,matched_attacks=len(pairs),
        operation=operations(events,first,last),
        onset_score_correlation_with_original=correlation(original,altered),
        onset_score_mean_absolute_change=float(np.abs(original-altered).mean()),
        ground_air_light_correlation=correlation(prediction['onsets'][frames,0,0],prediction['onsets'][frames,1,0]),
        ground_air_light_mean_absolute_difference=float(np.abs(prediction['onsets'][frames,0,0]-prediction['onsets'][frames,1,0]).mean()),
        region=[first,last],rule_rejections=audit['rule_rejection_count'])


def contract_audit(config, output):
    manifest = read(config['manifest']);root=Path(config['manifest']).parent
    records = {x['chart_sha256']:x for x in read(Path(config['tensors'])/'report.json')['charts']}
    stats = {role:dict(charts=0,events=Counter(),attack_positions=0,both_lane_positions=0,
        light_frames=0,light_both_frames=0,head_parameters_independent=True) for role in ('train','validation')}
    rows = [r for r in manifest['charts'] if r['training_eligible'] and r['split'] in stats]
    checked=[]
    for i,row in enumerate(rows):
        label=root/row['labels'];assert sha(label)==row['labels_sha256']
        chart=read(label);record=records[row['chart_sha256']];tensor=Path(config['tensors'])/record['file']
        assert sha(tensor)==record['sha256']
        fresh=frame_targets(chart)
        with np.load(tensor,allow_pickle=False) as old:
            for k in old.files:
                wanted=timing_features(chart,row['rating'],len(old['onsets'])) if k=='timing' else fresh[k]
                if not np.array_equal(old[k],wanted):raise ValueError(f'Tensor differs: {row["chart_sha256"]} {k}')
        stat=stats[row['split']];stat['charts']+=1;stat['events'].update(e['kind'] for e in chart['events'])
        op=operations([TrainingEvent.from_dict(e) for e in chart['events']]);stat['attack_positions']+=op['positions'];stat['both_lane_positions']+=op['both']
        light=fresh['onsets'][:,:,0]>.5
        stat['light_frames']+=int(light.any(1).sum());stat['light_both_frames']+=int(light.all(1).sum())
        checked.append(dict(chart_sha256=row['chart_sha256'],role=row['split'],tensor_sha256=record['sha256']))
        if (i+1)%40==0:print(json.dumps(dict(event='tensor_audit',charts=i+1,total=len(rows))),flush=True)
    for stat in stats.values():
        stat['both_lane_fraction']=stat['both_lane_positions']/max(1,stat['attack_positions'])
        stat['light_both_fraction']=stat['light_both_frames']/max(1,stat['light_frames'])
    report=dict(roles=stats,checked=checked,reconstructed_fields_equal=True,
        scope='cached labels -> tensors and lane distribution; source/audio musical correctness requires independent review',locked_test_opened=False)
    atomic_json(output/'data-contract.json',report)
    return report


def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    config=read(BASE/'baseline-lr3e4.json');frozen=read(BASE/'evaluation-lr3e4/frozen.json')
    checkpoint=Path(frozen['checkpoint']);assert sha(checkpoint)==frozen['checkpoint_sha256']
    initial=data_identity(config)
    atomic_json(args.output/'protocol.json',dict(checkpoint_sha256=sha(checkpoint),manifest_sha256=initial['manifest_sha256'],
        source_sha256=sha(Path(__file__)),roles=['train','validation'],validation_charts=22,train_probe_selection='8 evenly spaced rows sorted by chart hash',
        variants=['original',*VARIANTS],audio_shift_frames=813,phase_rotation_radians=float(np.pi/2),
        decoder=frozen['decoder'],thresholds_changed=False,optimizer_created=False,locked_test_opened=False,
        interpretation='counterfactual sensitivity; silence/phase removal are distribution shifts, not proof of causation alone',
        matching='attack positions grouped within 20ms; onset tolerance 50ms; exclude hold/gear; common interior crop 13.5 to duration-5.11 seconds'))
    contract=contract_audit(config,args.output)
    seed_all(config['seed']);model=JointChartModel().cuda().eval()
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False);model.load_state_dict(saved['model'])
    w=saved['model']['onsets.weight'].numpy()
    atomic_json(args.output/'head-weights.json',dict(light_ground_air_weight_cosine=float(np.dot(w[0],w[6])/(np.linalg.norm(w[0])*np.linalg.norm(w[6]))),
        light_ground_air_weight_l2_distance=float(np.linalg.norm(w[0]-w[6])),light_bias=saved['model']['onsets.bias'][[0,6]].tolist()))
    records=[];start=time.monotonic()
    for role in ('validation','train'):
        data=WindowDataset(config['manifest'],config['cache'],role,tensor_cache=config['tensors'])
        selected=range(len(data.rows)) if role=='validation' else [sorted(range(len(data.rows)),key=lambda i:data.rows[i]['chart_sha256'])[int(i)] for i in np.linspace(0,len(data.rows)-1,8)]
        for seq,ri in enumerate(selected):
            row=data.rows[ri];idx=next(i for i,(r,_) in enumerate(data.index) if r==ri)
            data[idx]  # verify cache identities with the training loader
            features,timing,target=data.loaded[ri]
            chart=read(data.root/row['labels']);truth=[TrainingEvent.from_dict(e) for e in chart['events']]
            original=infer(model,features,timing)
            replay=0.
            if role=='validation':
                path=BASE/'evaluation-lr3e4/predictions-validation'/(row['chart_sha256']+'.npz')
                metadata=read(path.with_suffix('.json'));assert sha(path)==metadata['sha256'] and metadata['identity']['checkpoint_sha256']==sha(checkpoint)
                with np.load(path) as old:replay=max(float(np.abs(original[k]-old[k]).max()) for k in original)
                if replay>1e-6:raise ValueError(f'Baseline replay changed: {replay}')
            baseline=compare(original,original,truth,chart['duration'],frozen['decoder'],target['onset_mask'])
            # Lane discrimination with timing supplied by the reference: a
            # diagnostic separates arranging from missing the onset entirely.
            light=target['onsets'][:,:,0]>.5
            single=light.sum(1)==1
            mask=target['onset_mask'][:,:,0].all(1)&single
            pred_lane=original['onsets'][:,:,0].argmax(1);true_lane=light.argmax(1)
            baseline['lane_at_reference_single_light']=dict(correct=int((pred_lane[mask]==true_lane[mask]).sum()),total=int(mask.sum()),
                true_ground=int((true_lane[mask]==0).sum()),predicted_ground=int((pred_lane[mask]==0).sum()),
                probability_margin_mean=float(np.abs(np.diff(original['onsets'][mask,:,0],axis=1)).mean()))
            report=dict(chart_sha256=row['chart_sha256'],title=row['title'],role=role,baseline_replay_max_absolute_error=replay,
                reference_operations=operations(truth),original=baseline,variants={})
            if role=='validation':
                for variant in VARIANTS:
                    audio=features;beats=timing
                    if variant=='audio_shift_8.13s':audio=np.roll(features,813,axis=2)
                    elif variant=='audio_silent':audio=np.zeros_like(features)
                    else:
                        beats=timing.copy()
                        if variant=='beat_phase_plus_quarter':beats[:,:2]=np.stack((timing[:,1],-timing[:,0]),axis=1)
                        else:beats[:,:2]=0
                    prediction=infer(model,audio,beats)
                    report['variants'][variant]=compare(prediction,original,truth,chart['duration'],frozen['decoder'],target['onset_mask'])
            records.append(report);atomic_json(args.output/(row['chart_sha256']+'.json'),report)
            print(json.dumps(dict(event='diagnosed',role=role,song=seq+1,seconds=time.monotonic()-start)),flush=True)
    summary={}
    for role in ('train','validation'):
        subset=[r for r in records if r['role']==role]
        values={}
        for variant in (('original',*VARIANTS) if role=='validation' else ('original',)):
            rows=[r['original'] if variant=='original' else r['variants'][variant] for r in subset]
            c=np.sum([r['attack_counts'] for r in rows],axis=0)
            both=sum(r['operation']['both'] for r in rows);positions=sum(r['operation']['positions'] for r in rows)
            values[variant]=dict(attack_50ms=result(c),both_lane_fraction=both/max(1,positions),attack_positions=positions,
                lane_set_accuracy_at_matched_attacks=sum(r['lane_set_correct_at_matched_attacks'] for r in rows)/max(1,sum(r['matched_attacks'] for r in rows)),
                mean_song_score_correlation=float(np.mean([r['onset_score_correlation_with_original'] for r in rows if r['onset_score_correlation_with_original'] is not None])),
                mean_song_score_absolute_change=float(np.mean([r['onset_score_mean_absolute_change'] for r in rows])),
                mean_song_ground_air_light_correlation=float(np.mean([r['ground_air_light_correlation'] for r in rows if r['ground_air_light_correlation'] is not None])))
        lane=[r['original']['lane_at_reference_single_light'] for r in subset]
        lane_sum={k:sum(r[k] for r in lane) for k in ('correct','total','true_ground','predicted_ground')}
        lane_sum['accuracy']=lane_sum['correct']/lane_sum['total']
        lane_sum['majority_lane_baseline']=max(lane_sum['true_ground'],lane_sum['total']-lane_sum['true_ground'])/lane_sum['total']
        summary[role]=dict(charts=len(subset),variants=values,lane_at_reference_single_light=lane_sum)
    assert data_identity(config)==initial
    atomic_json(args.output/'summary.json',dict(summary=summary,source_sha256=sha(Path(__file__)),input_identity_unchanged=True,
        locked_test_opened=False,training_started=False,checkpoints_modified=False,thresholds_changed=False,
        new_package_generated=False,seconds=time.monotonic()-start))
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
