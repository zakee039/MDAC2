"""One learned lane/type pattern per onset, with evidence-gated adaptive recall."""
from fractions import Fraction
import numpy as np
from .structured_patterns import PATTERNS,STATE_NAMES
from .training_chart import CLASSES,LANES,TrainingEvent
from .joint_decoding import decode_song,peaks,event_time
from .generated_constraints import constrain_decoded_events,select_generated_candidates

DEFAULT_CONFIG=dict(base_threshold=.35,floor_threshold=.15,threshold_step=.05,
                    nms_frames=4,quarters_per_block=4,max_fills_per_block=2,
                    activity_relative_floor=.001,flux_relative_floor=.25,
                    hold_threshold=.5,tail_threshold=.5,occupancy_threshold=.2)


def quarters_at_frames(timing,count):
    rows=timing['tempos']
    positions=np.array([v['seconds'] for v in rows])
    indexes=np.clip(np.searchsorted(positions,np.arange(count)*.01,side='right')-1,0,len(rows)-1)
    bpm=np.array([float(Fraction(v['bpm'])) for v in rows])[indexes]
    q0=np.array([float(Fraction(v['quarter'])) for v in rows])[indexes]
    return q0+(np.arange(count)*.01-positions[indexes])*bpm/60


def decode_structured_song(prediction,duration,timing,features,config=None):
    cfg={**DEFAULT_CONFIG,**(config or {})}
    if not 0<cfg['floor_threshold']<=cfg['base_threshold']<1 or cfg['threshold_step']<=0:
        raise ValueError('Bounded explicit adaptive thresholds required')
    probability=np.asarray(prediction['event_probability'])
    patterns=np.asarray(prediction['pattern_probability'])
    count=len(probability)
    if patterns.shape!=(count,len(PATTERNS)) or features.shape!=(3,96,count):
        raise ValueError('Structured prediction/feature geometry mismatch')
    if not all(np.isfinite(v).all() for v in prediction.values()):
        raise ValueError('Nonfinite structured prediction')
    decisions=patterns.argmax(-1)
    beat_quarters=quarters_at_frames(timing,count)
    blocks=np.floor(beat_quarters/cfg['quarters_per_block']).astype(np.int64)
    music=np.expm1(np.clip(features[0].astype(np.float32),0,25)).mean(0)
    flux=np.maximum(np.diff(features[0].astype(np.float32),axis=1,prepend=features[0,:,:1]),0).mean(0)
    active=music>=max(1e-5,float(np.percentile(music,90))*cfg['activity_relative_floor'])
    # Long notes remain separately represented; no new tail at a block boundary.
    hold_prediction={k:prediction[k] for k in ('onsets','offsets','hold_ends','hold_end_offsets','occupancy')}
    hold_prediction['onsets']=hold_prediction['onsets'].copy()
    hold_prediction['onsets'][:,:,[0,1,2,3,5]]=0
    hold_cfg=dict(thresholds=[.99,.99,.99,.99,cfg['hold_threshold'],.99],
        tail_threshold=cfg['tail_threshold'],occupancy_threshold=cfg['occupancy_threshold'],
        nms_frames=3,pair_tolerance_seconds=.02,precision_seconds=.001)
    holds,hold_audit=decode_song(hold_prediction,duration,hold_cfg)
    holds,hold_selection=constrain_decoded_events(holds,hold_prediction,duration,hold_cfg)
    held=np.zeros(count,dtype=bool)
    for e in holds:
        held[int(e.start*100):min(count,int(np.ceil(e.end*100)))]=True
    pool=peaks(probability,cfg['floor_threshold'],cfg['nms_frames'])
    selected={f:dict(source='base',threshold=cfg['base_threshold']) for f in pool if probability[f]>=cfg['base_threshold']}
    block_audit=[]
    def has_click(f):
        return any(state in (1,2,3,5) for state in PATTERNS[int(decisions[f])])
    for block in np.unique(blocks):
        frames=np.flatnonzero(blocks==block)
        first,last=int(frames[0]),int(frames[-1])+1
        if held[first:last].any() or any(first<=f<last and has_click(f) for f in selected):
            continue
        positive_flux=flux[first:last][flux[first:last]>0]
        required_flux=max(1e-6,float(np.median(positive_flux))*cfg['flux_relative_floor']) if len(positive_flux) else 1e-6
        eligible=[f for f in pool if first<=f<last and active[f] and flux[f]>=required_flux and has_click(f)]
        level=cfg['base_threshold']
        added=[]
        while not added and level>cfg['floor_threshold']+1e-9:
            level=max(cfg['floor_threshold'],round(level-cfg['threshold_step'],6))
            proposed=sorted((f for f in eligible if probability[f]>=level),key=lambda f:(-float(probability[f]),f))
            added=proposed[:cfg['max_fills_per_block']]
        for f in added:
            selected[f]=dict(source='adaptive',threshold=level)
        block_audit.append(dict(block=int(block),start=first*.01,end=min(duration,last*.01),
            threshold=level,added_frames=added,music_active=bool(active[first:last].any()),
            reason='evidence_supported_candidates' if added else 'no_supported_candidate_above_floor'))
    candidates=[]
    choices=[]
    for f,selection in sorted(selected.items()):
        decision=int(decisions[f]);pattern=PATTERNS[decision]
        members=[(side,STATE_NAMES[state]) for side,state in enumerate(pattern) if state]
        offset=np.mean([prediction['offsets'][f,side,CLASSES.index(kind)] for side,kind in members])
        seconds=event_time(f,offset,duration)
        score=float(probability[f]*patterns[f,decision])
        source=(f'structured-frame-{f}',)
        if pattern==(5,5):
            e=TrainingEvent(f'layout-{f}-gemini','gemini',LANES,seconds,seconds,source,'model','joint-tcn-structured-v2')
            candidates.append((e,score))
        else:
            for side,kind in members:
                group=f'layout-double-heavy-{f}' if pattern==(2,2) else None
                e=TrainingEvent(f'layout-{f}-{side}',kind,(LANES[side],),seconds,seconds,source,'model',
                    'joint-tcn-structured-v2',kind=='extra_heavy',group)
                candidates.append((e,score))
        choices.append(dict(frame=f,seconds=seconds,pattern=list(pattern),event_probability=float(probability[f]),
            pattern_probability=float(patterns[f,decision]),**selection))
    hold_scores={r['event_id']:r['score'] for r in hold_selection['candidate_scores']}
    candidates.extend((h,hold_scores[h.event_id]) for h in holds)
    events,selection=select_generated_candidates(candidates,duration)
    audit=dict(policy_id='structured-adaptive-v3',config=cfg,candidate_count=len(candidates),accepted_count=len(events),
        rule_rejection_count=selection['rejected_count'],
        rejected=[dict(code='rule_conflict',**r) for r in selection['rejected']],
        cross_window_holds=sum(e.start//16!=e.end//16 for e in events if e.kind=='hold'),
        adaptive_candidate_count=sum(x['source']=='adaptive' for x in choices),
        four_quarter_blocks=block_audit,choices=choices,hold_decoder=hold_audit,
        block_semantics='four-quarter timing blocks; not a claim that automatic downbeats/meter were human-confirmed')
    return events,audit


def operation_statistics(events):
    groups=[]
    for e in sorted((e for e in events if e.kind not in ('hold','gear')),key=lambda e:e.start):
        if not groups or e.start-groups[-1][0].start>.02:
            groups.append([e])
        else:
            groups[-1].append(e)
    both=sum(len({lane for e in g for lane in e.lanes})==2 for g in groups)
    return dict(attack_positions=len(groups),both_lane_positions=both,both_lane_fraction=both/max(1,len(groups)),
        gemini_positions=sum(any(e.kind=='gemini' for e in g) for g in groups),
        ground_only_positions=sum({l for e in g for l in e.lanes}=={'ground'} for g in groups),
        air_only_positions=sum({l for e in g for l in e.lanes}=={'air'} for g in groups))
