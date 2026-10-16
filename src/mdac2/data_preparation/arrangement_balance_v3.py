"""Reference-relative lane moments. Dual notes cannot replace single clicks."""
from collections import defaultdict
import numpy as np
import torch
from .arrangement_data_v3 import PATTERNS
from .joint_evaluation import result


def pattern_moments():
    values=[]
    for g,a,h in PATTERNS:
        values.append([int(g>0)+int(bool(h&1)),int(a>0)+int(bool(h&2)),
                       int(g>0 and a==0),int(a>0 and g==0),int(g>0 and a>0)])
    return np.asarray(values,np.float32)


def balance_loss(logits,tokens,valid):
    # Match the actual central-window reference, never impose 50:50 locally.
    table=torch.as_tensor(pattern_moments(),device=logits.device)
    predicted=(logits/.5).softmax(-1)@table
    target=table[tokens]
    count=valid.sum(1);mask=valid[:,:,None]
    diff=((predicted-target)*mask).sum(1)/count.clamp_min(1)[:,None]
    nonempty=count>0
    return (diff.square().mean(1)*nonempty).sum()/nonempty.sum().clamp_min(1)


def lane_stats(events):
    ground=air=0;attacks=defaultdict(set)
    for e in events:
        if e.kind=='gear':continue
        ground+=int('ground' in e.lanes);air+=int('air' in e.lanes)
        if e.kind!='hold':attacks[e.start].update(e.lanes)
    sg=sum(v=={'ground'} for v in attacks.values());sa=sum(v=={'air'} for v in attacks.values())
    dual=sum(len(v)==2 for v in attacks.values());longest=run=0;last=None
    windows=defaultdict(lambda:[0,0])
    for t,lanes in sorted(attacks.items()):
        side=next(iter(lanes)) if len(lanes)==1 else None
        run=run+1 if side is not None and side==last else int(side is not None)
        longest=max(longest,run);last=side
        windows[int(t//10)][0]+=int('ground' in lanes);windows[int(t//10)][1]+=int('air' in lanes)
    return dict(ground=ground,air=air,ground_share=ground/max(1,ground+air),single_ground=sg,single_air=sa,
                single_ground_share=sg/max(1,sg+sa),dual_fraction=dual/max(1,len(attacks)),
                longest_single_lane_run=longest,windows_10s={str(k):v for k,v in sorted(windows.items())})


def balance_error(prediction,reference):
    p,r=prediction,reference
    single=abs(p['single_ground_share']-r['single_ground_share']) if r['single_ground']+r['single_air'] else 0.
    return (abs(p['ground_share']-r['ground_share'])+single+abs(p['dual_fraction']-r['dual_fraction']))/3


def five_class_report(report):
    learned={k:v for k,v in report['six_classes'].items() if k!='gear'}
    report['five_classes']=learned
    report['macro_five_class_f1']=float(np.mean([v['f1'] for v in learned.values()]))
    report['micro_five_class']=result(np.sum([[v[k] for k in ('tp','fp','fn')] for v in learned.values()],axis=0))
    report['gear_policy']='excluded from learned targets and main score; postprocessing disabled'
    return report
