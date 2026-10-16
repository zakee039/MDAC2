"""Bounded whole-song orientation prior; no time/type/dual/hold changes.

Applied after learned class-prior correction. Only a single light/heavy click
may use its same-kind opposite-lane alternative. The prior is soft: confidence
cost competes with a broad 45--55 percent band, and the maximum logit boost is
bounded. Local alternation or quotas are never imposed.
"""
import numpy as np
from .arrangement_data_v3 import PATTERNS
from .arrangement_balance_v3 import pattern_moments


def apply_lane_prior(logits,strength=0.):
    scores=np.asarray(logits,np.float64)
    if not np.isfinite(scores).all() or strength<0:raise ValueError('Finite scores and nonnegative strength required')
    raw=scores.argmax(1);table=pattern_moments();eligible=[]
    for i,token in enumerate(raw):
        g,a,h=PATTERNS[token]
        if h==0 and ((g in (1,2) and a==0) or (a in (1,2) and g==0)):
            eligible.append((i,PATTERNS.index((a,g,0)),1 if a else -1))
    def summary(tokens):
        m=table[tokens].sum(0)
        return dict(ground_share=float(m[0]/max(1,m[0]+m[1])),single_ground_share=float(m[2]/max(1,m[2]+m[3])),dual_positions=int(m[4]))
    def penalty(stats):
        return sum(max(0.,.45-stats[k],stats[k]-.55)**2 for k in ('ground_share','single_ground_share'))
    before=summary(raw);best=(4*penalty(before),0.,scores.copy(),raw.copy(),0.)
    for bias in np.linspace(-strength,strength,17) if strength else []:
        adjusted=scores.copy()
        for i,mirror,direction in eligible:
            if bias*direction>0:adjusted[i,mirror]+=abs(bias)
        chosen=adjusted.argmax(1)
        # Mean confidence cost across every input onset, without adding events.
        cost=float(np.mean(scores[np.arange(len(raw)),raw]-scores[np.arange(len(raw)),chosen])) if len(raw) else 0.
        objective=cost+4*penalty(summary(chosen))
        if (objective,abs(bias))<(best[0],abs(best[1])):best=(objective,float(bias),adjusted,chosen,cost)
    _,bias,adjusted,chosen,cost=best
    for a,b in zip(raw,chosen):
        if a!=b:
            g,h,hold=PATTERNS[a];assert hold==0 and PATTERNS[b]==(h,g,0) and (g or h) in (1,2)
    return adjusted,dict(enabled=bool(strength),strength=strength,chosen_bias=bias,
        changed_orientation_count=int(np.sum(raw!=chosen)),eligible_count=len(eligible),confidence_cost=cost,
        before=before,after=summary(chosen),type_or_dual_or_hold_changed=False,onsets_changed=False,
        policy='bounded same-kind single light/heavy lane alternative; confidence cost plus soft whole-song 45-55 band; no local quota')


def decode_with_lane_prior(logits,log_lengths,times,duration,strength=0.):
    """Reject the prior if legal fallback would alter a type, hold or dual count."""
    from .arrangement_context_v3 import decode
    base_events,base_audit=decode(logits,log_lengths,times,duration)
    adjusted,prior=apply_lane_prior(logits,strength)
    if not prior['changed_orientation_count']:
        base_audit['lane_prior']=prior
        return base_events,base_audit
    events,audit=decode(adjusted,log_lengths,times,duration)
    def signature(values):
        return sorted((e.start,e.kind,e.end,','.join(e.lanes) if e.kind not in ('light','heavy') else str(len(e.lanes))) for e in values)
    def dual(values):
        positions={}
        for e in values:
            if e.kind!='hold':positions.setdefault(e.start,set()).update(e.lanes)
        return sorted(t for t,v in positions.items() if len(v)==2)
    if signature(events)!=signature(base_events) or dual(events)!=dual(base_events):
        prior['rejected_due_to_legal_fallback_type_or_dual_change']=True
        prior['applied_orientation_count']=0
        base_audit['lane_prior']=prior
        return base_events,base_audit
    prior['applied_orientation_count']=prior['changed_orientation_count']
    audit['lane_prior']=prior
    return events,audit
