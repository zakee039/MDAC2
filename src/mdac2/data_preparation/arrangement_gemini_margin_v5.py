"""A validated margin candidate only for already decoded, uncertain Gemini.
No event insertion, rhythmic snapping, hard quota, hold/special replacement.
"""
import numpy as np
from .arrangement_context_v4 import decode
from .arrangement_data_v3 import PATTERNS
GEMINI=PATTERNS.index((5,5,0))
SINGLES=tuple(PATTERNS.index(p) for p in ((1,0,0),(0,1,0),(2,0,0),(0,2,0)))

def decode_uncertain_gemini(logits,log_lengths,times,duration,margin=0.):
    if margin<0:raise ValueError('Nonnegative margin required')
    original=np.asarray(logits,np.float64)
    baseline,base_audit=decode(original,log_lengths,times,duration)
    chosen={round(d['start'],3):d['token'] for d in base_audit['decisions']}
    adjusted=original.copy();candidates=[]
    for i,t in enumerate(times):
        if chosen.get(round(float(t),3))!=GEMINI:continue
        token=SINGLES[int(np.argmax(original[i,list(SINGLES)]))]
        difference=float(original[i,GEMINI]-original[i,token])
        if 0<difference<margin:
            adjusted[i,token]=original[i].max()+1e-5
            candidates.append(round(float(t),3))
    report=dict(margin=margin,eligible_times=candidates,changes=0,onsets_changed=False,holds_or_specials_changed=False,
                policy='only decoded Gemini with a close single light/heavy alternative; validation margin, no quantity target')
    if not candidates:
        base_audit['gemini_margin']=report;return baseline,base_audit
    events,audit=decode(adjusted,log_lengths,times,duration)
    allowed=set(candidates)
    def signature(es,excluded):
        return sorted((e.start,e.end,e.kind,tuple(e.lanes)) for e in es if e.start not in excluded)
    # Any unexpected legal fallback change elsewhere rejects the whole candidate.
    safe=signature(events,allowed)==signature(baseline,allowed)
    for t in allowed:
        before=[e for e in baseline if e.start==t];after=[e for e in events if e.start==t]
        safe=safe and len(before)==1 and before[0].kind=='gemini' and len(after)==1 and after[0].kind in ('light','heavy') and len(after[0].lanes)==1
    if not safe:
        report['rejected_due_to_unexpected_change']=True;base_audit['gemini_margin']=report;return baseline,base_audit
    report['changes']=len(candidates);audit['gemini_margin']=report
    return events,audit
