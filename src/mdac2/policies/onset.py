"""Strongest supported candidates first; rests need no artificial notes."""
from bisect import bisect_left,insort
from fractions import Fraction
from mdac2.analysis.onset_candidates import OnsetSet
from mdac2.domain.onset_policy import OnsetPolicy,SelectionDecision,OnsetSelection


def select_onsets(onsets,*,policy=OnsetPolicy()):
    if not isinstance(onsets,OnsetSet) or not isinstance(policy,OnsetPolicy):raise ValueError('Typed candidates/policy required')
    if len(onsets.candidates)>20000:raise ValueError('Selection candidate limit exceeded')
    selected=[];decisions={};gap=Fraction(str(policy.minimum_gap_seconds))
    for c in sorted(onsets.candidates,key=lambda c:(-c.score,c.seconds,c.candidate_id)):
        reason='ACOUSTIC_SUPPORT_WITHIN_BUDGET'
        if c.quantized_q is None:reason='NEEDS_MANUAL_QUANTIZATION'
        elif c.score<policy.minimum_score:reason='BELOW_SCORE_THRESHOLD'
        elif c.quantized_q.denominator not in policy.allowed_denominators:reason='GRID_COMPLEXITY_LIMIT'
        else:
            # Density uses final quantized seconds, not the pre-quantization peak.
            time=c.seconds+c.quantization_error_seconds
            index=bisect_left(selected,time)
            if index and time-selected[index-1]<gap or index<len(selected) and selected[index]-time<gap:reason='MINIMUM_GAP'
            else:
                local=selected[bisect_left(selected,time-1):bisect_left(selected,time+1)]
                insort(local,time);left=0
                for right,value in enumerate(local):
                    while value-local[left]>=1:left+=1
                    if right-left+1>policy.max_notes_per_second:reason='LOCAL_DENSITY_LIMIT';break
            if reason=='ACOUSTIC_SUPPORT_WITHIN_BUDGET':insort(selected,time)
        decisions[c.candidate_id]=SelectionDecision(c.candidate_id,reason=='ACOUSTIC_SUPPORT_WITHIN_BUDGET',(reason,))
    return OnsetSelection(onsets.fingerprint,policy.fingerprint,tuple(decisions[c.candidate_id] for c in onsets.candidates))
