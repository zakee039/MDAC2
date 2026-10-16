"""Constraint selection for generated candidates, never for source annotation."""
import math
from .training_chart import validate_events


def select_candidates(candidates,duration):
    if any(not math.isfinite(score) or not 0<=score<=1 for _,score in candidates):
        raise ValueError('Finite probability scores required')
    ids=[event.event_id for event,_ in candidates]
    if len(ids)!=len(set(ids)):raise ValueError('Duplicate candidate id')
    accepted=[];rejected=[]
    for event,score in sorted(candidates,key=lambda pair:(-pair[1],pair[0].event_id)):
        conflicts=validate_events(accepted+[event],duration)
        if conflicts:
            rejected.append({'event_id':event.event_id,'score':score,'issues':conflicts})
        else:accepted.append(event)
    accepted.sort(key=lambda e:(e.start,e.event_id))
    return accepted,rejected
