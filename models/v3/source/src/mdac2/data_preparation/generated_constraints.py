"""Versioned generated-chart occupancy; frozen training labels stay immutable."""
from collections import defaultdict
import math
from .training_chart import CLASSES, LANES, validate_events

POLICY_ID = 'generated-single-position-v2'
# Engineering tolerance for almost-coincident model outputs, not a new beat grid.
NEARBY_SECONDS = .02


def validate_generated_events(events, duration, *, nearby_seconds=NEARBY_SECONDS):
    if not math.isfinite(nearby_seconds) or nearby_seconds<0:
        raise ValueError('Nonnegative finite occupancy tolerance required')
    issues=validate_events(events,duration)
    # Only Hold + one instantaneous object can share a lane. The base rules
    # still reject gear + Hold, exclusive special attacks, and overlapping Holds.
    instants=sorted((e for e in events if e.kind!='hold'),key=lambda e:(e.start,e.event_id))
    for i,left in enumerate(instants):
        for right in instants[i+1:]:
            if right.start-left.start>nearby_seconds+1e-9:
                break
            lanes=sorted(set(left.lanes)&set(right.lanes))
            if lanes:
                issues.append(dict(code='same_position_multiple_objects',event_ids=[left.event_id,right.event_id],
                    lanes=lanes,start=left.start,end=right.start,nearby_seconds=nearby_seconds))
    return issues


def select_generated_candidates(candidates, duration, *, nearby_seconds=NEARBY_SECONDS):
    """Greedy scored choice of complete objects; never moves or changes a type.

    Gemini is already one two-lane object. Explicit double-heavy groups remain
    atomic too, so losing one lane cannot silently turn them into single hits.
    This is not a globally optimal musical-arrangement claim.
    """
    ids=[e.event_id for e,_ in candidates]
    if len(ids)!=len(set(ids)):
        raise ValueError('Duplicate candidate identity')
    if any(not math.isfinite(score) or not 0<=score<=1 for _,score in candidates):
        raise ValueError('Finite probability scores required')
    groups=defaultdict(list)
    for event,score in candidates:
        key=('heavy_group',event.group_id) if event.kind=='heavy' and event.group_id else ('event',event.event_id)
        groups[key].append((event,score))
    for key,items in groups.items():
        if key[0]=='heavy_group':
            if len(items)!=2 or {e.lanes for e,_ in items}!={('ground',),('air',)} or len({e.start for e,_ in items})!=1:
                raise ValueError('Double-heavy group must have exactly two simultaneous opposite-lane members')
    ranked=sorted(groups.values(),key=lambda items:(-min(s for _,s in items),tuple(sorted(e.event_id for e,_ in items))))
    accepted=[]
    rejected=[]
    for items in ranked:
        group=[e for e,_ in items]
        issues=validate_generated_events(group,duration,nearby_seconds=nearby_seconds)
        for event in group:
            for other in accepted:
                issues.extend(validate_generated_events([other,event],duration,nearby_seconds=nearby_seconds))
        if issues:
            rejected.append(dict(event_ids=[e.event_id for e in group],score=min(s for _,s in items),issues=issues))
        else:
            accepted.extend(group)
    accepted.sort(key=lambda e:(e.start,e.event_id))
    if validate_generated_events(accepted,duration,nearby_seconds=nearby_seconds):
        raise AssertionError('Generated candidate selection left an occupancy conflict')
    return accepted,dict(policy_id=POLICY_ID,nearby_seconds=nearby_seconds,
        candidate_count=len(candidates),accepted_count=len(accepted),
        rejected_count=sum(len(r['event_ids']) for r in rejected),rejected=rejected,
        score_policy='weakest member of each atomic group; descending score then stable identity',
        candidate_scores=[dict(event_id=e.event_id,score=float(s)) for e,s in candidates])


def constrain_decoded_events(events, prediction, duration, decoder):
    """Recover frozen peak scores, then arbitrate the paired v1 proposals.

    The frozen decoder and its evaluated behavior are not modified. This is a
    new inference policy over its retained proposals, not the old test result.
    """
    from .joint_decoding import peaks,event_time
    heads={}
    tails={}
    for side,lane in enumerate(LANES):
        for kind_index,kind in enumerate(CLASSES):
            values=prediction['onsets'][:,side,kind_index]
            heads[lane,kind]=[(event_time(i,prediction['offsets'][i,side,kind_index],duration),float(values[i]))
                             for i in peaks(values,decoder['thresholds'][kind_index],decoder['nms_frames'])]
        values=prediction['hold_ends'][:,side]
        tails[lane]=[(event_time(i,prediction['hold_end_offsets'][i,side],duration),float(values[i]))
                     for i in peaks(values,decoder['tail_threshold'],decoder['nms_frames'])]
    # Paired endpoints are averaged; maximum movement is half pairing tolerance
    # plus the explicit millisecond rounding error.
    tolerance=decoder['pair_tolerance_seconds']/2+.001+1e-9
    def matched_score(points,time):
        matches=[(abs(t-time),-score,score) for t,score in points if abs(t-time)<=tolerance]
        if not matches:
            raise ValueError('Decoded event has no corresponding frozen prediction peak')
        return min(matches)[2]
    scored=[]
    for event in events:
        scores=[matched_score(heads[lane,event.kind],event.start) for lane in event.lanes]
        if event.kind=='hold':
            scores.extend(matched_score(tails[lane],event.end) for lane in event.lanes)
        scored.append((event,min(scores)))
    return select_generated_candidates(scored,duration)
