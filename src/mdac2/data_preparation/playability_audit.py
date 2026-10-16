"""Playtest checks beyond the immutable training-label compatibility rules.

Do not rewrite source labels or decoded events. Holds plus attacks remain a
supported form; distinct instantaneous attack types on one lane are reported.
Long rests are review findings, never automatically filled with notes.
"""
from collections import defaultdict
from .generated_constraints import POLICY_ID,validate_generated_events

ATTACKS = frozenset(('light', 'heavy', 'extra_heavy', 'gemini'))


def audit_playability(events, duration, *, nearby_seconds=.02, gap_seconds=2.):
    occupancy_issues=validate_generated_events(events,duration,nearby_seconds=nearby_seconds)
    cells = defaultdict(list)
    for event in events:
        if event.kind in ATTACKS:
            for lane in event.lanes:
                cells[event.start,lane].append(event)
    stacked = [dict(seconds=t,lane=lane,event_ids=[e.event_id for e in items],
                    kinds=[e.kind for e in items],count=len(items))
               for (t,lane),items in sorted(cells.items()) if len(items)>1]
    near_stacks=[]
    for lane in ('ground','air'):
        attacks=sorted((e for e in events if e.kind in ATTACKS and lane in e.lanes),key=lambda e:(e.start,e.event_id))
        for i,left in enumerate(attacks):
            for right in attacks[i+1:]:
                distance=right.start-left.start
                if distance>nearby_seconds+1e-9:
                    break
                if distance>0:
                    near_stacks.append(dict(lane=lane,start=left.start,end=right.start,
                        event_ids=[left.event_id,right.event_id],kinds=[left.kind,right.kind]))
    near_gemini = []
    for event in events:
        if event.kind != 'gemini':
            continue
        competing = [e for e in events if e.event_id!=event.event_id and e.kind in ATTACKS
                     and abs(e.start-event.start)<=nearby_seconds+1e-9]
        if competing:
            near_gemini.append(dict(event_id=event.event_id,seconds=event.start,
                nearby=[dict(event_id=e.event_id,kind=e.kind,seconds=e.start,lanes=list(e.lanes)) for e in competing]))
    # Union attack instants and held intervals. A long valid hold is not a rest.
    spans = sorted((e.start,e.end if e.kind=='hold' else e.start) for e in events if e.kind!='gear')
    coverage = []
    for start,end in spans:
        if coverage and start<=coverage[-1][1]:
            coverage[-1][1]=max(coverage[-1][1],end)
        else:
            coverage.append([start,end])
    gaps = [dict(start=left[1],end=right[0],seconds=round(right[0]-left[1],6))
            for left,right in zip(coverage,coverage[1:]) if right[0]-left[1]>=gap_seconds]
    tiny = [dict(event_id=e.event_id,start=e.start,end=e.end,duration=e.end-e.start,lanes=list(e.lanes))
            for e in events if e.kind=='hold' and e.end-e.start<.01-1e-9]
    return dict(schema_version=2,policy_id=POLICY_ID,logical_events=len(events),duration=duration,
        occupancy_issues=occupancy_issues,
        exact_attack_stacks=stacked,exact_attack_stack_lane_times=len(stacked),
        nearby_attack_stacks=near_stacks,
        excess_same_lane_attacks=sum(row['count']-1 for row in stacked),
        gemini_count=sum(e.kind=='gemini' for e in events),gemini_with_nearby_attack=near_gemini,
        nearby_seconds=nearby_seconds,minimum_reported_gap_seconds=gap_seconds,
        no_attack_or_hold_intervals=gaps,total_reported_gap_seconds=sum(g['seconds'] for g in gaps),
        sub_frame_holds=tiny,
        blocking_findings=sorted({issue['code'] for issue in occupancy_issues}),
        engineering_playtest_gate_passed=not occupancy_issues,
        caveats=['Passing this audit does not establish playable quality.',
                 'Rest intervals require musical review; do not fill them merely to increase density.',
                 'Hold plus attack is retained; training-label rules are not changed.'])
