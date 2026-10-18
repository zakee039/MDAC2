"""Explicit 240 BPM / 50 cells per quarter, plus bounded hold endpoint repair."""
from collections import defaultdict
from dataclasses import replace
from fractions import Fraction

from .generated_constraints import validate_generated_events
from .playtest_postprocess import (Policy,analyze_audio,add_climax_candidates,arrange_phrases,
                                  apply_repeats,repeat_candidates,structure_stats)

GRID=dict(version='fixed-240-50-r2',bpm=240,positions_per_beat=50,beats_per_measure=4,
          positions_per_measure=200,step_seconds=.005,origin_seconds=0.,song_tempo_estimated=False)


def grid_tick(t):
    value=Fraction(str(t))*200
    if value<0:raise ValueError('Negative time')
    return (2*value.numerator+value.denominator)//(2*value.denominator)


def on_grid(t):
    return abs(t*200-round(t*200))<1e-8


def issues(events,duration):
    return validate_generated_events(events,duration)+[
        dict(code='hold_under_30ms',event_ids=[e.event_id]) for e in events
        if e.kind=='hold' and e.end-e.start<.03-1e-8]


def snap_events(events,duration):
    if issues(events,duration):raise ValueError('Invalid chart before grid snapping')
    times=sorted({t for e in events for t in (e.start,e.end)})
    last=int(Fraction(str(duration))*200)
    mapping={t:min(last,grid_tick(t)) for t in times}
    def materialize(m):
        # Guard dataclass construction before checking minimum duration.
        if any(e.kind=='hold' and m[e.end]<=m[e.start] for e in events):return None
        return [replace(e,start=m[e.start]/200,end=m[e.end]/200) for e in events]
    out=materialize(mapping)
    if out is None:raise ValueError('Grid collapsed hold')
    errors=issues(out,duration);repairs=[]
    # Rounding can make two legal >20ms clicks exactly 20ms apart. Resolve on
    # neighbouring grid cells, retaining identities/groups and source order.
    for _ in range(len(times)*2):
        if not errors:break
        bad={eid for issue in errors for eid in issue['event_ids']}
        candidates={t for e in events if e.event_id in bad for t in (e.start,e.end)}
        best=None
        for t in sorted(candidates):
            ix=times.index(t)
            for delta in (-1,1):
                target=mapping[t]+delta
                if target<0 or target>last or abs(target/200-t)>.00750001:continue
                if ix and target<mapping[times[ix-1]]:continue
                if ix+1<len(times) and target>mapping[times[ix+1]]:continue
                proposal=dict(mapping);proposal[t]=target
                es=materialize(proposal)
                if es is None:continue
                es_issues=issues(es,duration)
                score=(len(es_issues),abs(target/200-t),t,target)
                if len(es_issues)<len(errors) and (best is None or score<best[0]):
                    best=(score,proposal,es,es_issues,t,target)
        if best is None:raise ValueError(f'Cannot safely snap chart to 5ms grid: {errors[:2]}')
        _,mapping,out,errors,t,target=best
        repairs.append(dict(original=t,target=target/200,reason='grid_occupancy_repair'))
    if errors:raise ValueError('Grid repair limit exceeded')
    return out,dict(grid=GRID,repairs=repairs,edits=[dict(original=t,target=mapping[t]/200,
        shift_ms=round((mapping[t]/200-t)*1000,6)) for t in times if abs(t-mapping[t]/200)>1e-9],
        nearest_shift_limit_ms=2.5,conflict_repair_limit_ms=7.5,event_count_preserved=True)


def align_holds(events,duration,tolerance=.1):
    """Fixed original endpoint groups. Note anchors first, then pair midpoints.

    Shared heads/tails remain atomic. Unanchored clusters never span >100ms;
    members of a single hold cannot collapse into the same endpoint cluster.
    """
    if issues(events,duration) or any(not on_grid(t) for e in events for t in (e.start,e.end)):
        raise ValueError('Legal grid chart required')
    nodes=defaultdict(list)
    for i,e in enumerate(events):
        if e.kind=='hold':
            nodes[e.start].append((i,'start'));nodes[e.end].append((i,'end'))
    mapping={t:t for t in nodes}
    attacks=sorted({e.start for e in events if e.kind in ('light','heavy','gemini')})
    protected={t for t in nodes if t in attacks}
    edits=[];rejected=[]
    def materialize(m):
        out=[]
        for e in events:
            if e.kind!='hold':out.append(e);continue
            a,b=m[e.start],m[e.end]
            if b-a<.03-1e-8:return None
            out.append(replace(e,start=a,end=b))
        return out
    anchor_candidates=[]
    for t in nodes:
        if t in protected:continue
        nearby=[a for a in attacks if abs(a-t)<=tolerance+1e-9]
        if nearby:
            a=min(nearby,key=lambda a:(abs(a-t),a))
            anchor_candidates.append((abs(a-t),t,a))
    for _,t,a in sorted(anchor_candidates):
        proposal=dict(mapping);proposal[t]=a;es=materialize(proposal)
        if es is None or issues(es,duration):
            rejected.append(dict(kind='note_anchor',source=t,target=a,reason='occupancy_or_length'));continue
        mapping=proposal;protected.add(t)
        edits.append(dict(kind='note_anchor',sources=[t],target=a,endpoints=nodes[t]))
    clusters={t:{t} for t in nodes if t not in protected}
    pairs=sorted((abs(a-b),a,b) for a in clusters for b in clusters if a<b and b-a<=tolerance+1e-9)
    for _,a,b in pairs:
        ca,cb=clusters[a],clusters[b]
        if ca==cb:continue
        merged=ca|cb
        if max(merged)-min(merged)>tolerance+1e-9:continue
        members=[i for t in merged for i,endpoint in nodes[t]]
        if len(members)!=len(set(members)):continue
        target=grid_tick(sum(merged)/len(merged))/200
        proposal=dict(mapping)
        for t in merged:proposal[t]=target
        es=materialize(proposal)
        if es is None or issues(es,duration):
            rejected.append(dict(kind='hold_pair',sources=sorted(merged),target=target,reason='occupancy_or_length'));continue
        mapping=proposal
        for t in merged:clusters[t]=merged
        edits.append(dict(kind='hold_pair',sources=sorted(merged),target=target,
                          endpoints=[p for t in sorted(merged) for p in nodes[t]]))
    out=materialize(mapping)
    if out is None or issues(out,duration):raise AssertionError('Invalid final alignment')
    changes=[dict(event_id=e.event_id,before=[e.start,e.end],after=[f.start,f.end])
             for e,f in zip(events,out) if (e.start,e.end)!=(f.start,f.end)]
    return out,dict(tolerance_seconds=tolerance,edits=edits,changes=changes,rejected=rejected,
        changed_holds=len(changes),extended=sum((f.end-f.start)>(e.end-e.start)+1e-8 for e,f in zip(events,out)),
        shortened=sum((f.end-f.start)<(e.end-e.start)-1e-8 for e,f in zip(events,out)),
        old_250ms_tail_extension=False,grid=GRID)


def process(events,features,prediction,duration,prior,policy=Policy()):
    before=structure_stats(events)
    analysis=analyze_audio(features,duration,policy)
    out,density=add_climax_candidates(events,prediction,analysis,duration,policy)
    out,phrasing=arrange_phrases(out,duration,prior,policy)
    out,snapping=snap_events(out,duration)
    out,alignment=align_holds(out,duration)
    times={e.event_id:(e.start,e.end) for e in out}
    out,repeats=apply_repeats(out,repeat_candidates(features,duration,policy),duration,policy)
    if any(times[e.event_id]!=(e.start,e.end) for e in out) or issues(out,duration):
        raise AssertionError('Final similarity stage changed aligned times or legality')
    return out,dict(policy=dict(version='playtest-grid-r2',grid=GRID,alignment_seconds=.1,old_250ms_tail_extension=False),
        stage_order=['gated_climax_candidates','single_lane_phrases','5ms_snap','hold_endpoint_alignment','similar_passage_mirroring','validation_only'],
        before=before,after=structure_stats(out),density=density,phrasing=phrasing,snapping=snapping,
        alignment=alignment,repeats=repeats,model_retrained=False,human_playtest='pending')
