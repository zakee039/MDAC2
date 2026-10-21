# Strategy-owned BPM guard snapshot. Frozen source is unchanged.
"""Shared current six-class postprocessing, independent of neural model choice.
Frozen release implementations are intentionally preserved as historical tools.
"""
from collections import defaultdict
from dataclasses import replace
from fractions import Fraction
from mdac2.data_preparation.fixed_grid_r2 import issues
from mdac2.data_preparation.hold_refinement_rc1 import extend_sustained_tails, dense_pairs, HoldPolicy
from mdac2.data_preparation.generated_constraints import NEARBY_SECONDS
from mdac2.data_preparation.playtest_postprocess import (Policy, analyze_audio, add_climax_candidates, arrange_phrases,
    repeat_candidates, apply_repeats, structure_stats)

STORAGE = dict(version='absolute-1ms-global-v1', step_seconds=.001, bpm=120,
               song_tempo_estimated=False, old_5ms_quantization=False)
POLICY_ID = 'global-postprocess-1ms-v2'

def ms_tick(t):
    value=Fraction(str(t))*1000
    if value<0: raise ValueError('Negative time')
    return (2*value.numerator+value.denominator)//(2*value.denominator)

def on_ms(t): return abs(t*1000-round(t*1000))<1e-7

def normalize_ms(events,duration):
    out=[replace(e,start=ms_tick(e.start)/1000,end=ms_tick(e.end)/1000) for e in events]
    if issues(out,duration): raise ValueError('Millisecond rounding caused conflicts')
    return out

def align_holds(events,duration,tolerance=.1):
    """Fixed original endpoint groups. Note anchors first, then pair midpoints.

    Shared heads/tails remain atomic. Unanchored clusters never span >100ms;
    members of a single hold cannot collapse into the same endpoint cluster.
    """
    if issues(events,duration) or any(not on_ms(t) for e in events for t in (e.start,e.end)):
        raise ValueError('Legal millisecond chart required')
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
        target=ms_tick(sum(merged)/len(merged))/1000
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
        old_250ms_tail_extension=False,grid=STORAGE)


def forbidden_lanes(event, events, tail_guard=.5):
    """Inclusive head, entire hold, and inclusive 500ms post-tail guard."""
    return {lane for h in events if h.kind == 'hold'
            and h.start-1e-8 <= event.start <= h.end+tail_guard+1e-8 for lane in h.lanes}


def avoidance_violations(events):
    return {(e.event_id,lane) for e in events if e.kind not in ('hold','extra_heavy')
            for lane in set(e.lanes) & forbidden_lanes(e,events)}


def prioritize_exclusive(events,duration):
    """Special attacks end all ongoing holds; never resume a cut hold.

    Remove simultaneous/nearby strikes on either lane. Duplicate nearby
    specials retain the earliest stable original identity. Future events past
    the 20ms instantaneous conflict neighbourhood are unaffected.
    """
    out=list(events);edits=[];dropped=[];kept=[]
    for special in sorted((e for e in events if e.kind=='extra_heavy'),key=lambda e:(e.start,e.event_id)):
        if any(abs(special.start-e.start)<=NEARBY_SECONDS+1e-9 for e in kept):
            out=[e for e in out if e.event_id!=special.event_id]
            dropped.append(dict(event_id=special.event_id,time=special.start,reason='nearby_exclusive_duplicate'))
        else:kept.append(special)
    for special in kept:
        proposal=[]
        for e in out:
            if e.kind=='extra_heavy':proposal.append(e);continue
            if e.kind=='hold' and e.start<=special.start<e.end:
                if special.start-e.start>=.03-1e-8:
                    proposal.append(replace(e,end=special.start))
                    edits.append(dict(event_id=e.event_id,before=[e.start,e.end],after=[e.start,special.start],
                                      special_id=special.event_id,reason='exclusive_cuts_hold'))
                else:dropped.append(dict(event_id=e.event_id,time=e.start,reason='exclusive_leaves_under_30ms_hold',special_id=special.event_id))
            elif e.kind!='hold' and abs(e.start-special.start)<=NEARBY_SECONDS+1e-9:
                dropped.append(dict(event_id=e.event_id,time=e.start,reason='exclusive_priority',special_id=special.event_id))
            else:proposal.append(e)
        out=proposal
    return out,dict(edits=edits,dropped=dropped,nearby_seconds=NEARBY_SECONDS,hold_resume=False)


def move_hold_clicks(events,duration):
    out,exclusive=prioritize_exclusive(events,duration)
    originals=list(out);groups=defaultdict(list);edits=[];dropped=[];plans=[]
    for e in originals:
        if e.kind in ('hold','extra_heavy'):continue
        key=('double',e.group_id) if e.kind=='heavy' and e.group_id else ('event',e.event_id)
        groups[key].append(e)
    for key,members in groups.items():
        members.sort(key=lambda e:(e.start,e.event_id));event=members[0]
        if key[0]=='double' and (len(members)!=2 or len({e.start for e in members})!=1
                or {e.lanes for e in members}!={('air',),('ground',)}):
            raise ValueError('Malformed double-heavy group')
        blocked=forbidden_lanes(event,originals)
        if not any(set(e.lanes)&blocked for e in members):continue
        plans.append((event,members,blocked,key[0]=='double'))
    # Remove all affected groups before applying proposals. A snapshot reserves
    # original positions even if that original is also scheduled for removal:
    # no iteration-order-dependent borrowing or later resurrection of notes.
    affected={e.event_id for _,members,_,_ in plans for e in members}
    out=[e for e in out if e.event_id not in affected]
    for event,members,blocked,double in sorted(plans,key=lambda p:(p[0].start,p[0].event_id)):
        ids={e.event_id for e in members};reason=None;neighbours=[];moved=None
        if len(blocked)==2:reason='both_lanes_guarded'
        elif event.kind not in ('light','heavy','gemini'):reason='unsupported_transfer_kind'
        else:
            target='air' if 'ground' in blocked else 'ground'
            base=next((e for e in members if e.lanes==(target,)),event)
            moved=replace(base,lanes=(target,),kind='light' if event.kind=='gemini' else event.kind,
                group_id=None,source_ids=tuple(dict.fromkeys(s for e in members for s in e.source_ids)))
            neighbours=sorted({e.event_id for e in originals+out if e.event_id not in ids and e.kind!='hold'
                and (target in e.lanes or e.exclusive) and abs(e.start-event.start)<=NEARBY_SECONDS+1e-9})
            if neighbours:reason='target_original_or_accepted_note'
            elif issues(out+[moved],duration):reason='occupancy_or_length'
        if reason:
            dropped.append(dict(event_ids=sorted(ids),time=event.start,reason=reason,blockers=neighbours))
            continue
        out.append(moved)
        edits.append(dict(event_id=event.event_id,event_ids=sorted(ids),retained_event_id=moved.event_id,
            time=event.start,from_kind=event.kind,to_kind=moved.kind,
            from_lanes=sorted({lane for e in members for lane in e.lanes}),to_lane=target,merged_into=None,
            gemini_to_opposite_light=event.kind=='gemini',double_heavy_to_opposite_heavy=double))
    out.sort(key=lambda e:(e.start,e.event_id))
    if avoidance_violations(out) or issues(out,duration):raise AssertionError('Strict transfer/drop left conflicts')
    return out,dict(edits=edits,dropped=dropped,rejected=dropped,exclusive=exclusive,tail_guard_seconds=.5,
        target_neighbour_seconds=NEARBY_SECONDS,original_positions_reserved=True,
        scope='inclusive hold head, entire hold, tail and 500ms after end; transfer or discard',
        remaining_violations=[])


def process(events,features,prediction,duration,prior):
    """One shared entry for ALL current automatic neural generation profiles."""
    before=structure_stats(events)
    events,exclusive_initial=prioritize_exclusive(events,duration)
    events=normalize_ms(events,duration)
    analysis=analyze_audio(features,duration,Policy())
    out,density=add_climax_candidates(events,prediction,analysis,duration,Policy())
    out,phrasing=arrange_phrases(out,duration,prior,Policy())
    out=normalize_ms(out,duration)
    out,tails=extend_sustained_tails(out,features,duration,HoldPolicy())
    out,alignment=align_holds(out,duration)
    out,avoidance=move_hold_clicks(out,duration)
    spacing=dict(policy='subsumed_by_all_hold_click_transfer_or_discard',minimum_click_gap_seconds=.505,
                 legacy_move_and_merge_disabled=True)
    times={e.event_id:(e.start,e.end) for e in out}
    proposal,repeats=apply_repeats(out,repeat_candidates(features,duration,Policy()),duration,Policy())
    if (not set(dense_pairs(proposal))<=set(dense_pairs(out))
            or not avoidance_violations(proposal)<=avoidance_violations(out)):
        repeats.update(applied=0,stage_rejected_by_global_hold_rules=True)
    else:out=proposal
    if issues(out,duration) or any(not on_ms(t) for e in out for t in (e.start,e.end)):
        raise AssertionError('Global final rule/millisecond gate failed')
    if any(times[e.event_id]!=(e.start,e.end) for e in out):raise AssertionError('Mirror changed timing')
    return out,dict(policy=dict(version=POLICY_ID,storage=STORAGE,alignment_seconds=.1,
        hold_click_gap_seconds=.505,hold_interior_opposite=True,post_tail_opposite_seconds=.5,
        gemini_in_guard='opposite_single_light',double_heavy_in_guard='opposite_single_heavy',
        blocked_transfer='discard',hold_head_included=True,exclusive_priority=True,
        old_250ms_tail_extension=False),
        before=before,after=structure_stats(out),density=density,phrasing=phrasing,tails=tails,alignment=alignment,
        exclusive_initial=exclusive_initial,spacing=spacing,avoidance=avoidance,repeats=repeats,final_dense_pairs=dense_pairs(out),
        final_avoidance_violations=sorted(avoidance_violations(out)),
        stage_order=['exclusive_priority','1ms','density','phrases','1ms','acoustic_tail','100ms_alignment',
                     'entire_hold_and_post_tail_opposite','similarity_with_global_gate','validation_only'],
        model_independent_rules=True,model_retrained=False,human_playtest='pending')

