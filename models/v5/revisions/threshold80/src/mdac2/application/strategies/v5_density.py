"""V5 revision 4: discard/retain > 0.8 admits a Boss; nearby Bosses merge."""
from mdac2.application.studio_timing_v4 import ATTACKS, _conflict, _four_groups, _priority
from mdac2.data_preparation.training_chart import validate_events
from .v5_events import SpecialEvent

DISCARD_TO_RETAIN_THRESHOLD = 0.8
BOSS_MERGE_GAP_MS = 500


def _thin(attacks, config):
    """Existing priorities and spacing, with deletion only: no lane/time edits.

    Hold occupancy is deliberately excluded from this density measurement.
    Otherwise every sparse click inside a hold would count as 100% discarded.
    Recheck four-chain exemptions after removing a member elsewhere.
    """
    current=list(attacks)
    while True:
        four=_four_groups(current,config);kept=[]
        for e in sorted(current,key=lambda e:(-_priority(e),e.start,e.event_id)):
            if any(any(_conflict(e,k,lane,[],config,current,four) for lane in e.lanes)
                   or validate_events([e,k],max(e.end,k.end)+1)
                   for k in kept if abs(e.start-k.start)<=2.001):continue
            kept.append(e)
        if len(kept)==len(current):return {e.event_id for e in kept}
        current=kept


def dense_groups(events, config, *, audit=None):
    attacks=sorted((e for e in events if e.kind in ATTACKS),key=lambda e:(e.start,e.event_id))
    four=_four_groups(attacks,config);spans=[]
    for lane in ('ground','air'):
        lane_events=[e for e in attacks if lane in e.lanes]
        for a,b in zip(lane_events,lane_events[1:]):
            if _conflict(a,b,lane,[],config,attacks,four):spans.append((a.start,b.start))
    segments=[]
    for a,b in sorted(spans):
        if segments and a<=segments[-1][1]+1e-9:
            segments[-1]=(segments[-1][0],max(b,segments[-1][1]))
        else:segments.append((a,b))
    kept=_thin(attacks,config);eligible=[];decisions=[]
    for a,b in segments:
        members=[e for e in attacks if a<=e.start<=b]
        retained=sum(e.event_id in kept for e in members);discarded=len(members)-retained
        admitted=discarded>retained*DISCARD_TO_RETAIN_THRESHOLD
        decisions.append(dict(start=a,end=b,source_note_count=len(members),retained=retained,
            discarded=discarded,discard_to_retain_ratio=discarded/retained if retained else None,
            action='boss' if admitted else 'discard_only'))
        if admitted:eligible.append((a,round(max(b,a+.001),3)))
    # Only admitted Boss intervals can initiate or extend a merge. Exactly
    # 500 ms remains separate; cascading merges use the current interval end.
    merged=[]
    for a,b in eligible:
        if merged and round(a*1000)-round(merged[-1][1]*1000)<BOSS_MERGE_GAP_MS:
            merged[-1]=(merged[-1][0],max(b,merged[-1][1]))
        else:merged.append((a,b))
    consumed=set();bosses=[];report=[]
    for index,(a,b) in enumerate(merged):
        # Include bridge notes once when the merged Boss spans their positions.
        members=[e for e in attacks if a<=e.start<=b]
        ids=tuple(e.event_id for e in members);consumed.update(ids)
        lane=max(('ground','air'),key=lambda lane:sum(lane in e.lanes for e in members))
        boss=SpecialEvent(f'v5-masher-{index}-{round(a*1000)}','masher',(lane,),a,b,
            ids,'0G','v5-dense-threshold',source_note_count=len(ids),target_hits=2*len(ids))
        bosses.append(boss)
        report.append(dict(event_id=boss.event_id,start=a,end=b,source_ids=list(ids),
            source_note_count=len(ids),target_hits=boss.target_hits))
    dropped={e.event_id for e in attacks if e.event_id not in kept and e.event_id not in consumed}
    if audit is not None:
        audit.update(discard_to_retain_threshold=DISCARD_TO_RETAIN_THRESHOLD,
            comparison='strictly_greater',merge_gap_ms=BOSS_MERGE_GAP_MS,merge_comparison='strictly_less',
            segments=decisions,eligible_before_merge=len(eligible),bosses_after_merge=len(bosses),
            discarded_without_boss=sorted(dropped),retention='same-lane spacing and existing priority; delete only')
    return [e for e in events if e.event_id not in consumed|dropped]+bosses,report
