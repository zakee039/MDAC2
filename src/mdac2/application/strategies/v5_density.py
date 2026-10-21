"""V5 revision 5: discard/retain > 0.8 admits a Boss, away from holds, and nearby Bosses merge."""
from mdac2.application.studio_timing_v4 import ATTACKS, _four_groups, _priority
from mdac2.data_preparation.training_chart import validate_events
from .v5_events import SpecialEvent
from .v5_spacing import DEFAULT_SPACING_SECONDS, conflict as _conflict

DISCARD_TO_RETAIN_THRESHOLD = 0.8
BOSS_MERGE_GAP_MS = 500
# Boss admission is decided by how long the merged interval lasts, not by how
# many hits 2N would ask for. Duration is what the player actually feels.
#   < 200 ms  : a rhythm accent. Kept, and normalised to a fixed 100 ms so the
#               whole tier reads the same instead of 90-121 ms noise.
#   200-2000  : the jackhammer band. Demoted back to density discarding: the
#               survivors of _thin stay as single notes, only the excess goes.
#   > 2000 ms : a real Boss. Left exactly as measured.
BOSS_SHORT_SPAN_MS = 200
BOSS_SHORT_DURATION_MS = 100
BOSS_DEMOTE_SPAN_MS = (200, 2000)
# Bosses stay clear of holds: the hold interval plus a 500 ms tail window. Notes
# no longer share this margin - hold endpoints only obey the ordinary same-lane
# floor, so this is purely a Boss rule now.
HOLD_MARGIN_SECONDS = .5


def _thin(attacks, config, spacing):
    """Existing priorities and spacing, with deletion only: no lane/time edits.

    Hold occupancy is deliberately excluded from this density measurement.
    Otherwise every sparse click inside a hold would count as 100% discarded.
    Recheck four-chain exemptions after removing a member elsewhere.
    """
    current=list(attacks)
    while True:
        four=_four_groups(current,config);kept=[]
        for e in sorted(current,key=lambda e:(-_priority(e),e.start,e.event_id)):
            if any(any(_conflict(e,k,lane,[],config,current,four,spacing) for lane in e.lanes)
                   or validate_events([e,k],max(e.end,k.end)+1)
                   for k in kept if abs(e.start-k.start)<=2.001):continue
            kept.append(e)
        if len(kept)==len(current):return {e.event_id for e in kept}
        current=kept


def _near_hold(a, b, holds):
    return any(a<=h.end+HOLD_MARGIN_SECONDS+1e-9 and b>=h.start-HOLD_MARGIN_SECONDS-1e-9 for h in holds)


def dense_groups(events, config, *, audit=None, holds=(), spacing=DEFAULT_SPACING_SECONDS):
    attacks=sorted((e for e in events if e.kind in ATTACKS),key=lambda e:(e.start,e.event_id))
    four=_four_groups(attacks,config);spans=[]
    for lane in ('ground','air'):
        lane_events=[e for e in attacks if lane in e.lanes]
        for a,b in zip(lane_events,lane_events[1:]):
            if _conflict(a,b,lane,[],config,attacks,four,spacing):spans.append((a.start,b.start))
    segments=[]
    for a,b in sorted(spans):
        if segments and a<=segments[-1][1]+1e-9:
            segments[-1]=(segments[-1][0],max(b,segments[-1][1]))
        else:segments.append((a,b))
    kept=_thin(attacks,config,spacing);eligible=[];decisions=[]
    for a,b in segments:
        members=[e for e in attacks if a<=e.start<=b]
        retained=sum(e.event_id in kept for e in members);discarded=len(members)-retained
        dense=discarded>retained*DISCARD_TO_RETAIN_THRESHOLD
        near=_near_hold(a,b,holds)
        admitted=dense and not near
        decisions.append(dict(start=a,end=b,source_note_count=len(members),retained=retained,
            discarded=discarded,discard_to_retain_ratio=discarded/retained if retained else None,
            action='boss' if admitted else 'hold_proximity' if dense else 'discard_only'))
        if admitted:eligible.append((a,round(max(b,a+.001),3)))
    # Only admitted Boss intervals can initiate or extend a merge. Exactly
    # 500 ms remains separate; cascading merges use the current interval end.
    merged=[]
    for a,b in eligible:
        if merged and round(a*1000)-round(merged[-1][1]*1000)<BOSS_MERGE_GAP_MS:
            merged[-1]=(merged[-1][0],max(b,merged[-1][1]))
        else:merged.append((a,b))
    consumed=set();bosses=[];report=[];fallback=[];shortened=[]
    for index,(a,b) in enumerate(merged):
        # Include bridge notes once when the merged Boss spans their positions.
        members=[e for e in attacks if a<=e.start<=b]
        ids=tuple(e.event_id for e in members)
        hits=2*len(ids)
        span_ms=round(b*1000)-round(a*1000)
        low,high=BOSS_DEMOTE_SPAN_MS
        if low<=span_ms<=high:
            # Not a Boss: the interval falls back to plain density discarding.
            # Nothing is consumed here, so _thin's survivors stay as single
            # notes and only the excess lands in `dropped` below. A refused
            # Boss must never make notes disappear.
            row=dict(event_id=None,start=a,end=b,source_ids=list(ids),
                source_note_count=len(ids),target_hits=hits,span_ms=span_ms,
                action='discard_only_boss_span',
                retained_ids=[e.event_id for e in members if e.event_id in kept])
            fallback.append(row);report.append(row);continue
        consumed.update(ids)
        lane=max(('ground','air'),key=lambda lane:sum(lane in e.lanes for e in members))
        end=b
        if span_ms<BOSS_SHORT_SPAN_MS:
            # Same tier, same length: a 90 ms and a 121 ms accent should not feel
            # like two different objects. Nothing survives inside the interval,
            # so a fixed 100 ms cannot collide with a kept note.
            end=round(a+BOSS_SHORT_DURATION_MS/1000,3)
            shortened.append(dict(start=a,measured_end=b,exported_end=end,span_ms=span_ms))
        boss=SpecialEvent(f'v5-masher-{index}-{round(a*1000)}','masher',(lane,),a,end,
            ids,'0G','v5-dense-threshold',source_note_count=len(ids),target_hits=hits)
        bosses.append(boss)
        report.append(dict(event_id=boss.event_id,start=a,end=end,source_ids=list(ids),
            source_note_count=len(ids),target_hits=boss.target_hits,span_ms=span_ms,action='boss'))
    dropped={e.event_id for e in attacks if e.event_id not in kept and e.event_id not in consumed}
    if audit is not None:
        audit.update(discard_to_retain_threshold=DISCARD_TO_RETAIN_THRESHOLD,
            comparison='strictly_greater',merge_gap_ms=BOSS_MERGE_GAP_MS,merge_comparison='strictly_less',
            same_lane_spacing_ms=round(spacing*1000),hold_margin_ms=round(HOLD_MARGIN_SECONDS*1000),
            segments=decisions,eligible_before_merge=len(eligible),bosses_after_merge=len(bosses),
            boss_span_policy=dict(short_below_ms=BOSS_SHORT_SPAN_MS,short_exported_ms=BOSS_SHORT_DURATION_MS,
                demoted_ms=list(BOSS_DEMOTE_SPAN_MS),kept_above_ms=BOSS_DEMOTE_SPAN_MS[1],
                rule='span < 200 ms kept as 100 ms accents; 200-2000 ms demoted to density discarding; > 2000 ms untouched'),
            bosses_before_span=len(merged),bosses_after_span=len(bosses),
            boss_span_fallbacks=[dict(start=r['start'],end=r['end'],source_note_count=r['source_note_count'],
                target_hits=r['target_hits'],span_ms=r['span_ms'],retained=len(r['retained_ids'])) for r in fallback],
            boss_span_restored_notes=sum(len(r['retained_ids']) for r in fallback),
            short_bosses_normalised=len(shortened),
            short_boss_spans_ms=[s['span_ms'] for s in shortened],
            discarded_without_boss=sorted(dropped),retention='same-lane spacing and existing priority; delete only')
    return [e for e in events if e.event_id not in consumed|dropped]+bosses,report
