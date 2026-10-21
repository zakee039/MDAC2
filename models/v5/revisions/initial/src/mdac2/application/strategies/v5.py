"""V5: V4 acoustic routing with independent holds and lossless dense groups."""
from dataclasses import replace
import numpy as np
from mdac2.application.studio_timing import settings
from mdac2.application.studio_timing_v4 import (
    ATTACKS, _build_grid, _nearest, _conflict, _four_groups, _resolve, violations)
from mdac2.data_preparation.global_postprocess import normalize_ms
from mdac2.data_preparation.playtest_postprocess import (
    Policy, analyze_audio, add_climax_candidates, arrange_phrases)
from .v5_events import SpecialEvent, issues, ordinary
from .v5_holds import refine


def dense_groups(events, config):
    """Close connected over-density spans over BOTH lanes; count identities once."""
    attacks=sorted((e for e in events if e.kind in ATTACKS),key=lambda e:(e.start,e.event_id))
    four=_four_groups(attacks,config);spans=[]
    for lane in ('ground','air'):
        lane_events=[e for e in attacks if lane in e.lanes]
        for a,b in zip(lane_events,lane_events[1:]):
            if _conflict(a,b,lane,[],config,attacks,four):spans.append((a.start,b.start))
    merged=[]
    for a,b in sorted(spans):
        if merged and a<=merged[-1][1]+1e-9:merged[-1]=(merged[-1][0],max(b,merged[-1][1]))
        else:merged.append((a,b))
    consumed=set();bosses=[];report=[]
    for index,(a,b) in enumerate(merged):
        members=[e for e in attacks if a<=e.start<=b]
        consumed.update(e.event_id for e in members)
        # Coincident attacks still need a non-zero interval for the BMS pair.
        end=round(max(b,a+.001),3)
        lane=max(('ground','air'),key=lambda lane:sum(lane in e.lanes for e in members))
        ids=tuple(e.event_id for e in members)
        boss=SpecialEvent(f'v5-masher-{index}-{round(a*1000)}','masher',(lane,),a,end,
                          ids,'0G','v5-dense-segment',source_note_count=len(ids),target_hits=2*len(ids))
        bosses.append(boss)
        report.append(dict(event_id=boss.event_id,start=a,end=end,source_ids=list(ids),
                           source_note_count=len(ids),target_hits=boss.target_hits))
    return [e for e in events if e.event_id not in consumed]+bosses,report


def soft_edges(events, features, duration):
    """Only quiet, non-silent single hits in the first/last 5 s qualify."""
    power=np.expm1(np.clip(np.asarray(features[0],dtype=float),0,25)).sum(0)
    reference=max(float(np.quantile(power,.9)),1e-12)
    changed=[];out=[]
    for e in events:
        index=min(len(power)-1,round(e.start*100))
        energy=float(np.max(power[max(0,index-3):min(len(power),index+4)]))
        if (e.kind in ('light','heavy') and (e.start<=5 or e.start>=duration-5)
                and reference*1e-6<energy<reference*.03):
            e=SpecialEvent(e.event_id,'music',e.lanes,e.start,e.end,e.source_ids,'23','v5-soft-edge')
            changed.append(dict(event_id=e.event_id,time=e.start,relative_energy=energy/reference))
        out.append(e)
    return out,changed


def run(events, features, prediction, duration, prior, timing):
    config=settings(timing)
    if config['enabled'] and config['subdivisions']!=12:
        raise ValueError('V5 requires confirmed 12-position BPM timing')
    out=normalize_ms(events,duration)
    analysis=analyze_audio(features,duration,Policy())
    out,density=add_climax_candidates(out,prediction,analysis,duration,Policy())
    out,phrasing=arrange_phrases(out,duration,prior,Policy())
    out,holds=refine(out,features,duration)
    out=normalize_ms(out,duration)
    collapsed=[]
    if config['enabled']:
        grid=_build_grid(duration,config);snapped=[]
        for e in out:
            a=_nearest(grid,e.start);b=_nearest(grid,e.end) if e.kind=='hold' else a
            if e.kind=='hold':
                # Rounding can turn a legal 501 ms gap into a blocked 500 ms.
                # Move the later head to the next grid cell, retaining its tail.
                previous=[h.end for h in snapped if h.kind=='hold' and
                          set(h.lanes)&set(e.lanes) and h.start<a]
                if previous and a<=max(previous)+.5+1e-8:
                    choices=[t for t in grid if t>max(previous)+.5+1e-8]
                    a=choices[0] if choices else b
            if e.kind=='hold' and b-a<.03:
                collapsed.append(e.event_id);continue
            snapped.append(replace(e,start=a,end=b))
        out=snapped
    out,dense=dense_groups(out,config)
    bosses=[e for e in out if e.kind=='masher']
    # At the exact audio boundary a zero-width cluster cannot extend past audio.
    bosses=[replace(e,start=round(max(0,duration-.001),3),end=round(duration,3))
            if e.end>duration else e for e in bosses]
    hs=[e for e in out if e.kind=='hold']
    blocked=[e for e in out if e.kind not in ('hold','masher') and
             any(h.start<=e.start<h.end for h in hs)]
    blocked_ids={e.event_id for e in blocked}
    normal=[e for e in out if e.kind!='masher' and e.event_id not in blocked_ids]
    normal,resolution=_resolve(normal,duration,config)
    normal,soft=soft_edges(normal,features,duration)
    out=sorted(normal+bosses,key=lambda e:(e.start,e.event_id))
    failures=issues(out,duration)+violations([ordinary(e) for e in normal],duration,config)
    report=dict(policy=dict(version='studio-v5-sustain-masher-1ms-v1',storage='1ms',
        hold_interior='both lanes: small masher only',masher_target='2N metadata; duration exported',
        model_retrained=False),holds=holds,dense_groups=dense,soft_edges=soft,
        density=density,phrasing=phrasing,blocked_inside_holds=[e.event_id for e in blocked],
        resolution=resolution,collapsed_holds=collapsed,
        stage_order=['v4_acoustic_routing','density','phrases','independent_vocal_mix_holds',
                     '1ms_and_optional_bpm12','dense_segments_to_masher','both_lane_hold_guard',
                     'v4_remaining_spacing','soft_edges','v5_gate'],human_playtest='pending')
    return out,report,dict(enabled=config['enabled'],subdivisions=12,stage_order=['v5_gate']),dict(
        engineering_playtest_gate_passed=not failures,issues=failures,human_playtest='pending')
