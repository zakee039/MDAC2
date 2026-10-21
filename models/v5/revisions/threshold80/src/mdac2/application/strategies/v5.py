"""V5: V4 acoustic routing, independent holds and thresholded dense groups."""
from dataclasses import replace
import numpy as np
from mdac2.application.studio_timing import settings
from mdac2.application.studio_timing_v4 import (
    _build_grid, _nearest, _resolve, violations)
from mdac2.data_preparation.global_postprocess import normalize_ms
from mdac2.data_preparation.playtest_postprocess import (
    Policy, analyze_audio, add_climax_candidates, arrange_phrases)
from .v5_events import SpecialEvent, issues, ordinary
from .v5_holds import refine
from .v5_density import dense_groups


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
    density_policy={}
    out,dense=dense_groups(out,config,audit=density_policy)
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
    report=dict(policy=dict(version='studio-v5-sustain-masher-1ms-v4',storage='1ms',
        hold_interior='both lanes: small masher only',masher_target='2N metadata; duration exported',
        model_retrained=False),holds=holds,dense_groups=dense,soft_edges=soft,
        density=density,density_policy=density_policy,phrasing=phrasing,blocked_inside_holds=[e.event_id for e in blocked],
        resolution=resolution,collapsed_holds=collapsed,
        stage_order=['v4_acoustic_routing','density','phrases','independent_vocal_mix_holds',
                     '1ms_and_optional_bpm12','density_threshold_then_boss_merge','both_lane_hold_guard',
                     'v4_remaining_spacing','soft_edges','v5_gate'],human_playtest='pending')
    return out,report,dict(enabled=config['enabled'],subdivisions=12,stage_order=['v5_gate']),dict(
        engineering_playtest_gate_passed=not failures,issues=failures,human_playtest='pending')
