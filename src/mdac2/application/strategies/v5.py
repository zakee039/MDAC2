"""V5: V4 acoustic routing, independent holds, probability interior reroute."""
from dataclasses import replace
import random
import numpy as np
from mdac2.application.studio_timing import settings
from mdac2.application.studio_timing_v4 import _build_grid, _nearest
from mdac2.data_preparation.global_postprocess import normalize_ms
from mdac2.data_preparation.playtest_postprocess import (
    Policy, analyze_audio, add_climax_candidates, arrange_phrases)
from .v5_events import SpecialEvent, issues, ordinary
from .v5_holds import refine
from .v5_density import HOLD_MARGIN_SECONDS, dense_groups
from .v5_reroute import reroute
from .v5_spacing import HOLD_SPACING_FACTOR, resolve as _resolve, spacing_seconds, violations


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


def run(events, features, prediction, duration, prior, timing, options=None):
    options=dict(options or {})
    spacing=spacing_seconds(options.get('spacing_ms'))
    rng=random.Random(options['seed']) if options.get('seed') is not None else random.Random()
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
                # Endpoints obey the ordinary same-lane floor, so a snapped head
                # only has to clear the previous hold end by one floor.
                previous=[h.end for h in snapped if h.kind=='hold' and
                          set(h.lanes)&set(e.lanes) and h.start<a]
                if previous and a<=max(previous)+spacing+1e-8:
                    choices=[t for t in grid if t>max(previous)+spacing+1e-8]
                    a=choices[0] if choices else b
            if e.kind=='hold' and b-a<.03:
                collapsed.append(e.event_id);continue
            snapped.append(replace(e,start=a,end=b))
        out=snapped
    density_policy={}
    out,dense=dense_groups(out,config,audit=density_policy,holds=[e for e in out if e.kind=='hold'],spacing=spacing)
    bosses=[e for e in out if e.kind=='masher']
    # At the exact audio boundary a zero-width cluster cannot extend past audio.
    bosses=[replace(e,start=round(max(0,duration-.001),3),end=round(duration,3))
            if e.end>duration else e for e in bosses]
    # Sustained lanes keep their window open: notes inside a hold are legal and
    # only spacing-bound. Interior hits are then added by the reroute stage.
    out,rerouted=reroute(out,spacing,rng)
    normal=[e for e in out if e.kind!='masher']
    normal,resolution=_resolve(normal,duration,config,spacing)
    normal,soft=soft_edges(normal,features,duration)
    out=sorted(normal+bosses,key=lambda e:(e.start,e.event_id))
    failures=issues(out,duration)+violations([ordinary(e) for e in normal],duration,config,spacing)
    report=dict(policy=dict(version='studio-v5-boss-span-1ms-v10',storage='1ms',
        same_lane_spacing_ms=round(spacing*1000),
        hold_interior_spacing_ms=round(spacing*HOLD_SPACING_FACTOR*1000),
        hold_edge_spacing_ms=round(spacing*1000),
        hold_margin_ms=round(HOLD_MARGIN_SECONDS*1000),
        boss_near_hold='forbidden',
        boss_span=dict(short_below_ms=density_policy.get('boss_span_policy',{}).get('short_below_ms'),
            short_exported_ms=density_policy.get('boss_span_policy',{}).get('short_exported_ms'),
            demoted_ms=density_policy.get('boss_span_policy',{}).get('demoted_ms'),
            rule='span < 200 ms -> kept as a fixed 100 ms accent; 200-2000 ms -> demoted to density discarding; > 2000 ms -> untouched'),
        bosses_demoted_by_span=len(density_policy.get('boss_span_fallbacks',[])),
        short_bosses_normalised=density_policy.get('short_bosses_normalised',0),
        notes_restored_by_span=density_policy.get('boss_span_restored_notes',0),
        hold_detection=dict(weak_head_accepted='vocal regions below the old 6% energy gate are kept when energy rises into them',
            head_extension='model hold heads are pulled back to the detected phrase start, at most 500 ms',
            version='v9'),
        hold_interior='notes allowed; same-lane floor doubled inside one hold',
        hold_edges='head and tail count as ordinary notes; no extra clearance',
        hold_reroute=dict(minimum_hold_ms=round(rerouted['minimal_hold_seconds']*1000),
            first_probability=rerouted['first_probability'],
            rule='first opposite-lane note 50%, second note certain, then settled'),
        masher_target='2N metadata; duration exported',model_retrained=False),holds=holds,dense_groups=dense,soft_edges=soft,
        density=density,density_policy=density_policy,phrasing=phrasing,hold_reroute=rerouted,
        resolution=resolution,collapsed_holds=collapsed,
        stage_order=['v4_acoustic_routing','density','phrases','independent_vocal_mix_holds',
                     '1ms_and_optional_bpm12','density_threshold_hold_margin_then_boss_merge',
                     'hold_head_extension','hold_reroute','v5_configurable_spacing','hold_edges_as_ordinary_notes',
                     'boss_span_policy','soft_edges','v5_gate'],human_playtest='pending')
    return out,report,dict(enabled=config['enabled'],subdivisions=12,stage_order=['v5_gate']),dict(
        engineering_playtest_gate_passed=not failures,issues=failures,human_playtest='pending')
