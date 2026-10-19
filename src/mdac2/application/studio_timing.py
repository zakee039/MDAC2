"""Optional studio beat grid, separate from frozen V2/V3 inference."""
from dataclasses import replace
from fractions import Fraction
import math
from statistics import median

DEFAULT=dict(enabled=False,bpm=120.,offset_ms=0.,tap_latency_ms=0.,subdivisions=12)


def rebase(value,delta_ms):
    out=settings(value);out['offset_ms']-=delta_ms
    if 'segments' in out:out['segments']=[{**s,'start_ms':s['start_ms']-delta_ms,'offset_ms':s['offset_ms']-delta_ms} for s in out['segments']]
    return settings(out)


def settings(value=None):
    if value is None:return DEFAULT.copy()
    segments=None
    if isinstance(value,dict) and 'segments' in value:
        value=dict(value);segments=value.pop('segments')
    # Already-generated v0.1 charts had four subdivisions but no explicit field.
    # Preserve their identity when exporting; confirmation upgrades new generation.
    if isinstance(value,dict) and 'subdivisions' not in value:value={**value,'subdivisions':4}
    if not isinstance(value,dict) or set(value)!=set(DEFAULT) or type(value['enabled']) is not bool:
        raise ValueError('BPM设置字段无效')
    if type(value['subdivisions']) is not int or value['subdivisions'] not in (4,12):raise ValueError('无效的拍内细分')
    out={'enabled':value['enabled'],'subdivisions':value['subdivisions']}
    for key,lo,hi in [('bpm',30,400),('offset_ms',-600000,600000),('tap_latency_ms',-1000,1000)]:
        v=value[key]
        if type(v) not in (float,int) or not math.isfinite(v) or not lo<=v<=hi:raise ValueError('BPM或偏移超出范围')
        out[key]=float(v)
    if segments is not None:
        if not isinstance(segments,list) or len(segments)>64:raise ValueError('最多64个变奏点')
        clean=[]
        for item in segments:
            if not isinstance(item,dict) or set(item)!={'start_ms','bpm','offset_ms','tap_latency_ms'}:raise ValueError('变奏点字段无效')
            start=item['start_ms']
            if type(start) not in (int,float) or not math.isfinite(start) or start!=round(start) or not -600000<=start<=600000:raise ValueError('变奏点位置无效')
            checked=settings({**DEFAULT,**{k:v for k,v in item.items() if k!='start_ms'}})
            clean.append(dict(start_ms=int(start),**{k:checked[k] for k in ('bpm','offset_ms','tap_latency_ms')}))
        if any(b['start_ms']-a['start_ms']<1 for a,b in zip(clean,clean[1:])):raise ValueError('变奏点必须按时间排序，不能重叠')
        out['segments']=clean
    return out


def fit_beats(beats):
    times=[float(t) for t in beats]
    if len(times)<4 or any(not math.isfinite(t) or t<0 for t in times) or any(b<=a for a,b in zip(times,times[1:])):
        raise ValueError('有效节拍不足，请手打测拍或手动填写')
    step=median([b-a for a,b in zip(times,times[1:])]);origin=times[0]
    indices=[0]
    for a,b in zip(times,times[1:]):indices.append(indices[-1]+max(1,round((b-a)/step)))
    # Missing detected beats can skip indices. Fit a single global grid, not a tempo map.
    for iteration in range(3):
        mx=sum(indices)/len(indices);my=sum(times)/len(times)
        variance=sum((i-mx)**2 for i in indices)
        if not variance:raise ValueError('节拍不足')
        residual=[t-(origin+i*step) for i,t in zip(indices,times)]
        good=[(i,t) for i,t,r in zip(indices,times,residual) if iteration==0 or abs(r-median(residual))<=max(.04,step*.15)]
        if len(good)<4:break
        mx=sum(i for i,t in good)/len(good);my=sum(t for i,t in good)/len(good)
        variance=sum((i-mx)**2 for i,t in good)
        if not variance:break
        step=sum((i-mx)*(t-my) for i,t in good)/variance;origin=my-mx*step
    bpm=round(60/step,6)
    if not 30<=bpm<=400:raise ValueError('自动测拍超出30–400 BPM，请手动修正')
    offset=(origin%step)*1000
    residuals=sorted(abs((t-origin)/step-round((t-origin)/step))*step for t in times)
    p95=residuals[min(len(residuals)-1,int(len(residuals)*.95))]*1000
    return dict(bpm=bpm,offset_ms=round(offset,3),beat_count=len(times),p95_ms=round(p95,2),
        needs_review=True,variable_tempo_warning=p95>50,beat_times=times)


def _nearest_pair_slots(original,subdivisions):
    """Choose a 2-note spacing of 3 or 6 positions while preserving order."""
    candidates=[(a,b) for a in range(subdivisions) for b in range(a+1,subdivisions)
                if b-a in (3,6)]
    return min(candidates,key=lambda pair:(sum(abs(a-b) for a,b in zip(original,pair)),
                                            abs((pair[1]-pair[0])-6),pair))


def downgrade_heavy_near_lights(events, config):
    """Prefer a same-lane light over a nearby heavy attack.

    This is a workbench adapter rule. It does not alter frozen model or
    global-postprocess sources. With a beat map, proximity means the same beat;
    without one it means 200ms in either direction. A converted member leaves
    its double-heavy group so the remaining heavy cannot be treated as an
    atomic pair after one side has changed type.
    """
    from collections import defaultdict

    config=settings(config)
    segments=[dict(start_ms=-float('inf'),**{k:config[k] for k in ('bpm','offset_ms')})]+config.get('segments',[])

    def beat_key(t):
        if not config['enabled']:
            return None
        chosen=0
        for index,item in enumerate(segments[1:],1):
            if t*1000>=item['start_ms']:chosen=index
            else:break
        segment=segments[chosen]
        step=Fraction(60,12)/Fraction(str(segment['bpm']))
        beat_step=step*12
        origin=Fraction(str(segment['offset_ms']))/1000
        index=math.floor(float((Fraction(str(t))-origin)/beat_step)+1e-8)
        return chosen,index

    lights=[e for e in events if e.kind=='light']
    heavies=[e for e in events if e.kind=='heavy']
    if not lights or not heavies:
        return list(events),dict(enabled=True,mode='same_beat' if config['enabled'] else 'nearby_200ms',
                                 window_ms=0 if config['enabled'] else 200,changed=[],dropped=[])
    group_members=defaultdict(list)
    for e in events:
        if e.kind=='heavy' and e.group_id:
            group_members[e.group_id].append(e.event_id)
    changed=[];changed_groups=set()
    for heavy in sorted(heavies,key=lambda e:(e.start,e.event_id)):
        candidates=[]
        for light in lights:
            if not set(heavy.lanes)&set(light.lanes):
                continue
            if config['enabled']:
                if beat_key(heavy.start)!=beat_key(light.start):
                    continue
                distance=abs(heavy.start-light.start)
            else:
                distance=abs(heavy.start-light.start)
                if distance>.2+1e-9:
                    continue
            candidates.append((distance,light.start,light.event_id,light))
        if not candidates:
            continue
        _,_,light_id,_=min(candidates, key=lambda item: item[:3])
        # Keep both source events. The heavy changes kind so it cannot cover
        # the existing light in the chart renderer.
        light_time=next(item[1] for item in candidates if item[2]==light_id)
        changed_groups.update(group_members.get(heavy.group_id,()))
        changed.append(dict(event_id=heavy.event_id,from_kind='heavy',to_kind='light',
                            light_event_id=light_id,action='downgrade',
                            distance_ms=round(abs(heavy.start-light_time)*1000,3)))
    out=[]
    changed_ids={row['event_id'] for row in changed if row['action']=='downgrade'}
    for event in events:
        if event.event_id in changed_ids:
            group_id=None
            out.append(replace(event,kind='light',group_id=group_id))
        elif event.group_id in changed_groups:
            out.append(replace(event,group_id=None))
        else:
            out.append(event)
    return out,dict(enabled=True,mode='same_beat' if config['enabled'] else 'nearby_200ms',
                     window_ms=0 if config['enabled'] else 200,changed=changed,
                     dropped=[],changed_count=len(changed_ids),
                     dropped_count=0,scope='same physical lane')


def uniformize_beat_notes(events,duration,config):
    """Spread supported same-track note groups across a 12-position beat.

    This adapter only runs after BPM snapping. Holds contribute their heads;
    tails keep the snapped endpoint chosen by the grid. A gemini is included
    once for each physical lane it occupies, with one shared chosen start time.
    """
    if config['subdivisions']!=12:
        return list(events),dict(enabled=False,reason='requires_12_positions')
    from collections import defaultdict
    from dataclasses import replace

    segments=[dict(start_ms=-float('inf'),**{k:config[k] for k in ('bpm','offset_ms')})]+config.get('segments',[])
    def segment_for(t):
        chosen=0
        for index,item in enumerate(segments[1:],1):
            if t*1000>=item['start_ms']:chosen=index
            else:break
        return chosen
    groups=defaultdict(list)
    details={}
    for event in events:
        segment_index=segment_for(event.start)
        segment=segments[segment_index]
        step=Fraction(60,12)/Fraction(str(segment['bpm']))
        beat_step=step*12
        origin=Fraction(str(segment['offset_ms']))/1000
        beat_position=float((Fraction(str(event.start))-origin)/beat_step)
        beat_index=math.floor(beat_position+1e-8)
        slot=round(float((Fraction(str(event.start))-(origin+beat_index*beat_step))/step))
        slot=max(0,min(11,slot))
        details[event.event_id]=(segment_index,beat_index,slot,step,origin)
        for lane in event.lanes:
            groups[(segment_index,beat_index,lane)].append(event)

    choices={};choice_cost={};dropped=set();conflicts=[];group_reports=[]
    for key,items in groups.items():
        unique={event.event_id:event for event in items}
        ordered=sorted(unique.values(),key=lambda event:(event.start,event.event_id))
        count=len(ordered);original=[details[event.event_id][2] for event in ordered]
        group=dict(segment=key[0],beat=key[1],lane=key[2],count=count)
        if count>4:
            dropped.update(unique)
            group['action']='drop_group'
            group_reports.append(group)
            continue
        if count==1:
            group['action']='keep'
            group_reports.append(group)
            continue
        slots=[0,4,8] if count==3 else [0,3,6,9] if count==4 else list(_nearest_pair_slots(original,12))
        segment_index,beat_index,_,step,origin=details[ordered[0].event_id]
        for event,slot in zip(ordered,slots):
            target=round(float((origin+(beat_index*12+slot)*step)*1000))/1000
            cost=abs(event.start-target)
            previous=choice_cost.get(event.event_id)
            if previous is None or cost<previous:
                choices[event.event_id]=target;choice_cost[event.event_id]=cost
            elif abs(choices[event.event_id]-target)>1e-9:
                conflicts.append(dict(event_id=event.event_id,kept=choices[event.event_id],discarded=target))
        group.update(action='redistribute',slots=slots,original_slots=original)
        group_reports.append(group)

    out=[]
    moved=0
    for event in events:
        if event.event_id in dropped:continue
        target=choices.get(event.event_id,event.start)
        if event.kind=='hold' and event.end-target<.03-1e-8:
            dropped.add(event.event_id)
            continue
        if abs(target-event.start)>1e-9:moved+=1
        out.append(replace(event,start=target,end=target if event.kind!='hold' else event.end))
    return out,dict(enabled=True,positions_per_beat=12,groups=group_reports,dropped=sorted(dropped),
                    moved=moved,conflicts=conflicts,scope='same physical lane; hold heads only')


def snap(events,duration,config):
    config=settings(config)
    if not config['enabled']:
        out,heavy_rule=downgrade_heavy_near_lights(events,config)
        return out,dict(enabled=False,heavy_conflicts=heavy_rule,
                         stage_order=['nearby_heavy_to_light','validation'])
    from mdac2.data_preparation.generated_constraints import select_generated_candidates
    from mdac2.data_preparation.global_postprocess import prioritize_exclusive,move_hold_clicks
    from mdac2.data_preparation.playability_audit import audit_playability
    from bisect import bisect_left
    # Each segment owns [start,end); the final segment also owns the song end.
    # A hold's head and tail independently use the map at their own times.
    segments=[dict(start_ms=-float('inf'),**{k:config[k] for k in ('bpm','offset_ms')})]+config.get('segments',[])
    grid=[]
    for index,segment in enumerate(segments):
        lower=max(0,segment['start_ms']/1000)
        upper=min(duration,segments[index+1]['start_ms']/1000) if index+1<len(segments) else duration
        if upper<lower:continue
        step=Fraction(60,config['subdivisions'])/Fraction(str(segment['bpm']))
        origin=Fraction(str(segment['offset_ms']))/1000
        low=math.ceil((Fraction(str(lower))-origin)/step);high=math.floor((Fraction(str(upper))-origin)/step)
        for ix in range(low,high+1):
            exact=origin+ix*step;t=round(exact*1000)/1000
            if lower<=t<=duration and (t<upper or upper==duration):grid.append(t)
    grid=sorted(set(grid))
    if not grid:raise ValueError('音频范围内没有网格点，请检查BPM和偏移')
    def point(t):
        i=bisect_left(grid,t)
        if i==0:return grid[0]
        if i==len(grid):return grid[-1]
        return grid[i-1] if t-grid[i-1]<grid[i]-t else grid[i]
    moved=[];dropped=[];shifts={}
    for e in events:
        a,b=point(e.start),point(e.end)
        if a>duration or b>duration or e.kind=='hold' and b-a<.03-1e-8:
            dropped.append(dict(event_id=e.event_id,reason='collapsed_or_outside_hold_grid'));continue
        moved.append(replace(e,start=a,end=b));shifts[e.event_id]=max(abs(a-e.start),abs(b-e.end))
    moved,uniformization=uniformize_beat_notes(moved,duration,config)
    moved,heavy_rule=downgrade_heavy_near_lights(moved,config)
    moved,exclusive=prioritize_exclusive(moved,duration)
    # Explicit specials > holds > clicks. Within a kind prefer the least-shifted original.
    scored=[(e,(1 if e.exclusive else .8 if e.kind=='hold' else .5)-min(.1,shifts[e.event_id]/10)) for e in moved]
    selected,collisions=select_generated_candidates(scored,duration)
    selected,avoidance=move_hold_clicks(selected,duration)
    if not selected:raise ValueError('BPM吸附后无可用音符，请调整BPM或关闭吸附')
    if any(abs(point(t)-t)>1e-9 for e in selected for t in (e.start,e.end)):raise AssertionError('Post-grid rule moved endpoint off grid')
    if not audit_playability(selected,duration)['engineering_playtest_gate_passed']:raise AssertionError('Grid conflicts remain')
    return selected,dict(enabled=True,policy='studio-tempo-map-v1' if config.get('segments') else 'studio-twelfth-beat-v2' if config['subdivisions']==12 else 'studio-quarter-beat-v1',settings=config,positions_per_beat=config['subdivisions'],
        step_seconds=float(step),storage_ms=1,max_storage_rounding_ms=.5,before=len(events),after=len(selected),
        max_shift_ms=round(max(shifts.values(),default=0)*1000,3),dropped=dropped,uniformization=uniformization,exclusive=exclusive,
        collisions=collisions,avoidance=avoidance,heavy_conflicts=heavy_rule,
        stage_order=['millisecond_grid','same_track_beat_uniformization','nearby_heavy_to_light','exclusive_priority',
                     'candidate_collision_selection','hold_avoidance','validation'])
