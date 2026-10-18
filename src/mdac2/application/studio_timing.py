"""Optional studio beat grid, separate from frozen V2/V3 inference."""
from dataclasses import replace
from fractions import Fraction
import math
from statistics import median

DEFAULT=dict(enabled=False,bpm=120.,offset_ms=0.,tap_latency_ms=0.,subdivisions=12)


def settings(value=None):
    if value is None:return DEFAULT.copy()
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


def snap(events,duration,config):
    config=settings(config)
    if not config['enabled']:return list(events),dict(enabled=False)
    from mdac2.data_preparation.generated_constraints import select_generated_candidates
    from mdac2.data_preparation.global_postprocess import prioritize_exclusive,move_hold_clicks
    from mdac2.data_preparation.playability_audit import audit_playability
    step=Fraction(60,config['subdivisions'])/Fraction(str(config['bpm']));origin=Fraction(str(config['offset_ms']))/1000
    low=math.ceil(-origin/step);high=math.floor((Fraction(str(duration))-origin)/step)
    if high<low:raise ValueError('音频范围内没有网格点，请检查BPM和偏移')
    def point(t):
        q=(Fraction(str(t))-origin)/step
        ix=min(high,max(low,math.floor(q+Fraction(1,2))))
        return float(Fraction(round((origin+ix*step)*1000),1000))
    moved=[];dropped=[];shifts={}
    for e in events:
        a,b=point(e.start),point(e.end)
        if a>duration or b>duration or e.kind=='hold' and b-a<.03-1e-8:
            dropped.append(dict(event_id=e.event_id,reason='collapsed_or_outside_hold_grid'));continue
        moved.append(replace(e,start=a,end=b));shifts[e.event_id]=max(abs(a-e.start),abs(b-e.end))
    moved,exclusive=prioritize_exclusive(moved,duration)
    # Explicit specials > holds > clicks. Within a kind prefer the least-shifted original.
    scored=[(e,(1 if e.exclusive else .8 if e.kind=='hold' else .5)-min(.1,shifts[e.event_id]/10)) for e in moved]
    selected,collisions=select_generated_candidates(scored,duration)
    selected,avoidance=move_hold_clicks(selected,duration)
    if not selected:raise ValueError('BPM吸附后无可用音符，请调整BPM或关闭吸附')
    if any(abs(point(t)-t)>1e-9 for e in selected for t in (e.start,e.end)):raise AssertionError('Post-grid rule moved endpoint off grid')
    if not audit_playability(selected,duration)['engineering_playtest_gate_passed']:raise AssertionError('Grid conflicts remain')
    return selected,dict(enabled=True,policy='studio-twelfth-beat-v2' if config['subdivisions']==12 else 'studio-quarter-beat-v1',settings=config,positions_per_beat=config['subdivisions'],
        step_seconds=float(step),storage_ms=1,max_storage_rounding_ms=.5,before=len(events),after=len(selected),
        max_shift_ms=round(max(shifts.values(),default=0)*1000,3),dropped=dropped,exclusive=exclusive,
        collisions=collisions,avoidance=avoidance)
