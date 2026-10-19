"""Small, explainable chart-only heuristic. Not Euterpe ML or a calibrated rating."""
from bisect import bisect_left, bisect_right
import math

VERSION = 'mdac-rule-rating-v1'


def level(value):
    if isinstance(value, bool):
        raise ValueError('定级应为 1–20 之间的数字，最多一位小数')
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError('定级应为 1–20 之间的数字，最多一位小数') from None
    if not math.isfinite(number) or not 1 <= number <= 20 or abs(number * 10 - round(number * 10)) > 1e-6:
        raise ValueError('定级应为 1–20 之间的数字，最多一位小数')
    return f'{number:.1f}'


def percentile(values, fraction):
    values = sorted(values)
    return values[min(len(values)-1, int((len(values)-1)*fraction))] if values else 0.


def estimate(chart):
    from mdac2.data_preparation.training_chart import TrainingEvent
    duration = float(chart['duration'])
    if not math.isfinite(duration) or not 0 < duration <= 600:
        raise ValueError('Invalid chart duration')
    events = sorted((TrainingEvent.from_dict(e) for e in chart['events']), key=lambda e:e.start)
    if any(e.end > duration + .001 for e in events):
        raise ValueError('Events outside audio')
    times = [e.start for e in events]
    prefix = [0.]
    weights = dict(light=1., heavy=1.2, extra_heavy=1.6, gemini=1., hold=.8, gear=.5)
    for e in events:
        prefix.append(prefix[-1] + weights[e.kind]*len(e.lanes))
    def rates(window):
        # Uniform-time sampling, so dense sections do not also get more votes.
        width = min(window, duration)
        starts = {min(i*.5, max(0., duration-width)) for i in range(math.ceil(duration*2))}
        return [(prefix[bisect_right(times,s+width)]-prefix[bisect_left(times,s)])/width for s in starts]
    short, sustained = rates(2.), rates(10.)
    peak = 0.
    for i, t in enumerate(times):
        j = bisect_left(times, t+2.)
        peak = max(peak, (prefix[j]-prefix[i])/min(2., duration))
    lane_peak = 0.
    for lane in ('ground','air'):
        lane_times = [e.start for e in events if e.kind!='gear' and lane in e.lanes]
        for i,t in enumerate(lane_times):
            lane_peak = max(lane_peak, (bisect_left(lane_times,t+2.)-i)/min(2.,duration))
    singles = [e for e in events if e.kind in ('light','heavy')]
    pairs = [(a,b) for a,b in zip(singles,singles[1:]) if 0 < b.start-a.start <= .25]
    alternating = sum(a.lanes!=b.lanes for a,b in pairs)/max(1,len(singles)-1)
    # Count occupied-lane attacks, including simultaneous hold heads, in one sweep.
    occupied = 0; ends = dict(ground=-1., air=-1.)
    for e in sorted(events,key=lambda e:(e.start,e.kind!='hold')):
        if e.kind=='hold':
            for lane in e.lanes:ends[lane]=max(ends[lane],e.end)
        elif e.kind!='gear' and any(end>e.start for end in ends.values()):occupied+=1
    density = .45*percentile(short,.9)+.35*percentile(sustained,.9)+.2*peak
    anchors = [(0.,1.),(1.,3.),(2.,5.),(4.,7.),(6.,9.),(8.,10.5),(12.,12.5),(18.,15.)]
    base = 15.
    for (x0,y0),(x1,y1) in zip(anchors,anchors[1:]):
        if density <= x1:
            base = y0+(y1-y0)*(density-x0)/(x1-x0);break
    lane_load = min(1.,max(0.,lane_peak-3.)*.15)
    pattern_load = .6*alternating + .8*occupied/max(1,len(events))
    score = round(min(15.,max(1.,base+lane_load+pattern_load)),1)
    return dict(algorithm=VERSION, value=score, calibrated=False,
                features=dict(events=len(events),duration=duration,
                    weighted_density=round(density,3),peak_2s=round(peak,3),
                    sustained_10s_p90=round(percentile(sustained,.9),3),
                    lane_peak_2s=lane_peak,fast_alternation=round(alternating,3),
                    attacks_during_holds=occupied),
                components=dict(density=round(base,3),lane_load=round(lane_load,3),patterns=round(pattern_load,3)))
