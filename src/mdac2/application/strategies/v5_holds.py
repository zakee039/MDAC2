"""Independent vocal/mix sustained-region detection; no learned weights change."""
from dataclasses import asdict, dataclass, replace
import numpy as np
from scipy.ndimage import uniform_filter1d
from mdac2.data_preparation.training_chart import TrainingEvent


@dataclass(frozen=True)
class HoldPolicy:
    minimum_vocal_seconds: float = .45
    minimum_mix_seconds: float = .70
    vocal_cosine: float = .88
    mix_cosine: float = .93
    reference_seconds: float = .12
    boundary_seconds: float = .08
    relative_floor: float = .004
    continuation_db: float = -12.
    maximum_seconds: float = 8.
    minimum_extension_seconds: float = .08
    # Weak syllables are still syllables. A phrase head often sits far below the
    # track's own 90th percentile, so the old flat .06 refused the opening line
    # of a verse while accepting its louder continuation. The gate now asks
    # whether energy rises into the region (a new attack) rather than how loud
    # the region is in absolute terms.
    vocal_relative_energy: float = .02
    mix_relative_energy: float = .12
    rise_ratio: float = 1.15
    rise_seconds: float = .10
    # A model hold can start after the syllable it belongs to. The head is pulled
    # back to the detected phrase start, at most this far, never over the
    # previous same-lane event.
    head_extension_seconds: float = .08
    head_extension_limit: float = .50


def detect(features, duration, policy=HoldPolicy()):
    """Fixed spectral anchors reject changing notes, even at constant energy.

    Each stream is scanned independently at the cached 100 Hz acoustic rate.
    Short attacks and silences do not become holds. A region ends at confirmed
    silence, spectral change, or an explicit bounded-analysis boundary.
    """
    f=np.asarray(features,dtype=np.float32)
    if f.ndim!=3 or f.shape[:2]!=(3,96) or not np.isfinite(f).all():
        raise ValueError('Finite three-stream 96-bin features required')
    if abs(f.shape[2]/100-duration)>.03:raise ValueError('Acoustic duration mismatch')
    power=np.expm1(np.clip(f,0,25))
    step=round(policy.rise_seconds*100)
    result=[]
    for source,name in ((1,'vocals'),(0,'mix')):
        # The running-sum filter can leave tiny negative residuals after a loud
        # region returns to silence. Power and square-root anchors stay >= 0.
        smooth=np.maximum(uniform_filter1d(power[source],5,axis=1,mode='nearest'),0)
        energy=smooth.sum(0)
        floor=max(1e-6,float(np.quantile(energy,.9))*policy.relative_floor)
        vectors=np.sqrt(np.maximum(smooth,0))
        vectors/=np.maximum(np.linalg.norm(vectors,axis=0),1e-12)
        minimum=round(100*(policy.minimum_vocal_seconds if source==1 else policy.minimum_mix_seconds))
        cosine=policy.vocal_cosine if source==1 else policy.mix_cosine
        reference=round(policy.reference_seconds*100);confirmation=round(policy.boundary_seconds*100)
        cursor=0;n=f.shape[2]
        while cursor+minimum<=n:
            if energy[cursor]<floor:cursor+=1;continue
            start=cursor
            ref=smooth[:,start:min(n,start+reference)].mean(1)
            template=np.sqrt(ref);template/=max(float(np.linalg.norm(template)),1e-12)
            ref_energy=max(float(ref.sum()),floor)
            limit=min(n,start+round(policy.maximum_seconds*100))
            good=(template@vectors[:,start:limit]>=cosine)&(energy[start:limit]>=max(floor,ref_energy*10**(policy.continuation_db/10)))
            bad=0;end=limit;reason='audio_end' if limit==n else 'analysis_limit'
            for local,ok in enumerate(good):
                bad=0 if ok else bad+1
                if bad>=confirmation:
                    end=start+local-bad+1;reason='acoustic_boundary';break
            if end-start>=minimum and float(good[:end-start].mean())>=.9:
                # How the region begins matters: a phrase head climbs out of a
                # quieter bed, a reverb tail only decays.
                before=float(energy[max(0,start-step):start].mean()) if start>0 else 0.
                after=float(energy[start:min(n,start+step)].mean()) if start<n else 0.
                rise=after/max(before,floor) if after else 0.
                result.append(dict(source=name,start=round(start/100,3),end=round(min(duration,end/100),3),
                    support=round(float(good[:end-start].mean()),4),boundary=reason,rise=round(rise,3),
                    relative_energy=round(float(energy[start:end].mean()/max(np.quantile(energy,.9),1e-12)),4)))
                cursor=max(end,start+1)
            else:cursor=start+max(1,min(reference,max(1,end-start)))
    return sorted(result,key=lambda r:(r['start'],r['source'])),dict(policy=asdict(policy),
        streams=['vocals','mix'],hop_seconds=.01,source_identity='acoustic evidence, not labeled source truth')


def refine(events, features, duration, policy=HoldPolicy()):
    regions,report=detect(features,duration,policy)
    holds=[e for e in events if e.kind=='hold'];changes=[];added=[];rejected=[]
    # A late head loses the syllable it was written for. Pull it back to the
    # detected phrase start when the model started clearly later, stopping at
    # the previous same-lane event so nothing is overlapped into.
    for index,h in enumerate(list(holds)):
        covering=[r for r in regions if r['start']<=h.start-policy.head_extension_seconds
                  and r['end']>=h.start and r['boundary']!='analysis_limit']
        if not covering:continue
        lead=max(covering,key=lambda r:r['start'])['start']
        lead=max(lead,h.start-policy.head_extension_limit,0.)
        previous=max([e.end for e in events if e.event_id!=h.event_id and set(e.lanes)&set(h.lanes)
                      and e.end<=h.start+.001]+[0.])
        if previous:lead=max(lead,previous+.02)
        if h.start-lead<policy.head_extension_seconds:continue
        holds[index]=replace(h,start=round(lead,3))
        changes.append(dict(event_id=h.event_id,kind='head',before=h.start,after=round(lead,3)))
    # Existing heads retain their model identity. Small holds are eligible too.
    for index,h in enumerate(list(holds)):
        matches=[r for r in regions if r['start']<=h.end-.03 and r['end']>=h.end+policy.minimum_extension_seconds
                 and r['end']>h.start and r['boundary']!='analysis_limit']
        if not matches:continue
        r=max(matches,key=lambda r:(r['end'],r['source']=='vocals'))
        end=r['end']
        blockers=[o.start for o in holds if o.event_id!=h.event_id and set(o.lanes)&set(h.lanes) and o.start>h.start]
        if blockers:end=min(end,min(blockers)-.501)
        if end<h.end+policy.minimum_extension_seconds:continue
        holds[index]=replace(h,end=round(end,3))
        changes.append(dict(event_id=h.event_id,kind='tail',before=h.end,after=round(end,3),evidence=r))
    # A detector can add an entirely missing region; it never needs a predicted
    # hold as a seed. Vocal candidates win duplicate coverage from the mix.
    for r in sorted(regions,key=lambda r:(r['source']!='vocals',r['start'])):
        a,b=r['start'],r['end']
        if r['boundary']=='analysis_limit':
            rejected.append(dict(region=r,reason='unconfirmed_bounded_endpoint'));continue
        threshold=policy.vocal_relative_energy if r['source']=='vocals' else policy.mix_relative_energy
        if r['relative_energy']<threshold and r['rise']<policy.rise_ratio:
            rejected.append(dict(region=r,reason='weak_decay_not_new_attack'));continue
        # Do not append a series of synthetic holds to fill an existing hold.
        covered=sum(max(0,min(b,h.end)-max(a,h.start)) for h in holds)
        if covered>=.5*(b-a):continue
        options=[]
        for lane in ('air','ground'):
            left,right=a,b
            for h in sorted(holds,key=lambda h:h.start):
                if lane not in h.lanes:continue
                if h.start<=left<h.end+.501:left=round(h.end+.501,3)
                elif left<h.start and right>h.start-.501:right=round(h.start-.501,3)
            minimum=policy.minimum_vocal_seconds if r['source']=='vocals' else policy.minimum_mix_seconds
            if right-left>=minimum:options.append((lane,left,right))
        if not options:
            rejected.append(dict(region=r,reason='existing_hold_occupancy'));continue
        lane,a,b=max(options,key=lambda item:(item[2]-item[1],-sum(item[0] in h.lanes for h in holds)))
        identity=f"v5-sustain-{r['source']}-{round(a*1000)}"
        holds.append(TrainingEvent(identity,'hold',(lane,),a,b,(identity,),'0F','v5-acoustic-'+r['source']))
        added.append(dict(event_id=identity,lane=lane,start=a,end=b,evidence=r))
    report.update(regions=regions,extended=[c for c in changes if c['kind']=='tail'],
        heads=[c for c in changes if c['kind']=='head'],added=added,rejected=rejected,
        before_holds=sum(e.kind=='hold' for e in events),after_holds=len(holds))
    return sorted([e for e in events if e.kind!='hold']+holds,key=lambda e:(e.start,e.event_id)),report
