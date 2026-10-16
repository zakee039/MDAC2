"""Sustained-energy Hold proposals; energy is not note transcription."""
from array import array
from dataclasses import asdict,dataclass
from fractions import Fraction as F
import math
import sys
from mdac2.domain.artifact import content_hash,_hash
from mdac2.domain.validation import validate_chart


@dataclass(frozen=True,slots=True)
class SustainEvidence:
    audio_hash:str
    timing_fingerprint:str
    frames:tuple[tuple[F,F,float],...]
    policy_hash:str

    def __post_init__(self):
        for h in (self.audio_hash,self.timing_fingerprint,self.policy_hash):_hash(h)
        if type(self.frames) is not tuple or len(self.frames)>30000:raise ValueError('Bounded immutable sustain frames required')
        previous=None
        for frame in self.frames:
            if type(frame) is not tuple or len(frame)!=3:raise ValueError('Sustain frame required')
            a,b,v=frame
            if type(a) is not F or type(b) is not F or a<0 or b<=a or previous is not None and a!=previous:raise ValueError('Contiguous exact sustain frame coordinates required')
            if type(v) not in (int,float) or not math.isfinite(v) or not 0<=v<=1:raise ValueError('Bounded sustained energy required')
            previous=b

    @property
    def fingerprint(self):return content_hash({'audio':self.audio_hash,'timing':self.timing_fingerprint,'policy':self.policy_hash,'frames':[[str(a),str(b),v] for a,b,v in self.frames]})


def sustain_evidence(audio,pcm,timing):
    if type(pcm) is not bytes or len(pcm)!=audio.sample_count*audio.channels*2 or len(pcm)>128*1024*1024 or audio.duration>600 or audio.channels>8:raise ValueError('Bounded matching PCM16 required')
    if F(audio.sample_count,audio.sample_rate)!=timing.mapping.audio_duration:raise ValueError('Sustain audio/timing mismatch')
    values=array('h');values.frombytes(pcm)
    if sys.byteorder!='little':values.byteswap()
    hop=max(1,round(audio.sample_rate*.02));frames=[]
    for a in range(0,audio.sample_count,hop):
        b=min(audio.sample_count,a+hop);frame=values[a*audio.channels:b*audio.channels]
        rms=math.sqrt(sum((v/32768)**2 for v in frame)/len(frame))
        frames.append((F(a,audio.sample_rate),F(b,audio.sample_rate),min(1.,rms/.1)))
    return SustainEvidence(audio.audio_hash,timing.fingerprint,tuple(frames),content_hash({'version':'sustain-rms-1','hop_seconds':.02,'scale':.1,'channels':'mean_power'}))


@dataclass(frozen=True,slots=True)
class HoldPolicy:
    minimum_seconds:float=.3
    maximum_seconds:float=4.
    minimum_strength:float=.5
    tail_grid:F=F(1,4)
    version:str='hold-evidence-1'

    def __post_init__(self):
        if self.version!='hold-evidence-1' or any(type(v) not in (float,int) or not math.isfinite(v) for v in (self.minimum_seconds,self.maximum_seconds,self.minimum_strength)):raise ValueError('Finite Hold policy required')
        if not 0<self.minimum_seconds<=self.maximum_seconds<=8 or not 0<self.minimum_strength<=1 or type(self.tail_grid) is not F or not 0<self.tail_grid<=1 or self.tail_grid.denominator>192:raise ValueError('Bounded Hold policy required')

    @property
    def fingerprint(self):return content_hash({**asdict(self),'tail_grid':str(self.tail_grid)})


@dataclass(frozen=True,slots=True)
class HoldCandidate:
    event_id:str
    end_q:F|None
    reason:str
    evidence_hash:str
    policy_hash:str
    base_hash:str


def propose_holds(chart,timing,evidence,*,policy=HoldPolicy()):
    from bisect import bisect_right
    if type(evidence) is not SustainEvidence or evidence.timing_fingerprint!=timing.fingerprint or timing.review!='accepted':raise ValueError('Current approved sustained evidence required')
    if len(chart.events)>2048 or validate_chart(chart,timing.mapping,timing_revision=timing.revision_id,allow_empty=True).has_errors:raise ValueError('Invalid/oversized Hold input')
    starts=[a for a,_,_ in evidence.frames];result=[];base_hash=content_hash(chart.to_dict());evidence_hash=evidence.fingerprint
    for e in chart.events:
        end_q=None;reason='NO_SUSTAINED_SUPPORT'
        if e.locked or e.source_kind=='manual':reason='MANUAL_OR_LOCKED_PRESERVED'
        elif e.semantic!='tap' or len(e.lane_mask)!=1:reason='EXISTING_OR_DOUBLE_LANE_SEMANTIC_PRESERVED'
        else:
            head=timing.mapping.t(e.q);index=bisect_right(starts,head)-1;end=head
            if index>=0:
                for a,b,strength in evidence.frames[index:]:
                    if not a<=end<b or strength<policy.minimum_strength:break
                    end=min(b,head+F(str(policy.maximum_seconds)),timing.mapping.time_bounds[1])
                    if end<b or end-head>=F(str(policy.maximum_seconds)):break
            if end-head>=F(str(policy.minimum_seconds)):
                q=math.floor(timing.mapping.q(end)/policy.tail_grid)*policy.tail_grid
                if q>e.q and timing.mapping.t(q)-head>=F(str(policy.minimum_seconds)):
                    collisions=[other for other in chart.events if other.event_id!=e.event_id and set(e.lane_mask)&set(other.lane_mask) and
                        (e.q<=other.q<q or other.semantic=='hold' and other.q<=e.q<other.end_q)]
                    if collisions:reason='TAIL_OR_LANE_CONFLICT'
                    else:end_q=q;reason='SUSTAINED_ENERGY_CANDIDATE_REQUIRES_REVIEW'
                else:reason='TAIL_GRID_TOO_SHORT'
        result.append(HoldCandidate(e.event_id,end_q,reason,evidence_hash,policy.fingerprint,base_hash))
    return tuple(result)
