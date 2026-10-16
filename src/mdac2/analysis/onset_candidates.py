"""Versioned PCM energy-rise baseline: candidates, never note-density decisions."""
from array import array
from dataclasses import asdict,dataclass
from fractions import Fraction
import math
from statistics import median
import sys
from mdac2.domain.artifact import content_hash,_hash
from mdac2.domain.timing import TimingError
from mdac2.domain.timing_edits import TimingRevision
from mdac2.ports.timing import TimingInput


@dataclass(frozen=True,slots=True)
class CandidatePolicy:
    version:str='energy-rise-1'
    hop_seconds:float=.01
    minimum_rise:float=.012
    relative_threshold:float=.12
    median_multiplier:float=3
    minimum_gap_seconds:float=.03
    grid_quarters:Fraction=Fraction(1,4)
    maximum_quantization_seconds:float=.035
    max_seconds:int=600

    def __post_init__(self):
        if self.version!='energy-rise-1':raise ValueError('Unknown candidate policy')
        for v in (self.hop_seconds,self.minimum_rise,self.relative_threshold,self.median_multiplier,self.minimum_gap_seconds,self.maximum_quantization_seconds):
            if type(v) not in (float,int) or not math.isfinite(v) or v<=0:raise ValueError('Positive finite candidate budget required')
        if not .002<=self.hop_seconds<=.1 or self.relative_threshold>1 or self.minimum_gap_seconds<self.hop_seconds:raise ValueError('Invalid temporal/relative threshold')
        if type(self.grid_quarters) is not Fraction or not 0<self.grid_quarters<=4 or self.grid_quarters.denominator>192:raise ValueError('Bounded rational grid required')
        if type(self.max_seconds) is not int or not 0<self.max_seconds<=600:raise ValueError('Bounded duration required')

    @property
    def fingerprint(self):return content_hash({**asdict(self),'grid_quarters':str(self.grid_quarters)})


@dataclass(frozen=True,slots=True)
class OnsetCandidate:
    candidate_id:str
    seconds:Fraction
    raw_q:Fraction|None
    quantized_q:Fraction|None
    quantization_error_seconds:Fraction|None
    score:float
    energy_rise:float
    resolution_seconds:Fraction
    evidence_role:str='mix'
    reasons:tuple[str,...]=()

    def __post_init__(self):
        _hash(self.candidate_id)
        if type(self.seconds) is not Fraction or self.seconds<0 or type(self.resolution_seconds) is not Fraction or self.resolution_seconds<=0:raise ValueError('Exact candidate time/resolution required')
        if any(v is not None and type(v) is not Fraction for v in (self.raw_q,self.quantized_q,self.quantization_error_seconds)):raise ValueError('Exact optional coordinates required')
        if (self.quantized_q is None)!=(self.quantization_error_seconds is None) or self.quantized_q is not None and self.raw_q is None:raise ValueError('Incomplete quantization evidence')
        if type(self.score) not in (float,int) or not math.isfinite(self.score) or not 0<=self.score<=1:raise ValueError('Bounded heuristic score required')
        if type(self.energy_rise) not in (float,int) or not math.isfinite(self.energy_rise) or self.energy_rise<0:raise ValueError('Finite onset energy required')
        if self.evidence_role!='mix' or type(self.reasons) is not tuple or any(type(r) is not str or not r for r in self.reasons):raise ValueError('Explicit mix evidence required')

    def to_dict(self):
        d=asdict(self)
        for k in ('seconds','raw_q','quantized_q','quantization_error_seconds','resolution_seconds'):d[k]=None if d[k] is None else str(d[k])
        d['reasons']=list(self.reasons)
        return d


@dataclass(frozen=True,slots=True)
class OnsetSet:
    audio_hash:str
    timing_fingerprint:str
    policy_hash:str
    candidates:tuple[OnsetCandidate,...]
    mode:str='mix_only'

    def __post_init__(self):
        for h in (self.audio_hash,self.timing_fingerprint,self.policy_hash):_hash(h)
        if self.mode!='mix_only' or type(self.candidates) is not tuple or any(type(c) is not OnsetCandidate for c in self.candidates):raise ValueError('Immutable mix candidates required')
        if any(a.seconds>=b.seconds for a,b in zip(self.candidates,self.candidates[1:])) or len({c.candidate_id for c in self.candidates})!=len(self.candidates):raise ValueError('Unique increasing candidates required')

    def to_dict(self):return {'schema_version':1,'audio_hash':self.audio_hash,'timing_fingerprint':self.timing_fingerprint,'policy_hash':self.policy_hash,'mode':self.mode,'candidates':[c.to_dict() for c in self.candidates]}
    @property
    def fingerprint(self):return content_hash(self.to_dict())


def quantize(seconds,timing,policy):
    try:raw=timing.mapping.q(seconds)
    except TimingError:return None,None,None,('OUTSIDE_TIMING_COVERAGE',)
    value=raw/policy.grid_quarters
    nearest=math.floor(value+Fraction(1,2))*policy.grid_quarters
    try:error=timing.mapping.t(nearest)-seconds
    except TimingError:return raw,None,None,('GRID_OUTSIDE_TIMING_COVERAGE',)
    if abs(error)>Fraction(str(policy.maximum_quantization_seconds)):return raw,None,None,('QUANTIZATION_LIMIT',)
    return raw,nearest,error,()


def detect_candidates(audio,pcm,timing,*,policy=CandidatePolicy()):
    if not isinstance(audio,TimingInput) or not isinstance(timing,TimingRevision) or not isinstance(policy,CandidatePolicy):raise ValueError('Typed audio/timing/policy required')
    if type(pcm) is not bytes or len(pcm)!=audio.sample_count*audio.channels*2:raise ValueError('Matching PCM16 required')
    if Fraction(audio.sample_count,audio.sample_rate)!=timing.mapping.audio_duration:raise ValueError('Timing duration differs from audio')
    if audio.duration>policy.max_seconds or audio.channels>8:raise ValueError('Onset analysis resource bound exceeded')
    values=array('h');values.frombytes(pcm)
    if sys.byteorder!='little':values.byteswap()
    hop=max(1,round(audio.sample_rate*policy.hop_seconds));energy=[]
    for start in range(0,audio.sample_count,hop):
        frame=values[start*audio.channels:min(audio.sample_count,start+hop)*audio.channels]
        # Squared channels retain anti-phase stereo transients.
        energy.append(math.sqrt(sum(v*v for v in frame)/len(frame))/32768)
    novelty=[max(0,v-(energy[i-1] if i else 0)) for i,v in enumerate(energy)]
    peak=max(novelty,default=0);chosen=[]
    for i,value in enumerate(novelty):
        local=novelty[max(0,i-50):min(len(novelty),i+51)]
        threshold=max(policy.minimum_rise,peak*policy.relative_threshold,median(local)*policy.median_multiplier)
        if value<threshold or i and value<=novelty[i-1] or i+1<len(novelty) and value<novelty[i+1]:continue
        if chosen and (i-chosen[-1])*hop/audio.sample_rate<policy.minimum_gap_seconds:
            if value>novelty[chosen[-1]]:chosen[-1]=i
        else:chosen.append(i)
    result=[]
    for i in chosen:
        seconds=Fraction(i*hop,audio.sample_rate);raw,q,error,reasons=quantize(seconds,timing,policy)
        identity=content_hash({'audio':audio.audio_hash,'timing':timing.fingerprint,'policy':policy.fingerprint,'sample':i*hop})
        result.append(OnsetCandidate(identity,seconds,raw,q,error,novelty[i]/peak,novelty[i],Fraction(hop,audio.sample_rate),
            reasons=('RULE_ENERGY_RISE_NOT_CALIBRATED_PROBABILITY',*reasons)))
    return OnsetSet(audio.audio_hash,timing.fingerprint,policy.fingerprint,tuple(result))
