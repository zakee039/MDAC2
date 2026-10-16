"""Continuous per-candidate evidence, never a whole-bar vocal gate."""
from dataclasses import asdict,dataclass
from fractions import Fraction
import hashlib
import math
from mdac2.application.stem_pipeline import samples
from mdac2.domain.artifact import content_hash
from mdac2.domain.stems import StemBundle
from mdac2.analysis.onset_candidates import OnsetSet


@dataclass(frozen=True,slots=True)
class EvidencePolicy:
    version:str='stem-energy-1'
    baseline_weight:float=.5
    vocal_weight:float=.3
    drum_weight:float=.2
    window_seconds:float=.04
    soft_energy_scale:float=.04
    required:tuple[str,...]=()

    def __post_init__(self):
        if self.version!='stem-energy-1':raise ValueError('Unknown stem evidence policy')
        values=(self.baseline_weight,self.vocal_weight,self.drum_weight,self.window_seconds,self.soft_energy_scale)
        if any(type(v) not in (float,int) or not math.isfinite(v) or v<0 for v in values):raise ValueError('Finite nonnegative evidence parameters required')
        if self.baseline_weight<=0 or not 0<self.window_seconds<=.5 or self.soft_energy_scale<=0:raise ValueError('Positive baseline/window/scale required')
        if type(self.required) is not tuple or len(set(self.required))!=len(self.required) or any(c not in ('vocals','drums') for c in self.required):raise ValueError('Explicit supported required capabilities')

    @property
    def fingerprint(self):return content_hash({**asdict(self),'required':list(self.required)})


@dataclass(frozen=True,slots=True)
class FusedCandidate:
    candidate_id:str
    score:float
    support:tuple[tuple[str,float],...]
    reasons:tuple[str,...]


@dataclass(frozen=True,slots=True)
class StemEvidence:
    onset_fingerprint:str
    stem_fingerprint:str
    policy_hash:str
    missing_optional:tuple[str,...]
    candidates:tuple[FusedCandidate,...]

    @property
    def fingerprint(self):
        return content_hash({'onsets':self.onset_fingerprint,'stems':self.stem_fingerprint,'policy':self.policy_hash,
            'missing_optional':list(self.missing_optional),'candidates':[{'candidate_id':c.candidate_id,'score':c.score,
                'support':[[role,v] for role,v in c.support],'reasons':list(c.reasons)} for c in self.candidates]})


def fuse_evidence(onsets,bundle,pcm_by_role,*,policy=EvidencePolicy()):
    if not isinstance(onsets,OnsetSet) or not isinstance(bundle,StemBundle) or not isinstance(policy,EvidencePolicy):raise ValueError('Typed evidence inputs required')
    if onsets.audio_hash!=bundle.audio_hash:raise ValueError('Stem evidence belongs to another audio')
    optional=tuple(c for c in ('vocals','drums') if c not in policy.required)
    capabilities=bundle.require(policy.required,optional)
    if not capabilities.available:raise ValueError('REQUIRED_STEM_CAPABILITY_MISSING')
    if type(pcm_by_role) is not dict or set(pcm_by_role)!={t.role for t in bundle.tracks}:raise ValueError('Exact bundle PCM inputs required')
    audio={}
    for track in bundle.tracks:
        pcm=pcm_by_role[track.role]
        if type(pcm) is not bytes or len(pcm)!=track.sample_count*len(track.channel_layout)*2 or hashlib.sha256(pcm).hexdigest()!=track.pcm_sha256:raise ValueError('Stem evidence PCM hash/size mismatch')
        if track.offset_samples!=0:raise ValueError('Align stems before evidence fusion')
        audio[track.role]=(track,samples(pcm))
    outputs=[];weights={'vocals':policy.vocal_weight,'drums':policy.drum_weight}
    for candidate in onsets.candidates:
        support=[];weighted=policy.baseline_weight*candidate.score;total=policy.baseline_weight
        for capability,role in capabilities.roles:
            track,values=audio[role];channels=len(track.channel_layout)
            if candidate.seconds>=Fraction(track.sample_count,track.sample_rate):raise ValueError('Candidate outside stem audio')
            center=round(candidate.seconds*track.sample_rate);width=max(1,round(policy.window_seconds*track.sample_rate))
            before=values[max(0,center-width)*channels:center*channels]
            after=values[center*channels:min(track.sample_count,center+width)*channels]
            def rms(frame):return math.sqrt(sum(v*v for v in frame)/len(frame))/32768 if frame else 0
            energy=max(0,rms(after)-rms(before))
            strength=energy/(energy+policy.soft_energy_scale)
            support.append((capability,strength));weighted+=weights[capability]*strength;total+=weights[capability]
        outputs.append(FusedCandidate(candidate.candidate_id,weighted/total,tuple(support),
            ('CONTINUOUS_LOCAL_ENERGY_NOT_SYLLABLE_DETECTION',*('MISSING_OPTIONAL_'+c.upper() for c in capabilities.missing_optional))))
    return StemEvidence(onsets.fingerprint,bundle.fingerprint,policy.fingerprint,capabilities.missing_optional,tuple(outputs))
