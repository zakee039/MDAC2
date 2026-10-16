"""Variable semantic stems. Capability absence is never an empty audio track."""
from dataclasses import asdict, dataclass, fields
import math
import re
from mdac2.domain.artifact import _hash,content_hash


def role_name(value):
    if type(value) is not str or not re.fullmatch('[a-z][a-z0-9_]{0,31}',value):raise ValueError('Invalid stem role/capability')


def strings(values):
    if type(values) is not tuple or len(set(values))!=len(values):raise ValueError('Unique immutable names required')
    for value in values:role_name(value)


def decode(cls,value):
    if type(value) is not dict or set(value)!={f.name for f in fields(cls)}:raise ValueError('Unknown stem fields')
    return value


@dataclass(frozen=True,slots=True)
class OutputRole:
    role:str
    capabilities:tuple[str,...]

    def __post_init__(self):
        role_name(self.role);strings(self.capabilities)
        if not self.capabilities:raise ValueError('Explicit stem capability required')


@dataclass(frozen=True,slots=True)
class OutputProfile:
    profile_id:str
    roles:tuple[OutputRole,...]

    def __post_init__(self):
        if type(self.profile_id) is not str or not self.profile_id.strip():raise ValueError('Profile identity required')
        if type(self.roles) is not tuple or not self.roles or any(type(r) is not OutputRole for r in self.roles):raise ValueError('Immutable output roles required')
        if len({r.role for r in self.roles})!=len(self.roles):raise ValueError('Duplicate output roles')
        capabilities=[c for r in self.roles for c in r.capabilities]
        if len(capabilities)!=len(set(capabilities)):raise ValueError('Ambiguous capability mapping')

    def to_dict(self):return {'profile_id':self.profile_id,'roles':[{'role':r.role,'capabilities':list(r.capabilities)} for r in self.roles]}

    @classmethod
    def from_dict(cls,value):
        decode(cls,value)
        if type(value['roles']) is not list:raise ValueError('Serialized role list required')
        roles=[]
        for r in value['roles']:
            decode(OutputRole,r)
            if type(r['capabilities']) is not list:raise ValueError('Serialized capabilities required')
            roles.append(OutputRole(r['role'],tuple(r['capabilities'])))
        return cls(value['profile_id'],tuple(roles))


@dataclass(frozen=True,slots=True)
class StemTrack:
    role:str
    audio_hash:str
    pcm_sha256:str
    sample_rate:int
    sample_count:int
    channel_layout:tuple[str,...]
    rms:float
    peak:float
    provenance:str
    source_hashes:tuple[str,...]=()
    derivation:str|None=None
    offset_samples:int=0

    def __post_init__(self):
        role_name(self.role);_hash(self.audio_hash);_hash(self.pcm_sha256)
        if any(type(v) is not int or v<=0 for v in (self.sample_rate,self.sample_count)):raise ValueError('Positive sample dimensions required')
        if type(self.offset_samples) is not int:raise ValueError('Integer offset required')
        if type(self.channel_layout) is not tuple or not self.channel_layout or len(set(self.channel_layout))!=len(self.channel_layout) or any(type(c) is not str or not c for c in self.channel_layout):raise ValueError('Explicit unique channel layout required')
        if any(type(v) not in (float,int) or not math.isfinite(v) for v in (self.rms,self.peak)) or not 0<=self.rms<=self.peak<=1:raise ValueError('Normalized finite energy required')
        if type(self.source_hashes) is not tuple or len(set(self.source_hashes))!=len(self.source_hashes):raise ValueError('Immutable unique sources required')
        for h in self.source_hashes:_hash(h)
        if self.audio_hash in self.source_hashes:raise ValueError('Self-derived stem')
        if self.provenance=='model_output':
            if self.source_hashes or self.derivation is not None:raise ValueError('Native model output cannot claim a derivation')
        elif self.provenance=='derived':
            if not self.source_hashes or type(self.derivation) is not str or not self.derivation.strip():raise ValueError('Derived audio requires exact sources and recipe')
        else:raise ValueError('Unknown stem provenance')

    def to_dict(self):return {**asdict(self),'channel_layout':list(self.channel_layout),'source_hashes':list(self.source_hashes)}

    @classmethod
    def from_dict(cls,value):
        decode(cls,value)
        if any(type(value[k]) is not list for k in ('channel_layout','source_hashes')):raise ValueError('Serialized stem arrays required')
        return cls(**{**value,'channel_layout':tuple(value['channel_layout']),'source_hashes':tuple(value['source_hashes'])})


@dataclass(frozen=True,slots=True)
class CapabilityReport:
    available:bool
    missing_required:tuple[str,...]
    missing_optional:tuple[str,...]
    roles:tuple[tuple[str,str],...]


@dataclass(frozen=True,slots=True)
class StemBundle:
    audio_hash:str
    profile:OutputProfile
    tracks:tuple[StemTrack,...]
    schema_version:int=1

    def __post_init__(self):
        _hash(self.audio_hash)
        if type(self.schema_version) is not int or self.schema_version!=1:raise ValueError('Unknown stem bundle schema')
        if type(self.profile) is not OutputProfile or type(self.tracks) is not tuple or any(type(t) is not StemTrack for t in self.tracks):raise ValueError('Typed immutable stems required')
        if tuple(t.role for t in self.tracks)!=tuple(r.role for r in self.profile.roles):raise ValueError('Tracks must exactly match the ordered profile; no missing/extra roles')
        if len({t.audio_hash for t in self.tracks})!=len(self.tracks):raise ValueError('Distinct stem identities required')
        if len({(t.sample_rate,t.sample_count,t.channel_layout,t.offset_samples) for t in self.tracks})!=1:raise ValueError('Stems have incompatible time/channel coordinates')

    def require(self,required=(),optional=()):
        strings(required);strings(optional)
        if set(required)&set(optional):raise ValueError('Required/optional capabilities overlap')
        capabilities={c:r.role for r in self.profile.roles for c in r.capabilities}
        missing=tuple(c for c in required if c not in capabilities)
        return CapabilityReport(not missing,missing,tuple(c for c in optional if c not in capabilities),tuple((c,capabilities[c]) for c in (*required,*optional) if c in capabilities))

    def to_dict(self):return {'schema_version':self.schema_version,'audio_hash':self.audio_hash,'profile':self.profile.to_dict(),'tracks':[t.to_dict() for t in self.tracks]}

    @classmethod
    def from_dict(cls,value):
        decode(cls,value)
        if type(value['tracks']) is not list:raise ValueError('Serialized track list required')
        return cls(value['audio_hash'],OutputProfile.from_dict(value['profile']),tuple(StemTrack.from_dict(t) for t in value['tracks']),value['schema_version'])

    @property
    def fingerprint(self):return content_hash(self.to_dict())
