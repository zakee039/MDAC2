"""Demucs contract boundary. The injected backend must load local files only."""
from dataclasses import asdict,dataclass
import math
import re
from mdac2.domain.artifact import canonical_json,strict_json
from mdac2.domain.stems import OutputProfile,OutputRole
from mdac2.ports.stems import PCMStem
from mdac2.ports.timing import TimingInput


@dataclass(frozen=True,slots=True)
class SeparationOptions:
    device:str='cpu'
    precision:str='float32'
    segment_seconds:float=7.8
    overlap:float=.25
    shifts:int=0
    seed:int=0

    def __post_init__(self):
        if type(self.device) is not str or not re.fullmatch(r'cpu|cuda:[0-9]+',self.device):raise ValueError('Explicit device required')
        if self.precision not in ('float32','float16') or self.device=='cpu' and self.precision=='float16':raise ValueError('Unsupported precision')
        if any(type(v) not in (float,int) or not math.isfinite(v) for v in (self.segment_seconds,self.overlap)):raise ValueError('Finite options required')
        if not 0<self.segment_seconds<=60 or not 0<=self.overlap<1:raise ValueError('Invalid segment/overlap')
        if type(self.shifts) is not int or not 0<=self.shifts<=10 or type(self.seed) is not int or not 0<=self.seed<2**32:raise ValueError('Explicit bounded deterministic shift settings required')


@dataclass(frozen=True,slots=True)
class SeparationRun:
    outputs:tuple[PCMStem,...]
    identity:bytes

    @property
    def producer(self):return strict_json(self.identity)['producer']
    @property
    def options(self):return strict_json(self.identity)['options']


FOUR_STEMS=OutputProfile('demucs-four-pcm16-v1',tuple(OutputRole(r,(r,)) for r in ('drums','bass','other','vocals')))


class LocalSeparationBoundary:
    def __init__(self,registry,model_id,backend,*,profile,native_roles,options=SeparationOptions()):
        if not isinstance(options,SeparationOptions) or not isinstance(profile,OutputProfile):raise ValueError('Typed profile/options required')
        if type(native_roles) is not tuple or len(native_roles)!=len(profile.roles) or len(set(native_roles))!=len(native_roles) or any(type(r) is not str or not r for r in native_roles):raise ValueError('Explicit unique native output order required')
        for key in ('name','version','engine','engine_version'):
            if type(getattr(backend,key,None)) is not str or not getattr(backend,key).strip():raise ValueError('Wrapper and actual engine identities required')
        self.registry,self.model_id,self.backend=registry,model_id,backend
        self.profile,self.native_roles,self.options=profile,native_roles,options

    def run(self,audio,pcm):
        if not isinstance(audio,TimingInput) or type(pcm) is not bytes or len(pcm)!=audio.sample_count*audio.channels*2:raise ValueError('Matching canonical PCM16 required')
        path,before=self.registry.require(self.model_id)
        options={**asdict(self.options),'output_format':'pcm_s16le','allow_download':False,
            'backend_policy':strict_json(canonical_json(getattr(self.backend,'policy',{})))}
        result=self.backend.infer(checkpoint=path,audio=audio,pcm=pcm,options=dict(options))
        if type(result) is not dict or set(result)!={'stems','model_hash','effective_options','engine','engine_version'}:raise ValueError('Unknown separation result')
        if result['model_hash']!=before.model_hash or result['effective_options']!=options:raise ValueError('Model/precision/device fallback forbidden')
        if (result['engine'],result['engine_version'])!=(self.backend.engine,self.backend.engine_version):raise ValueError('Actual engine differs from declared backend')
        stems=result['stems']
        if type(stems) is not dict or set(stems)!=set(self.native_roles):raise ValueError('Missing/extra native stems')
        outputs=[]
        for native,role in zip(self.native_roles,self.profile.roles):
            stem=stems[native]
            if type(stem) is not PCMStem or stem.role!=role.role:raise ValueError('Incorrect explicit stem role mapping')
            if (stem.sample_rate,stem.sample_count,stem.channels,stem.offset_samples)!=(audio.sample_rate,audio.sample_count,audio.channels,0):raise ValueError('Output alignment/dimensions changed')
            outputs.append(stem)
        _,after=self.registry.require(self.model_id)
        if before!=after:raise ValueError('Separation weights changed during inference')
        producer={'name':'mdac2.separation/'+self.backend.name,'version':self.backend.version,'model_hash':before.model_hash,
            'dependencies':{**dict(before.dependencies),'engine':self.backend.engine,'engine_version':self.backend.engine_version}}
        return SeparationRun(tuple(outputs),canonical_json({'producer':producer,'options':{**options,'output_profile':self.profile.to_dict(),'native_roles':list(self.native_roles)}}))

    def separate(self,audio,pcm):return self.run(audio,pcm).outputs


class DemucsProvider(LocalSeparationBoundary):
    def __init__(self,registry,model_id,backend,options=SeparationOptions()):
        super().__init__(registry,model_id,backend,profile=FOUR_STEMS,native_roles=('drums','bass','other','vocals'),options=options)
