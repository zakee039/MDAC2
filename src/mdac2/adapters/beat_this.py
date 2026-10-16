"""BeatThis integration boundary, tested with injected backends only.

There is deliberately no default upstream loader: upstream load_checkpoint can
fall back to a URL on FileNotFoundError. A real backend must be separately
validated as local-only before injection (T039).
"""
from dataclasses import dataclass,asdict
import re
from typing import Protocol
from mdac2.ports.timing import TimingInput,TimingObservation


@dataclass(frozen=True,slots=True)
class BeatThisOptions:
    device:str='cpu'
    precision:str='float32'
    postprocessor:str='minimal'

    def __post_init__(self):
        if not re.fullmatch(r'cpu|cuda:[0-9]+',self.device):raise ValueError('Explicit supported device required')
        if self.precision not in {'float32','float16'} or self.device=='cpu' and self.precision=='float16':raise ValueError('Unsupported precision/device')
        if self.postprocessor!='minimal':raise ValueError('DBN is not enabled in this contract')


class Backend(Protocol):
    name:str
    version:str
    def infer(self,*,checkpoint,pcm,sample_rate,channels,options,include_logits)->dict:...


class BeatThisProvider:
    def __init__(self,registry,model_id,backend:Backend,options=BeatThisOptions()):
        if not isinstance(options,BeatThisOptions):raise ValueError('BeatThis options required')
        if not all(type(v) is str and v.strip() for v in (backend.name,backend.version)):raise ValueError('Backend identity required')
        self.registry,self.model_id,self.backend,self.options=registry,model_id,backend,options

    def observe(self,audio:TimingInput,pcm:bytes,*,include_logits=False):
        if not isinstance(audio,TimingInput) or type(pcm) is not bytes or len(pcm)!=audio.sample_count*audio.channels*2:
            raise ValueError('Canonical interleaved PCM16 and matching audio dimensions required')
        if type(include_logits) is not bool:raise ValueError('Explicit frame output choice required')
        path,capability=self.registry.require(self.model_id)
        options=asdict(self.options)
        result=self.backend.infer(checkpoint=path,pcm=pcm,sample_rate=audio.sample_rate,channels=audio.channels,
            options=dict(options),include_logits=include_logits)
        if type(result) is not dict or set(result)!={'beat_times','downbeat_times','frame_logits','effective_options','model_hash','warnings'}:
            raise ValueError('Unknown backend result contract')
        if result['model_hash']!=capability.model_hash:raise ValueError('Backend weight identity mismatch')
        if result['effective_options']!=options:raise ValueError('Implicit backend fallback is forbidden')
        if (result['frame_logits'] is not None)!=include_logits:raise ValueError('Requested frame output contract mismatch')
        # Recheck after inference to reject changed/removed checkpoints.
        _,after=self.registry.require(self.model_id)
        if after!=capability:raise ValueError('Model capability changed during inference')
        producer={'name':'mdac2.beat_this/'+self.backend.name,'version':self.backend.version,
            'model_hash':capability.model_hash,'dependencies':dict(capability.dependencies)}
        warnings=list(result['warnings'])
        if not result['beat_times']:warnings.append('NO_BEATS_OBSERVED')
        return TimingObservation.create(audio,producer=producer,options={**options,'include_logits':include_logits,'adapter_version':'1'},
            beat_times=result['beat_times'],downbeat_times=result['downbeat_times'],frame_logits=result['frame_logits'],warnings=warnings)
