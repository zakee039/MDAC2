"""Providers produce explicit ordered PCM16 outputs for atomic validation."""
from dataclasses import dataclass
from typing import Protocol
from mdac2.domain.stems import OutputProfile,role_name
from mdac2.ports.timing import TimingInput


@dataclass(frozen=True,slots=True)
class PCMStem:
    role:str
    pcm:bytes
    sample_rate:int
    sample_count:int
    channels:int
    offset_samples:int=0

    def __post_init__(self):
        role_name(self.role)
        if any(type(v) is not int or v<=0 for v in (self.sample_rate,self.sample_count,self.channels)):raise ValueError('Positive PCM dimensions required')
        if type(self.pcm) is not bytes or len(self.pcm)!=self.sample_count*self.channels*2:raise ValueError('Matching interleaved PCM16 required')
        if type(self.offset_samples) is not int:raise ValueError('Explicit sample offset required')


class StemProvider(Protocol):
    profile:OutputProfile
    def separate(self,audio:TimingInput,pcm:bytes)->tuple[PCMStem,...]:...
