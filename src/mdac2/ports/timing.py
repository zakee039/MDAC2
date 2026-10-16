"""Immutable raw tracker observations; no timing authority or review approval."""
from dataclasses import dataclass
import math
import re
from typing import Protocol
from mdac2.domain.artifact import ArtifactEnvelope,canonical_json,strict_json


def _seconds(value):
    if type(value) not in (int,float) or not math.isfinite(value) or value<0:
        raise ValueError('Finite nonnegative seconds required')
    return value


def _times(values,duration):
    if type(values) is not list:raise ValueError('Observation times must be a list')
    if any(_seconds(t)>duration for t in values) or any(a>=b for a,b in zip(values,values[1:])):
        raise ValueError('Observation times must increase inside audio')


@dataclass(frozen=True,slots=True)
class TimingInput:
    audio_hash:str
    sample_rate:int
    sample_count:int
    channels:int

    def __post_init__(self):
        if type(self.audio_hash) is not str or not re.fullmatch('[0-9a-f]{64}',self.audio_hash):raise ValueError('Audio artifact hash required')
        for value in (self.sample_rate,self.sample_count,self.channels):
            if type(value) is not int or value<=0:raise ValueError('Positive audio dimensions required')

    @property
    def duration(self):return self.sample_count/self.sample_rate


@dataclass(frozen=True,slots=True)
class TimingObservation:
    encoded:bytes

    def __post_init__(self):
        if type(self.encoded) is not bytes:raise ValueError('Immutable observation bytes required')
        envelope=ArtifactEnvelope.from_json(self.encoded)
        data=envelope.to_dict();p=envelope.payload
        if data['artifact_type']!='timing_observation' or len(data['inputs'])!=1:raise ValueError('Observation must bind one audio input')
        if type(p) is not dict or set(p)!={'schema_version','audio','beat_times','downbeat_times','frame_logits','warnings'} or type(p['schema_version']) is not int or p['schema_version']!=1:
            raise ValueError('Unknown observation schema')
        if type(p['audio']) is not dict or set(p['audio'])!={'audio_hash','sample_rate','sample_count','channels'}:raise ValueError('Invalid observation audio')
        audio=TimingInput(**p['audio'])
        if data['inputs']!=[audio.audio_hash]:raise ValueError('Audio input identity mismatch')
        _times(p['beat_times'],audio.duration);_times(p['downbeat_times'],audio.duration)
        if type(p['warnings']) is not list or any(type(v) is not str or not v.strip() for v in p['warnings']):raise ValueError('Invalid warnings')
        frames=p['frame_logits']
        if frames is not None:
            if type(frames) is not dict or set(frames)!={'start_seconds','step_seconds','beat','downbeat'}:raise ValueError('Unknown frame schema')
            start=_seconds(frames['start_seconds']);step=_seconds(frames['step_seconds'])
            if step<=0 or type(frames['beat']) is not list or type(frames['downbeat']) is not list or len(frames['beat'])!=len(frames['downbeat']):raise ValueError('Invalid frame axes')
            if not frames['beat'] or start+(len(frames['beat'])-1)*step>audio.duration+1e-9:raise ValueError('Frames outside audio')
            if any(type(v) not in (int,float) or not math.isfinite(v) for v in frames['beat']+frames['downbeat']):raise ValueError('Finite logits required')
        object.__setattr__(self,'encoded',envelope.to_json())

    @classmethod
    def create(cls,audio,*,producer,options,beat_times,downbeat_times,frame_logits=None,warnings=()):
        from dataclasses import asdict
        envelope=ArtifactEnvelope.create(artifact_type='timing_observation',producer=producer,options=options,inputs=[audio.audio_hash],
            payload={'schema_version':1,'audio':asdict(audio),'beat_times':list(beat_times),'downbeat_times':list(downbeat_times),'frame_logits':frame_logits,'warnings':list(warnings)})
        return cls(envelope.to_json())

    @property
    def artifact(self):return ArtifactEnvelope.from_json(self.encoded)
    @property
    def payload(self):return self.artifact.payload


class TimingProvider(Protocol):
    def observe(self,audio:TimingInput,pcm:bytes,*,include_logits:bool=False)->TimingObservation:...
