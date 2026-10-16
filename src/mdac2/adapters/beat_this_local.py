"""Explicit CPU-only BeatThis 1.1.0 backend; no network loader."""
import hashlib
import inspect
from importlib import metadata
import numpy as np
import torch
from beat_this.inference import Audio2Frames
from beat_this.model.beat_tracker import BeatThis
from beat_this.model.postprocessor import Postprocessor
from beat_this.preprocessing import LogMelSpect
from beat_this.utils import replace_state_dict_key


class LocalFrames(Audio2Frames):
    def __init__(self,path):
        self.device=torch.device('cpu');self.float16=False
        # File object + weights_only means a missing local path cannot become a URL.
        with path.open('rb') as stream:checkpoint=torch.load(stream,map_location='cpu',weights_only=True)
        hparams={k:v for k,v in checkpoint['hyper_parameters'].items() if k in inspect.signature(BeatThis).parameters}
        self.model=BeatThis(**hparams)
        state=replace_state_dict_key(checkpoint['state_dict'],'model.','')
        self.model.load_state_dict(state)
        self.model=self.model.to(self.device).eval()
        self.spect=LogMelSpect(device=self.device)


class LocalBeatThisBackend:
    name='beat-this-local-cpu'
    version=metadata.version('beat-this')

    def infer(self,*,checkpoint,pcm,sample_rate,channels,options,include_logits):
        if options!={'device':'cpu','precision':'float32','postprocessor':'minimal'}:raise ValueError('Only verified CPU float32 minimal profile enabled')
        torch.set_num_threads(2)
        frames=LocalFrames(checkpoint)
        signal=np.frombuffer(pcm,dtype='<i2').reshape(-1,channels).astype(np.float32)/32768
        beat_logits,downbeat_logits=frames(signal,sample_rate)
        processor=Postprocessor(type='minimal')
        beats,downbeats=processor(beat_logits,downbeat_logits)
        with checkpoint.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
        # Upstream default LogMelSpect uses 50 frames/second; verified installed
        # source fingerprint is recorded by the experiment runner.
        frame_data={'start_seconds':0,'step_seconds':.02,'beat':beat_logits.cpu().tolist(),'downbeat':downbeat_logits.cpu().tolist()} if include_logits else None
        return {'beat_times':beats.tolist(),'downbeat_times':downbeats.tolist(),'frame_logits':frame_data,
            'effective_options':options,'model_hash':digest,'warnings':['AUTOMATIC_OBSERVATION_REQUIRES_REVIEW']}
