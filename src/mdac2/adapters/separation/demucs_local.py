"""Verified official HTDemucs checkpoint, local-only weights-only CPU inference."""
from fractions import Fraction
import hashlib
import math
from importlib.metadata import version
import random
from mdac2.ports.stems import PCMStem

OFFICIAL_SHA256='8726e21a993978c7ba086d3872e7608d7d5bfca646ca4aca459ffda844faa8b4'


class LocalDemucsBackend:
    name='local-htdemucs';version='1';engine='demucs';engine_version='4.0.1'
    policy={'model_rate':44100,'resampler':'julius.resample_frac','input_length':'ceil','output_length':'exact-original',
        'mono':'duplicate-input-average-output','normalization':'all-channel-mean-std','dc_restore':'equal-share-four-stems',
        'pcm_quantization':'nearest-reject-overflow','cpu_threads':4,'max_seconds':60,'max_chunk_evaluations':512,'checkpoint_loader':'weights-only-fixed-allowlist'}

    def infer(self,*,checkpoint,audio,pcm,options):
        if options.get('allow_download') is not False or options.get('backend_policy')!=self.policy:raise ValueError('Explicit offline backend policy required')
        if options['device']!='cpu' or options['precision']!='float32' or options['output_format']!='pcm_s16le':raise ValueError('This verified backend supports CPU float32 PCM16 only')
        if audio.channels not in (1,2) or not 8000<=audio.sample_rate<=96000 or audio.duration>60 or len(pcm)!=audio.sample_count*audio.channels*2:raise ValueError('Demucs experimental input bounds exceeded')
        if not .1<=options['segment_seconds']<=7.8:raise ValueError('HTDemucs segment outside verified bounds')
        segment=int(44100*options['segment_seconds']);stride=int((1-options['overlap'])*segment)
        # Demucs adds up to half a second for each random-shift invocation.
        length=math.ceil(audio.sample_count*44100/audio.sample_rate)+(22050 if options['shifts'] else 0)
        if stride<1 or math.ceil(length/stride)*max(1,options['shifts'])>self.policy['max_chunk_evaluations']:
            raise ValueError('Demucs chunk evaluation limit exceeded')
        with checkpoint.open('rb') as stream:
            digest=hashlib.file_digest(stream,'sha256').hexdigest()
            if digest!=OFFICIAL_SHA256:raise ValueError('Unrecognized HTDemucs checkpoint')
            stream.seek(0)
            if version('demucs')!=self.engine_version:raise ValueError('Demucs engine version mismatch')
            import numpy as np
            import torch
            import julius
            from demucs.htdemucs import HTDemucs
            from demucs.apply import apply_model
            allowed=[HTDemucs,Fraction,np.dtype,(np._core.multiarray.scalar,'numpy.core.multiarray.scalar'),np.dtypes.Float64DType]
            with torch.serialization.safe_globals(allowed):
                package=torch.load(stream,map_location='cpu',weights_only=True)
        if package['klass'] is not HTDemucs or package['kwargs']['sources']!=['drums','bass','other','vocals']:raise ValueError('Unexpected checkpoint architecture or roles')
        threads=torch.get_num_threads();state=random.getstate()
        try:
            torch.set_num_threads(self.policy['cpu_threads']);random.seed(options['seed'])
            with torch.random.fork_rng(devices=[]),torch.inference_mode():
                torch.manual_seed(options['seed'])
                model=HTDemucs(*package['args'],**package['kwargs'])
                model.load_state_dict(package['state'],strict=True);model.eval()
                samples=np.frombuffer(pcm,dtype='<i2').reshape(-1,audio.channels).T.copy()
                mix=torch.from_numpy(samples).to(torch.float32)/32768
                if audio.channels==1:mix=mix.repeat(2,1)
                if audio.sample_rate!=44100:mix=julius.resample_frac(mix,audio.sample_rate,44100,full=True)
                mean=mix.mean();scale=mix.std(unbiased=False).clamp_min(1e-8)
                separated=apply_model(model,((mix-mean)/scale)[None],shifts=options['shifts'],split=True,
                    overlap=options['overlap'],progress=False,device='cpu',num_workers=0,segment=options['segment_seconds'])[0]
                separated=separated*scale+mean/len(model.sources)
                if audio.sample_rate!=44100:separated=julius.resample_frac(separated,44100,audio.sample_rate,output_length=audio.sample_count)
                if audio.channels==1:separated=separated.mean(dim=1,keepdim=True)
                if tuple(separated.shape)!=(4,audio.channels,audio.sample_count) or not torch.isfinite(separated).all():raise ValueError('Invalid native separated output')
                quantized=torch.round(separated*32768)
                if quantized.min()<-32768 or quantized.max()>32767:raise ValueError('Stem PCM overflow; no implicit clipping or independent rescaling')
                values=quantized.to(torch.int16).cpu().numpy()
                stems={role:PCMStem(role,values[i].T.astype('<i2',copy=False).tobytes(),audio.sample_rate,audio.sample_count,audio.channels) for i,role in enumerate(model.sources)}
        finally:
            random.setstate(state);torch.set_num_threads(threads)
        return {'stems':stems,'model_hash':digest,'effective_options':dict(options),'engine':self.engine,'engine_version':self.engine_version}
