"""Pinned author MelBandRoformer through an explicit audio-separator engine."""
import hashlib
import math
from importlib.metadata import version
from pathlib import Path
from mdac2.ports.stems import PCMStem

WEIGHTS_SHA256='87201f4d31afb5bc79993230fc49446918425574db48c01c405e44f365c7559e'
CONFIG_SHA256='87aabb300193b019159269b69d6fe5f313aae4201a22b4af6268bc2846ab2fe1'


class LocalRoformerBackend:
    name='local-audio-separator';version='1';engine='audio-separator.MelBandRoformer';engine_version='0.47.0'
    policy={'config_sha256':CONFIG_SHA256,'native_model_roles':['vocals'],'model_rate':44100,
        'resampler':'julius.resample_frac','input_length':'ceil','output_length':'exact-original','mono':'duplicate-input-average-output',
        'normalization':'none','chunk_padding':'right-zero-to-configured-segment','overlap_add':'uniform-normalized',
        'match_input_audio_length':True,'zero_dc':False,'cpu_threads':4,'max_seconds':60,'max_chunks':512,
        'pcm_quantization':'nearest-reject-overflow','loader':'weights-only-direct-constructor-no-fallback',
        'derivations':{'accompaniment':{'operation':'mix_minus_vocals_pcm16','sources':['mix','vocals']}}}

    def __init__(self,config_path):self.config_path=Path(config_path)

    def infer(self,*,checkpoint,audio,pcm,options):
        if options.get('allow_download') is not False or options.get('backend_policy')!=self.policy:raise ValueError('Explicit offline RoFormer policy required')
        if options['device']!='cpu' or options['precision']!='float32' or options['output_format']!='pcm_s16le' or options['shifts']!=0:raise ValueError('Verified RoFormer supports CPU float32 without random shifts')
        if audio.channels not in (1,2) or not 8000<=audio.sample_rate<=96000 or audio.duration>60 or len(pcm)!=audio.sample_count*audio.channels*2:raise ValueError('RoFormer input resource bound exceeded')
        if not .1<=options['segment_seconds']<=8:raise ValueError('RoFormer segment outside verified bounds')
        length=math.ceil(audio.sample_count*44100/audio.sample_rate)
        chunk=round(options['segment_seconds']*44100);step=max(1,round(chunk*(1-options['overlap'])))
        positions=range(0,max(1,length-chunk+1),step)
        if len(positions)+int(positions[-1]+chunk<length)>self.policy['max_chunks']:raise ValueError('RoFormer chunk limit exceeded')
        config_bytes=self.config_path.read_bytes()
        if hashlib.sha256(config_bytes).hexdigest()!=CONFIG_SHA256:raise ValueError('RoFormer config identity mismatch')
        with checkpoint.open('rb') as stream:
            digest=hashlib.file_digest(stream,'sha256').hexdigest()
            if digest!=WEIGHTS_SHA256:raise ValueError('Unrecognized RoFormer checkpoint')
            stream.seek(0)
            if version('audio-separator')!=self.engine_version:raise ValueError('RoFormer engine version mismatch')
            import torch
            import numpy as np
            import yaml
            import julius
            from audio_separator.separator.uvr_lib_v5.roformer.mel_band_roformer import MelBandRoformer
            class AuthorConfigLoader(yaml.SafeLoader):pass
            AuthorConfigLoader.add_constructor('tag:yaml.org,2002:python/tuple',lambda loader,node:tuple(loader.construct_sequence(node)))
            config=yaml.load(config_bytes,Loader=AuthorConfigLoader)['model']
            if config['num_stems']!=1 or config['sample_rate']!=44100 or not config['stereo']:raise ValueError('Unexpected author model dimensions')
            weights=torch.load(stream,map_location='cpu',weights_only=True)
        threads=torch.get_num_threads()
        try:
            torch.set_num_threads(self.policy['cpu_threads'])
            with torch.random.fork_rng(devices=[]),torch.inference_mode():
                torch.manual_seed(options['seed'])
                model=MelBandRoformer(**config,match_input_audio_length=True,zero_dc=False)
                model.load_state_dict(weights,strict=True);del weights;model.eval()
                original=np.frombuffer(pcm,dtype='<i2').reshape(-1,audio.channels).T.copy()
                mix=torch.from_numpy(original).to(torch.float32)/32768
                if audio.channels==1:mix=mix.repeat(2,1)
                if audio.sample_rate!=44100:mix=julius.resample_frac(mix,audio.sample_rate,44100,full=True)
                length=mix.shape[-1];chunk=round(options['segment_seconds']*44100);step=max(1,round(chunk*(1-options['overlap'])))
                positions=range(0,max(1,length-chunk+1),step)
                if len(positions)>self.policy['max_chunks']:raise ValueError('RoFormer chunk limit exceeded')
                starts=list(positions)
                if starts[-1]+chunk<length:starts.append(starts[-1]+step)
                if len(starts)>self.policy['max_chunks']:raise ValueError('RoFormer chunk limit exceeded')
                vocals=torch.zeros_like(mix);counts=torch.zeros(length)
                for start in starts:
                    end=min(length,start+chunk);piece=mix[:,start:end]
                    if piece.shape[-1]<chunk:piece=torch.nn.functional.pad(piece,(0,chunk-piece.shape[-1]))
                    predicted=model(piece[None])[0]
                    if tuple(predicted.shape)!=(2,chunk) or not torch.isfinite(predicted).all():raise ValueError('Invalid native RoFormer output')
                    vocals[:,start:end]+=predicted[:,:end-start];counts[start:end]+=1
                if not (counts>0).all():raise ValueError('Uncovered RoFormer output samples')
                vocals/=counts
                if audio.sample_rate!=44100:vocals=julius.resample_frac(vocals,44100,audio.sample_rate,output_length=audio.sample_count)
                if audio.channels==1:vocals=vocals.mean(dim=0,keepdim=True)
                if tuple(vocals.shape)!=(audio.channels,audio.sample_count):raise ValueError('RoFormer output length mismatch')
                rounded=torch.round(vocals*32768)
                if not torch.isfinite(rounded).all() or rounded.min()<-32768 or rounded.max()>32767:raise ValueError('RoFormer PCM overflow; no implicit clipping')
                quantized=rounded.to(torch.int32).cpu().numpy()
                accompaniment=original.astype(np.int32)-quantized
                if min(quantized.min(),accompaniment.min())<-32768 or max(quantized.max(),accompaniment.max())>32767:raise ValueError('RoFormer PCM overflow; no implicit clipping')
                stems={name:PCMStem(role,value.T.astype('<i2').tobytes(),audio.sample_rate,audio.sample_count,audio.channels) for name,role,value in
                    (('Vocals','vocals',quantized),('Instrumental','accompaniment',accompaniment))}
        finally:torch.set_num_threads(threads)
        return {'stems':stems,'model_hash':digest,'effective_options':dict(options),'engine':self.engine,'engine_version':self.engine_version}
