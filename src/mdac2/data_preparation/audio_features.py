"""Offline, versioned three-stream features. Inference only; never optimizes."""
from pathlib import Path
import hashlib
import json
import math
import os
import subprocess
import time

FEATURE_CONFIG = {
    'schema_version':1,'streams':['mix','vocals','accompaniment'],
    'sample_rate':24000,'hop_samples':240,'n_fft':1024,'mel_bins':96,
    'fmin':30.,'fmax':12000.,'center':True,'window':'hann_periodic',
    'stereo':'mean_channel_power','scaling':'log1p(10000 * mel_power)',
    'mel_normalization':'slaney','storage':'float16',
    'separator':'MelBandRoformer-author-87201f4d',
    'native_stem':'vocals','accompaniment':'original_mix_minus_vocals_float32',
    'model_rate':44100,'chunk_samples':352800,'overlap':.25,
    'overlap_add':'uniform_normalized','precision':'autocast_bfloat16_float32_stft_and_output','device':'cuda',
    'inference_batch_size':4,
    'normalization':'none','inference_only':True,
}


def config_hash(config=FEATURE_CONFIG):
    return hashlib.sha256(json.dumps(config,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


class OfflineFeatureBuilder:
    def __init__(self, *, checkpoint, config_path, ffmpeg):
        import torch
        import yaml
        import librosa
        from importlib.metadata import version
        from mdac2.adapters.separation.roformer_local import WEIGHTS_SHA256, CONFIG_SHA256
        from audio_separator.separator.uvr_lib_v5.roformer.mel_band_roformer import MelBandRoformer
        self.torch=torch;self.ffmpeg=Path(ffmpeg)
        if not torch.cuda.is_available():raise RuntimeError('Verified CUDA preparation runtime required')
        if sha(checkpoint)!=WEIGHTS_SHA256 or sha(config_path)!=CONFIG_SHA256:
            raise ValueError('Separation model/config changed')
        if version('audio-separator')!='0.47.0':raise ValueError('Separator engine version changed')
        class Loader(yaml.SafeLoader):pass
        Loader.add_constructor('tag:yaml.org,2002:python/tuple',lambda loader,node:tuple(loader.construct_sequence(node)))
        config=yaml.load(Path(config_path).read_bytes(),Loader=Loader)['model']
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        torch.manual_seed(20260927)
        self.model=MelBandRoformer(**config,match_input_audio_length=True,zero_dc=False)
        self.model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True),strict=True)
        self.model.eval().cuda()
        self.mel=torch.from_numpy(librosa.filters.mel(sr=24000,n_fft=1024,n_mels=96,
            fmin=30.,fmax=12000.,htk=False,norm='slaney')).cuda()
        self.window=torch.hann_window(1024,device='cuda')
        self.environment={'torch':torch.__version__,'cuda':torch.version.cuda,
            'device':torch.cuda.get_device_name(),'audio_separator':version('audio-separator'),
            'numpy':version('numpy'),'librosa':version('librosa'),'julius':version('julius'),
            'ffmpeg_sha256':sha(ffmpeg),'weights_sha256':WEIGHTS_SHA256,'author_config_sha256':CONFIG_SHA256}

    def decode(self,path,limit_seconds=None):
        import numpy as np
        cmd=[str(self.ffmpeg),'-v','error','-i',str(path),'-map','0:a:0']
        if limit_seconds is not None:cmd+=['-t',str(limit_seconds)]
        cmd+=['-ar','44100','-ac','2','-f','f32le','pipe:1']
        proc=subprocess.run(cmd,capture_output=True,check=True,timeout=120)
        result=np.frombuffer(proc.stdout,dtype='<f4').reshape(-1,2).T.copy()
        if not result.size or result.shape[-1]>44100*1800 or not np.isfinite(result).all():
            raise ValueError('Invalid decoded song')
        return result

    def separate(self,mix):
        import numpy as np
        torch=self.torch;chunk=FEATURE_CONFIG['chunk_samples'];step=round(chunk*.75)
        length=mix.shape[-1];starts=list(range(0,max(1,length-chunk+1),step))
        if starts[-1]+chunk<length:starts.append(starts[-1]+step)
        vocals=np.zeros_like(mix);counts=np.zeros(length,dtype=np.float32)
        with torch.inference_mode():
            for batch_start in range(0,len(starts),FEATURE_CONFIG['inference_batch_size']):
                positions=starts[batch_start:batch_start+FEATURE_CONFIG['inference_batch_size']]
                pieces=[]
                for start in positions:
                    piece=torch.from_numpy(mix[:,start:min(length,start+chunk)]).cuda()
                    pieces.append(torch.nn.functional.pad(piece,(0,chunk-piece.shape[-1])))
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    prediction=self.model(torch.stack(pieces))
                if tuple(prediction.shape)!=(len(positions),2,chunk) or not torch.isfinite(prediction).all():
                    raise ValueError('Invalid model output')
                prediction=prediction.float().cpu().numpy()
                for j,start in enumerate(positions):
                    end=min(length,start+chunk)
                    vocals[:,start:end]+=prediction[j,:,:end-start]
                    counts[start:end]+=1
        if not (counts>0).all():raise ValueError('Uncovered output samples')
        vocals/=counts
        accompaniment=mix-vocals
        if not np.isfinite(vocals).all() or not np.isfinite(accompaniment).all():
            raise ValueError('Nonfinite separated audio')
        return vocals,accompaniment

    def features(self,streams,duration):
        import numpy as np
        import julius
        torch=self.torch;frames=math.ceil(duration*100-1e-8);values=[]
        with torch.inference_mode():
            for stream in streams:
                audio=torch.from_numpy(stream).cuda()
                audio=julius.resample_frac(audio,44100,24000,full=True)
                stft=torch.stft(audio,n_fft=1024,hop_length=240,win_length=1024,
                               window=self.window,center=True,return_complex=True)
                power=stft.abs().square().mean(dim=0)
                mel=torch.log1p(10000*(self.mel@power))[:,:frames]
                if mel.shape[-1]!=frames or not torch.isfinite(mel).all():
                    raise ValueError('Invalid feature geometry')
                values.append(mel.cpu().numpy().astype(np.float16))
        return np.stack(values)

    def process(self,row,destination,limit_seconds=None):
        import numpy as np
        import soundfile as sf
        torch=self.torch;started=time.perf_counter();torch.cuda.reset_peak_memory_stats()
        source=Path(row['audio_path_absolute'])
        if sha(source)!=row['audio_sha256']:raise ValueError('Source audio changed')
        mix=self.decode(source,limit_seconds)
        duration=mix.shape[-1]/44100
        if limit_seconds is None and abs(duration-row['pcm']['seconds'])>2/44100:
            raise ValueError('Decoded audio duration changed')
        decoded_at=time.perf_counter()
        vocals,accompaniment=self.separate(mix)
        separated_at=time.perf_counter()
        features=self.features((mix,vocals,accompaniment),duration)
        featured_at=time.perf_counter()
        destination=Path(destination);destination.mkdir(parents=True,exist_ok=False)
        np.save(destination/'features.npy',features,allow_pickle=False)
        sf.write(destination/'vocals.wav',vocals.T,44100,subtype='FLOAT')
        sf.write(destination/'accompaniment.wav',accompaniment.T,44100,subtype='FLOAT')
        reloaded,_=sf.read(destination/'vocals.wav',dtype='float32',always_2d=True)
        if not np.array_equal(reloaded.T,vocals):raise ValueError('Stem write verification failed')
        max_error=float(np.max(np.abs(mix-vocals-accompaniment)))
        if max_error>1e-6:raise ValueError('Stem reconstruction engineering check failed')
        values={'schema_version':1,'config':FEATURE_CONFIG,'config_sha256':config_hash(),
            'implementation_sha256':sha(Path(__file__)),
            'source_pcm_sha256':row['pcm']['sha256'],'source_audio_sha256':row['audio_sha256'],
            'duration':duration,'sample_count':mix.shape[-1],'feature_shape':list(features.shape),
            'frame_seconds':.01,'feature_origin_seconds':0.,'environment':self.environment,
            'files':{name:sha(destination/name) for name in ('features.npy','vocals.wav','accompaniment.wav')},
            'diagnostics':{'seconds':time.perf_counter()-started,'cuda_peak_bytes':torch.cuda.max_memory_allocated(),
                'decode_seconds':decoded_at-started,'separation_seconds':separated_at-decoded_at,
                'features_seconds':featured_at-separated_at,
                'reconstruction_max_error':max_error,'vocals_peak':float(np.max(np.abs(vocals))),
                'accompaniment_peak':float(np.max(np.abs(accompaniment))),
                'vocals_rms':float(np.sqrt(np.mean(vocals**2))),
                'accompaniment_rms':float(np.sqrt(np.mean(accompaniment**2)))},
            'human_quality_review':'pending','production_model_selected':False,'training_started':False,
            'partial_excerpt':limit_seconds is not None}
        (destination/'metadata.json').write_text(json.dumps(values,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        return values


def validate_cache(path, row, *, verify_hashes=True):
    import numpy as np
    path=Path(path);data=json.loads((path/'metadata.json').read_text(encoding='utf-8'))
    if data['config_sha256']!=config_hash() or data['config']!=FEATURE_CONFIG or data.get('implementation_sha256')!=sha(Path(__file__)) or data['source_pcm_sha256']!=row['pcm']['sha256'] or data['partial_excerpt']:
        raise ValueError('Incompatible feature cache')
    features=np.load(path/'features.npy',mmap_mode='r',allow_pickle=False)
    if list(features.shape)!=data['feature_shape'] or features.dtype!=np.float16 or not np.isfinite(features).all():
        raise ValueError('Corrupt feature tensor')
    if verify_hashes and any(sha(path/name)!=value for name,value in data['files'].items()):
        raise ValueError('Feature/stem checksum mismatch')
    return data
