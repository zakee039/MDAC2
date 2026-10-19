"""Non-destructive source intervals and bounded local structure suggestions."""
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import soundfile as sf
from mdac2.data_preparation.audio_features import sha


def settings(value, duration):
    if value is None:return dict(start_ms=0,end_ms=round(duration*1000))
    if not isinstance(value,dict) or set(value)!={'start_ms','end_ms'}:raise ValueError('裁剪设置无效')
    if any(type(v) not in (int,float) or not math.isfinite(v) or v!=round(v) for v in value.values()):raise ValueError('裁剪位置必须为整数毫秒')
    a,b=value['start_ms'],value['end_ms']
    if not 0<=a<b<=round(duration*1000) or b-a<min(1000,round(duration*1000)):raise ValueError('裁剪范围至少1秒，且必须位于原曲内')
    return dict(start_ms=int(a),end_ms=int(b))


def prepare(prepared, folder, interval):
    folder=Path(folder);source=Path(prepared['path']);record=source/'audio.json'
    if sha(record)!=prepared['record_sha256']:raise ValueError('原始音轨记录已变化')
    info=json.loads(record.read_text(encoding='utf-8'))
    if sha(source/'mix.wav')!=info['wav_sha256']:raise ValueError('原始音轨已变化')
    interval=settings(interval,info['duration'])
    x,rate=sf.read(source/'mix.wav',dtype='float32',always_2d=True)
    a=round(interval['start_ms']*rate/1000);b=min(len(x),round(interval['end_ms']*rate/1000))
    x=x[a:b];folder.mkdir(parents=True,exist_ok=True)
    if (folder/'audio.json').exists():
        saved=json.loads((folder/'audio.json').read_text(encoding='utf-8'))
        if saved.get('trim')!=interval or saved.get('parent_wav_sha256')!=info['wav_sha256'] or sha(folder/'mix.wav')!=saved['wav_sha256']:raise ValueError('裁剪缓存身份不匹配')
        return saved
    if (folder/'mix.wav').exists():raise ValueError('存在未完成裁剪缓存，请保留诊断后重试新目录')
    sf.write(folder/'mix.wav',x,rate,subtype='FLOAT')
    saved={**info,'wav_sha256':sha(folder/'mix.wav'),'pcm_sha256':hashlib.sha256(x.astype('<f4').tobytes()).hexdigest(),
        'duration':len(x)/rate,'sample_count':len(x),'peak':float(np.abs(x).max()),'trim':interval,
        'parent_wav_sha256':info['wav_sha256'],'pcm_policy':'source interval; nearest sample; no pad/gain/fade',
        'source_start_sample':a,'source_end_sample':b}
    (folder/'audio.json').write_text(json.dumps(saved),encoding='utf-8')
    return saved


def waveform(path, start=0., end=None, bins=1600):
    with sf.SoundFile(path) as f:
        duration=len(f)/f.samplerate;end=duration if end is None else min(duration,float(end))
        start=max(0.,float(start))
        if not math.isfinite(start) or not math.isfinite(end) or end<=start:raise ValueError('无效波形区间')
        f.seek(round(start*f.samplerate));x=f.read(min(len(f)-f.tell(),round((end-start)*f.samplerate)),dtype='float32',always_2d=True)
    # Peak over both channels avoids cancelling opposite-phase stereo.
    x=np.max(np.abs(x),axis=1);edges=np.linspace(0,len(x),min(bins,len(x))+1,dtype=int)
    return dict(start=start,end=end,duration=duration,peaks=[round(float(x[a:b].max()),5) for a,b in zip(edges,edges[1:])])


def suggest(path):
    """Explainable candidate, not a trained chorus classifier. Abstain if unclear."""
    import librosa
    from scipy.ndimage import gaussian_filter1d
    x,sr=sf.read(path,dtype='float32',always_2d=True)
    y=librosa.resample(x.mean(axis=1),orig_sr=sr,target_sr=11025);sr=11025;hop=5512
    duration=len(y)/sr
    if duration<40 or np.max(np.abs(y))<1e-5:return dict(candidate_ms=None,method='structure-candidate-v1',reason='音乐过短或过静，请手动设置出点')
    spectrum=np.abs(librosa.stft(y,n_fft=2048,hop_length=hop))
    chroma=librosa.feature.chroma_stft(S=spectrum**2,sr=sr,n_fft=2048,hop_length=hop)
    chroma/=np.maximum(np.linalg.norm(chroma,axis=0),1e-8)
    energy=gaussian_filter1d(np.sqrt(np.mean(spectrum**2,axis=0)),2)
    energy/=max(float(np.percentile(energy,90)),1e-8)
    similarity=chroma.T@chroma;n=len(energy);repeat=np.zeros(n)
    # Similar harmonic neighbourhoods at least 20 seconds apart support repeats.
    for i in range(8,n-8):
        candidates=[np.mean([similarity[i+k,j+k] for k in range(-8,9)]) for j in range(8,n-8,2) if abs(j-i)>40]
        repeat[i]=max(candidates,default=0)
    novelty=np.zeros(n)
    for i in range(8,n-8):novelty[i]=1-float(np.dot(chroma[:,i-8:i].mean(1),chroma[:,i:i+8].mean(1)))
    # First sustained, energetic, repeated passage; end must have a structural change.
    candidates=[]
    for i in range(40,min(n-8,360)):
        if i*hop/sr<35:continue
        body=slice(max(0,i-24),i)
        drop=float(energy[max(0,i-8):i].mean()-energy[i:i+8].mean())
        rep=float(repeat[body].mean());power=float(energy[body].mean())
        if rep>.72 and power>.65 and (drop>.12 or novelty[i]>max(.15,np.percentile(novelty,80))):
            candidates.append((i,rep,power,drop))
    if not candidates:return dict(candidate_ms=None,method='structure-candidate-v1',reason='未找到可靠的首段高潮结束候选，请试听后手动设置')
    first=candidates[0][0]
    i,rep,power,drop=max((c for c in candidates if c[0]<=first+12),key=lambda c:c[3]+novelty[c[0]])
    # Refine to local quiet edge, within +/- one second, never force beat snapping.
    center=i*hop/sr;lo=max(0,int((center-1)*sr));hi=min(len(y),int((center+1)*sr))
    frame=max(1,int(.02*sr));levels=[np.mean(y[a:a+frame]**2) for a in range(lo,hi,frame)]
    end=(lo+int(np.argmin(levels))*frame)/sr
    return dict(candidate_ms=round(end*1000),method='structure-candidate-v1',needs_review=True,
        reason='重复和声、持续能量与结构变化的本地候选；不保证等于副歌，请试听确认',
        evidence=dict(repetition=rep,energy=power,energy_drop=drop))
