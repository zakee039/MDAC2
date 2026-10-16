"""Bounded bar-synchronous chroma, log-Mel and spectral-rise features."""
from dataclasses import asdict,dataclass
from fractions import Fraction
from importlib import metadata
import math
from mdac2.domain.artifact import ArtifactEnvelope
from mdac2.domain.structure import Span
from mdac2.domain.timing_edits import TimingRevision
from mdac2.ports.timing import TimingInput


@dataclass(frozen=True,slots=True)
class FeaturePolicy:
    version:str='bar-features-1'
    sample_rate:int=22050
    n_fft:int=1024
    hop_length:int=256
    n_mels:int=32
    cells_per_bar:int=16
    silence_rms:float=.002
    max_bars:int=512
    max_seconds:int=600
    max_spectral_cells:int=20000000

    def __post_init__(self):
        if self.version!='bar-features-1':raise ValueError('Unknown feature policy')
        for value in (self.sample_rate,self.n_fft,self.hop_length,self.n_mels,self.cells_per_bar,self.max_bars,self.max_seconds,self.max_spectral_cells):
            if type(value) is not int or value<=0:raise ValueError('Positive integer feature configuration required')
        if self.sample_rate!=22050 or self.n_fft not in (1024,2048) or not 128<=self.hop_length<=self.n_fft or not 12<=self.n_mels<=64 or self.cells_per_bar not in (8,16,32) or self.max_bars>512 or self.max_seconds>600:raise ValueError('Unsupported feature resource configuration')
        if type(self.silence_rms) not in (int,float) or not math.isfinite(self.silence_rms) or not 0<self.silence_rms<1:raise ValueError('Explicit silence threshold required')
        if self.max_spectral_cells>20000000:raise ValueError('Spectral memory limit exceeded')


def bar_spans(timing,policy):
    if timing.review!='accepted':raise ValueError('Approved timing required for structural features')
    mapping=timing.mapping;lo,hi=map(mapping.q,mapping.time_bounds);bars=[]
    meters=mapping.meter_segments
    for i,meter in enumerate(meters):
        left=lo if i==0 else max(lo,meter.start_q)
        right=min(hi,meters[i+1].start_q) if i+1<len(meters) else hi
        q=meter.start_q+math.ceil((left-meter.start_q)/meter.bar_quarters)*meter.bar_quarters
        while q+meter.bar_quarters<=right:
            bars.append((Span(q,q+meter.bar_quarters),(meter.numerator,meter.denominator)))
            if len(bars)>policy.max_bars:raise ValueError('Bar feature limit exceeded')
            q+=meter.bar_quarters
    if not bars:raise ValueError('No complete bar in approved timing coverage')
    return bars


def extract_features(audio,pcm,timing,*,policy=FeaturePolicy()):
    if not isinstance(audio,TimingInput) or not isinstance(timing,TimingRevision) or not isinstance(policy,FeaturePolicy):raise ValueError('Typed feature inputs required')
    if type(pcm) is not bytes or len(pcm)!=audio.sample_count*audio.channels*2:raise ValueError('Matching PCM16 required')
    if Fraction(audio.sample_count,audio.sample_rate)!=timing.mapping.audio_duration:raise ValueError('Audio/timing duration mismatch')
    if audio.duration>policy.max_seconds or audio.channels>2:raise ValueError('Feature audio resource bound exceeded')
    if len(pcm)>128*1024*1024 or (math.ceil(audio.duration*policy.sample_rate/policy.hop_length)+1)*(policy.n_fft//2+1)>policy.max_spectral_cells:
        raise ValueError('Feature matrix memory bound exceeded')
    bars=bar_spans(timing,policy)
    # Optional analysis dependencies import only when used, never in manual UI.
    import numpy as np
    import librosa
    from scipy.signal import resample_poly
    original=np.frombuffer(pcm,dtype='<i2').reshape(-1,audio.channels).astype(np.float32)/32768
    divisor=math.gcd(audio.sample_rate,policy.sample_rate)
    power=None
    for channel in range(audio.channels):
        signal=resample_poly(original[:,channel],policy.sample_rate//divisor,audio.sample_rate//divisor)
        spectrum=np.abs(librosa.stft(signal,n_fft=policy.n_fft,hop_length=policy.hop_length,center=True,pad_mode='constant'))**2
        if power is None:power=spectrum
        else:power+=spectrum
    power/=audio.channels
    mel=np.log1p(1000*librosa.feature.melspectrogram(S=power,sr=policy.sample_rate,n_fft=policy.n_fft,n_mels=policy.n_mels,fmin=30,fmax=policy.sample_rate/2,htk=False,norm='slaney'))
    chroma=librosa.feature.chroma_stft(S=power,sr=policy.sample_rate,n_fft=policy.n_fft,tuning=0,norm=2)
    onset=np.maximum(0,np.diff(mel,axis=1,prepend=mel[:,:1])).mean(axis=0)
    times=np.arange(power.shape[1])*policy.hop_length/policy.sample_rate
    def normalized(vector):
        norm=float(np.linalg.norm(vector))
        return (vector/norm).astype(float).tolist() if norm>1e-12 else [0.]*len(vector)
    rows=[]
    for span,meter in bars:
        cells=[]
        for i in range(policy.cells_per_bar):
            q0=span.start_q+(span.end_q-span.start_q)*Fraction(i,policy.cells_per_bar)
            q1=span.start_q+(span.end_q-span.start_q)*Fraction(i+1,policy.cells_per_bar)
            t0,t1=timing.mapping.t(q0),timing.mapping.t(q1)
            a,b=np.searchsorted(times,[float(t0),float(t1)],side='left')
            a=min(int(a),len(times)-1);b=max(a+1,min(int(b),len(times)))
            segment=original[max(0,round(t0*audio.sample_rate)):min(audio.sample_count,round(t1*audio.sample_rate))]
            rms=float(np.sqrt(np.mean(segment**2))) if segment.size else 0.
            silent=rms<policy.silence_rms
            cells.append({'start_q':str(q0),'end_q':str(q1),'start_seconds':str(t0),'end_seconds':str(t1),
                'silent':silent,'rms':rms,'chroma':[0.]*12 if silent else normalized(chroma[:,a:b].mean(axis=1)),
                'mel':[0.]*policy.n_mels if silent else normalized(mel[:,a:b].mean(axis=1)),
                'onset':0. if silent else float(onset[a:b].mean())})
        rows.append({'span':span.to_dict(),'meter':list(meter),'cells':cells,'silent':all(c['silent'] for c in cells)})
    producer={'name':'mdac2.librosa-bar-features','version':'1','model_hash':None,
        'dependencies':{name:metadata.version(name) for name in ('librosa','numpy','scipy','numba')}}
    return ArtifactEnvelope.create(artifact_type='similarity_features',producer=producer,inputs=[audio.audio_hash],
        options={**asdict(policy),'timing_fingerprint':timing.fingerprint,'channel_combination':'mean_channel_power',
            'resampling':'scipy-resample-poly','stft_center':True,'chroma_tuning':0,'normalization':'cell-l2'},
        payload={'schema_version':1,'audio_hash':audio.audio_hash,'timing_fingerprint':timing.fingerprint,
            'cells_per_bar':policy.cells_per_bar,'n_mels':policy.n_mels,'bars':rows,
            'warnings':['PARTIAL_EDGE_BARS_EXCLUDED','FEATURES_ARE_NOT_REUSE_PERMISSION']})
