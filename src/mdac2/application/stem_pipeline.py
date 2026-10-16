"""One atomic bundle of PCM16 WAV sidecars with embedded audio identities."""
from array import array
from dataclasses import asdict,dataclass
import hashlib
import io
import math
import sys
import wave
from mdac2.domain.artifact import ArtifactEnvelope,canonical_json
from mdac2.domain.stems import StemBundle,StemTrack
from mdac2.ports.stems import PCMStem


@dataclass(frozen=True,slots=True)
class AlignmentPolicy:
    version:str='1'
    max_shift_samples:int=1
    search_samples:int=64
    window_samples:int=8192
    max_reconstruction_relative_rms:float=.5
    minimum_correlation:float=.5
    max_total_pcm_bytes:int=128*1024*1024

    def __post_init__(self):
        if self.version!='1':raise ValueError('Unknown alignment policy')
        if type(self.max_shift_samples) is not int or self.max_shift_samples<0:raise ValueError('Nonnegative alignment bound required')
        if type(self.search_samples) is not int or not self.max_shift_samples<self.search_samples<=256:raise ValueError('Bounded alignment search required')
        if type(self.window_samples) is not int or not 2*self.search_samples+2<=self.window_samples<=16384:raise ValueError('Bounded alignment window required')
        if type(self.max_reconstruction_relative_rms) not in (int,float) or not math.isfinite(self.max_reconstruction_relative_rms) or self.max_reconstruction_relative_rms<=0:raise ValueError('Finite reconstruction budget required')
        if type(self.minimum_correlation) not in (int,float) or not 0<self.minimum_correlation<=1:raise ValueError('Positive correlation threshold required')
        if type(self.max_total_pcm_bytes) is not int or not 0<self.max_total_pcm_bytes<=512*1024*1024:raise ValueError('Bounded PCM memory budget required')


def samples(pcm):
    data=array('h');data.frombytes(pcm)
    if sys.byteorder!='little':data.byteswap()
    return data


def alignment(reference,outputs,channels,policy):
    source=samples(reference);tracks=[samples(o.pcm) for o in outputs]
    # Energy is computed per interleaved channel, without stereo cancellation.
    reconstruction=array('i',(sum(t[i] for t in tracks) for i in range(len(source))))
    energy=sum(v*v for v in source)
    residual=sum((v-r)**2 for v,r in zip(source,reconstruction))
    if not energy:
        if any(reconstruction):raise ValueError('Non-silent reconstruction of silent input')
        return {'status':'silent_unmeasurable','shift_samples':None,'correlation':None,'relative_rms':0,
            'scope':'summed-output-only','warnings':['SILENCE_HAS_NO_ALIGNMENT_EVIDENCE']}
    relative=math.sqrt(residual/energy)
    if relative>policy.max_reconstruction_relative_rms:raise ValueError('Reconstruction exceeds policy budget')
    # Inspect the strongest channel near its strongest transient. This diagnostic
    # is bounded, and cannot prove that individual separated stems are correct.
    channel=max(range(channels),key=lambda c:sum(v*v for v in source[c::channels]))
    x=source[channel::channels];y=reconstruction[channel::channels]
    if len(x)<=2*policy.search_samples+2:raise ValueError('Audio too short to measure alignment')
    center=max(range(len(x)),key=lambda i:abs(x[i]))
    start=max(0,min(center-policy.window_samples//2,len(x)-policy.window_samples))
    stop=min(len(x),start+policy.window_samples)
    lo=start+policy.search_samples;hi=stop-policy.search_samples
    ref=x[lo:hi];norm=sum(v*v for v in ref)
    if not norm:raise ValueError('No alignment evidence in bounded window')
    scores=[]
    for lag in range(-policy.search_samples,policy.search_samples+1):
        shifted=y[lo+lag:hi+lag];power=sum(v*v for v in shifted)
        score=sum(a*b for a,b in zip(ref,shifted))/math.sqrt(norm*power) if power else 0
        scores.append((score,-abs(lag),lag))
    score,_,lag=max(scores)
    if score<policy.minimum_correlation or abs(lag)>policy.max_shift_samples:raise ValueError('Unverified or excessive stem alignment offset')
    return {'status':'measured','shift_samples':lag,'correlation':score,'relative_rms':relative,
        'scope':'summed-output-strongest-channel-window','window_start_sample':lo,'window_end_sample':hi,
        'warnings':['SUM_ALIGNMENT_DOES_NOT_PROVE_INDIVIDUAL_STEM_QUALITY']}


def validate_derivations(reference,outputs,derivations):
    if not derivations:return
    if derivations!={'accompaniment':{'operation':'mix_minus_vocals_pcm16','sources':['mix','vocals']}}:
        raise ValueError('Unsupported explicit stem derivation')
    roles={o.role:o for o in outputs}
    if not {'vocals','accompaniment'}<=roles.keys():raise ValueError('Missing derivation sources')
    mix=samples(reference);vocals=samples(roles['vocals'].pcm);accompaniment=samples(roles['accompaniment'].pcm)
    if len(mix)!=len(vocals) or len(mix)!=len(accompaniment) or any(m-v!=a for m,v,a in zip(mix,vocals,accompaniment)):
        raise ValueError('Derived stem differs from declared exact residual')


def publish_stems(store,audio,profile,outputs,*,producer,options,policy=AlignmentPolicy()):
    store.get('audio',audio.artifact_hash)
    canonical=audio.payload['canonical'];rate=canonical['sample_rate'];count=canonical['sample_count'];layout=tuple(canonical['channel_layout'])
    if type(outputs) is not tuple or any(type(o) is not PCMStem for o in outputs):raise ValueError('Immutable PCM outputs required')
    if tuple(o.role for o in outputs)!=tuple(r.role for r in profile.roles):raise ValueError('Provider output order/roles differ from profile')
    if len(outputs)>16 or count*len(layout)*2+sum(len(o.pcm) for o in outputs)>policy.max_total_pcm_bytes:raise ValueError('Stem PCM resource budget exceeded')
    for o in outputs:
        if (o.sample_rate,o.sample_count,o.channels,o.offset_samples)!=(rate,count,len(layout),0):raise ValueError('No implicit stem crop, pad, resample or delay compensation')
    with wave.open(str(store._destination('audio',audio.artifact_hash)/'audio.wav'),'rb') as wav:
        if (wav.getframerate(),wav.getnframes(),wav.getnchannels(),wav.getsampwidth())!=(rate,count,len(layout),2):raise ValueError('Input WAV differs from canonical identity')
        reference=wav.readframes(count)
    derivations=options.get('backend_policy',{}).get('derivations',{})
    validate_derivations(reference,outputs,derivations)
    if derivations and tuple(o.role for o in outputs)!=('vocals','accompaniment'):raise ValueError('Derived pair requires vocals-first source order')
    report=alignment(reference,outputs,len(layout),policy)
    if derivations:report['warnings'].append('RESIDUAL_RECONSTRUCTION_IS_AN_IDENTITY_NOT_SEPARATION_QUALITY')
    files={};assets=[];tracks=[]
    for output in outputs:
        values=samples(output.pcm);rms=math.sqrt(sum(v*v for v in values)/len(values))/32768;peak=max(abs(v) for v in values)/32768
        pcm_hash=hashlib.sha256(output.pcm).hexdigest()
        derived=output.role in derivations
        sources=[audio.artifact_hash,tracks[0].audio_hash] if derived else [audio.artifact_hash]
        asset=ArtifactEnvelope.create(artifact_type='stem_audio',producer=producer,options={**options,'role':output.role},inputs=sources,
            payload={'pcm_sha256':pcm_hash,'sample_rate':rate,'sample_count':count,'channel_layout':list(layout),'pcm_format':'s16le'})
        assets.append(asset.to_dict())
        tracks.append(StemTrack(output.role,asset.artifact_hash,pcm_hash,rate,count,layout,rms,peak,
            'derived' if derived else 'model_output',tuple(sources) if derived else (),derivations[output.role]['operation'] if derived else None))
        stream=io.BytesIO()
        with wave.open(stream,'wb') as writer:writer.setparams((len(layout),2,rate,0,'NONE','none'));writer.writeframes(output.pcm)
        files[output.role+'.wav']=stream.getvalue()
    bundle=StemBundle(audio.artifact_hash,profile,tuple(tracks))
    envelope=ArtifactEnvelope.create(artifact_type='stems',producer=producer,
        options={**options,'output_profile':profile.to_dict(),'alignment_policy':asdict(policy)},inputs=[audio.artifact_hash],
        payload={'bundle':bundle.to_dict(),'audio_assets':assets,'alignment':report,**({'derivations':derivations} if derivations else {}),
            'files':[{'name':name,'sha256':hashlib.sha256(blob).hexdigest(),'byte_count':len(blob)} for name,blob in sorted(files.items())]})
    # Embedded per-track audio envelopes are addressable within this bundle;
    # there are no separately visible half-published stem artifacts.
    store.publish(envelope,files)
    return envelope


def load_stems(store,identity):
    envelope=store.get('stems',identity);bundle=StemBundle.from_dict(envelope.payload['bundle'])
    assets=envelope.payload['audio_assets']
    if len(assets)!=len(bundle.tracks):raise ValueError('Incomplete embedded stem assets')
    if envelope.to_dict()['inputs']!=[bundle.audio_hash] or {f['name'] for f in envelope.payload['files']}!={t.role+'.wav' for t in bundle.tracks}:raise ValueError('Inconsistent bundle references')
    outputs=[];derivations=envelope.payload.get('derivations',{})
    for track,data in zip(bundle.tracks,assets):
        asset=ArtifactEnvelope.from_json(canonical_json(data))
        sources=[bundle.audio_hash]
        if track.role in derivations:
            sources.append(bundle.tracks[0].audio_hash)
            if track.provenance!='derived' or track.source_hashes!=tuple(sources) or track.derivation!='mix_minus_vocals_pcm16':raise ValueError('Invalid derived stem provenance')
        elif track.provenance!='model_output':raise ValueError('Unexplained derived stem')
        if asset.artifact_hash!=track.audio_hash or asset.to_dict()['artifact_type']!='stem_audio' or asset.to_dict()['inputs']!=sources:raise ValueError('Embedded audio identity mismatch')
        if asset.payload!={'pcm_sha256':track.pcm_sha256,'sample_rate':track.sample_rate,'sample_count':track.sample_count,'channel_layout':list(track.channel_layout),'pcm_format':'s16le'}:raise ValueError('Embedded audio metadata mismatch')
        with wave.open(str(store._destination('stems',identity)/(track.role+'.wav')),'rb') as wav:
            if (wav.getframerate(),wav.getnframes(),wav.getnchannels(),wav.getsampwidth())!=(track.sample_rate,track.sample_count,len(track.channel_layout),2):raise ValueError('Stem dimensions mismatch')
            pcm=wav.readframes(track.sample_count)
            if hashlib.sha256(pcm).hexdigest()!=track.pcm_sha256:raise ValueError('Stem PCM identity mismatch')
            if derivations:outputs.append(PCMStem(track.role,pcm,track.sample_rate,track.sample_count,len(track.channel_layout)))
    if derivations:
        store.get('audio',bundle.audio_hash)
        with wave.open(str(store._destination('audio',bundle.audio_hash)/'audio.wav'),'rb') as wav:reference=wav.readframes(wav.getnframes())
        validate_derivations(reference,outputs,derivations)
    return bundle,envelope.payload['alignment']
