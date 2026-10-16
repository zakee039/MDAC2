"""One isolated model/sample run, writes observations and bounded metrics."""
import argparse
from dataclasses import asdict
from fractions import Fraction
import hashlib
import io
import wave
import importlib.metadata as metadata
import json
from pathlib import Path
import socket
import sys
import time

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'src'))
from mdac2.adapters.beat_this import BeatThisProvider
from mdac2.infrastructure.model_registry import ModelSpec,ModelRegistry
from mdac2.ports.timing import TimingInput
from mdac2.analysis.timing_fit import fit_timing
from mdac2.adapters.audio_io import AudioDecoder
from mdac2.infrastructure.artifact_store import ArtifactStore
from local_backend import LocalBeatThisBackend
from samples import generate
import mir_eval
import psutil
import beat_this.inference
import beat_this.preprocessing


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--model',choices=['final0','small0'],required=True)
    parser.add_argument('--case',choices=['fixed_4_4','pickup_4_4','compound_6_8'],required=True);args=parser.parse_args()
    root=ROOT/'tools/private/beat-this'
    record=next(r for r in json.loads((root/'download-manifest.json').read_text()) if r['model_id']==args.model)
    dependencies=tuple((n,metadata.version(n)) for n in ('beat-this','torch','torchaudio','numpy','soxr','einops','rotary-embedding-torch'))
    spec=ModelSpec(args.model,record['relative_path'],record['sha256'],record['source'],'MIT',('inference',),dependencies)
    pcm,rate,count,beats,downbeats=generate(args.case)
    wav=io.BytesIO()
    with wave.open(wav,'wb') as writer:
        writer.setparams((2,2,rate,0,'NONE','none'));writer.writeframes(pcm)
    binary=ROOT/'tools/.deps/ffmpeg/extracted/ffmpeg-9.0.1-essentials_build/bin'
    decoder=AudioDecoder(binary/'ffmpeg.exe',binary/'ffprobe.exe')
    audio_artifact=decoder.import_audio(wav.getvalue(),ArtifactStore(root/'sample-store'))
    audio=TimingInput(audio_artifact.artifact_hash,rate,count,2)
    provider=BeatThisProvider(ModelRegistry(root,[spec]),args.model,LocalBeatThisBackend())
    # The entire inference call fails if any code tries to open a socket.
    original=socket.socket
    def forbidden(*a,**k):raise RuntimeError('Network forbidden during inference')
    socket.socket=forbidden
    started=time.perf_counter()
    try:observation=provider.observe(audio,pcm,include_logits=True)
    finally:socket.socket=original
    elapsed=time.perf_counter()-started
    data=observation.payload
    import numpy as np
    fit=fit_timing(observation,pulse_unit=Fraction(3,8) if args.case=='compound_6_8' else Fraction(1,4),meter=(6,8) if args.case=='compound_6_8' else (4,4))
    metrics={'model':args.model,'case':args.case,'model_hash':record['sha256'],'audio_artifact_hash':audio.audio_hash,'audio_pcm_hash':hashlib.sha256(pcm).hexdigest(),
        'elapsed_seconds':elapsed,'peak_working_set_bytes':psutil.Process().memory_info().peak_wset,
        'device':'cpu','threads':2,'precision':'float32','network_blocked_during_inference':True,
        'beat_f_measure_70ms':mir_eval.beat.f_measure(beats,np.asarray(data['beat_times']),f_measure_threshold=.07),
        'downbeat_f_measure_70ms':mir_eval.beat.f_measure(downbeats,np.asarray(data['downbeat_times']),f_measure_threshold=.07),
        'beat_count':len(data['beat_times']),'downbeat_count':len(data['downbeat_times']),
        'reference_beats':beats.tolist(),'reference_downbeats':downbeats.tolist(),
        'fit_accepted':fit.accepted,'fit_reason':fit.reason,'fit_p95_seconds':fit.p95_seconds,'fit_max_seconds':fit.max_seconds,
        'dependencies':dict(dependencies),'source_files':{Path(module.__file__).name:hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() for module in (beat_this.inference,beat_this.preprocessing)}}
    stem=args.model+'-'+args.case
    (root/(stem+'.observation.json')).write_bytes(observation.encoded)
    (root/(stem+'.metrics.json')).write_text(json.dumps(metrics,indent=2),encoding='utf-8')
    print(json.dumps(metrics),flush=True)

if __name__=='__main__':main()
