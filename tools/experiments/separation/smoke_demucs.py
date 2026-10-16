"""Real offline Demucs smoke on original synthetic PCM, not a music benchmark."""
from pathlib import Path
import hashlib
from importlib.metadata import version
import io
import json
import math
import socket
import struct
import sys
import tempfile
import threading
import time
import wave
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'src'))
from mdac2.adapters.audio_io import AudioDecoder
from mdac2.adapters.separation.demucs import DemucsProvider
from mdac2.adapters.separation.demucs_local import LocalDemucsBackend,OFFICIAL_SHA256
from mdac2.application.stem_pipeline import publish_stems,load_stems
from mdac2.infrastructure.artifact_store import ArtifactStore
from mdac2.infrastructure.model_registry import ModelRegistry,ModelSpec
from mdac2.ports.timing import TimingInput


def main():
    import torch
    import numpy
    import julius
    import psutil
    from demucs.apply import apply_model
    from demucs.htdemucs import HTDemucs
    private=ROOT/'tools/private/separation';manifest=json.loads((private/'demucs-manifest.json').read_text())
    deps=tuple((name,version(name)) for name in ('demucs','torch','torchaudio','numpy','julius','einops','omegaconf'))
    spec=ModelSpec(manifest['model_id'],manifest['file'],OFFICIAL_SHA256,manifest['url'],'MIT',('inference',),deps)
    provider=DemucsProvider(ModelRegistry(private,[spec]),spec.model_id,LocalDemucsBackend())
    rate=48000;count=rate*3;data=bytearray()
    for i in range(count):
        t=i/rate;envelope=min(1,t/.05,(3-t)/.05)
        pulse=.045*math.exp(-((t% .5)/.018)**2)*math.sin(2*math.pi*1300*t)
        a=round(32768*envelope*(.07*math.sin(2*math.pi*220*t)+.035*math.sin(2*math.pi*440*t)+pulse))
        b=round(32768*envelope*(.07*math.sin(2*math.pi*220*t+.2)+.035*math.sin(2*math.pi*660*t)+pulse))
        data.extend(struct.pack('<hh',a,b))
    stream=io.BytesIO()
    with wave.open(stream,'wb') as wav:wav.setparams((2,2,rate,0,'NONE','none'));wav.writeframes(data)
    ffmpeg=ROOT/'tools/.deps/ffmpeg/extracted/ffmpeg-9.0.1-essentials_build/bin'
    store=ArtifactStore(private/'smoke-store');audio=AudioDecoder(ffmpeg/'ffmpeg.exe',ffmpeg/'ffprobe.exe').import_audio(stream.getvalue(),store)
    with wave.open(str(store._destination('audio',audio.artifact_hash)/'audio.wav'),'rb') as wav:pcm=wav.readframes(wav.getnframes())
    inp=TimingInput(audio.artifact_hash,rate,count,2);attempts=[]
    def denied(*args,**kwargs):attempts.append('NETWORK_ATTEMPT');raise RuntimeError('Inference network disabled')
    originals=(socket.socket.connect,socket.socket.connect_ex,socket.create_connection,socket.getaddrinfo)
    socket.socket.connect=socket.socket.connect_ex=socket.create_connection=socket.getaddrinfo=denied
    peak=[0];stop=threading.Event();process=psutil.Process()
    def monitor():
        while not stop.wait(.02):peak[0]=max(peak[0],process.memory_info().rss)
    watcher=threading.Thread(target=monitor,daemon=True);watcher.start();started=time.perf_counter()
    try:
        run=provider.run(inp,pcm);elapsed=time.perf_counter()-started
        bundle=publish_stems(store,audio,provider.profile,run.outputs,producer=run.producer,options=run.options)
        tracks,alignment=load_stems(store,bundle.artifact_hash)
        failures={}
        with tempfile.TemporaryDirectory() as directory:
            registry=ModelRegistry(Path(directory),[spec]);test=DemucsProvider(registry,spec.model_id,LocalDemucsBackend())
            for case in ('missing','corrupt'):
                if case=='corrupt':(Path(directory)/spec.relative_path).write_bytes(b'not a checkpoint')
                try:test.run(inp,pcm)
                except ValueError as error:failures[case]=str(error)
                else:raise AssertionError('Invalid weights accepted')
        if attempts:raise AssertionError('Backend attempted network')
    finally:
        stop.set();watcher.join();socket.socket.connect,socket.socket.connect_ex,socket.create_connection,socket.getaddrinfo=originals
    report={'schema_version':1,'scope':'real-htdemucs-original-synthetic-3-second-smoke','music_quality_evaluated':False,
        'audio_hash':audio.artifact_hash,'bundle_hash':bundle.artifact_hash,'checkpoint_sha256':OFFICIAL_SHA256,'dependencies':dict(deps),
        'wall_seconds':elapsed,'sampled_peak_process_rss_bytes':peak[0],'network_attempts':len(attempts),'failures':failures,
        'alignment':alignment,'tracks':[{'role':t.role,'sample_rate':t.sample_rate,'sample_count':t.sample_count,'rms':t.rms,'peak':t.peak} for t in tracks.tracks],
        'normalization_policy':provider.backend.policy}
    (private/'demucs-smoke.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
