"""Actual RoFormer on the exact Demucs smoke input, with network disabled."""
from pathlib import Path
from importlib.metadata import version
import json
import socket
import sys
import threading
import time
import wave
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'src'))
from mdac2.adapters.separation.audio_separator import AudioSeparatorProvider
from mdac2.adapters.separation.demucs import SeparationOptions
from mdac2.adapters.separation.roformer_local import LocalRoformerBackend,WEIGHTS_SHA256
from mdac2.application.stem_pipeline import publish_stems,load_stems
from mdac2.infrastructure.artifact_store import ArtifactStore
from mdac2.infrastructure.model_registry import ModelRegistry,ModelSpec
from mdac2.ports.timing import TimingInput


def main():
    import torch,numpy,julius,psutil
    from audio_separator.separator.uvr_lib_v5.roformer.mel_band_roformer import MelBandRoformer
    root=ROOT/'tools/private/separation';store=ArtifactStore(root/'smoke-store')
    manifest=json.loads((root/'roformer-manifest.json').read_text());previous=json.loads((root/'demucs-smoke.json').read_text())
    audio=store.get('audio',previous['audio_hash']);canonical=audio.payload['canonical']
    with wave.open(str(store._destination('audio',audio.artifact_hash)/'audio.wav'),'rb') as wav:pcm=wav.readframes(wav.getnframes())
    inp=TimingInput(audio.artifact_hash,canonical['sample_rate'],canonical['sample_count'],len(canonical['channel_layout']))
    deps=tuple((name,version(name)) for name in ('audio-separator','torch','torchaudio','numpy','julius','einops','librosa','rotary-embedding-torch','audioread'))
    spec=ModelSpec(manifest['model_id'],manifest['file'],WEIGHTS_SHA256,manifest['url'],'MIT',('inference',),deps)
    provider=AudioSeparatorProvider(ModelRegistry(root,[spec]),spec.model_id,LocalRoformerBackend(root/'roformer-author-config.yaml'),
        native_labels=('Vocals','Instrumental'),options=SeparationOptions(segment_seconds=3,overlap=.25))
    attempts=[]
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
        if attempts:raise AssertionError('RoFormer attempted network')
    finally:
        stop.set();watcher.join();socket.socket.connect,socket.socket.connect_ex,socket.create_connection,socket.getaddrinfo=originals
    report={'schema_version':1,'scope':'real-roformer-same-original-synthetic-3-second-smoke','music_quality_evaluated':False,
        'audio_hash':audio.artifact_hash,'bundle_hash':bundle.artifact_hash,'checkpoint_sha256':WEIGHTS_SHA256,'dependencies':dict(deps),
        'wall_seconds':elapsed,'sampled_peak_process_rss_bytes':peak[0],'network_attempts':len(attempts),'alignment':alignment,
        'tracks':[{'role':t.role,'provenance':t.provenance,'source_hashes':list(t.source_hashes),'sample_rate':t.sample_rate,'sample_count':t.sample_count,'rms':t.rms,'peak':t.peak} for t in tracks.tracks],
        'options':run.options}
    (root/'roformer-smoke.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
