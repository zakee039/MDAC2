"""Local-only paired engineering comparison; no human labels or quality ranking.

Prepare with the core environment and explicit user-project audio paths. Run each
backend in its own frozen environment. All audio and reports stay under private/.
"""
import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import socket
import sys
import threading
import time
import wave

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'src'))
from mdac2.adapters.audio_io import AudioDecoder
from mdac2.infrastructure.artifact_store import ArtifactStore

PRIVATE = ROOT / 'tools/private/separation/local-comparison'


def write(name, value):
    PRIVATE.mkdir(parents=True, exist_ok=True)
    (PRIVATE / name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def prepare(paths):
    ffmpeg = ROOT / 'tools/.deps/ffmpeg/extracted/ffmpeg-9.0.1-essentials_build/bin'
    decoder = AudioDecoder(ffmpeg / 'ffmpeg.exe', ffmpeg / 'ffprobe.exe')
    store = ArtifactStore(PRIVATE / 'store')
    clips = []
    for index, path in enumerate(paths):
        data = Path(path).read_bytes()
        original, _ = decoder.decode(data)
        transform = original.payload['transform']
        rate, count = transform['source_rate'], transform['source_sample_count']
        start = (30 + index * 15) * rate
        length = 8 * rate
        if start + length > count:
            raise ValueError('Requested eight-second excerpt outside input')
        audio = decoder.import_audio(data, store, crop_head=start, crop_tail=count-start-length, gain=.5)
        clips.append({'id': f'clip-{index+1}', 'source_path': str(Path(path).resolve()),
            'source_sha256': hashlib.sha256(data).hexdigest(), 'audio_hash': audio.artifact_hash,
            'transform': audio.payload['transform'], 'semantic_slice': 'unreviewed',
            'human_reference': None, 'pretrained_overlap': 'unknown'})
    manifest = {'schema_version': 1, 'scope': 'local-project-audio-engineering-only',
        'use_basis': 'Existing user project assets, analyzed locally under continued development instruction',
        'copyright_license': 'not asserted', 'redistribution': False,
        'source_gain': .5, 'reason_for_gain': 'Shared explicit headroom before both models; never per-stem normalization',
        'clips': clips}
    write('inputs.json', manifest)
    print(json.dumps({'prepared_clips': len(clips), 'audio_hashes': [c['audio_hash'] for c in clips]}))


def run(backend):
    import numpy as np
    import torch, julius, psutil
    from mdac2.adapters.separation.demucs import DemucsProvider, SeparationOptions
    from mdac2.application.stem_pipeline import publish_stems, load_stems
    from mdac2.infrastructure.model_registry import ModelRegistry, ModelSpec
    from mdac2.ports.timing import TimingInput
    model_root = ROOT / 'tools/private/separation'
    if backend == 'demucs':
        from demucs.apply import apply_model
        from demucs.htdemucs import HTDemucs
        from mdac2.adapters.separation.demucs_local import LocalDemucsBackend
        engine = LocalDemucsBackend()
        names = ('demucs', 'torch', 'torchaudio', 'numpy', 'julius', 'einops', 'omegaconf')
    else:
        from audio_separator.separator.uvr_lib_v5.roformer.mel_band_roformer import MelBandRoformer
        from mdac2.adapters.separation.roformer_local import LocalRoformerBackend
        from mdac2.adapters.separation.audio_separator import AudioSeparatorProvider
        engine = LocalRoformerBackend(model_root / 'roformer-author-config.yaml')
        names = ('audio-separator', 'torch', 'torchaudio', 'numpy', 'julius', 'einops', 'librosa', 'audioread', 'rotary-embedding-torch')
    model = json.loads((model_root / f'{backend}-manifest.json').read_text())
    deps = tuple((name, version(name)) for name in names)
    spec = ModelSpec(model['model_id'], model['file'], model['sha256'], model['url'], 'MIT', ('inference',), deps)
    registry = ModelRegistry(model_root, [spec])
    options = SeparationOptions(segment_seconds=7.8, overlap=.25)
    provider = (DemucsProvider(registry, spec.model_id, engine, options) if backend == 'demucs' else
        AudioSeparatorProvider(registry, spec.model_id, engine, native_labels=('Vocals', 'Instrumental'), options=options))
    raw_manifest = (PRIVATE / 'inputs.json').read_bytes()
    manifest = json.loads(raw_manifest)
    store = ArtifactStore(PRIVATE / 'store')
    attempts = []
    def denied(*args, **kwargs):
        attempts.append('NETWORK_ATTEMPT')
        raise RuntimeError('Inference network disabled')
    originals = socket.socket.connect, socket.socket.connect_ex, socket.create_connection, socket.getaddrinfo
    socket.socket.connect = socket.socket.connect_ex = socket.create_connection = socket.getaddrinfo = denied
    results = []
    try:
        for clip in manifest['clips']:
            audio = store.get('audio', clip['audio_hash'])
            c = audio.payload['canonical']
            with wave.open(str(store._destination('audio', audio.artifact_hash) / 'audio.wav'), 'rb') as wav:
                pcm = wav.readframes(wav.getnframes())
            inp = TimingInput(audio.artifact_hash, c['sample_rate'], c['sample_count'], len(c['channel_layout']))
            process = psutil.Process(); peak = [process.memory_info().rss]; stop = threading.Event()
            def monitor():
                while not stop.wait(.02):
                    peak[0] = max(peak[0], process.memory_info().rss)
            watcher = threading.Thread(target=monitor, daemon=True); watcher.start()
            started = time.perf_counter()
            row = {'id': clip['id'], 'audio_hash': audio.artifact_hash}
            try:
                output = provider.run(inp, pcm)
                row['inference_wall_seconds'] = time.perf_counter()-started
                artifact = publish_stems(store, audio, provider.profile, output.outputs, producer=output.producer, options=output.options)
                bundle, alignment = load_stems(store, artifact.artifact_hash)
                diagnostics = []
                for stem in output.outputs:
                    values = np.frombuffer(stem.pcm, dtype='<i2').astype(np.float64).reshape(-1, inp.channels)/32768
                    hop = inp.sample_rate//100
                    usable = len(values)//hop*hop
                    rms = np.sqrt(np.mean(values[:usable].reshape(-1, hop, inp.channels)**2, axis=(1,2)))
                    rise = np.maximum(0, np.diff(rms))
                    diagnostics.append({'role': stem.role, 'frame_rms_rise_sum_10ms': float(rise.sum()),
                        'max_frame_rms_rise_10ms': float(rise.max(initial=0))})
                row.update(status='succeeded', bundle_hash=artifact.artifact_hash, alignment=alignment,
                    tracks=[{'role': t.role, 'provenance': t.provenance, 'rms': t.rms, 'peak': t.peak,
                             'sample_count': t.sample_count, 'sample_rate': t.sample_rate} for t in bundle.tracks],
                    transient_diagnostics=diagnostics, effective_options=output.options)
            except Exception as exc:
                row.update(status='failed', error_type=type(exc).__name__, error=str(exc))
            finally:
                stop.set(); watcher.join(); peak[0] = max(peak[0], process.memory_info().rss)
            row.update(total_wall_seconds=time.perf_counter()-started, sampled_peak_process_rss_bytes=peak[0])
            results.append(row)
            print(json.dumps({k: row[k] for k in ('id', 'status', 'total_wall_seconds', 'sampled_peak_process_rss_bytes')}), flush=True)
            write(f'{backend}.json', {'schema_version': 1, 'backend': backend, 'input_manifest_sha256': hashlib.sha256(raw_manifest).hexdigest(),
                'dependencies': dict(deps), 'model_sha256': model['sha256'],
                'network_attempts': len(attempts), 'quality_evaluated': False, 'human_listening_performed': False,
                'diagnostic_scope': 'Energy rises are not onset accuracy; summed alignment does not establish individual stem quality',
                'results': results})
    finally:
        socket.socket.connect, socket.socket.connect_ex, socket.create_connection, socket.getaddrinfo = originals
    if attempts or any(row['status'] != 'succeeded' for row in results):
        raise SystemExit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('prepare', 'demucs', 'roformer'))
    parser.add_argument('paths', nargs='*')
    args = parser.parse_args()
    if args.stage == 'prepare':
        if not 1 <= len(args.paths) <= 10: parser.error('Provide one to ten explicit local project audio files')
        prepare(args.paths)
    else: run(args.stage)
