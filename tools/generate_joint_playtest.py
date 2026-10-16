"""Local new-song playtest with frozen joint weights; never trains or promotes."""
import argparse
from collections import Counter
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from mdac2.data_preparation.audio_features import sha

FFMPEG = ROOT/'tools/.deps/ffmpeg/extracted/ffmpeg-9.0.1-essentials_build/bin/ffmpeg.exe'
EXPERIMENT = ROOT/'tools/private/community/supervised-joint-v1-20260927'


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    os.replace(temporary, path)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def run_ffmpeg(arguments):
    return subprocess.run([str(FFMPEG), '-v', 'error', '-nostdin', *map(str, arguments)],
                          check=True, capture_output=True, timeout=180).stdout


def deduplicate_beats(observed):
    """Reject a close extra pulse only when a coherent complete beat grid exists.

    The grid is a candidate association device, not a hard quantizer: retained
    observations keep their original seconds. Failure leaves observations intact.
    """
    import numpy as np
    from scipy.optimize import minimize_scalar
    b=np.asarray(observed,dtype=np.float64)
    center=60/np.median(np.diff(b))
    objective=lambda bpm: -abs(np.exp(2j*np.pi*b*bpm/60).mean())
    grid=np.arange(center*.85,center*1.15,.01)
    candidate=min(grid,key=objective)
    optimum=minimize_scalar(objective,bounds=(candidate-.02,candidate+.02),method='bounded')
    bpm=float(optimum.x)
    phase=float(np.angle(np.exp(2j*np.pi*b*bpm/60).mean())/(2*np.pi))
    q=np.rint(b*bpm/60-phase).astype(int)
    groups=[np.flatnonzero(q==i) for i in np.unique(q)]
    complete=bool(np.all(np.diff(np.unique(q))==1))
    close=all(float(np.ptp(b[g]))<=.45*60/bpm for g in groups)
    eligible=bool(-optimum.fun>=.75 and complete and close and len(groups)>=.8*len(b))
    indices=[int(min(g,key=lambda k:abs(b[k]*bpm/60-phase-q[k]))) for g in groups] if eligible else list(range(len(b)))
    return b[indices],dict(applied=eligible,candidate_bpm=bpm,phase_seconds=phase*60/bpm,
        grid_coherence=float(-optimum.fun),complete_grid=complete,close_duplicate_groups=close,
        raw_count=len(b),retained_count=len(indices),removed_seconds=[float(b[i]) for i in range(len(b)) if i not in indices],
        policy='coherence >= .75; no missing pulse slots; duplicate span <= .45 period; retain >= .8 observations; nearest grid candidate retains original time')


def audio(args):
    import numpy as np
    import soundfile as sf
    if (args.output/'audio.json').exists():
        record = read(args.output/'audio.json')
        if sha(args.source) != record['source_sha256'] or sha(args.output/'mix.wav') != record['wav_sha256']:
            raise ValueError('Existing audio identity changed')
        return record
    run_ffmpeg(['-n', '-i', args.source, '-map', '0:a:0', '-vn', '-ar', 44100, '-ac', 2,
                '-map_metadata', -1, '-c:a', 'pcm_f32le', args.output/'mix.wav'])
    signal, rate = sf.read(args.output/'mix.wav', dtype='float32', always_2d=True)
    if rate != 44100 or signal.shape[1] != 2 or not np.isfinite(signal).all():
        raise ValueError('Invalid extracted audio')
    record = dict(source=str(args.source), source_sha256=sha(args.source), wav_sha256=sha(args.output/'mix.wav'),
                  pcm_sha256=hashlib.sha256(signal.astype('<f4').tobytes()).hexdigest(),
                  pcm_policy='44100 Hz stereo float32 interleaved; no crop/pad/gain',
                  sample_rate=rate, sample_count=len(signal), duration=len(signal)/rate,
                  peak=float(np.abs(signal).max()), ffmpeg_sha256=sha(FFMPEG))
    save(args.output/'audio.json', record)
    return record


def beats(args):
    import numpy as np
    from mdac2.adapters.beat_this_local import LocalBeatThisBackend
    from mdac2.analysis.timing_fit import fit_timing
    from mdac2.ports.timing import TimingInput, TimingObservation
    record = read(args.output/'audio.json')
    model = ROOT/'tools/private/beat-this/final0.ckpt'
    expected = next(r for r in read(model.parent/'download-manifest.json') if r['model_id']=='final0')
    if sha(model) != expected['sha256']:
        raise ValueError('Beat model hash mismatch')
    if (args.output/'timing.json').exists():
        cached = read(args.output/'timing.json')
        if cached['audio_sha256'] != record['wav_sha256'] or cached['model_sha256'] != expected['sha256']:
            raise ValueError('Stale beat estimate')
        return
    pcm = run_ffmpeg(['-i', args.output/'mix.wav', '-f', 's16le', '-ar', 44100, '-ac', 2, 'pipe:1'])
    backend = LocalBeatThisBackend()
    options = dict(device='cpu', precision='float32', postprocessor='minimal')
    result = backend.infer(checkpoint=model, pcm=pcm, sample_rate=44100, channels=2,
                           options=options, include_logits=False)
    observation = TimingObservation.create(TimingInput(record['wav_sha256'], 44100, len(pcm)//4, 2),
        producer=dict(name=backend.name, version=backend.version, model_hash=result['model_hash'], dependencies={}),
        options=options, beat_times=result['beat_times'], downbeat_times=result['downbeat_times'],
        warnings=result['warnings'])
    (args.output/'beat-observation.json').write_bytes(observation.encoded)
    fit = fit_timing(observation)
    observed = np.asarray(result['beat_times'], dtype=np.float64)
    if len(observed) < 6:
        raise ValueError('Too few beats for a reviewable automatic timing input')
    original_fit = fit
    deduplication = dict(applied=False)
    if not fit.accepted:
        observed,deduplication=deduplicate_beats(observed)
        cleaned = TimingObservation.create(TimingInput(**observation.payload['audio']),
            producer=dict(name='playtest-beat-candidate-deduplication',version='1',model_hash=result['model_hash'],dependencies={}),
            options=deduplication,beat_times=observed.tolist(),downbeat_times=[],
            warnings=['AUTOMATIC_CANDIDATE_DEDUPLICATION_REQUIRES_REVIEW'])
        (args.output/'beat-observation-deduplicated.json').write_bytes(cleaned.encoded)
        fit=fit_timing(cleaned)
    if fit.accepted:
        anchors = fit.mapping.anchors
        tempos = [dict(seconds=float(a.t), quarter=str(a.q), bpm=str(60*(b.q-a.q)/(b.t-a.t)))
                  for a,b in zip(anchors, anchors[1:])]
        policy = 'existing robust fixed/two-segment fitter; end slopes extend to audio boundaries'
    else:
        # Explicit experimental fallback, never marks a rejected fit as approved.
        # Consecutive observed pulses define local quarter-note intervals.
        tempos = [dict(seconds=float(a), quarter=str(i), bpm=str(Fraction(60)/Fraction(str(float(b-a)))))
                  for i,(a,b) in enumerate(zip(observed,observed[1:]))]
        policy = 'experimental observed-beat interpolation; consecutive pulses interpreted as quarters; edge slopes extended'
    value = dict(audio_sha256=record['wav_sha256'], model_sha256=expected['sha256'],
        beat_count=len(observed), raw_downbeat_count=len(result['downbeat_times']),
        median_bpm=float(60/np.median(np.diff(observed))), first_beat=float(observed[0]),
        last_beat=float(observed[-1]), fit_accepted=fit.accepted, fit_reason=fit.reason,
        fit_type=fit.model_type, fit_p95_seconds=fit.p95_seconds, fit_max_seconds=fit.max_seconds,
        raw_fit=dict(accepted=original_fit.accepted,p95_seconds=original_fit.p95_seconds,max_seconds=original_fit.max_seconds),
        deduplication=deduplication,
        timing_policy=policy, tempos=tempos, pulse_interpretation='quarter', meter_candidate=[4,4],
        human_confirmed=False, warnings=list(fit.warnings)+['AUTOMATIC_TIMING_USED_FOR_EXPLICIT_PLAYTEST'])
    save(args.output/'timing.json', value)
    print(json.dumps({k:v for k,v in value.items() if k!='tempos'}, ensure_ascii=False), flush=True)


def features(args):
    from mdac2.data_preparation.audio_features import OfflineFeatureBuilder, validate_cache
    record = read(args.output/'audio.json')
    row = dict(audio_path_absolute=str(args.output/'mix.wav'), audio_sha256=record['wav_sha256'],
               pcm=dict(sha256=record['pcm_sha256'], seconds=record['duration']))
    destination = args.output/'features'
    if destination.exists():
        validate_cache(destination, row)
        print('Reused hash-verified new-song features', flush=True)
        return
    engine = OfflineFeatureBuilder(checkpoint=ROOT/'tools/private/separation/MelBandRoformer.ckpt',
        config_path=ROOT/'tools/private/separation/roformer-author-config.yaml', ffmpeg=FFMPEG)
    stage = args.output/'features.staging'
    if stage.exists():
        raise ValueError('Inspect unfinished feature staging before retrying')
    record = engine.process(row, stage)
    os.replace(stage, destination)
    print(json.dumps(record['diagnostics']), flush=True)


def predict(args):
    import numpy as np
    import torch
    from mdac2.data_preparation import joint_decoding
    from mdac2.data_preparation.joint_model import JointChartModel
    from mdac2.data_preparation.supervised_training import seed_all
    from mdac2.data_preparation.training_tensors import timing_features, window_starts, slice_window
    from mdac2.data_preparation.training_chart import validate_events
    from mdac2.data_preparation.generated_constraints import POLICY_ID,constrain_decoded_events
    record = read(args.output/'audio.json')
    timing = read(args.output/'timing.json')
    frozen = read(args.frozen)
    features(args)  # Verify the complete new-song cache before consuming it.
    if timing['audio_sha256']!=record['wav_sha256'] or sha(args.output/'mix.wav')!=record['wav_sha256']:
        raise ValueError('Timing or waveform identity changed')
    if (args.output/'chart.json').exists() or (args.output/'prediction.npz').exists():
        raise ValueError('Prediction already exists; use a new version for another inference')
    if sha(args.checkpoint) != frozen['checkpoint_sha256'] or sha(Path(joint_decoding.__file__)) != frozen['decoder_code_sha256']:
        raise ValueError('Frozen model/decoder identity mismatch')
    seed_all(20260927)
    torch.set_num_threads(4)
    model = JointChartModel().cuda().eval()
    model.load_state_dict(torch.load(args.checkpoint, map_location='cpu', weights_only=False)['model'])
    feature = np.load(args.output/'features/features.npy', mmap_mode='r', allow_pickle=False)
    count = feature.shape[-1]
    beat = timing_features(dict(timing=timing), args.difficulty, count)
    np.save(args.output/'timing-input.npy', beat, allow_pickle=False)
    arrays = dict(onsets=np.zeros((count,2,6),np.float32), offsets=np.zeros((count,2,6),np.float32),
                  hold_ends=np.zeros((count,2),np.float32), hold_end_offsets=np.zeros((count,2),np.float32),
                  occupancy=np.zeros((count,2),np.float32), vocal_activity_prior=np.zeros(count,np.float32))
    coverage = np.zeros(count,np.uint8)
    starts = window_starts(count)
    for first in range(0,len(starts),8):
        part = starts[first:first+8]
        x = torch.from_numpy(np.stack([slice_window(feature,s,time_axis=2)[0] for s in part]).astype(np.float32)).cuda()
        t = torch.from_numpy(np.stack([slice_window(beat,s)[0] for s in part])).cuda()
        with torch.inference_mode(), torch.autocast('cuda',dtype=torch.bfloat16):
            out = model(x,t)
        out = {k:v.float() for k,v in out.items()}
        values = dict(onsets=out['onset_logits'].sigmoid().cpu().numpy(), offsets=out['offsets'].cpu().numpy(),
                      hold_ends=out['hold_end_logits'].sigmoid().cpu().numpy(),
                      hold_end_offsets=out['hold_end_offsets'].cpu().numpy(),
                      occupancy=out['occupancy_logits'].sigmoid().cpu().numpy(),
                      vocal_activity_prior=out['vocal_activity_prior'].cpu().numpy())
        if not all(np.isfinite(v).all() for v in values.values()):
            raise ValueError('Nonfinite prediction')
        for i,start in enumerate(part):
            joint_decoding.place_core(arrays,coverage,start,{k:v[i] for k,v in values.items()})
    if not np.all(coverage==1):
        raise ValueError('Incomplete center coverage')
    np.savez_compressed(args.output/'prediction.npz',**arrays)
    events,audit = joint_decoding.decode_song(arrays,record['duration'],frozen['decoder'])
    if validate_events(events,record['duration']):
        raise ValueError('Unresolved chart conflicts')
    save(args.output/'decoder-v1-proposals.json',dict(events=[e.to_dict() for e in events],audit=audit))
    events,selection=constrain_decoded_events(events,arrays,record['duration'],frozen['decoder'])
    audit=dict(base_decoder=audit,occupancy_selection=selection,accepted_count=len(events),
               cross_window_holds=sum(e.start//16!=e.end//16 for e in events if e.kind=='hold'))
    result = dict(title=args.title, creator=args.creator, scene='scene_01', duration=record['duration'],
                  generation_policy=POLICY_ID,
                  difficulty_input=args.difficulty, difficulty_human_rated=False,
                  checkpoint_sha256=sha(args.checkpoint), frozen_sha256=sha(args.frozen),
                  timing_sha256=sha(args.output/'timing.json'), prediction_sha256=sha(args.output/'prediction.npz'),
                  events=[e.to_dict() for e in events], audit=audit, center_coverage='exactly once',
                  windows=len(starts), class_counts=dict(Counter(e.kind for e in events)),
                  production_promoted=False, human_listening='pending', game_acceptance='pending')
    save(args.output/'chart.json',result)
    print(json.dumps(dict(events=len(events),classes=result['class_counts'],windows=len(starts),
                          cross_window_holds=audit['cross_window_holds']),ensure_ascii=False),flush=True)


def package(args):
    import numpy as np
    import soundfile as sf
    from mdac2.application.export_builder import default_cover
    from mdac2.data_preparation.training_chart import TrainingEvent
    from mdac2.data_preparation.six_export import write_bms, read_bms, semantics, PROFILE_ID
    from mdac2.data_preparation.playability_audit import audit_playability
    chart = read(args.output/'chart.json')
    record = read(args.output/'audio.json')
    events = [TrainingEvent.from_dict(e) for e in chart['events']]
    playability = audit_playability(events,record['duration'])
    save(args.output/'playability-audit.json',playability)
    if not playability['engineering_playtest_gate_passed']:
        raise ValueError('Playtest blocked: generated occupancy conflicts; see playability-audit.json')
    package_dir = args.output/'package'
    package_dir.mkdir(exist_ok=False)
    text = write_bms(events,record['duration'])
    for value in (args.title,args.creator):
        if any(c in value for c in '\r\n\0'):
            raise ValueError('Metadata must be one line')
    text = text.replace('#TITLE MDAC2 six-class review',f'#TITLE {args.title}')
    text = text.replace('#ARTIST MDAC2', '#ARTIST Unknown\n#LEVELDESIGN '+args.creator)
    text = text.replace('#PLAYLEVEL 9.5',f'#PLAYLEVEL {args.difficulty:g}')
    (package_dir/'map2.bms').write_text(text,encoding='utf-8')
    if semantics(read_bms(text,record['duration'])) != semantics(events):
        raise ValueError('BMS changed predicted events')
    run_ffmpeg(['-n','-i',args.output/'mix.wav','-map_metadata',-1,'-c:a','libvorbis','-q:a',6,package_dir/'music.ogg'])
    run_ffmpeg(['-n','-i',args.output/'mix.wav','-t',15,'-map_metadata',-1,'-c:a','libvorbis','-q:a',6,package_dir/'demo.ogg'])
    (package_dir/'cover.png').write_bytes(default_cover())
    metadata = dict(name=args.title,author='Unknown',levelDesigner=args.creator,scene='scene_01',
                    bpm='120',difficulty1='0',difficulty2=f'{args.difficulty:g}',difficulty3='0',difficulty4='0',unlockLevel=0)
    save(package_dir/'info.json',metadata)
    decoded = np.frombuffer(run_ffmpeg(['-i',package_dir/'music.ogg','-f','f32le','-ar',44100,'-ac',2,'pipe:1']),dtype='<f4').reshape(-1,2)
    original,_ = sf.read(args.output/'mix.wav',dtype='float32',always_2d=True)
    if abs(len(decoded)-len(original))>1024 or not np.isfinite(decoded).all():
        raise ValueError('Encoded music length/values changed')
    # Verify codec delay from actual waveform, independently of container timestamps.
    from scipy.signal import correlate, correlation_lags
    limit = min(len(original),44100*20)
    x=original[:limit].mean(axis=1);y=decoded[:limit].mean(axis=1)
    correlation=correlate(y,x,mode='full',method='fft'); lags=correlation_lags(len(y),len(x),mode='full')
    near=np.abs(lags)<=2048
    lag=int(lags[near][np.argmax(correlation[near])])
    if abs(lag)>2:
        raise ValueError(f'Encoded music has unexpected waveform offset: {lag} samples')
    name = f'{args.title}_mikee_MDAC2试玩.mdm' if args.creator=='mikee' else f'{args.title}_MDAC2试玩.mdm'
    target=args.output/name
    with zipfile.ZipFile(target,'x',compression=zipfile.ZIP_DEFLATED) as archive:
        for p in sorted(package_dir.iterdir()):
            archive.write(p,p.name)
    with zipfile.ZipFile(target) as archive:
        if archive.testzip() is not None or set(archive.namelist()) != {'info.json','map2.bms','music.ogg','demo.ogg','cover.png'}:
            raise ValueError('Package resources failed verification')
        for p in package_dir.iterdir():
            if archive.read(p.name)!=p.read_bytes():
                raise ValueError('Archived resource differs')
    report = dict(package=str(target),package_sha256=sha(target),bytes=target.stat().st_size,
        profile=PROFILE_ID,creator=args.creator,scene='scene_01',duration=record['duration'],
        logical_events=len(events),physical_lane_events=sum(len(e.lanes) for e in events),
        class_counts=chart['class_counts'],bms_semantic_roundtrip=True,
        music_length_error_samples=len(decoded)-len(original),music_alignment_offset_samples=lag,
        rules_passed=True,game_verified=False,human_listening='pending',production_promoted=False,
        bms_bpm_policy='120 BPM is a 1 ms timestamp representation, not the inferred musical BPM',
        resources={p.name:sha(p) for p in package_dir.iterdir()})
    save(args.output/'package-report.json',report)
    print(json.dumps(report,ensure_ascii=False),flush=True)


def main():
    sys.stdout.reconfigure(encoding='utf-8',errors='backslashreplace')
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['audio','beats','features','predict','package'],required=True)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--title',required=True)
    parser.add_argument('--creator',required=True)
    parser.add_argument('--difficulty',type=float,default=9.5)
    parser.add_argument('--checkpoint',type=Path,default=EXPERIMENT/'formal-lr3e4/best.pt')
    parser.add_argument('--frozen',type=Path,default=EXPERIMENT/'evaluation-lr3e4/frozen.json')
    args=parser.parse_args()
    args.source=args.source.resolve();args.output=args.output.resolve()
    if not 8.5<=args.difficulty<=10.5:
        raise ValueError('Difficulty outside trained range')
    args.output.mkdir(parents=True,exist_ok=True)
    identity=dict(source=str(args.source),source_sha256=sha(args.source),title=args.title,creator=args.creator,
                  difficulty=args.difficulty,checkpoint_sha256=sha(args.checkpoint),frozen_sha256=sha(args.frozen))
    identity_path=args.output/'request.json'
    if identity_path.exists() and read(identity_path)!=identity:
        raise ValueError('Output belongs to another generation request')
    if not identity_path.exists():save(identity_path,identity)
    started=time.perf_counter()
    globals()[args.stage](args)
    save(args.output/f'{args.stage}-status.json',dict(stage=args.stage,status='completed',
        seconds=time.perf_counter()-started,command=[sys.executable,*sys.argv],script_sha256=sha(Path(__file__))))


if __name__=='__main__':
    main()
