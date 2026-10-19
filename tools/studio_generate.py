"""Durable studio worker: source audio -> R2/V2/V3 profiles -> independent export.

Adapter only. Never modifies the release-bound neural/postprocessing modules.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import sys
import time
import zipfile
from contextlib import nullcontext
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from generate_joint_playtest import audio,features as prepare_features,save,read,run_ffmpeg
from mdac2.data_preparation.audio_features import sha,validate_cache
from mdac2.application.studio_metadata import opening_talk,SCENES,clean,MODEL_LEVELS
from mdac2.application.studio_models import PROFILES,PIPELINE,select_onsets,postprocess


def progress(out,stage,message):save(out/'progress.json',dict(stage=stage,message=message,updated=time.time()))


def generate(request,out):
    import numpy as np
    import torch
    from mdac2.data_preparation.dual_models import OnsetModel
    from mdac2.data_preparation.arrangement_context_v6 import ContextArrangement,predict,decode
    from mdac2.data_preparation.training_tensors import window_starts,slice_window
    from mdac2.data_preparation.joint_decoding import place_core
    from mdac2.data_preparation.supervised_training import seed_all
    from mdac2.data_preparation.global_postprocess import process as v3_process
    from mdac2.data_preparation.playtest_postprocess import process as v2_process
    from mdac2.data_preparation.six_export import write_bms,read_bms,semantics
    from mdac2.data_preparation.playability_audit import audit_playability
    model_id=request['metadata']['model']
    if model_id not in PROFILES:raise ValueError('不支持的模型')
    progress(out,'identity','正在核验模型与音乐')
    manifest_path=ROOT/'models'/PROFILES[model_id]['base']/'manifest.json';manifest=read(manifest_path)
    assert sha(manifest_path)==request['model_manifest_sha256'],'模型版本已变化'
    for name,item in manifest['postprocessing']['sources'].items():assert sha(ROOT/name)==item['sha256'],name
    for c in manifest['checkpoints'].values():assert sha(ROOT/c['path'])==c['sha256']
    prior_path=ROOT/manifest['postprocessing']['prior_path'];assert sha(prior_path)==manifest['postprocessing']['prior_sha256']
    shared=request.get('studio_pipeline')
    if shared is not None:
        assert shared==PIPELINE,'未知工作台方案'
        common_path=ROOT/'models/v3/manifest.json';common=read(common_path)
        assert sha(common_path)==request['postprocess_manifest_sha256'],'全局后处理身份已变化'
        for name,item in common['postprocessing']['sources'].items():assert sha(ROOT/name)==item['sha256'],name
        prior_path=ROOT/common['postprocessing']['prior_path']
        assert sha(prior_path)==common['postprocessing']['prior_sha256']
    elif model_id=='r2':raise ValueError('R2需要当前工作台方案')
    source=Path(request['source']);assert sha(source)==request['source_sha256']
    device_name=request.get('device') or os.environ.get('MDAC2_DEVICE','cuda')
    if device_name not in ('cpu','cuda'):raise ValueError('Unsupported inference device')
    device=torch.device(device_name)
    if device.type=='cuda' and not torch.cuda.is_available():raise ValueError('CUDA不可用，请检查GPU环境后重试')
    cache=Path(request['cache']);cache.mkdir(parents=True,exist_ok=True)
    # Existing whole-song caches are read-only. Never use prepared training data.
    from mdac2.application.studio_trim import settings as trim_settings,prepare as prepare_trim
    prepared=request.get('prepared_audio')
    trim=request.get('trim')
    cropped=bool(trim and prepared and trim!=trim_settings(None,prepared['duration']))
    cached=None
    for p in (ROOT/'tools/private/playtests').glob('*/audio.json'):
        r=read(p)
        if not cropped and r.get('source_sha256')==request['source_sha256'] and (p.parent/'features/metadata.json').exists():
            assert sha(p.parent/'mix.wav')==r['wav_sha256']
            validate_cache(p.parent/'features',dict(pcm=dict(sha256=r['pcm_sha256'])))
            cached=p.parent;break
    if cached is None:
        progress(out,'audio','正在准备音频')
        if (cache/'mix.wav').exists() and not (cache/'audio.json').exists():
            (cache/'mix.wav').rename(cache/f'mix-interrupted-{time.time_ns()}.wav')
        prepared=request.get('prepared_audio')
        if cropped:
            prepare_trim(prepared,cache,trim)
        if prepared and not cropped and not (cache/'audio.json').exists():
            ready=Path(prepared['path']);metadata=ready/'audio.json'
            assert sha(metadata)==prepared['record_sha256'],'导入音轨记录已变化'
            info=read(metadata)
            assert info['source_sha256']==request['source_sha256'] and sha(ready/'mix.wav')==info['wav_sha256'],'导入音轨校验失败'
            shutil.copyfile(ready/'mix.wav',cache/'mix.wav')
            save(cache/'audio.json',info)
        record=audio(SimpleNamespace(source=source,output=cache))
        if not 0<record['duration']<=600:raise ValueError('支持10分钟以内的音乐')
        progress(out,'features','正在分离人声并提取特征；完成后可复用')
        if (cache/'features.staging').exists():(cache/'features.staging').rename(cache/f'features-interrupted-{time.time_ns()}')
        if device.type=='cpu':
            from mdac2.application.studio_cpu_features import CPUFeatureBuilder
            row=dict(audio_path_absolute=str(cache/'mix.wav'),audio_sha256=record['wav_sha256'],
                     pcm=dict(sha256=record['pcm_sha256'],seconds=record['duration']))
            CPUFeatureBuilder(checkpoint=ROOT/'tools/private/separation/MelBandRoformer.ckpt',
                              config_path=ROOT/'tools/private/separation/roformer-author-config.yaml',
                              ffmpeg=ROOT/'tools/.deps/ffmpeg/extracted/ffmpeg-9.0.1-essentials_build/bin/ffmpeg.exe').process(row,cache/'features')
        else:
            prepare_features(SimpleNamespace(output=cache))
        cached=cache
    record=read(cached/'audio.json')
    if not 0<record['duration']<=600:raise ValueError('支持10分钟以内的音乐')
    if device.type=='cpu':
        from mdac2.application.studio_cpu_features import validate_cpu_cache
        validate_cpu_cache(cached/'features',dict(pcm=dict(sha256=record['pcm_sha256'])))
    else:
        validate_cache(cached/'features',dict(pcm=dict(sha256=record['pcm_sha256'])))
    f=np.load(cached/'features/features.npy',mmap_mode='r',allow_pickle=False);duration=record['duration']
    seed_all(20260928);torch.set_num_threads(4)
    
    progress(out,'onsets','正在采音')
    onset_item=manifest['checkpoints']['onset']
    prediction_path=cache/('onsets-'+onset_item['sha256'][:16]+'.npz')
    prediction_meta=prediction_path.with_suffix('.json')
    if prediction_path.exists() and prediction_meta.exists():
        pi=read(prediction_meta)
        assert pi['features_sha256']==sha(cached/'features/features.npy') and pi['sha256']==sha(prediction_path)
        with np.load(prediction_path,allow_pickle=False) as z:pred={k:z[k] for k in ('probability','offsets')}
    else:
        saved=torch.load(ROOT/onset_item['path'],map_location='cpu',weights_only=False)
        for name,digest in saved['identity']['code'].items():assert sha(ROOT/'src/mdac2/data_preparation'/name)==digest
        onset=OnsetModel().to(device).eval();onset.load_state_dict(saved['model'])
        n=f.shape[-1];pred=dict(probability=np.zeros(n,np.float32),offsets=np.zeros(n,np.float32));coverage=np.zeros(n,np.int32)
        with torch.inference_mode():
            for start in window_starts(n):
                window,_=slice_window(f[1:],start,time_axis=2)
                with (torch.autocast('cuda',dtype=torch.bfloat16) if device.type=='cuda' else nullcontext()):
                    o=onset(torch.as_tensor(window[None],device=device,dtype=torch.float32))
                place_core(pred,coverage,start,dict(probability=o['logits'][0].float().sigmoid().cpu().numpy(),offsets=o['offsets'][0].float().cpu().numpy()))
        assert np.all(coverage==1)
        temp=prediction_path.with_name(prediction_path.stem+'.partial.npz');np.savez_compressed(temp,**pred);os.replace(temp,prediction_path)
        save(prediction_meta,dict(features_sha256=sha(cached/'features/features.npy'),sha256=sha(prediction_path)))
        del onset,saved
        if device.type=='cuda':torch.cuda.empty_cache()
    times,boosted,routing=select_onsets(model_id,f,pred,duration)
    progress(out,'arrangement','正在生成轨道、物件与长条')
    saved=torch.load(ROOT/manifest['checkpoints']['arrangement']['path'],map_location='cpu',weights_only=False)
    for name,digest in saved['identity']['code'].items():assert sha(ROOT/'src/mdac2/data_preparation'/name)==digest
    model=ContextArrangement().to(device).eval();model.load_state_dict(saved['model'])
    scores=predict(model,f,times,duration,device=str(device),amp=device.type=='cuda');events,decoding=decode(scores['logits'],scores['log_lengths'],times,duration)
    save(out/'neural-chart.json',dict(duration=duration,events=[e.to_dict() for e in events],audit=decoding))
    progress(out,'postprocess','正在执行后处理和冲突检查')
    processor=postprocess if shared else (v3_process if model_id=='v3' else v2_process)
    after,report=processor(events,f,boosted,duration,read(prior_path))
    from mdac2.application.studio_timing import settings,snap
    timing=settings(request.get('timing'))
    after,grid_report=snap(after,duration,timing)
    grid_report['adapter_sha256']=sha(ROOT/'src/mdac2/application/studio_timing.py')
    report['timing_adapter']=grid_report
    report['full_stage_order']=['model_selection_and_decoding',*report.get('stage_order',[]),
                                *grid_report.get('stage_order',[])]
    save(out/'beat-grid.json',grid_report)
    audit=audit_playability(after,duration)
    if not audit['engineering_playtest_gate_passed']:raise ValueError('谱面冲突检查未通过')
    bms=write_bms(after,duration);assert semantics(after)==semantics(read_bms(bms,duration))
    save(out/'chart.json',dict(duration=duration,events=[e.to_dict() for e in after],class_counts=dict(Counter(e.kind for e in after)),
        model_id=model_id,metadata=request['metadata'],timing=timing,trim=trim))
    save(out/'postprocess.json',report);save(out/'audit.json',audit)
    (out/'chart.bms').write_text(bms,encoding='utf-8')
    save(out/'result.json',dict(status='succeeded',events=len(after),duration=duration,model=model_id,
        cached_audio=str(cached),audio_record_sha256=sha(cached/'audio.json'),wav_sha256=record['wav_sha256'],
        features_sha256=sha(cached/'features/features.npy'),chart_sha256=sha(out/'chart.json'),
        model_manifest_sha256=sha(manifest_path),routing=routing,worker_sha256=sha(Path(__file__)),
        studio_pipeline=shared,base_model=PROFILES[model_id]['base'],
        postprocess_manifest_sha256=request.get('postprocess_manifest_sha256'),
        selection_adapter_sha256=sha(ROOT/'src/mdac2/application/studio_models.py'),
        beat_grid_sha256=sha(out/'beat-grid.json'),timing_adapter_sha256=grid_report['adapter_sha256']))


def export(request,out):
    import numpy as np
    import soundfile as sf
    from mdac2.data_preparation.six_export import write_bms,read_bms,semantics
    from mdac2.data_preparation.training_chart import TrainingEvent
    from mdac2.application.export_builder import default_cover
    from scipy.signal import correlate,correlation_lags
    generation=Path(request['generation']);result=read(generation/'result.json');chart=read(generation/'chart.json')
    assert result['status']=='succeeded' and sha(generation/'chart.json')==result['chart_sha256']
    meta=request['metadata'];assert meta==chart['metadata'],'信息与已生成谱面不一致，请重新制谱'
    from mdac2.application.studio_difficulty import level
    rating=request.get('difficulty')
    if rating:
        assert rating['chart_sha256']==result['chart_sha256'] and rating['generation']==generation.name,'定级与谱面不一致'
    difficulty=level(rating['value'] if rating else MODEL_LEVELS[meta['model']])
    from mdac2.application.studio_timing import settings
    assert settings(request.get('timing'))==settings(chart.get('timing')),'BPM设置已变化，请重新制谱'
    from mdac2.application.studio_trim import settings as trim_settings
    assert request.get('trim')==chart.get('trim') or chart.get('trim') is None and request.get('trim')==trim_settings(None,chart['duration']),'裁剪设置已变化，请重新制谱'
    source=Path(result['cached_audio']);record=read(source/'audio.json')
    assert sha(source/'audio.json')==result['audio_record_sha256'] and sha(source/'mix.wav')==result['wav_sha256']
    for key in ('title','author','creator'):clean(meta[key],empty=key=='author')
    from mdac2.application.studio_scenes import export_scene
    scene_settings=export_scene(meta['scene'])
    progress(out,'export','正在编码音乐与打包')
    package=out/'package';package.mkdir()
    es=[TrainingEvent.from_dict(e) for e in chart['events']]
    text=write_bms(es,chart['duration']).replace('#TITLE MDAC2 six-class review','#TITLE '+meta['title'])
    text=text.replace('#ARTIST MDAC2','#ARTIST '+(meta['author'] or 'Unknown')+'\n#LEVELDESIGN '+meta['creator'])
    text=text.replace('#GENRE scene_01','#GENRE '+scene_settings['scene'])
    text=text.replace('#PLAYLEVEL 9.5','#PLAYLEVEL '+difficulty)
    assert semantics(es)==semantics(read_bms(text,chart['duration']))
    (package/'map2.bms').write_text(text,encoding='utf-8')
    save(package/'map2.talk',opening_talk())
    for name,extra in [('music',[]),('demo',['-t','15'])]:
        run_ffmpeg(['-n','-i',source/'mix.wav',*extra,'-map_metadata',-1,'-c:a','libvorbis','-q:a',6,package/(name+'.ogg')])
    cover=request.get('cover')
    if cover:
        path=Path(cover['path']);assert sha(path)==cover['sha256'],'封面文件校验失败'
        (package/'cover.png').write_bytes(path.read_bytes())
    else:(package/'cover.png').write_bytes(default_cover())
    save(package/'info.json',dict(name=meta['title'],author=meta['author'] or 'Unknown',levelDesigner=meta['creator'],
        **scene_settings,bpm='120',difficulty1='0',difficulty2=difficulty,difficulty3='0',difficulty4='0',unlockLevel=0))
    original,_=sf.read(source/'mix.wav',dtype='float32',always_2d=True)
    decoded=np.frombuffer(run_ffmpeg(['-i',package/'music.ogg','-f','f32le','-ar',44100,'-ac',2,'pipe:1']),dtype='<f4').reshape(-1,2)
    assert abs(len(original)-len(decoded))<=1024 and np.isfinite(decoded).all()
    n=min(len(original),len(decoded),44100*20);x=original[:n].mean(1);y=decoded[:n].mean(1)
    lags=correlation_lags(len(y),len(x));corr=correlate(y,x,method='fft');near=abs(lags)<=2048
    lag=int(lags[near][np.argmax(corr[near])]);assert abs(lag)<=2,'音频编码偏移异常'
    target=out/'chart.mdm'
    with zipfile.ZipFile(target,'x',compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(package.iterdir()):z.write(p,p.name)
    with zipfile.ZipFile(target) as z:assert z.testzip() is None
    save(out/'result.json',dict(status='succeeded',package_sha256=sha(target),events=len(es),scene=scene_settings['scene'],
        selected_scene=meta['scene'],scene_egg=scene_settings.get('sceneEgg'),
        music_alignment_samples=lag,music_length_difference=len(decoded)-len(original),talk_included=True,
        talk_game_visual_acceptance='pending',source_generation=str(generation),difficulty=difficulty,
        difficulty_assessment=rating,worker_sha256=sha(Path(__file__))))


def main():
    p=argparse.ArgumentParser();p.add_argument('--request',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();request=read(a.request);a.output.mkdir(parents=True,exist_ok=True)
    # Also protects against an orphan process after abrupt server termination.
    lock=Path(request['gpu_lock']);lock.parent.mkdir(parents=True,exist_ok=True)
    with lock.open('a+b') as f:
        if os.name=='nt':
            import msvcrt
            f.seek(0);f.write(b'0');f.flush();f.seek(0)
            try:msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1)
            except OSError:raise RuntimeError('另一个制谱任务仍在运行，请稍后重试')
        try:
            (generate if request['operation']=='generate' else export)(request,a.output)
        except Exception as exc:
            # No credentials enter worker requests. Keep local traceback for diagnosis.
            import traceback
            (a.output/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8')
            save(a.output/'result.json',dict(status='failed',message=str(exc)[:500] or type(exc).__name__))
            raise


if __name__=='__main__':main()
