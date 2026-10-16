"""Review an existing private playtest cache with the frozen dual baseline."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
from collections import Counter
import html
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from mdac2.data_preparation.audio_features import sha,validate_cache
from mdac2.data_preparation.dual_training import identity,make_model
from mdac2.data_preparation.dual_inference import acoustic_predictions,onset_times,arrange
from mdac2.data_preparation.supervised_training import atomic_json,seed_all
from mdac2.data_preparation.six_export import write_bms,read_bms,semantics
from mdac2.data_preparation.playability_audit import audit_playability
from build_supervised_review import make_clicks,LABELS
from build_dual_review import timeline

def main(experiment,evaluation,cached,output):
    output.mkdir(parents=True,exist_ok=False);seed_all(20260928)
    frozen=json.loads((evaluation/'frozen.json').read_text(encoding='utf-8'))
    audio=json.loads((cached/'audio.json').read_text(encoding='utf-8'))
    if sha(cached/'mix.wav')!=audio['wav_sha256']:raise ValueError('Original cached audio changed')
    metadata=validate_cache(cached/'features',{'pcm':{'sha256':audio['pcm_sha256']}})
    configs={s:json.loads((experiment/f'{s}.json').read_text(encoding='utf-8')) for s in ('onset','arrangement')}
    models={}
    for stage,c in configs.items():
        path=Path(frozen['checkpoints'][stage]['path'])
        if sha(path)!=frozen['checkpoints'][stage]['sha256']:raise ValueError('Frozen checkpoint changed')
        saved=torch.load(path,map_location='cpu',weights_only=False)
        if saved['identity']!=identity(c):raise ValueError('Training code/data identity changed')
        model=make_model(stage).cuda();model.load_state_dict(saved['model']);models[stage]=model
    duration=audio['duration'];features=np.load(cached/'features/features.npy',mmap_mode='r')
    pred=acoustic_predictions(models['onset'],models['arrangement'],features,int(np.ceil(duration*100-1e-8)))
    np.savez_compressed(output/'prediction.npz',**pred)
    times=onset_times(pred,frozen['threshold'],duration)
    events,decoding=arrange(models['arrangement'],pred['acoustic'],times,duration)
    playback=audit_playability(events,duration)
    if playback['occupancy_issues']:raise ValueError('Occupancy conflicts')
    chart=dict(title='琵琶行 双模型待听评',creator='mikee',scene='scene_01',duration=duration,
               events=[e.to_dict() for e in events],source_audio=audio,decoding=decoding,
               checkpoint_identity=frozen['checkpoints'],threshold=frozen['threshold'],bpm_used=False,
               grid=False,no_gap_filling=True,human_listening='pending',game_acceptance='pending',production_promoted=False)
    atomic_json(output/'chart.json',chart)
    if events:
        bms=write_bms(events,duration)
        bms=bms.replace('#TITLE MDAC2 six-class review','#TITLE 琵琶行 双模型待听评')
        bms=bms.replace('#ARTIST MDAC2','#ARTIST Unknown\n#LEVELDESIGN mikee')
        if semantics(events)!=semantics(read_bms(bms,duration)):raise ValueError('BMS roundtrip mismatch')
        (output/'experimental.bms').write_text(bms,encoding='utf-8')
    ffmpeg=ROOT/'tools/.deps/ffmpeg/extracted/ffmpeg-9.0.1-essentials_build/bin/ffmpeg.exe'
    def ff(args):subprocess.run([str(ffmpeg),'-hide_banner','-loglevel','error','-n',*args],check=True,capture_output=True,timeout=120)
    ff(['-i',str(cached/'mix.wav'),'-ar','24000','-ac','2','-c:a','pcm_f32le',str(output/'mix.wav')])
    make_clicks(output/'clicks.wav',chart['events'],0,duration)
    ff(['-i',str(output/'mix.wav'),'-i',str(output/'clicks.wav'),'-filter_complex','amix=inputs=2:duration=first:normalize=0',
        '-c:a','pcm_f32le',str(output/'prediction-mix.wav')])
    positions={}
    for e in events:
        if e.kind not in ('hold','gear'):positions.setdefault(e.start,set()).update(e.lanes)
    counts=dict(Counter(e.kind for e in events));dual=sum(len(lanes)==2 for lanes in positions.values())
    report=dict(status='completed',source='existing user-provided song cache; no retraining or reseparation',
                feature_metadata_sha256=sha(cached/'features/metadata.json'),checkpoints=frozen['checkpoints'],
                threshold=frozen['threshold'],logical_class_counts=counts,attack_positions=len(positions),
                dual_attack_positions=dual,dual_attack_fraction=dual/max(1,len(positions)),playability=playback,
                bpm_used=False,human_listening='pending',game_acceptance='pending',mdm_package_created=False,
                old_v1_v2_untouched=True,production_promoted=False,source_sha256=sha(Path(__file__)))
    atomic_json(output/'report.json',report)
    cards=[]
    for start in range(0,int(duration),24):
        end=min(duration,start+24)
        cards.append(f'<article data-start="{start}" data-end="{end}"><h2>{start}–{end:.3f} 秒 · 实际预测</h2><button data-seek="{start}">从这里播放</button>{timeline(chart["events"],start,end,"预测")}</article>')
    page=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>琵琶行 · 双模型实际预测</title><style>
body{{font:16px/1.6 system-ui;max-width:1180px;margin:32px auto;padding:0 20px;color:#172033;background:#f1f5f9}}article{{background:white;padding:20px;margin:20px 0;border-radius:10px}}audio,svg{{width:100%}}.player{{position:sticky;top:0;background:#fff7ed;padding:18px;z-index:2}}button{{padding:8px 15px}}pre{{white-space:pre-wrap}}
</style><h1>琵琶行：双模型实际预测</h1><p>这首没有参考谱，所有时间轴和击打均为模型预测。复用原分轨缓存，阈值来自22首验证；没有针对本曲调参或填补空档。待听评，尚未生成新版游戏包。</p>
<p>逻辑物件：{'、'.join(LABELS[k]+' '+str(v) for k,v in counts.items())}；击打位置 {len(positions)}，双轨 {dual/max(1,len(positions)):.1%}。有 {len(playback['no_attack_or_hold_intervals'])} 段≥2秒空档，是否合理需结合音乐判断。</p>
<div class="player"><strong>原曲＋实际预测击打（完整146秒）</strong><audio id="prediction" controls preload="metadata" src="prediction-mix.wav"></audio><details><summary>只听原曲</summary><audio id="original" controls preload="none" src="mix.wav"></audio></details></div>
{''.join(cards)}<p><a href="chart.json">完整预测事件</a> · <a href="report.json">规则与空档报告</a> · <a href="experimental.bms">实验BMS</a></p>
<script>const p=document.querySelector('#prediction'),o=document.querySelector('#original');for(const b of document.querySelectorAll('button'))b.onclick=()=>{{o.pause();p.currentTime=Number(b.dataset.seek);p.play()}};p.onplay=()=>o.pause();o.onplay=()=>p.pause();</script></html>'''
    (output/'琵琶行-实际预测.html').write_text(page,encoding='utf-8')
    print(json.dumps(dict(path=output.as_posix(),counts=counts,dual_fraction=report['dual_attack_fraction']),ensure_ascii=True))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--experiment',type=Path,required=True);p.add_argument('--evaluation',type=Path,required=True)
    p.add_argument('--cached',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();main(a.experiment.resolve(),a.evaluation.resolve(),a.cached.resolve(),a.output.resolve())
