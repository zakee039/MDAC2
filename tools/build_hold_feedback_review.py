"""Independent local review of hold-feedback candidates; no web server needed."""
import argparse
import html
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from mdac2.data_preparation.audio_features import sha
from build_supervised_review import make_clicks
from diagnose_hold_feedback_r1 import timeline


def main(prediction, baseline, cached, output):
    read=lambda p:json.loads(p.read_text(encoding='utf-8'))
    after,before=read(prediction/'chart.json'),read(baseline/'chart.json')
    audio=read(cached/'audio.json')
    assert after['source_audio']==before['source_audio']==audio
    assert sha(cached/'mix.wav')==audio['wav_sha256']
    output.mkdir(parents=True,exist_ok=False)
    commands=[]
    def ff(args):
        cmd=[str(ROOT/'tools/.deps/ffmpeg/extracted/ffmpeg-9.0.1-essentials_build/bin/ffmpeg.exe'),'-hide_banner','-loglevel','error','-n',*args]
        commands.append(cmd);subprocess.run(cmd,check=True,capture_output=True,timeout=120)
    ff(['-i',str(cached/'mix.wav'),'-ar','24000','-ac','2','-c:a','pcm_f32le',str(output/'original.wav')])
    for name,chart in [('r1',before),('candidate',after)]:
        make_clicks(output/f'{name}-clicks.wav',chart['events'],0,chart['duration'])
        ff(['-i',str(output/'original.wav'),'-i',str(output/f'{name}-clicks.wav'),'-filter_complex',
            'amix=inputs=2:duration=first:normalize=0','-c:a','pcm_f32le',str(output/f'{name}.wav')])
    cards=[]
    for start,end,label in [(6,11,'长条内部点击：8.645 秒移到天空'),(252,262,'持续音尾部：256.310 → 259.725 秒')]:
        cards.append(f'<article><h2>{html.escape(label)}</h2><button data-seek="{start}">从 {start} 秒播放</button>'+
            timeline(after['events'],start,end,'候选 · 实际出片')+timeline(before['events'],start,end,'R1 · 保留对照')+'</article>')
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>Snow · R1 长条修订候选</title>
<style>body{font:16px system-ui;background:#eef3f5;color:#193044;margin:24px auto;max-width:1100px}header,article{background:white;padding:24px;margin:16px;border-radius:12px}svg{width:100%}button,select{font:inherit;padding:8px}audio{width:100%}.player{position:sticky;top:0;background:white;padding:12px;z-index:1}</style>
<header><h1>Snow Song Show · R1 长条修订候选</h1><p>保留 V2 权重和 R1。只修持续音提前收尾与长条内部点击过密；图2的长条链保持不变。新版本待听评、待实机反馈，未正式定为 R2。</p>
<p>蓝条为长按，红线为独立起点，深色线为终点。听感叠加音仅辅助试听；游戏包仍使用原曲。</p></header>
<div class="player"><select id="mode"><option value="candidate.wav">候选＋击打</option><option value="r1.wav">R1＋击打</option><option value="original.wav">只听原曲</option></select><audio id="player" controls preload="metadata" src="candidate.wav"></audio><span id="clock">0.000 秒</span></div>'''+''.join(cards)+'''
<p><a href="chart.json">候选事件</a> · <a href="report.json">修改依据</a> · <a href="experimental.bms">候选 BMS</a></p>
<script>const p=document.querySelector('#player'),s=document.querySelector('#mode');s.onchange=()=>{let t=p.currentTime,playing=!p.paused;p.src=s.value;p.addEventListener('loadedmetadata',()=>{p.currentTime=Math.min(t,p.duration);if(playing)p.play()},{once:true})};for(const b of document.querySelectorAll('[data-seek]'))b.onclick=()=>{p.currentTime=+b.dataset.seek;p.play()};p.ontimeupdate=()=>document.querySelector('#clock').textContent=p.currentTime.toFixed(3)+' 秒';</script></html>'''
    for name in ('chart.json','report.json','experimental.bms'):shutil.copyfile(prediction/name,output/name)
    (output/'index.html').write_text(page,encoding='utf-8')
    import soundfile as sf
    frames={name:sf.info(output/name).frames for name in ('original.wav','r1.wav','candidate.wav')}
    assert len(set(frames.values()))==1 and abs(frames['original.wav']/24000-after['duration'])<1/24000
    audit=dict(status='completed',command=[sys.executable,*sys.argv],ffmpeg_commands=commands,frames=frames,
        files={p.name:sha(p) for p in output.iterdir() if p.is_file()},tool_sha256=sha(Path(__file__)),
        local_file_review=True,http_server_started=False,human_listening='pending')
    (output/'audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(output/'index.html')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for arg in ('prediction','baseline','cached','output'):p.add_argument('--'+arg,type=Path,required=True)
    a=p.parse_args();main(a.prediction,a.baseline,a.cached,a.output)
