"""Compare actual v2 and vocal-routing charts with auditable voice-mode bands."""
import argparse
import html
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from mdac2.data_preparation.audio_features import sha
from mdac2.data_preparation.training_chart import TrainingEvent
from mdac2.data_preparation.six_export import read_bms, semantics
from mdac2.data_preparation.generated_constraints import validate_generated_events
from build_supervised_review import make_clicks
from build_dual_review import timeline


def main(prediction, cached, output, title, previous):
    read = lambda p: json.loads(p.read_text(encoding='utf-8'))
    base, chart = [read(prediction/name) for name in ('base-chart.json', 'chart.json')]
    report = read(prediction/'report.json')
    prior_chart = read(previous/'chart.json')
    routing = read(prediction/'routing.json')
    validation_path = prediction.parent/'validation/validation.json'
    validation = read(validation_path)
    if validation['identity']['policy'] != routing['policy']:
        raise ValueError('Validation and preview routing policy differ')
    v0 = validation['metrics']['v2']['sampling']
    v1 = validation['metrics']['vocal_trial']['sampling']
    audio = read(cached/'audio.json')
    if chart['source_audio'] != audio or base['source_audio'] != audio or prior_chart['source_audio'] != audio or sha(cached/'mix.wav') != audio['wav_sha256']:
        raise ValueError('Preview source mismatch')
    if sha(prediction/'report.json') != chart['postprocess_report_sha256']:
        raise ValueError('Report mismatch')
    duration = chart['duration']
    events = [TrainingEvent.from_dict(e) for e in chart['events']]
    if validate_generated_events(events, duration) or semantics(events) != semantics(read_bms((prediction/'experimental.bms').read_text(encoding='utf-8'), duration)):
        raise ValueError('Chart/BMS/occupancy mismatch')
    output.mkdir(parents=True, exist_ok=False)
    ffmpeg = ROOT/'tools/.deps/ffmpeg/extracted/ffmpeg-9.0.1-essentials_build/bin/ffmpeg.exe'
    commands = []
    def ff(args):
        cmd = [str(ffmpeg), '-hide_banner', '-loglevel', 'error', '-n', *args]
        commands.append(cmd)
        subprocess.run(cmd, check=True, capture_output=True, timeout=120)
    for source, target in [('mix.wav', 'original.wav'), ('features/vocals.wav', 'vocals.wav')]:
        ff(['-i', str(cached/source), '-ar', '24000', '-ac', '2', '-c:a', 'pcm_f32le', str(output/target)])
    for key, data in [('before', base), ('after', chart), ('r1', prior_chart)]:
        make_clicks(output/f'{key}-clicks.wav', data['events'], 0, duration)
        ff(['-i', str(output/'original.wav'), '-i', str(output/f'{key}-clicks.wav'),
            '-filter_complex', 'amix=inputs=2:duration=first:normalize=0', '-c:a', 'pcm_f32le', str(output/f'{key}.wav')])
    ff(['-i', str(output/'vocals.wav'), '-i', str(output/'after-clicks.wav'),
        '-filter_complex', 'amix=inputs=2:duration=first:normalize=0', '-c:a', 'pcm_f32le', str(output/'vocals-clicks.wav')])
    for name in ('chart.json', 'base-chart.json', 'report.json', 'routing.json', 'postprocessing.json', 'experimental.bms'):
        shutil.copyfile(prediction/name, output/name)
    def band(start, end):
        rects = []
        for spans, y, color in [(routing['vocal_mode_intervals'], 5, '#b94369'), (routing['vocal_intervals'], 27, '#cc9a16')]:
            for a,b in spans:
                a,b = max(a,start),min(b,end)
                if b > a:
                    rects.append(f'<rect x="{100+(a-start)/(end-start)*1000:.1f}" y="{y}" width="{(b-a)/(end-start)*1000:.1f}" height="14" fill="{color}"/>')
        return '<svg viewBox="0 0 1120 48"><text x="0" y="17">人声优先</text><text x="0" y="39">人声活动</text>'+''.join(rects)+'</svg>'
    cards = []
    for start in range(0, int(duration), 16):
        end = min(start+16, duration)
        cards.append(f'<article data-start="{start}" data-end="{end}"><h2>{start}–{end:.2f} 秒</h2><button data-seek="{start}">从这里播放</button>{band(start,end)}'
            f'<h3>r2 人声前瞻＋5ms网格 · 实际出片</h3>{timeline(chart["events"],start,end,"实验")}'
            f'<details><summary>展开已保留的 v2 · 实际出片</summary>{timeline(base["events"],start,end,"v2")}</details><details><summary>展开人声 r1 实际出片</summary>{timeline(prior_chart["events"],start,end,"r1")}</details></article>')
    page = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>{html.escape(title)} · 人声前瞻与5ms网格对照</title><style>
body{{font:16px/1.65 system-ui;background:#eff3f5;color:#172536;max-width:1150px;margin:30px auto;padding:0 20px}}
article,.summary{{background:white;padding:20px;margin:18px 0;border-radius:12px}}svg,audio{{width:100%}}svg text{{font-size:13px}}
.player{{position:sticky;top:0;background:#fff3df;padding:16px;z-index:3;border:1px solid #c6a66f;border-radius:9px}}
button,select{{padding:8px;margin:4px}}.mode{{font-weight:bold;color:#923553}}a{{color:#175e91}}
</style><h1>{html.escape(title)} · 人声前瞻与5ms网格 r2</h1>
<p>v2、r1、r2 都是真实预测，没有参考谱。旧版原样保留，r2 待听评，不自动替换。人声活动是分离音轨的声学判断，可能受乐器残留影响。</p>
<section class="summary">采音落点 {routing['baseline_candidates']} → {routing['selected_candidates']}；移除 {routing['removed']}，新增 {routing['added']}。
人声优先模式共 {routing['vocal_mode_seconds']:.1f} 秒；用未来一秒确认，从实际起点生效，末次活动结束立即恢复普通采音，不再保持两秒。
<p>本轮 22 曲验证，v2 → r2：采音精确率 {v0['precision']:.2%} → {v1['precision']:.2%}，召回率 {v0['recall']:.2%} → {v1['recall']:.2%}，F1 {v0['f1']:.2%} → {v1['f1']:.2%}。误点 {v0['fp']} → {v1['fp']}，漏点 {v0['fn']} → {v1['fn']}。没有人声来源标注，不能据此判断人声采音已合格。</p>
<p>固定存储 BPM 240；每拍 50 格，精度 5ms，未估计歌曲 BPM。长条对齐调整 {report['alignment']['changed_holds']} 根；旧 250ms 补尾已停用。5ms 吸附的手感与性能收益尚待实机确认。</p>
<p>粉色条：前瞻确认的人声优先片段；黄色条：人声音轨活动证据。空白模式使用原采音选择。可单听分离人声，检查残留、弱字、起音以及切换处的空档。</p></section>
<div class="player"><select id="mode"><option value="after.wav">r2 人声前瞻＋5ms网格＋击打</option><option value="r1.wav">r1 人声优先＋击打</option><option value="before.wav">v2＋击打</option><option value="original.wav">只听原曲</option><option value="vocals.wav">只听分离人声</option><option value="vocals-clicks.wav">分离人声＋实验击打</option></select>
<audio id="player" controls preload="metadata" src="after.wav"></audio><span id="clock">0.00 秒</span> · <span class="mode" id="routing"></span></div>
{''.join(cards)}<p><a href="routing.json">人声活动与切换记录</a> · <a href="chart.json">实验事件</a> · <a href="experimental.bms">实验 BMS</a></p>
<script>const p=document.querySelector('#player'),s=document.querySelector('#mode'),v={json.dumps(routing['vocal_mode_intervals'])},a={json.dumps(routing['vocal_intervals'])};
function inside(t,spans){{return spans.some(x=>x[0]<=t&&t<x[1])}}
s.onchange=()=>{{let t=p.currentTime,playing=!p.paused;p.src=s.value;p.addEventListener('loadedmetadata',()=>{{p.currentTime=Math.min(t,p.duration);if(playing)p.play()}},{{once:true}})}};
for(const b of document.querySelectorAll('[data-seek]'))b.onclick=()=>{{p.currentTime=Number(b.dataset.seek);p.play()}};
p.ontimeupdate=()=>{{let t=p.currentTime;document.querySelector('#clock').textContent=t.toFixed(2)+' 秒';document.querySelector('#routing').textContent=inside(t,v)?(inside(t,a)?'人声优先 · 检测到活动':'人声优先 · 片段内短停顿'):'普通采音';for(const c of document.querySelectorAll('article')){{let start=+c.dataset.start,end=+c.dataset.end,x=100+1000*Math.max(0,Math.min(1,(t-start)/(end-start)));for(const l of c.querySelectorAll('.playhead'))l.setAttribute('d',`M${{x}} 20V112`)}}}};
</script></html>'''
    (output/'index.html').write_text(page, encoding='utf-8')
    import soundfile as sf
    frames = {n: sf.info(output/n).frames for n in ['original.wav','vocals.wav','before.wav','after.wav','r1.wav','vocals-clicks.wav']}
    if len(set(frames.values())) != 1 or abs(frames['original.wav']/24000-duration) > 1/24000:
        raise ValueError('Media duration mismatch')
    audit = dict(status='completed', command=[sys.executable,*sys.argv], ffmpeg_commands=commands, frames=frames,
        source_chart_sha256=sha(prediction/'chart.json'), source_tool_sha256=sha(Path(__file__)), validation_sha256=sha(validation_path),
        files={p.name: sha(p) for p in output.iterdir() if p.is_file()}, human_listening='pending', model_registry_unchanged=True)
    (output/'audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(dict(status='completed', page=str(output/'index.html'))))


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    for name in ['prediction','cached','output','previous']:
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--title', required=True)
    a=p.parse_args()
    main(a.prediction.resolve(),a.cached.resolve(),a.output.resolve(),a.title,a.previous.resolve())
