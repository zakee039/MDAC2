"""Build a full-song original/edited prediction comparison, never reference notes."""
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
from mdac2.data_preparation.playtest_postprocess import structure_stats
from build_supervised_review import make_clicks
from build_dual_review import timeline


def main(prediction, cached, output, title):
    load = lambda p: json.loads(p.read_text(encoding='utf-8'))
    base = load(prediction/'base-chart.json')
    chart = load(prediction/'chart.json')
    report = load(prediction/'report.json')
    audio = load(cached/'audio.json')
    if base['source_audio'] != audio or chart['source_audio'] != audio or sha(cached/'mix.wav') != audio['wav_sha256']:
        raise ValueError('Preview audio identity mismatch')
    if chart['postprocess_report_sha256'] != sha(prediction/'report.json'):
        raise ValueError('Postprocessing report identity mismatch')
    for content, key in [(base, 'before'), (chart, 'after')]:
        stats = structure_stats([TrainingEvent.from_dict(e) for e in content['events']])
        # JSON stringifies histogram integer keys; canonical serialization keeps
        # this check strict without confusing key representation with semantics.
        if json.dumps(stats, sort_keys=True) != json.dumps(report[key], sort_keys=True):
            # Normalize dictionary keys once through JSON, then compare values.
            if json.loads(json.dumps(stats)) != report[key]:
                raise ValueError('Prediction/report structure mismatch')
    output.mkdir(parents=True, exist_ok=False)
    duration = chart['duration']
    revision = report['policy']['version'].removeprefix('playtest-postprocess-')
    ffmpeg = ROOT/'tools/.deps/ffmpeg/extracted/ffmpeg-9.0.1-essentials_build/bin/ffmpeg.exe'
    commands = []
    def ff(args):
        command = [str(ffmpeg), '-hide_banner', '-loglevel', 'error', '-n', *args]
        commands.append(command)
        subprocess.run(command, check=True, capture_output=True, timeout=120)
    ff(['-i', str(cached/'mix.wav'), '-ar', '24000', '-ac', '2', '-c:a', 'pcm_f32le', str(output/'original.wav')])
    for key, c in [('before', base), ('after', chart)]:
        make_clicks(output/f'{key}-clicks.wav', c['events'], 0, duration)
        ff(['-i', str(output/'original.wav'), '-i', str(output/f'{key}-clicks.wav'),
            '-filter_complex', 'amix=inputs=2:duration=first:normalize=0',
            '-c:a', 'pcm_f32le', str(output/f'{key}.wav')])
    for name in ('chart.json', 'base-chart.json', 'report.json', 'experimental.bms'):
        shutil.copyfile(prediction/name, output/name)
    summary = report['after']
    cards = []
    for start in range(0, int(duration), 16):
        end = min(duration, start+16)
        cards.append(f'<article data-start="{start}" data-end="{end}"><h2>{start}–{end:.2f} 秒</h2>'
                     f'<button data-seek="{start}">从此处播放当前选择</button><h3>编排 {revision} · 实际出片</h3>'
                     f'{timeline(chart["events"], start, end, revision)}'
                     f'<details><summary>展开原版 v1 实际出片</summary>{timeline(base["events"], start, end, "v1")}</details></article>')
    sections = '；'.join(f'{s["start"]:.1f}–{s["end"]:.1f} 秒' for s in report['density']['sections']) or '本曲没有满足保守条件的高强度片段'
    repeats = '；'.join(f'{p["source"][0]:.1f}–{p["source"][1]:.1f} → {p["target"][0]:.1f}–{p["target"][1]:.1f} 秒'
                        for p in report['repeats']['proposals'] if p['applied']) or '没有通过自动应用检查的重复段'
    page = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>{html.escape(title)} · 编排对照</title><style>
body{{font:16px/1.65 system-ui;max-width:1160px;margin:32px auto;padding:0 20px;color:#182b39;background:#eef3f6}}
article,.summary{{background:white;padding:20px;border-radius:12px;margin:18px 0}}audio,svg{{width:100%}}
.player{{position:sticky;top:0;background:#e6f7f1;padding:16px;z-index:2;border:1px solid #89b5a6;border-radius:8px}}
button,select{{padding:9px;margin:3px;border-radius:5px}}details{{margin-top:16px}}h3{{margin-bottom:0;color:#176248}}a{{color:#145e9b}}
</style><h1>{html.escape(title)} · 编排 {revision}</h1>
<p>这里对比两份实际预测。默认播放新版：原曲＋击打提示。模型仍为 v1；编排修订另行保存，待试听和实机反馈。</p>
<section class="summary">同轨三连 {report['before']['three_runs']} → {summary['three_runs']} 组；四连 {report['before']['four_runs']} → {summary['four_runs']} 组；长条尾补齐 {report['tails']['extended']} 处；新增点击 {report['density']['added']} 个。
<p>高强度片段候选：{sections}。这是声音强度规则，尚不是经人工验证的高潮检测模型。</p>
<p>最后执行的相似段镜像：{repeats}。</p></section>
<div class="player"><label for="mode">试听版本 </label><select id="mode"><option value="after.wav">编排 {revision}＋击打</option><option value="before.wav">原版 v1＋击打</option><option value="original.wav">只听原曲</option></select>
<audio id="player" controls preload="metadata" src="after.wav"></audio><span id="clock">0.00 秒</span></div>
{''.join(cards)}<p><a href="chart.json">新版事件</a> · <a href="report.json">逐项改动记录</a> · <a href="experimental.bms">新版 BMS</a></p>
<script>
const player=document.querySelector('#player'),mode=document.querySelector('#mode');
mode.addEventListener('change',()=>{{const t=player.currentTime,playing=!player.paused;player.src=mode.value;player.addEventListener('loadedmetadata',()=>{{player.currentTime=Math.min(t,player.duration);if(playing)player.play()}},{{once:true}})}});
for(const b of document.querySelectorAll('[data-seek]'))b.onclick=()=>{{player.currentTime=Number(b.dataset.seek);player.play()}};
player.addEventListener('timeupdate',()=>{{document.querySelector('#clock').textContent=player.currentTime.toFixed(2)+' 秒';for(const c of document.querySelectorAll('article')){{let start=Number(c.dataset.start),end=Number(c.dataset.end),x=100+1000*Math.max(0,Math.min(1,(player.currentTime-start)/(end-start)));for(const l of c.querySelectorAll('.playhead'))l.setAttribute('d',`M${{x}} 20V112`)}}}});
</script></html>'''
    (output/'index.html').write_text(page, encoding='utf-8')
    import soundfile as sf
    frames = {name: sf.info(output/name).frames for name in ('original.wav', 'before.wav', 'after.wav')}
    if len(set(frames.values())) != 1 or abs(frames['original.wav']/24000-duration) > 1/24000:
        raise ValueError('Preview media duration mismatch')
    audit = dict(status='completed', source_chart_sha256=sha(prediction/'chart.json'),
                 base_chart_sha256=sha(prediction/'base-chart.json'),
                 files={p.name: sha(p) for p in output.iterdir() if p.is_file()},
                 frames=frames, command=[sys.executable, *sys.argv], ffmpeg_commands=commands,
                 human_listening='pending', no_reference_chart_displayed=True)
    (output/'audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(dict(status='completed', path=str(output/'index.html'), frames=frames)))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--prediction', type=Path, required=True)
    p.add_argument('--cached', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--title', required=True)
    a = p.parse_args()
    main(a.prediction.resolve(), a.cached.resolve(), a.output.resolve(), a.title)
