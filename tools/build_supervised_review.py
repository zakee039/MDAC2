"""Local listening clips and synchronized six-class prediction/reference cards."""
import argparse
from collections import Counter
import html
import json
from pathlib import Path
import subprocess
import sys
import wave
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from mdac2.data_preparation.audio_features import sha
from mdac2.data_preparation.supervised_training import atomic_json
from mdac2.data_preparation.six_export import write_bms, read_bms, semantics
from mdac2.data_preparation.training_chart import TrainingEvent

COLORS = dict(light='#15803d',heavy='#c2410c',extra_heavy='#b91c1c',gemini='#7c3aed',hold='#0369a1',gear='#475569')
LABELS = dict(light='轻击',heavy='重击',extra_heavy='特重击',gemini='Gemini',hold='长条',gear='齿轮')


def svg_chart(reference, predicted, start, end):
    def x(t):
        return 130+(t-start)/(end-start)*840
    parts = ['<svg viewBox="0 0 1000 220" role="img" aria-label="参考谱与预测谱的双轨时间轴">']
    for y,label in ((48,'参考 空中'),(82,'参考 地面'),(148,'预测 空中'),(182,'预测 地面')):
        parts.append(f'<text x="8" y="{y+4}" font-size="14">{label}</text><line x1="130" x2="970" y1="{y}" y2="{y}" stroke="#cbd5e1"/>')
    for t in np.arange(np.ceil(start),end,1):
        parts.append(f'<line x1="{x(t):.2f}" x2="{x(t):.2f}" y1="28" y2="195" stroke="#e2e8f0"/><text x="{x(t):.2f}" y="214" font-size="11">{t:.0f}s</text>')
    for group,shift in ((reference,0),(predicted,100)):
        for event in group:
            if event['start']>=end or event['end']<start:
                continue
            left,right = x(max(start,event['start'])),x(min(end,event['end']))
            color = COLORS[event['kind']]
            title = html.escape(f"{LABELS[event['kind']]} {event['start']:.3f}–{event['end']:.3f}s")
            for lane in event['lanes']:
                y = (82 if lane=='ground' else 48)+shift
                if event['kind']=='hold':
                    parts.append(f'<rect x="{left:.2f}" y="{y-5}" width="{max(2,right-left):.2f}" height="10" fill="{color}" opacity=".6"><title>{title}</title></rect>')
                else:
                    parts.append(f'<circle cx="{left:.2f}" cy="{y}" r="4" fill="{color}"><title>{title}</title></circle>')
    parts.append('<line class="playhead" x1="130" x2="130" y1="20" y2="195" stroke="#111827" stroke-width="2"/></svg>')
    return ''.join(parts)


def choose_examples(config,evaluation):
    manifest = json.loads(Path(config['manifest']).read_text(encoding='utf-8'))
    rows = [r for r in manifest['charts'] if r['training_eligible'] and r['split']=='validation']
    candidates = {k:[] for k in ('vocal_onset','vocal_sustain','interlude','possible_leak')}
    charts = {}
    for row in rows:
        reference = json.loads((Path(config['manifest']).parent/row['labels']).read_text(encoding='utf-8'))
        charts[row['chart_sha256']] = (row,reference)
        features = np.load(Path(config['cache'])/row['pcm']['sha256']/'features.npy',mmap_mode='r')
        # Energy/flux and cross-stream correlation are selection proxies, not
        # vocal source truth or a listening verdict.
        for first in range(200,features.shape[-1]-800,400):
            part = features[:,:,first:first+800].astype(np.float32)
            energy = np.expm1(np.clip(part,0,25)).mean(axis=1)
            activity = (energy[1]/(energy[0]+1e-5))/(energy[1]/(energy[0]+1e-5)+.1)
            ratio = float(np.mean(energy[1]/(energy[0]+1e-5)))
            vocal_flux = float(np.maximum(np.diff(part[1],axis=1),0).mean())
            accompaniment_flux = float(np.maximum(np.diff(part[2],axis=1),0).mean())
            correlation = float(np.corrcoef(part[1].reshape(-1),part[2].reshape(-1))[0,1])
            if not np.isfinite(correlation):
                correlation = 0.
            diagnostics = dict(vocal_energy_ratio=ratio,mean_vocal_activity=float(activity.mean()),
                               vocal_flux=vocal_flux,accompaniment_flux=accompaniment_flux,
                               stream_feature_correlation=correlation)
            entry = dict(chart_sha256=row['chart_sha256'],start=first/100,end=(first+800)/100,proxy=diagnostics)
            candidates['vocal_onset'].append((vocal_flux*float(activity.mean()),entry))
            candidates['vocal_sustain'].append((float(activity.mean())/(vocal_flux+.01),entry))
            candidates['interlude'].append(((1-float(activity.mean()))*accompaniment_flux,entry))
            candidates['possible_leak'].append((correlation*min(.3,max(0.,ratio))*accompaniment_flux,entry))
    examples = []
    used = []
    for kind,values in candidates.items():
        values.sort(key=lambda p:p[0],reverse=True)
        _,entry = next((v for v in values if all(v[1]['chart_sha256']!=u['chart_sha256'] or
                          abs(v[1]['start']-u['start'])>=8 for u in used)),values[0])
        example = dict(focus=kind,**entry)
        examples.append(example)
        used.append(example)
    for focus,kind in (('extra_heavy','extra_heavy'),('gemini','gemini'),('double_heavy','heavy'),('cross_window_hold','hold')):
        found = None
        for row,chart in charts.values():
            for event in chart['events']:
                if event['kind']!=kind:
                    continue
                if focus=='double_heavy' and not any(e['kind']=='heavy' and e['start']==event['start'] and e['lanes']!=event['lanes'] for e in chart['events']):
                    continue
                if focus=='cross_window_hold' and event['start']//16==event['end']//16:
                    continue
                found = dict(focus=focus,chart_sha256=row['chart_sha256'],start=max(0.,event['start']-3),
                             end=min(chart['duration'],max(event['start']+5,event['end']+3)),proxy=dict(reference_event_id=event['event_id']))
                break
            if found:
                break
        if found:
            examples.append(found)
    return examples,charts


def make_clicks(path, events, start, end):
    rate = 24000
    signal = np.zeros((int(np.ceil((end-start)*rate)),2),np.float32)
    for event in events:
        marks = [event['start']] + ([event['end']] if event['kind']=='hold' else [])
        for j,t in enumerate(marks):
            if not start<=t<end:
                continue
            first = int(round((t-start)*rate))
            length = min(int(rate*.035),len(signal)-first)
            if length<=0:
                continue
            frequency = 660 if j else {'light':1100,'heavy':800,'extra_heavy':500,'gemini':1500,'hold':1200,'gear':1900}[event['kind']]
            click = .15*np.sin(2*np.pi*frequency*np.arange(length)/rate)*np.exp(-np.arange(length)/(rate*.008))
            for lane in event['lanes']:
                signal[first:first+length,0 if lane=='ground' else 1] += click
    with wave.open(str(path),'wb') as output:
        output.setnchannels(2)
        output.setsampwidth(2)
        output.setframerate(rate)
        output.writeframes((np.clip(signal,-1,1)*32767).astype('<i2').tobytes())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--evaluation',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding='utf-8'))
    frozen = json.loads((args.evaluation/'frozen.json').read_text(encoding='utf-8'))
    if frozen['config_sha256'] != sha(args.config):
        raise ValueError('Review requires this frozen configuration')
    args.output.mkdir(parents=True,exist_ok=False)
    examples,charts = choose_examples(config,args.evaluation)
    ffmpeg = ROOT/'tools/.deps/ffmpeg/extracted/ffmpeg-9.0.1-essentials_build/bin/ffmpeg.exe'
    ffprobe = ffmpeg.with_name('ffprobe.exe')
    titles = dict(vocal_onset='人声起音候选',vocal_sustain='人声长音候选',interlude='间奏候选',possible_leak='分离漏音疑似候选',
                  extra_heavy='排他特重击',gemini='Gemini成对',double_heavy='双重击',cross_window_hold='跨窗口长条')
    cards,records = [],[]
    for i,example in enumerate(examples):
        row,reference = charts[example['chart_sha256']]
        decoded = json.loads((args.evaluation/'charts-validation'/(row['chart_sha256']+'.json')).read_text(encoding='utf-8'))
        start,end = example['start'],example['end']
        sources = [('mix',Path(row['audio_path_absolute'])),('vocals',Path(config['cache'])/row['pcm']['sha256']/'vocals.wav'),
                   ('accompaniment',Path(config['cache'])/row['pcm']['sha256']/'accompaniment.wav')]
        clips = {}
        for name,source in sources:
            target = args.output/f'{i:02d}-{name}.wav'
            subprocess.run([str(ffmpeg),'-hide_banner','-loglevel','error','-n','-ss',f'{start:.6f}','-i',str(source),
                '-t',f'{end-start:.6f}','-ac','2','-ar','24000','-c:a','pcm_f32le',str(target)],check=True,timeout=60,
                stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
            probe = json.loads(subprocess.run([str(ffprobe),'-v','error','-show_entries','format=duration','-of','json',str(target)],
                                              check=True,capture_output=True,timeout=15).stdout)
            if abs(float(probe['format']['duration'])-(end-start))>.01:
                raise ValueError('Review clip duration mismatch')
            clips[name] = dict(file=target.name,sha256=sha(target),source=str(source),source_sha256=sha(source))
        make_clicks(args.output/f'{i:02d}-prediction-clicks.wav',decoded['events'],start,end)
        clips['prediction_clicks'] = dict(file=f'{i:02d}-prediction-clicks.wav',sha256=sha(args.output/f'{i:02d}-prediction-clicks.wav'))
        overlay_path = args.output/f'{i:02d}-prediction-mix.wav'
        subprocess.run([str(ffmpeg),'-hide_banner','-loglevel','error','-n','-i',str(args.output/clips['mix']['file']),
            '-i',str(args.output/clips['prediction_clicks']['file']),'-filter_complex','amix=inputs=2:duration=first:normalize=0',
            '-c:a','pcm_f32le',str(overlay_path)],check=True,timeout=60,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
        clips['prediction_mix'] = dict(file=overlay_path.name,sha256=sha(overlay_path),gain_normalization=False,click_amplitude=.15)
        # Full-song experimental BMS, with millisecond rounding already stated
        # in the decoder. Keep logical groups in the JSON sidecar.
        export_path = args.output/(row['chart_sha256']+'.bms')
        if not export_path.exists() and decoded['events']:
            event_objects = [TrainingEvent.from_dict(e) for e in decoded['events']]
            text = write_bms(event_objects,reference['duration'])
            recovered = read_bms(text,reference['duration'])
            if semantics(event_objects)!=semantics(recovered):
                raise ValueError('Prediction BMS semantics mismatch')
            export_path.write_text(text,encoding='utf-8')
            atomic_json(export_path.with_suffix('.json'),decoded)
        reference_events = [e for e in reference['events'] if e['start']<end and e['end']>=start]
        predicted_events = [e for e in decoded['events'] if e['start']<end and e['end']>=start]
        record = dict(**example,title=row['title'],role='validation',clips=clips,
                      reference_counts=dict(Counter(e['kind'] for e in reference_events)),
                      predicted_counts=dict(Counter(e['kind'] for e in predicted_events)),
                      reference=reference_events,predicted=predicted_events,
                      human_listening='pending',behavior_verdict='pending listening; proxies do not prove vocal source',
                      bms=export_path.name if export_path.exists() else None,game_acceptance='pending')
        records.append(record)
        players = ''.join(f'<label>{label}<audio controls preload="none" data-role="{name}" src="{clips[name]["file"]}"></audio></label>'
                          for name,label in [('mix','原曲'),('vocals','人声'),('accompaniment','伴奏'),('prediction_mix','原曲＋预测击打')])
        counts_text = f'参考 {len(reference_events)} 个逻辑物件；预测 {len(predicted_events)} 个。'
        link = f'<a href="{export_path.name}">实验BMS</a> · <a href="{export_path.with_suffix(".json").name}">预测旁表</a>' if export_path.exists() else '空预测：没有BMS'
        cards.append(f'<article data-start="{start}" data-length="{end-start}"><h2>{i+1}. {titles[example["focus"]]}</h2><p>{html.escape(row["title"])} · {start:.3f}–{end:.3f}s · 验证集</p><p class="pending">待听评。场景由能量／起音／相关性或参考物件自动筛选，尚未人工确认。</p><p>{counts_text}</p><div class="players">{players}</div>{svg_chart(reference_events,predicted_events,start,end)}<p>{link}</p><details><summary>分项计数与筛选证据</summary><pre>{html.escape(json.dumps({k:record[k] for k in ("reference_counts","predicted_counts","proxy")},ensure_ascii=False,indent=2))}</pre></details></article>')
    legend = ' · '.join(f'<span style="color:{c}">{LABELS[k]}</span>' for k,c in COLORS.items())
    page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>MDAC2 监督预测听评</title><style>body{max-width:1180px;margin:35px auto;padding:0 24px;background:#f1f5f9;color:#0f172a;font:16px/1.6 system-ui}article{background:white;border-radius:14px;padding:24px;margin:25px 0}.players{display:grid;grid-template-columns:repeat(2,1fr);gap:14px}audio{width:100%;display:block}.pending{color:#92400e}svg{width:100%;margin-top:18px}pre{white-space:pre-wrap}a{color:#0369a1}@media(max-width:800px){.players{grid-template-columns:1fr}}</style><h1>MDAC2 监督预测听评</h1><p>联合模型学习采音与排键，三流为原曲／人声／伴奏。这里只提供验证集审阅材料，尚未通过真人听评或真实游戏验收。</p><p>左右声道的击打声分别表示地面／空中；长条尾点采用较低音高提示。播放器从片段起点开始，图上的时间为全曲秒数。Gemini和双重击保留不同类型。实验BMS附预测旁表，尚不是游戏资源包。</p><p>''' + legend + '</p>' + ''.join(cards) + '''<script>
const tracks=[...document.querySelectorAll('audio')];
for(const card of document.querySelectorAll('article')){
 for(const a of card.querySelectorAll('audio')){
  a.addEventListener('play',()=>{for(const other of tracks)if(other!==a)other.pause()});
  a.addEventListener('timeupdate',()=>{const x=130+Math.min(1,a.currentTime/Number(card.dataset.length))*840;const line=card.querySelector('.playhead');line.setAttribute('x1',x);line.setAttribute('x2',x)});
 }
}
</script></html>'''
    (args.output/'预测听评.html').write_text(page,encoding='utf-8')
    atomic_json(args.output/'review.json',dict(frozen_sha256=sha(args.evaluation/'frozen.json'),examples=records,
        html_sha256=sha(args.output/'预测听评.html'),human_listening='pending',game_acceptance='pending',production_promoted=False,
        ffmpeg_sha256=sha(ffmpeg),clip_gain_normalization=False,clips_float32_preserve_peaks=True,
        overlay_sync='offline 24kHz waveform mix; player and SVG share currentTime; device output latency not calibrated'))
    print(json.dumps(dict(examples=len(records),players=len(records)*4,human_listening='pending'),ensure_ascii=True))


if __name__ == '__main__':
    main()
