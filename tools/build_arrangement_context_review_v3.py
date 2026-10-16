"""Prediction-first listening review with longer excerpts and a complete song."""
import argparse
from collections import Counter
import html
import json
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from build_supervised_review import choose_examples,make_clicks,COLORS,LABELS
from mdac2.data_preparation.supervised_training import atomic_json
from mdac2.data_preparation.audio_features import sha
from mdac2.data_preparation.six_export import write_bms,read_bms,semantics
from mdac2.data_preparation.training_chart import TrainingEvent
from mdac2.data_preparation.generated_constraints import validate_generated_events


def timeline(events,start,end,label):
    def x(t):return 100+(t-start)/(end-start)*1000
    pieces=[f'<svg viewBox="0 0 1120 150" role="img" aria-label="{label}">']
    for y,lane,title in ((45,'air','空中'),(92,'ground','地面')):
        pieces.append(f'<text x="3" y="{y+5}">{label} {title}</text><path d="M100 {y}H1100" stroke="#cbd5e1"/>')
        for e in events:
            if lane not in e['lanes'] or e['start']>=end or e['end']<start:continue
            left,right=x(max(start,e['start'])),x(min(end,e['end']));color=COLORS[e['kind']]
            title_text=html.escape(f"{LABELS[e['kind']]} {e['start']:.3f}–{e['end']:.3f}s")
            if e['kind']=='hold':pieces.append(f'<rect x="{left}" y="{y-8}" width="{max(2,right-left)}" height="16" fill="{color}" opacity=".5"><title>{title_text}</title></rect>')
            else:pieces.append(f'<circle cx="{left}" cy="{y}" r="5" fill="{color}"><title>{title_text}</title></circle>')
    step=max(1,round((end-start)/12))
    for t in range(int(start)+1,int(end),step):
        pieces.append(f'<text x="{x(t)}" y="135" font-size="12">{t}s</text>')
    pieces.append('<path class="playhead" d="M100 20V112" stroke="#0f172a" stroke-width="2"/></svg>')
    return ''.join(pieces)


def main(experiment,evaluation,output):
    output.mkdir(parents=True,exist_ok=False)
    config=json.loads((experiment/'onset.json').read_text(encoding='utf-8'))
    report=json.loads((evaluation/'validation.json').read_text(encoding='utf-8'))
    examples,charts=choose_examples(config,evaluation)
    ffmpeg=ROOT/'tools/.deps/ffmpeg/extracted/ffmpeg-9.0.1-essentials_build/bin/ffmpeg.exe'
    def ff(args):subprocess.run([str(ffmpeg),'-hide_banner','-loglevel','error','-n',*args],check=True,capture_output=True,timeout=120)
    focus_names=dict(vocal_onset='人声起音候选',vocal_sustain='人声长音候选',interlude='间奏候选',possible_leak='疑似分离漏音',
                     extra_heavy='特重击参考段',gemini='Gemini参考段',double_heavy='双重击参考段',cross_window_hold='跨窗长条参考段',full_song='整曲连续审阅')
    # Continuous listening exposes gaps and repetitive placement hidden by 8s clips.
    first=examples[0];row,ref=charts[first['chart_sha256']]
    examples=[dict(focus='full_song',chart_sha256=row['chart_sha256'],start=0.,end=ref['duration'],proxy={'selection':'same chart as vocal-onset example'})]+examples
    cards=[];records=[]
    for i,example in enumerate(examples):
        row,reference=charts[example['chart_sha256']]
        if example['focus']=='full_song':start,end=0.,reference['duration']
        else:
            center=(example['start']+example['end'])/2;start=max(0.,center-12);end=min(reference['duration'],start+24)
        decoded=json.loads((evaluation/'charts-predicted_onsets'/(row['chart_sha256']+'.json')).read_text(encoding='utf-8'))
        events=[TrainingEvent.from_dict(e) for e in decoded['events']]
        if validate_generated_events(events,reference['duration']):raise ValueError('Prediction violates rules')
        export=output/(row['chart_sha256']+'.bms')
        if events and not export.exists():
            bms=write_bms(events,reference['duration'])
            if semantics(events)!=semantics(read_bms(bms,reference['duration'])):raise ValueError('BMS round-trip mismatch')
            export.write_text(bms,encoding='utf-8');atomic_json(export.with_suffix('.json'),decoded)
        sources={'mix':Path(row['audio_path_absolute']),
                 'vocals':Path(config['cache'])/row['pcm']['sha256']/'vocals.wav',
                 'accompaniment':Path(config['cache'])/row['pcm']['sha256']/'accompaniment.wav'}
        clips={}
        for name,path in sources.items():
            target=output/f'{i:02d}-{name}.wav'
            trim=f'aresample=24000,atrim=start_sample={round(start*24000)}:end_sample={round(end*24000)},asetpts=PTS-STARTPTS'
            ff(['-i',str(path),'-af',trim,'-ac','2','-ar','24000','-c:a','pcm_f32le',str(target)])
            clips[name]=dict(file=target.name,sha256=sha(target),source=path.as_posix())
        clicks=output/f'{i:02d}-prediction-clicks.wav';make_clicks(clicks,decoded['events'],start,end)
        overlay=output/f'{i:02d}-prediction-mix.wav'
        ff(['-i',str(output/clips['mix']['file']),'-i',str(clicks),'-filter_complex','amix=inputs=2:duration=first:normalize=0','-c:a','pcm_f32le',str(overlay)])
        clips['prediction']=dict(file=overlay.name,sha256=sha(overlay))
        predicted=[e for e in decoded['events'] if e['start']<end and e['end']>=start]
        truth=[e for e in reference['events'] if e['start']<end and e['end']>=start]
        record=dict(**example,actual_start=start,actual_end=end,title=row['title'],clips=clips,
                    predicted_counts=dict(Counter(e['kind'] for e in predicted)),reference_counts=dict(Counter(e['kind'] for e in truth)),
                    predicted=predicted,reference=truth,human_listening='pending',game_acceptance='pending',
                    bms=export.name if export.exists() else None)
        records.append(record)
        players=''.join(f'<label>{label}<audio controls preload="none" src="{clips[key]["file"]}"></audio></label>' for key,label in [('mix','原曲'),('vocals','人声'),('accompaniment','伴奏')])
        link=f'<a href="{export.name}">整曲实验BMS</a> · <a href="{export.with_suffix(".json").name}">完整预测事件</a>' if export.exists() else '本曲空预测'
        cards.append(f'<article data-length="{end-start}"><h2>{i+1}. {focus_names[example["focus"]]}</h2><p>{html.escape(row["title"])} · {start:.3f}–{end:.3f}s · 验证集 · 待听评</p>'
                     f'<h3>实际预测：原曲＋模型击打</h3><audio controls preload="none" src="{overlay.name}"></audio>'
                     f'{timeline(predicted,start,end,"预测")}<p>预测物件 {len(predicted)}；{"、".join(LABELS[k]+" "+str(v) for k,v in record["predicted_counts"].items())}</p>'
                     f'<p>{link}</p><details><summary>展开参考谱对照（不是模型预测）</summary>{timeline(truth,start,end,"参考")}<p>参考物件 {len(truth)}</p></details>'
                     f'<details><summary>展开原曲／人声／伴奏试听与筛选依据</summary>{players}<pre>{html.escape(json.dumps(example["proxy"],ensure_ascii=False,indent=2))}</pre></details></article>')
    m=report['free_running']['predicted_onsets'];legend=' · '.join(f'<span style="color:{c}">{LABELS[k]}</span>' for k,c in COLORS.items())
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>MDAC2 排键 v3 实际预测审阅</title><style>
body{font:16px/1.65 system-ui;margin:32px auto;padding:0 24px;max-width:1180px;color:#172033;background:#f1f5f9}article{padding:24px;background:white;border:2px solid #cbd5e1;border-radius:12px;margin:24px 0}h3{color:#164e63}audio,svg{width:100%}details{margin:18px 0;background:#f8fafc;padding:12px}summary{cursor:pointer}pre{white-space:pre-wrap}.pending{padding:18px;background:#fff7ed;border-left:5px solid #c2410c}a{color:#0369a1}
</style><h1>排键 v3：实际预测审阅</h1><p class="pending">本页默认展示和播放的是模型实际预测。参考谱折叠在单独的对照中。训练执行通过不代表听感合格或游戏可用；当前均待人工验收。v3齿轮已移出监督且后处理未启用；采音保持旧模型，仍可能产生齿轮来源落点。参考对照保留原谱齿轮供识别，主指标排除齿轮。人声／间奏场景仅由代理指标筛选，没有声部真值。</p>'''
    page+=f'<p>22首验证整链：采音F1 {m["sampling"]["f1"]:.2%}；五类宏F1 {m["macro_five_class_f1"]:.2%}；双轨击打比例 {m["dual_attack_fraction"]:.2%}。这些指标不能代替试听。</p><p>地面点击在左声道、空中在右声道；长条尾为低音提示。无BPM参考、无节拍吸附、无空段强填；BMS的固定120仅为按绝对秒序列化的载体，不是模型输入。整曲和24秒片段来自同一份预测。</p><p>{legend}</p>'
    page+=''.join(cards)+'''<script>
const tracks=[...document.querySelectorAll('audio')];
for(const card of document.querySelectorAll('article'))for(const player of card.querySelectorAll('audio')){
player.addEventListener('play',()=>{for(const p of tracks)if(p!==player)p.pause()});
player.addEventListener('timeupdate',()=>{let x=100+1000*Math.min(1,player.currentTime/Number(card.dataset.length));for(const line of card.querySelectorAll('.playhead'))line.setAttribute('d',`M${x} 20V112`)});
}
</script></html>'''
    (output/'预测听评.html').write_text(page,encoding='utf-8')
    atomic_json(output/'review.json',dict(examples=records,prediction_first=True,reference_collapsed=True,
                 frozen_sha256=sha(evaluation/'frozen.json'),html_sha256=sha(output/'预测听评.html'),
                 human_listening='pending',game_acceptance='pending',production_promoted=False,
                 sync='offline 24kHz overlay; complete-song and excerpt events come from same JSON'))
    print(json.dumps(dict(examples=len(records),path=(output/'预测听评.html').as_posix()),ensure_ascii=True))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--experiment',type=Path,required=True);p.add_argument('--evaluation',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();main(a.experiment.resolve(),a.evaluation.resolve(),a.output.resolve())
