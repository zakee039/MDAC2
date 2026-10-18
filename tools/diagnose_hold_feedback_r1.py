"""Read-only diagnostics for the user's two Snow Song Show hold examples."""
import argparse
from collections import Counter
from datetime import datetime, timedelta, timezone
import hashlib
import html
import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from mdac2.data_preparation.training_chart import TrainingEvent
from mdac2.data_preparation.generated_constraints import validate_generated_events


def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def cropped(events,start,end):
    return [e for e in events if e['start']<end and e['end']>=start]


def hold_stats(events,start,end):
    lanes={lane:sorted([e for e in events if e['kind']=='hold' and lane in e['lanes'] and e['end']>start and e['start']<end],key=lambda e:e['start']) for lane in ['air','ground']}
    overlaps=[]
    for a in lanes['air']:
        for g in lanes['ground']:
            left,right=max(a['start'],g['start'],start),min(a['end'],g['end'],end)
            if right>left:overlaps.append(dict(start=left,end=right,air=a['event_id'],ground=g['event_id']))
    gaps={lane:[dict(left=a['event_id'],right=b['event_id'],tail=a['end'],head=b['start'],
                     gap_ms=round((b['start']-a['end'])*1000,6)) for a,b in zip(es,es[1:])] for lane,es in lanes.items()}
    return dict(interval=[start,end],holds={k:len(v) for k,v in lanes.items()},same_lane_gaps=gaps,
                cross_lane_overlaps=overlaps,dual_hold_seconds=sum(x['end']-x['start'] for x in overlaps))


def dense_clicks(chart):
    records=[]
    for hold in chart['events']:
        if hold['kind']!='hold':continue
        for lane in hold['lanes']:
            notes=sorted([n for n in chart['events'] if n['kind']!='hold' and lane in n['lanes'] and hold['start']<n['start']<hold['end']],key=lambda n:n['start'])
            for a,b in zip(notes,notes[1:]):
                if b['start']-a['start']<=.5000001:
                    target='air' if lane=='ground' else 'ground'
                    other=[n for n in chart['events'] if target in n['lanes'] and n['kind']!='hold' and abs(n['start']-b['start'])<=.0200001]
                    other_holds=[n['event_id'] for n in chart['events'] if n['kind']=='hold' and target in n['lanes'] and n['start']<b['start']<n['end']]
                    records.append(dict(hold=hold['event_id'],lane=lane,notes=[a,b],gap_ms=round((b['start']-a['start'])*1000,6),target_notes=other,target_holds=other_holds))
    return records


def timeline(events,start,end,title):
    x=lambda t:100+(t-start)/(end-start)*1000
    pieces=[f'<h3>{html.escape(title)}</h3><svg viewBox="0 0 1140 180" role="img" aria-label="{html.escape(title)}">']
    for lane,y in [('air',40),('ground',100)]:
        pieces.append(f'<text x="8" y="{y+5}">{"天空" if lane=="air" else "地面"}</text><path d="M100 {y}H1100" stroke="#b5c2cf"/>')
        for e in cropped(events,start,end):
            if lane not in e['lanes']:continue
            left,right=x(max(start,e['start'])),x(min(end,e['end']))
            desc=html.escape(f"{e['event_id']} {e['kind']} {e['start']:.3f}–{e['end']:.3f}")
            if e['kind']=='hold':
                pieces.append(f'<rect x="{left}" y="{y-8}" width="{max(1,right-left)}" height="16" fill="#74b2d0"><title>{desc}</title></rect>')
                if start<=e['start']<=end:pieces.append(f'<path d="M{left} {y-17}v34" stroke="#b62c45" stroke-width="2"/>')
                if start<=e['end']<=end:pieces.append(f'<path d="M{right-1} {y-12}v24" stroke="#26384c" stroke-width="1.5"/>')
            else:pieces.append(f'<circle cx="{left}" cy="{y}" r="4" fill="#128248"><title>{desc}</title></circle>')
    for t in np.arange(np.ceil(start),end+1e-8,1):pieces.append(f'<text x="{x(t)-15}" y="155">{t:.0f}s</text>')
    pieces.append('</svg>');return ''.join(pieces)


def main(output):
    if output.exists():raise FileExistsError('Preserve existing diagnosis')
    exp=ROOT/'tools/private/community/vocal-grid-r2-20261017'
    recipe=read(ROOT/'recipes/R1/manifest.json')
    registry=read(ROOT/'models/registry.json')
    if sha(ROOT/'recipes/R1/manifest.json')!=registry['recipes']['R1']['manifest_sha256']:raise ValueError('R1 identity changed')
    original=read(exp/'snow/neural-chart.json');final=read(exp/'snow/chart.json');post=read(exp/'snow/postprocessing.json')
    if sha(exp/'snow/chart.json')!=read(exp/'snow-package/package-report.json')['prediction_chart_sha256']:raise ValueError('Approved chart changed')
    holds=[e for e in final['events'] if e['event_id']=='dual-1056-0'];assert len(holds)==1
    tail_hold=holds[0];next_note=next(e for e in final['events'] if e['event_id']=='dual-1057-0')
    with np.load(exp/'snow/arrangement-scores.npz') as a:
        predicted_length=float(np.expm1(a['log_lengths'][1056,1]));predicted_start=float(a['times'][1056])
    assert abs(predicted_start+predicted_length-256.309)<.00051
    f=np.load(ROOT/'tools/private/playtests/20261016-snow-song-show-v1/features/features.npy',mmap_mode='r')
    energy=[]
    for a,b in [(252.565,256.310),(256.310,259.725),(259.725,260.5)]:
        p=np.expm1(np.asarray(f[:,:,round(a*100):round(b*100)],np.float64))/10000
        energy.append(dict(interval=[a,b],mean_mel_power_db={k:float(v) for k,v in zip(['mix','vocals','accompaniment'],10*np.log10(np.maximum(p.mean((1,2)),1e-12)))}))
    neural_stats=hold_stats(original['events'],172.,176.)
    final_stats=hold_stats(final['events'],172.,176.)
    rule_issues=validate_generated_events([TrainingEvent.from_dict(e) for e in final['events']],final['duration'])
    assert not rule_issues
    changes=[c for c in post['alignment']['changes'] if any(171.99<t<176.14 for t in c['before']+c['after'])]
    dense={s:dense_clicks(read(exp/s/'chart.json')) for s in ['pipa','snow']}
    sources={str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'recipes/R1/manifest.json',exp/'snow/chart.json',exp/'snow/neural-chart.json',exp/'snow/arrangement-scores.npz',exp/'snow/postprocessing.json',Path(__file__)]}
    report=dict(status='diagnosed_no_chart_change',created_at=datetime.now(timezone(timedelta(hours=8))).isoformat(),
        recipe='R1',base_model='v2',source_sha256=sources,
        image1=dict(hold=tail_hold,next_note=next_note,gap_seconds=next_note['start']-tail_hold['end'],
            neural_start=predicted_start,neural_duration=predicted_length,neural_end=predicted_start+predicted_length,
            postprocess_endpoint_move_ms=1,nearby_alignment_limit_ms=100,energy=energy,
            mechanism='per-lane duration regression at hold head; no independent continuous sound-offset detector; no later tail edit here',
            caveat='Continuing energy supports non-silence, not proof of a single stable musical source or a gold hold boundary.'),
        image2=dict(neural=neural_stats,final=final_stats,alignment_changes=changes,
            same_lane_illegal_overlap=False,same_lane_zero_gap_retrigger=True,
            mechanism='Model produces staggered two-lane holds; 100ms endpoint averaging closes same-lane gaps without merging event identities.',
            visualization='Old preview rectangles lack distinct head/tail markers, hiding repeated boundaries.'),
        feedback2=dict(request='Same-lane instantaneous clicks strictly inside a hold should be more than 500ms apart; prefer opposite lane, merge compatible coincident click.',
            strict_5ms_minimum_seconds=.505,measure='Consecutive instantaneous events strictly inside each same-lane hold; excludes endpoint attacks and separately reports grouped objects.',
            findings=dense,implementation='not applied; R1 frozen',
            guardrails=['Never turn a click into Gemini merely to merge.', 'Preserve atomic Gemini and double-heavy semantics.',
                'Check occupancy and density on target lane; two occupied lanes may prevent a safe move.',
                'Merge only genuinely co-located compatible events, not arbitrary notes elsewhere on the other lane.']),
        proposed_next_revision=dict(status='not implemented or named as a release',
            areas=['Audio-supported sustained tail endpoint checking, separate from 100ms local snapping.',
                'Same-lane hold continuation vs deliberate retrigger classification before endpoint snapping.',
                'Staggered cross-lane hold load and clear handoff diagnostics.',
                'Hold-internal click relocation/compatible merge and post-mirror final checks.']),
        actual_command=[sys.executable,*sys.argv])
    output.mkdir(parents=True)
    attachments=[]
    for i,name in enumerate(['codex-clipboard-5badc8dc-0180-419b-a6ed-766607e8463c.png','codex-clipboard-1745e659-23e5-4571-8273-870bc1e86a41.png'],1):
        p=Path('C:/Users/mamin/AppData/Local/Temp')/name
        if p.exists():
            dst=output/f'user-image-{i}.png';shutil.copyfile(p,dst);attachments.append(dict(file=dst.name,sha256=sha(dst)))
    report['attachments']=attachments
    (output/'diagnosis.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    rows=''.join(f"<tr><td>{html.escape(e['event_id'])}</td><td>{html.escape('/'.join(e['lanes']))}</td><td>{e['start']:.3f}</td><td>{e['end']:.3f}</td></tr>" for e in cropped(final['events'],171.99,176.01) if e['kind']=='hold')
    page='''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>R1 长条反馈定位</title><style>body{font:16px/1.7 system-ui;max-width:1150px;margin:28px auto;padding:20px;color:#24354b}svg{width:100%;background:#f3f7fa}svg text{font-size:14px}td,th{padding:5px 14px;border-bottom:1px solid #ddd}h2{margin-top:36px}</style><h1>R1 长条反馈定位 · 原事件未修改</h1><p>蓝条是每一根实际长条；红线标记长条头，深色线标记尾。相邻条即使没有间隙也保留各自边界。</p><h2>图2：交错双轨长条与零间隙重复起按</h2><p>172–176秒双轨同时占用：模型原始2.934秒，R1最终3.440秒。地面7根独立长条被首尾对齐，形成171.995–178.340秒连续占用；数据仍有7个头。没有同轨区间相交的硬规则错误，但手感约束不足。</p>'''
    page+=timeline(original['events'],171,179,'模型原始输出')+timeline(final['events'],171,179,'R1 最终输出（增加头尾标记）')
    page+='<table><tr><th>事件</th><th>轨道</th><th>头／秒</th><th>尾／秒</th></tr>'+rows+'</table>'
    page+='<h2>图1：预测时长提前结束</h2><p>空中长条252.565–256.310秒，下一点击259.725秒，相差3.415秒。原模型时长3.744591秒，后处理仅取整约1ms，没有切掉后半段。100ms附近吸附不会跨过3.415秒。区间伴奏仍有能量，但能量本身不能证明应该保持同一根长条。</p>'
    page+=timeline(final['events'],252,261,'R1 末段原事件')
    page+='<h2>长条内点击规则反馈</h2><p>按严格内部、同轨相邻点击扫描：琵琶行0对，Snow 1对（8.295和8.645秒，相隔350ms）。头尾边界和跨轨长条另列，不能混作一种问题。新规则尚未应用；R1已冻结，后续修订另建版本。</p><p><a href="diagnosis.json">完整定位数据与身份</a></p></html>'
    (output/'index.html').write_text(page,encoding='utf-8')
    print(json.dumps(dict(status=report['status'],image1_gap=report['image1']['gap_seconds'],image2_dual_seconds=[neural_stats['dual_hold_seconds'],final_stats['dual_hold_seconds']],dense_pairs={k:len(v) for k,v in dense.items()},page=str(output/'index.html'))))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();main(a.output.resolve())
