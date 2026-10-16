"""Prepare immutable six-class labels and family roles, never train a model."""
import argparse
from collections import Counter, defaultdict
from fractions import Fraction
import hashlib
import html
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from mdac2.data_preparation.community import digest, write_json
from mdac2.data_preparation.community_labels import convert, LOADER_COMMIT
from mdac2.data_preparation.training_chart import CLASSES, POLICY_ID


def split_roles(charts, seed):
    families=sorted({c['provisional_family'] for c in charts},
                    key=lambda f:hashlib.sha256(f'{seed}:{f}'.encode()).hexdigest())
    n=len(families);test=max(1,round(n*.1));validation=max(1,round(n*.1))
    if n < 3: raise ValueError('At least three independent families required')
    return {f:('locked_test' if i<test else 'validation' if i<test+validation else 'train')
            for i,f in enumerate(families)}


def build(selection_path, output, seed=20260927):
    selection_path=selection_path.resolve(strict=True);output=output.resolve()
    selection_bytes=selection_path.read_bytes();selection=json.loads(selection_bytes)
    if selection['status']!='completed' or selection['errors']:
        raise ValueError('Verified selection required')
    source=Path(selection['source_root']);output.mkdir(parents=True,exist_ok=False)
    (output/'labels').mkdir()
    roles=split_roles(selection['charts'],seed)
    rows=[];class_counts=defaultdict(Counter);issues=Counter();examples=[];example_roles=set()
    for index,row in enumerate(selection['charts'],1):
        raw_path=source/row['raw_label_path']
        if digest(raw_path)!=row['raw_label_sha256'] or digest(Path(row['chart_path_absolute']))!=row['chart_sha256']:
            raise ValueError('Source label/chart changed')
        raw=json.loads(raw_path.read_text(encoding='utf-8'))
        result=convert(raw,chart_hash=row['chart_sha256'],duration=row['pcm']['seconds'])
        label_path=output/'labels'/(row['chart_sha256']+'.json');write_json(label_path,result)
        role=roles[row['provisional_family']]
        counts=Counter(e['kind'] for e in result['events'])
        if result['training_eligible']:class_counts[role].update(counts)
        problems=Counter(i.get('reason',i.get('code')) for i in result['unresolved']+result['issues'])
        issues.update(problems)
        out={**row,'split':role,'training_eligible':result['training_eligible'],
             'labels':label_path.relative_to(output).as_posix(),'labels_sha256':digest(label_path),
             'audio_path_absolute':(source/row['audio_path']).as_posix(),
             'class_counts':dict(counts),'label_problems':dict(problems),
             'difficulty_policy':'modern_then_legacy_then_bms; conflict retained, not human reviewed',
             'family_review':'automatic_exact_pcm_and_normalized_title; variants may remain',
             'reference_kind':'community_supervision_not_gold',
             'use_basis':'user-authorized local preparation and planned training; no redistribution'}
        rows.append(out)
        if role=='train':
            for e in result['events']:
                category=e['source_role']
                if category in {'0C','0I','note','boss_single_melee','masher_start','boss_masher',
                                'hold','tap_hold','gemini_pair','0H','0A','0D'} and category not in example_roles:
                    examples.append({'title':row['title'],'chart_sha256':row['chart_sha256'],
                                     'audio_path':out['audio_path_absolute'],'start':max(0,e['start']-2),
                                     'end':min(row['pcm']['seconds'],e['end']+2),'focus':category,
                                     'events':[v for v in result['events'] if e['start']-2<=v['start']<=e['end']+2],
                                     'source':[v for v in result.get('source_events',[]) if e['start']-2<=v['seconds']<=e['end']+2]})
                    example_roles.add(category)
        if index%50==0:print(f'Labels {index}/{len(selection["charts"])}',flush=True)
    if selection_path.read_bytes()!=selection_bytes:raise ValueError('Selection changed')
    # Repeated PCM and provisional families must share roles, even for quarantined references.
    for key in ('provisional_family','audio_sha256'):
        seen={}
        for r in rows:
            if r[key] in seen and seen[r[key]]!=r['split']:raise ValueError('Split leakage')
            seen[r[key]]=r['split']
    pcm_roles={}
    for r in rows:
        pcm=r['pcm']['sha256']
        if pcm in pcm_roles and pcm_roles[pcm]!=r['split']:raise ValueError('PCM split leakage')
        pcm_roles[pcm]=r['split']
    summary={'charts':len(rows),'eligible_charts':sum(r['training_eligible'] for r in rows),
             'quarantined_charts':sum(not r['training_eligible'] for r in rows),
             'eligible_by_role':dict(Counter(r['split'] for r in rows if r['training_eligible'])),
             'families_by_role':dict(Counter(roles.values())),
             'class_counts_by_role':dict(class_counts),'problems':dict(issues),
             'difficulty_conflicts':sum('difficulty_metadata_conflict' in r['issues'] for r in rows),
             'unique_audio':len(pcm_roles),'review_examples':len(examples)}
    report={'schema_version':1,'policy':POLICY_ID,'classes':CLASSES,'loader_commit':LOADER_COMMIT,
            'selection_path':selection_path.as_posix(),'selection_sha256':hashlib.sha256(selection_bytes).hexdigest(),
            'seed':seed,'training_started':False,'split_frozen':True,
            'family_review':'provisional_automatic; do not claim fully independent human-reviewed test set',
            'code':{p.name:digest(p) for p in [Path(__file__),ROOT/'src/mdac2/data_preparation/community_labels.py',
                      ROOT/'src/mdac2/data_preparation/training_chart.py']},
            'summary':summary,'charts':rows}
    write_json(output/'manifest.json',report);write_json(output/'review-examples.json',examples)
    lines=['# 训练前标签准备报告','',f"候选{len(rows)}张；初版可用{summary['eligible_charts']}张；隔离待审查{summary['quarantined_charts']}张。",
           '', '未开始训练。可用表示自动规则检查通过，不等于人工采音／游戏验收。',
           '歌曲族按固定种子分配，审阅例子仅来自训练角色。翻唱／重混的族关系仍可能需要人工合并；合并后必须重新冻结并换版本。',
           '', '| 数据角色 | 可用谱数 | 六类事件数 |','|---|---|---|']
    for role in ('train','validation','locked_test'):
        lines.append(f"| {role} | {summary['eligible_by_role'].get(role,0)} | {dict(class_counts[role])} |")
    lines+=['','## 隔离问题','',*['- '+k+'：'+str(v) for k,v in issues.items()],
            '', '## 每谱结果','', '| 歌曲 | 等级 | 角色 | 状态 | 问题 |','|---|---|---|---|---|']
    for r in rows:
        title=str(r['title']).replace('|','／').replace('\n',' ')
        lines.append(f"| {title} | {r['rating']} | {r['split']} | {'可用' if r['training_eligible'] else '待审查'} | {r['label_problems']} |")
    (output/'标签准备报告.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    write_review(output,examples)
    print(json.dumps(summary,ensure_ascii=False),flush=True)
    return report


def write_review(output, examples):
    cards=[]
    colors={'light':'#7dd3fc','heavy':'#fb923c','extra_heavy':'#ef4444','gemini':'#c084fc','hold':'#4ade80','gear':'#94a3b8'}
    for i,x in enumerate(examples):
        width=1100;span=max(.001,x['end']-x['start']);svg=[]
        for e in x['events']:
            pos=20+(e['start']-x['start'])/span*(width-40)
            for lane in e['lanes']:
                y=50 if lane=='air' else 115
                length=max(7,(e['end']-e['start'])/span*(width-40))
                tooltip=html.escape(f"{e['kind']} {e['start']:.3f}–{e['end']:.3f} 来源{e['source_code']}")
                svg.append(f'<rect x="{pos:.2f}" y="{y}" width="{length:.2f}" height="20" fill="{colors[e["kind"]]}"><title>{tooltip}</title></rect>')
        source='；'.join(f"{v['seconds']:.3f}s {v['channel']}:{v['code']}" for v in x['source'])
        uri=Path(x['audio_path']).as_uri()+f"#t={x['start']:.3f},{x['end']:.3f}"
        cards.append(f'<section><h2>{i+1}. {html.escape(str(x["title"]))} · {html.escape(x["focus"])}</h2><p>{x["start"]:.3f}–{x["end"]:.3f}秒 · 训练角色审阅片段</p><audio controls preload="none" src="{html.escape(uri,quote=True)}"></audio><svg viewBox="0 0 {width} 155"><text x="0" y="38">空中</text><text x="0" y="103">地面</text>{"".join(svg)}</svg><details><summary>原始标记与转换事件</summary><p>{html.escape(source)}</p><pre>{html.escape(json.dumps(x["events"],ensure_ascii=False,indent=2))}</pre></details></section>')
    page='<!doctype html><meta charset="utf-8"><title>六类标签归并审阅</title><style>body{background:#111827;color:#e5e7eb;font:16px system-ui;max-width:1200px;margin:32px auto;padding:20px}section{background:#1f2937;padding:24px;border-radius:12px;margin:24px 0}svg{width:100%;fill:#cbd5e1}pre{white-space:pre-wrap;font-size:12px}audio{width:100%}</style><h1>六类标签归并审阅</h1><p>蓝：轻击　橙：重击　红：排他特重击　紫：Gemini　绿：长条　灰：齿轮。鼠标悬停查看事件。原谱未改写，未开始训练。</p><p>请在本机浏览器直接打开此文件以播放本地音频。本页是审阅材料，不表示已有人工通过结论。</p>'+''.join(cards)
    (output/'归并审阅.html').write_text(page,encoding='utf-8')


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8',errors='backslashreplace')
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--selection',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seed',type=int,default=20260927)
    a=p.parse_args();build(a.selection,a.output,a.seed)
