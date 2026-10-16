"""Versioned rating-range selection over an existing community preparation."""
import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from decimal import Decimal
import json
from pathlib import Path
import unicodedata

from .community import digest, pcm_identity, rating_value, resolve_music, write_json


def in_range(value, minimum, maximum):
    lower, upper = rating_value(minimum), rating_value(maximum)
    if lower is None or upper is None or lower > upper:
        raise ValueError('Ordered finite nonnegative rating bounds required')
    rating = rating_value(value)
    return rating is not None and lower <= rating <= upper


def group_families(rows):
    parent = {r['chart_sha256']: r['chart_sha256'] for r in rows}
    def find(key):
        while key != parent[key]:
            parent[key] = parent[parent[key]]; key = parent[key]
        return key
    groups = defaultdict(list)
    for row in rows:
        groups[('chart', row['chart_sha256'])].append(row)
        if 'pcm' in row: groups[('pcm', row['pcm']['sha256'])].append(row)
        title = ''.join(c for c in unicodedata.normalize('NFKC', str(row['title'])).casefold() if c.isalnum())
        if title: groups[('title', title)].append(row)
    for group in groups.values():
        for row in group[1:]:
            a, b = find(group[0]['chart_sha256']), find(row['chart_sha256'])
            parent[max(a,b)] = min(a,b)
    for row in rows:
        row['provisional_family'] = 'community-'+find(row['chart_sha256'])[:20]
        row['family_review'] = 'pending'


def select_range(root, output, *, minimum, maximum, ffmpeg, ffprobe):
    in_range(0, minimum, maximum)
    root = root.resolve(strict=True); output = output.resolve()
    inventory_path = root/'inventory.json'
    inventory_hash = digest(inventory_path)
    inventory = json.loads(inventory_path.read_text(encoding='utf-8'))
    if inventory['status'] != 'completed':
        raise ValueError('Completed source preparation required')
    output.mkdir(parents=True, exist_ok=False)
    original_ids = {r['chart_sha256'] for r in inventory['charts'] if r['selected_9_10']}
    rows = [deepcopy(r) for r in inventory['charts'] if in_range(r['rating'], minimum, maximum)]
    identity = {name: digest(path) for name, path in (('ffmpeg_sha256',ffmpeg),('ffprobe_sha256',ffprobe))}
    same_decoder = all(inventory.get('preparation',{}).get(k)==v for k,v in identity.items())
    cache = {r['audio_sha256']:r['pcm'] for r in inventory['charts'] if same_decoder and 'pcm' in r}
    errors = []; decoded = 0; counts = Counter(); object_charts = defaultdict(set)
    for index, row in enumerate(rows, 1):
        row.pop('selected_9_10', None)
        row['selected'] = True
        row['training_eligible'] = False
        row['split'] = 'unassigned_architecture_pending'
        chart_path = root/'expanded'/row['package_hash']/row['chart_path']
        if digest(chart_path) != row['chart_sha256']:
            raise ValueError('Source chart hash mismatch')
        row['chart_path_absolute'] = chart_path.as_posix()
        raw_path = root/row['raw_label_path']
        raw = json.loads(raw_path.read_text(encoding='utf-8'))
        row['raw_label_sha256'] = digest(raw_path)
        for cell in raw['cells']:
            if 'code' in cell:
                key = cell['channel']+':'+cell['code']
                counts[key] += 1; object_charts[key].add(row['chart_sha256'])
        try:
            music = resolve_music(root/'expanded'/row['package_hash'])
            audio_hash = digest(music)
            if audio_hash not in cache:
                cache[audio_hash] = pcm_identity(music,ffmpeg,ffprobe); decoded += 1
            row['audio_sha256'] = audio_hash
            row['audio_path'] = music.relative_to(root).as_posix()
            row['pcm'] = cache[audio_hash]
            row['audio_status'] = 'verified'
            if 'missing_or_ambiguous_music' in row['issues']:
                row['issues'].remove('missing_or_ambiguous_music')
                row['resolved_preparation_issues'] = ['missing_or_ambiguous_music']
        except Exception as exc:
            row.pop('pcm',None)
            row['audio_status'] = 'error'
            row['audio_error'] = str(exc)
            errors.append({'chart_sha256':row['chart_sha256'],'error':str(exc)})
        if index%25==0: print(f'Checked {index}/{len(rows)} charts',flush=True)
    if digest(inventory_path)!=inventory_hash: raise ValueError('Source inventory changed')
    group_families(rows)
    selected_ids = {r['chart_sha256'] for r in rows}
    audio = {r['pcm']['sha256']:r['pcm'] for r in rows if 'pcm' in r}
    summary = {'charts':len(rows),'unique_charts':len(selected_ids),'unique_pcm':len(audio),
        'audio_hours':round(sum(p['seconds'] for p in audio.values())/3600,4),
        'families':len({r['provisional_family'] for r in rows}),
        'added_from_9_10':len(selected_ids-original_ids),'removed_from_9_10':len(original_ids-selected_ids),
        'difficulty_conflicts':sum('difficulty_metadata_conflict' in r['issues'] for r in rows),
        'parse_errors':sum(bool(r['parse_errors']) for r in rows),'newly_decoded_audio':decoded,
        'audio_verified_charts':sum(r['audio_status']=='verified' for r in rows),'errors':len(errors)}
    report = {'schema_version':1,'status':'completed' if not errors else 'completed_with_errors',
        'source_root':root.as_posix(),'source_inventory_sha256':inventory_hash,
        'selector_sha256':digest(Path(__file__)), 'preparation_module_sha256':digest(Path(__file__).with_name('community.py')), 'decoder':identity,
        'rating_policy':{'minimum':str(minimum),'maximum':str(maximum),'inclusive':True,
                         'priority':['modern','legacy','bms']},
        'training_started':False,'architecture_status':'pending','summary':summary,'errors':errors,'charts':rows}
    write_json(output/'selection.json',report)
    write_json(output/'object-inventory.json',{'scope':'raw markers, not decoded gameplay events',
        'counts':dict(sorted(counts.items())),
        'chart_counts':{k:len(v) for k,v in sorted(object_charts.items())}})
    clean=lambda s:str(s).replace('|','／').replace('\n',' ').replace('\r',' ')
    lines=[f'# {minimum}～{maximum} 级候选谱面（含两端）','',
        f"{len(rows)} 张谱面，{len(audio)} 份不同规范音频，约 {summary['audio_hours']:.2f} 小时。",
        '', f"主音频核验通过 {summary['audio_verified_charts']} 张，失败 {len(errors)} 张。未开始训练；等级冲突保留待复核，物件归并表未应用。",
        '', '| 序号 | 歌曲 | 等级 | BMS 等级 | 等级冲突 | 音频 | 原谱 |','|---|---|---|---|---|---|---|']
    for index,row in enumerate(sorted(rows,key=lambda r:(Decimal(r['rating']),str(r['title']))),1):
        conflict='是' if 'difficulty_metadata_conflict' in row['issues'] else ''
        lines.append(f"| {index} | {clean(row['title'])} | {row['rating']} | {clean(row['rating_sources']['bms'])} | {conflict} | {row['audio_status']} | [BMS](<{row['chart_path_absolute']}>) |")
    (output/'候选谱面清单.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(summary),flush=True)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','output','ffmpeg','ffprobe'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--minimum',required=True);p.add_argument('--maximum',required=True)
    args=vars(p.parse_args())
    return 0 if select_range(**args)['status']=='completed' else 1
