"""Copy and inventory community packages without interpreting game objects."""
import argparse
from collections import Counter, defaultdict
from decimal import Decimal, InvalidOperation
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import unicodedata
import zipfile


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            result.update(chunk)
    return result.hexdigest()


def write_json(path, value):
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    os.replace(temp, path)


def safe_members(archive):
    members = archive.infolist()
    if len(members) > 20000 or sum(m.file_size for m in members) > 8*1024**3:
        raise ValueError('Archive size budget exceeded')
    seen = set()
    for member in members:
        name = member.filename.replace('\\', '/')
        path = PurePosixPath(name)
        mode = member.external_attr >> 16
        if (path.is_absolute() or not path.parts or any(p in ('..', '.') or ':' in p or
                p.rstrip(' .') != p or p.split('.')[0].upper() in {'CON','PRN','AUX','NUL',*[f'COM{i}' for i in range(1,10)],*[f'LPT{i}' for i in range(1,10)]}
                for p in path.parts) or stat.S_ISLNK(mode)):
            raise ValueError('Unsafe archive member: '+name)
        key = name.rstrip('/').casefold()
        if key in seen:
            raise ValueError('Duplicate archive member: '+name)
        seen.add(key)
        if member.flag_bits & 1:
            raise ValueError('Encrypted archive member')
    return members


def extract(archive_path, target):
    if target.exists():
        raise FileExistsError(target)
    staging = target.with_name(target.name+'.staging')
    staging.mkdir()
    try:
        with zipfile.ZipFile(archive_path) as archive:
            members = safe_members(archive)
            for member in members:
                destination = staging.joinpath(*PurePosixPath(member.filename.replace('\\','/')).parts)
                if member.is_dir():
                    destination.mkdir(parents=True, exist_ok=True)
                    continue
                destination.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member) as src, destination.open('xb') as dst:
                    shutil.copyfileobj(src, dst, 1024*1024)
            # Reading every member above verifies ZIP CRC before publication.
        os.replace(staging, target)
    except Exception:
        # Preserve staging for diagnosis; no successful directory is published.
        raise


def decode_text(raw):
    for encoding in ('utf-8-sig', 'gb18030'):
        try:
            return raw.decode(encoding), encoding
        except UnicodeDecodeError:
            pass
    raise ValueError('Unsupported text encoding')


def resolve_music(root):
    """Find the unique main song, excluding demos and unrelated audio assets."""
    candidates = [p for p in root.rglob('*') if p.is_file()
                  and p.stem.casefold() == 'music'
                  and p.suffix.casefold() in {'.ogg', '.mp3', '.wav', '.flac', '.m4a'}]
    if len(candidates) != 1:
        raise ValueError('Missing or ambiguous music')
    return candidates[0]


def rating_value(raw):
    try:
        value = Decimal(str(raw))
        return value if value.is_finite() and value >= 0 else None
    except InvalidOperation:
        return None


def inspect_bms(path):
    text, encoding = decode_text(path.read_bytes())
    headers = defaultdict(list); channels = Counter(); objects = Counter()
    cells = []; errors = []
    for line_no, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line or line.startswith('*') or line.startswith('//'):
            continue
        match = re.fullmatch(r'#(\d{3})([0-9A-Za-z]{2}):(.+)', line)
        if match:
            measure, channel, data = int(match[1]), match[2].upper(), match[3].strip().upper()
            channels[channel] += 1
            if channel == '02':
                if rating_value(data) is None or Decimal(data) <= 0:
                    errors.append({'line': line_no, 'reason': 'invalid measure length'})
                cells.append({'measure': measure, 'channel': channel, 'raw': data, 'line': line_no})
            elif len(data)%2 or not re.fullmatch('[0-9A-Z]+', data):
                errors.append({'line': line_no, 'reason': 'invalid raw cells'})
            else:
                count = len(data)//2
                for index in range(count):
                    code = data[2*index:2*index+2]
                    if code != '00':
                        cells.append({'measure': measure, 'channel': channel, 'index': index,
                                      'divisions': count, 'code': code, 'line': line_no})
                        objects[code] += 1
        elif (match := re.fullmatch(r'#([A-Za-z0-9]+)(?:\s+(.*))?', line)):
            headers[match[1].upper()].append((match[2] or '').strip())
        else:
            errors.append({'line': line_no, 'reason': 'unparsed directive', 'raw':line})
    return {'schema_version': 1, 'encoding': encoding, 'headers': dict(headers),
            'channels': dict(channels), 'raw_code_counts': dict(objects), 'cells': cells,
            'errors': errors, 'semantics': 'raw_only_unverified', 'gold_eligible': False}


def pcm_identity(path, ffmpeg, ffprobe):
    probe = subprocess.run([str(ffprobe), '-v','error','-show_streams','-show_format','-of','json',str(path)],
                           capture_output=True, timeout=60, check=True)
    metadata = json.loads(probe.stdout)
    streams = [s for s in metadata['streams'] if s['codec_type']=='audio']
    if len(streams) != 1:
        raise ValueError('Exactly one music audio stream required')
    # Standard identity only; do not persist PCM/features before architecture agreement.
    with tempfile.TemporaryFile() as errors:
        process = subprocess.Popen([str(ffmpeg), '-v','error','-i',str(path),'-map','0:a:0',
            '-ar','48000','-ac','2','-f','s16le','pipe:1'], stdout=subprocess.PIPE, stderr=errors)
        sha = hashlib.sha256(); byte_count = 0
        try:
            for chunk in iter(lambda: process.stdout.read(1024*1024), b''):
                byte_count += len(chunk); sha.update(chunk)
                if byte_count > 48000*4*1800:
                    raise ValueError('Audio exceeds 30 minute preparation budget')
            code = process.wait(timeout=60)
            if code or not byte_count or byte_count%4:
                raise ValueError('Audio decoding failed')
        finally:
            if process.poll() is None:
                process.kill(); process.wait()
            process.stdout.close()
    return {'sha256': sha.hexdigest(), 'sample_rate': 48000, 'channels': 2,
            'frames': byte_count//4, 'seconds': byte_count/(4*48000),
            'source_sample_rate': streams[0].get('sample_rate'), 'source_channels': streams[0].get('channels'),
            'source_codec': streams[0].get('codec_name')}


def prepare(source, target, *, ffmpeg, ffprobe):
    source, target = source.resolve(strict=True), target.resolve()
    if source == target or source in target.parents or target in source.parents:
        raise ValueError('Separate source and preparation roots required')
    target.mkdir(parents=True, exist_ok=False)
    copied = target/'copied'; expanded = target/'expanded'; labels = target/'raw-labels'
    for directory in (copied, expanded, labels): directory.mkdir()
    files = sorted(p for p in source.rglob('*') if p.is_file())
    report = {'schema_version':1, 'source':str(source), 'status':'running',
              'scope':'user-authorized local preparation; no public redistribution license asserted',
              'preparation':{'code_sha256':digest(Path(__file__)), 'python':sys.version,
                  'ffmpeg_sha256':digest(ffmpeg) if ffmpeg.is_file() else None,
                  'ffprobe_sha256':digest(ffprobe) if ffprobe.is_file() else None,
                  'pcm_policy':'FFmpeg decoded PCM16 little-endian; 48000Hz stereo; no crop/pad/gain',
                  'rating_policy':'9 <= rating < 11; modern metadata > legacy metadata > BMS'},
              'training_started':False, 'files':[], 'packages':[], 'charts':[], 'errors':[]}
    archive_queue = []
    for path in files:
        relative = path.relative_to(source)
        destination = copied/relative; destination.parent.mkdir(parents=True, exist_ok=True)
        before = digest(path); shutil.copy2(path, destination)
        if digest(destination) != before or digest(path) != before:
            raise ValueError('Source changed or copy mismatch: '+str(relative))
        report['files'].append({'relative_path':relative.as_posix(), 'sha256':before, 'bytes':destination.stat().st_size})
        if path.suffix.lower() in ('.mdm','.zip'):
            archive_queue.append((destination, before, relative.as_posix(), 0))
    write_json(target/'inventory.json', report)
    audio_cache = {}; extracted_hashes = set()
    for number, (path, sha, origin, depth) in enumerate(archive_queue, 1):
        package = {'source':origin, 'sha256':sha, 'status':'pending', 'depth':depth}
        report['packages'].append(package)
        try:
            if depth > 5: raise ValueError('Nested archive depth budget exceeded')
            root = expanded/sha
            if sha not in extracted_hashes:
                extract(path, root); extracted_hashes.add(sha)
                for nested in sorted(root.rglob('*')):
                    if nested.is_file() and nested.suffix.lower() in ('.zip','.mdm'):
                        archive_queue.append((nested,digest(nested),origin+'!'+nested.relative_to(root).as_posix(),depth+1))
            package['status'] = 'extracted'
            package['files'] = [{'path':p.relative_to(root).as_posix(),'sha256':digest(p),'bytes':p.stat().st_size}
                                for p in sorted(root.rglob('*')) if p.is_file()]
            info_paths = list(root.rglob('info.json'))
            info = json.loads(decode_text(info_paths[0].read_bytes())[0]) if len(info_paths)==1 else {}
            package['info_schema'] = 'modern' if 'meta' in info else 'legacy'
            for chart_path in sorted(root.rglob('*.bms')):
                raw = inspect_bms(chart_path); chart_sha = digest(chart_path)
                write_json(labels/(sha+'-'+chart_sha+'.json'), raw)
                key = chart_path.stem
                modern = info.get('meta',{}).get('maps',{}).get(key,{}).get('rating')
                legacy = info.get('difficulty'+key.removeprefix('map'))
                header = raw['headers'].get('PLAYLEVEL', [None])[-1]
                rating = modern if modern is not None else legacy if legacy is not None else header
                value = rating_value(rating); selected = value is not None and Decimal(9)<=value<Decimal(11)
                issues = []
                if len({str(rating_value(v)) for v in (modern,legacy,header) if rating_value(v) is not None})>1:
                    issues.append('difficulty_metadata_conflict')
                title = info.get('meta',{}).get('name',info.get('name',raw['headers'].get('TITLE',['unknown'])[-1]))
                row = {'package_hash':sha,'source':origin,'chart_path':chart_path.relative_to(root).as_posix(),
                    'chart_sha256':chart_sha,'raw_label_path':(labels/(sha+'-'+chart_sha+'.json')).relative_to(target).as_posix(),
                    'title':title,'author':info.get('meta',{}).get('author',info.get('author')),
                    'charter':raw['headers'].get('LEVELDESIGN', []),
                    'rating':str(rating),'rating_sources':{'modern':modern,'legacy':legacy,'bms':header},
                    'selected_9_10':selected,'issues':issues,'channels':raw['channels'],
                    'raw_code_counts':raw['raw_code_counts'],'parse_errors':raw['errors'],
                    'reference_review':'pending','gold_eligible':False,'public_license':'unknown',
                    'training_eligible':False,'split':'unassigned_architecture_pending'}
                if selected:
                    try:
                        music = resolve_music(root)
                    except ValueError:
                        row['issues'].append('missing_or_ambiguous_music')
                    else:
                        audio_sha = digest(music); row['audio_sha256']=audio_sha
                        row['audio_path']=music.relative_to(target).as_posix()
                        if audio_sha not in audio_cache:
                            audio_cache[audio_sha]=pcm_identity(music,ffmpeg,ffprobe)
                        row['pcm']=audio_cache[audio_sha]
                report['charts'].append(row)
        except Exception as exc:
            package['status']='failed'
            package['error']=type(exc).__name__+': '+str(exc)
            report['errors'].append({'source':origin,'error':package['error']})
        if number%10==0:
            write_json(target/'inventory.json',report)
            print(f'Prepared {number}/{len(archive_queue)} packages',flush=True)
    selected = [r for r in report['charts'] if r['selected_9_10']]
    pcm_groups = defaultdict(list); title_groups = defaultdict(list)
    for row in selected:
        if 'pcm' in row: pcm_groups[row['pcm']['sha256']].append(row)
        normalized = ''.join(c for c in unicodedata.normalize('NFKC',str(row['title'])).casefold() if c.isalnum())
        title_groups[normalized].append(row)
    # Connected groups for identical PCM OR same normalized title. Provisional:
    # remixes/covers and different naming still require family review before split.
    parent = {r['chart_sha256']:r['chart_sha256'] for r in selected}
    def find(x):
        while parent[x]!=x:
            parent[x]=parent[parent[x]]; x=parent[x]
        return x
    for group in [*pcm_groups.values(),*title_groups.values()]:
        for row in group[1:]:
            a,b=find(group[0]['chart_sha256']),find(row['chart_sha256'])
            parent[max(a,b)]=min(a,b)
    for row in selected:
        row['provisional_family']='community-'+find(row['chart_sha256'])[:20]
        row['family_review']='pending'
    report['summary']={'copied_files':len(report['files']),'packages':len(report['packages']),
        'extracted_packages':sum(p['status']=='extracted' for p in report['packages']),
        'charts':len(report['charts']),'selected_9_10':len(selected),
        'selected_unique_chart_hashes':len({r['chart_sha256'] for r in selected}),
        'selected_unique_pcm':len(pcm_groups),'provisional_families':len({r['provisional_family'] for r in selected}),
        'difficulty_conflicts':sum('difficulty_metadata_conflict' in r['issues'] for r in selected),
        'selected_parse_errors':sum(bool(r['parse_errors']) for r in selected), 'errors':len(report['errors'])}
    report['status']='completed' if not report['errors'] else 'completed_with_errors'
    write_json(target/'selected-9-10.json',{'schema_version':1,'rating_policy':'9 <= Decimal(rating) < 11; modern > legacy > BMS',
        'training_started':False,'architecture_status':'pending_user_review','charts':selected})
    write_json(target/'inventory.json',report)
    write_views(target,report)
    print(json.dumps(report['summary']),flush=True)
    return report


def write_views(target, report):
    selected=[r for r in report['charts'] if r['selected_9_10']]
    unique={r['pcm']['sha256']:r['pcm'] for r in selected if 'pcm' in r}
    hours=sum(p['seconds'] for p in unique.values())/3600
    def clean(value):
        return str(value).replace('|','／').replace('\n',' ').replace('\r',' ')
    lines=['# 9～10 级社区谱候选清单','',
        f"全部 {len(report['packages'])} 包；候选 {len(selected)} 张；不同规范音频 {len(unique)} 份，约 {hours:.2f} 小时。",
        '', '仅完成准备，未启动训练。等级冲突、对象映射、作品族与训练架构等待确认；未冻结 train/validation/test。',
        '', '| 序号 | 名称 | 选用等级 | BMS 等级 | 复核提示 | 原始谱面 |',
        '|---|---|---|---|---|---|']
    for index,row in enumerate(sorted(selected,key=lambda r:(Decimal(r['rating']),str(r['title']))),1):
        bms=(target/'expanded'/row['package_hash']/row['chart_path']).resolve().as_posix()
        issue='、'.join(row['issues']) or '对象/作品族待复核'
        lines.append(f"| {index} | {clean(row['title'])} | {row['rating']} | {clean(row['rating_sources']['bms'])} | {clean(issue)} | [BMS](<{bms}>) |")
    (target/'候选谱面清单.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    summary=['# 社区谱准备验收','',json.dumps(report['summary'],ensure_ascii=False,indent=2),
        '',f'规范音频总时长（去除精确 PCM 重复）：{hours:.2f} 小时。',
        '', '原件只读复制；全部包 CRC 验证后展开。哈希与来源保存在 inventory.json；原始标签在 raw-labels。',
        '', '社区监督不是独立复核 gold。切分、语义映射和特征契约尚未冻结，training_started=false。']
    (target/'准备验收.md').write_text('\n'.join(summary)+'\n',encoding='utf-8')


def refresh_labels(target):
    """Reparse existing verified copies after a parser fix; no recopy/decode."""
    target=target.resolve(strict=True)
    report=json.loads((target/'inventory.json').read_text(encoding='utf-8'))
    report.setdefault('preparation', {})['label_parser_sha256']=digest(Path(__file__))
    for row in report['charts']:
        path=target/'expanded'/row['package_hash']/row['chart_path']
        if digest(path)!=row['chart_sha256']:
            raise ValueError('Raw chart changed after preparation')
        raw=inspect_bms(path)
        write_json(target/row['raw_label_path'],raw)
        row['parse_errors']=raw['errors'];row['channels']=raw['channels'];row['raw_code_counts']=raw['raw_code_counts']
        row['rating_sources']['bms']=raw['headers'].get('PLAYLEVEL',[None])[-1]
    selected=[r for r in report['charts'] if r['selected_9_10']]
    report['summary']['selected_parse_errors']=sum(bool(r['parse_errors']) for r in selected)
    write_json(target/'inventory.json',report)
    selection=json.loads((target/'selected-9-10.json').read_text(encoding='utf-8'))
    selection['charts']=selected;write_json(target/'selected-9-10.json',selection)
    write_views(target,report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True);parser.add_argument('--target',type=Path,required=True)
    parser.add_argument('--ffmpeg',type=Path,required=True);parser.add_argument('--ffprobe',type=Path,required=True)
    args=parser.parse_args()
    report=prepare(args.source,args.target,ffmpeg=args.ffmpeg,ffprobe=args.ffprobe)
    return 0 if report['status']=='completed' else 1
