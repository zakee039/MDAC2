"""Portable studio files alongside the existing workspace snapshot (no keys)."""
import json
from pathlib import Path
import shutil

from mdac2.infrastructure.artifact_store import _plain


def copy_tree(source, target):
    _plain(source)
    for item in source.rglob('*'):
        _plain(item)
        if item.is_file() and item.name not in ('gpu.lock','runtime.lock'):
            dest=target/item.relative_to(source);dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(item,dest)


def snapshot(connection, runtime, stage):
    rows=[json.loads(r[0]) for r in connection.execute('SELECT data FROM studio_attempts')]
    if any(r['state'] in ('queued','running') for r in rows):
        raise ValueError('请等待自动制谱/导出完成或取消后再备份')
    directory=runtime/'studio'
    if not directory.exists():return
    # Studio files are outside the immutable artifact store; require quiescence.
    lease=directory/'runtime.lock'
    if lease.exists():
        import os
        if os.name=='nt':
            import msvcrt
            with lease.open('r+b') as f:
                try:msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1)
                except OSError:raise ValueError('请先关闭自动制谱工作台服务再备份') from None
    copy_tree(directory,stage/'studio')
    mappings={str(directory.resolve()):'studio'}
    # Generation may have reused a read-only whole-song cache outside runtime.
    for result in (stage/'studio').glob('*/*/result.json'):
        data=json.loads(result.read_text(encoding='utf-8'))
        if data.get('cached_audio'):
            cache=Path(data['cached_audio']).resolve()
            if not cache.is_relative_to(directory.resolve()) and str(cache) not in mappings:
                import hashlib
                relative='studio/restored-cache/'+hashlib.sha256(str(cache).encode()).hexdigest()
                copy_tree(cache,stage/relative);mappings[str(cache)]=relative
    (stage/'studio').mkdir(exist_ok=True)
    (stage/'studio/backup-paths.json').write_text(json.dumps(mappings),encoding='utf-8')


def relocate(connection, stage, target):
    mapping_file=stage/'studio/backup-paths.json'
    if not mapping_file.exists():return
    mappings=json.loads(mapping_file.read_text(encoding='utf-8'))
    if not isinstance(mappings,dict):raise ValueError('Invalid studio path map')
    for old,relative in mappings.items():
        if not isinstance(relative,str) or not relative.startswith('studio') or '\\' in relative or any(p in ('..','.') for p in relative.split('/')) or not (target/relative).resolve().is_relative_to((target/'studio').resolve()):
            raise ValueError('Invalid studio restore path')
    def change(value):
        if isinstance(value,dict):return {k:change(v) for k,v in value.items()}
        if isinstance(value,list):return [change(v) for v in value]
        if isinstance(value,str):
            for old,relative in sorted(mappings.items(),key=lambda p:len(p[0]),reverse=True):
                normalized=value.replace('\\','/');prefix=old.replace('\\','/')
                if normalized==prefix or normalized.startswith(prefix+'/'):
                    return str(target/relative/normalized[len(prefix):].lstrip('/'))
        return value
    for table,key in [('studio_projects','project_id'),('studio_attempts','attempt_id')]:
        for identity,data in connection.execute(f'SELECT {key},data FROM {table}').fetchall():
            connection.execute(f'UPDATE {table} SET data=? WHERE {key}=?',(json.dumps(change(json.loads(data))),identity))
    for path in (stage/'studio').glob('*/*/*.json'):
        if path.name in ('request.json','result.json'):
            data=change(json.loads(path.read_text(encoding='utf-8')))
            path.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
    connection.commit()
