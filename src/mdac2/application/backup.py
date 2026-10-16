"""Consistent whole-workspace backups, restored only to a new directory.

The snapshot includes every project and registered artifact, and recursively adds
unregistered but referenced artifacts. Staging, queue transport and models are
excluded. Immutable files may be copied after SQLite's online backup completes.
"""
from dataclasses import asdict, replace
import hashlib
import os
from pathlib import Path
import shutil
import sqlite3
from uuid import uuid4
import wave

from mdac2.domain.artifact import canonical_json, strict_json
from mdac2.application.jobs_contract import Run, public_run
from mdac2.infrastructure.artifact_store import ArtifactStore, _plain, _name
from mdac2.infrastructure.project_repository import RepositoryError


_SCHEMA_TABLES = {'asset_schema', 'manual_schema', 'job_schema', 'job_event_schema', 'publication_schema'}
_TABLES = {'projects','revisions','project_audio','revision_timing','jobs','runs','job_outbox',
           'job_events','artifacts','artifact_inputs','run_outputs',*_SCHEMA_TABLES}


def _connection(path):
    connection=sqlite3.connect(path)
    connection.row_factory=sqlite3.Row
    return connection


def _check_database(c):
    if c.execute('PRAGMA user_version').fetchone()[0]!=1:
        raise RepositoryError('Unknown backup database schema')
    if c.execute('PRAGMA integrity_check').fetchone()[0]!='ok' or c.execute('PRAGMA foreign_key_check').fetchone():
        raise RepositoryError('Invalid backup database')
    tables={r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")}
    if not {'projects','revisions'}<=tables or tables-_TABLES:
        raise RepositoryError('Unknown backup tables')
    if c.execute("SELECT 1 FROM sqlite_master WHERE type IN ('trigger','view')").fetchone():
        raise RepositoryError('Unexpected backup executable schema')
    for name in tables&_SCHEMA_TABLES:
        if [tuple(r) for r in c.execute(f'SELECT version FROM {name}')]!=[(1,)]:
            raise RepositoryError('Unknown backup extension schema')
    return tables


def _references(c, tables):
    roots={("chart",r[0]) for r in c.execute('SELECT artifact_hash FROM revisions')}
    for table,kind in (('project_audio','audio'),('revision_timing','timing')):
        if table in tables:
            roots.update((kind,r[0]) for r in c.execute(f'SELECT artifact_hash FROM {table}'))
    if 'artifacts' in tables:
        roots.update(tuple(r) for r in c.execute('SELECT artifact_type,artifact_hash FROM artifacts'))
    if 'jobs' in tables:
        for r in c.execute('SELECT request FROM jobs'):
            roots.update((None,h) for h in strict_json(r[0])['inputs'])
    return roots


def _closure(store, roots):
    index={}
    for kind in store.artifacts.iterdir():
        _plain(kind)
        if not kind.is_dir():raise ValueError('Unexpected artifact entry')
        for item in kind.iterdir():
            _plain(item)
            if item.name in index:raise ValueError('Ambiguous artifact identity')
            index[item.name]=kind.name
    found={}
    pending=list(roots)
    while pending:
        kind,identity=pending.pop()
        actual=index.get(identity)
        if actual is None or kind is not None and kind!=actual:
            raise ValueError('Missing artifact reference')
        if identity in found:continue
        envelope=store.get(actual,identity)
        found[identity]=(actual,envelope)
        pending.extend((None,h) for h in envelope.to_dict()['inputs'])
    return found


def _audio_check(store, kind, identity, envelope):
    if kind!='audio':return
    canonical=envelope.payload['canonical']
    with wave.open(str(store._destination(kind,identity)/'audio.wav'),'rb') as wav:
        if (wav.getnframes(),wav.getframerate(),wav.getnchannels(),wav.getsampwidth())!=(canonical['sample_count'],canonical['sample_rate'],len(canonical['channel_layout']),2):
            raise ValueError('Backup audio disagrees with manifest')
        frames=0
        while data:=wav.readframes(65536):frames+=len(data)//(2*wav.getnchannels())
        if frames!=canonical['sample_count']:raise ValueError('Truncated backup audio')
    # All-zero PCM is valid silence, not a corruption heuristic.


def _new_stage(target):
    target=Path(target).absolute()
    if target.exists():raise FileExistsError('Backup/restore target must not exist')
    for parent in reversed((target.parent,*target.parent.parents)):
        if parent.exists():_plain(parent)
    target.parent.mkdir(parents=True,exist_ok=True)
    stage=target.with_name(target.name+'.staging-'+str(uuid4()))
    stage.mkdir()
    return target,stage


def _entry(path, relative):
    _plain(path)
    with path.open('rb') as stream: digest=hashlib.file_digest(stream,'sha256').hexdigest()
    return {'path':relative,'sha256':digest,'byte_count':path.stat().st_size}


def _write_manifest(stage, projects):
    files=[_entry(p,p.relative_to(stage).as_posix()) for p in sorted(stage.rglob('*')) if p.is_file()]
    data={'schema_version':1,'scope':'workspace','projects':projects,'files':files}
    ArtifactStore._write(stage/'backup.json',canonical_json(data))
    return data


def _verify_snapshot(root):
    root=Path(root).absolute();_plain(root);_plain(root/'backup.json')
    manifest=strict_json((root/'backup.json').read_bytes())
    if type(manifest) is not dict or set(manifest)!={'schema_version','scope','projects','files'} or type(manifest['schema_version']) is not int or manifest['schema_version']!=1 or manifest['scope']!='workspace':
        raise ValueError('Unknown backup manifest')
    expected=set()
    for entry in manifest['files']:
        if type(entry) is not dict or set(entry)!={'path','sha256','byte_count'}:raise ValueError('Invalid backup entry')
        name=entry['path']
        if type(name) is not str or '\\' in name or name in expected:raise ValueError('Invalid backup path')
        parts=name.split('/')
        for part in parts:_name(part)
        if parts[0] not in {'state.sqlite','store'}:raise ValueError('Unexpected backup file')
        path=root.joinpath(*parts)
        for parent in (path,*path.parents):
            if parent==root:break
            _plain(parent)
        if _entry(path,name)!=entry:raise ValueError('Backup hash/length mismatch')
        expected.add(name)
    actual=set()
    for p in root.rglob('*'):
        _plain(p)
        if p.is_file():actual.add(p.relative_to(root).as_posix())
    if actual!=expected|{'backup.json'} or 'state.sqlite' not in expected:
        raise ValueError('Unexpected or missing backup file')
    c=_connection(root/'state.sqlite')
    try:
        tables=_check_database(c)
        projects=[dict(r) for r in c.execute('SELECT * FROM projects ORDER BY project_id')]
        if projects!=manifest['projects']:raise ValueError('Backup project index mismatch')
        store=ArtifactStore(root/'store')
        found=_closure(store,_references(c,tables))
        for identity,(kind,envelope) in found.items():_audio_check(store,kind,identity,envelope)
        from mdac2.domain.chart import ChartRevision
        from mdac2.domain.validation import validate_chart
        for row in c.execute('SELECT project_id,revision_id,artifact_hash FROM revisions'):
            chart=ChartRevision.from_json(canonical_json(found[row['artifact_hash']][1].payload))
            if chart.revision_id!=row['revision_id']:raise ValueError('Chart revision identity mismatch')
            if 'revision_timing' in tables:
                bound=c.execute('SELECT artifact_hash FROM revision_timing WHERE revision_id=?',(chart.revision_id,)).fetchone()
                if bound is not None:
                    from mdac2.application.manual_workflow import timing_from
                    timing=timing_from(found[bound[0]][1].payload)
                    if validate_chart(chart,timing.mapping,timing_revision=timing.revision_id,allow_empty=True).has_errors:
                        raise ValueError('Invalid restored chart/timing binding')
        if 'artifact_inputs' in tables:
            for row in c.execute('SELECT artifact_hash FROM artifacts'):
                actual=[r[0] for r in c.execute('SELECT input_hash FROM artifact_inputs WHERE artifact_hash=? ORDER BY position',(row[0],))]
                if actual!=found[row[0]][1].to_dict()['inputs']:
                    raise ValueError('Artifact dependency index mismatch')
    finally:c.close()
    return manifest


def backup_workspace(repository,store,target):
    """Copies a consistent snapshot of ALL projects; never overwrites a target."""
    resolved=Path(target).absolute()
    if resolved.is_relative_to(store.root) or resolved.is_relative_to(repository.database.absolute().parent):
        raise ValueError('Backup target must be outside the source runtime')
    target,stage=_new_stage(target)
    snapshot=_connection(stage/'state.sqlite')
    try:
        with repository._connection() as source:source.backup(snapshot)
        tables=_check_database(snapshot)
        roots=_references(snapshot,tables)
        projects=[dict(r) for r in snapshot.execute('SELECT * FROM projects ORDER BY project_id')]
    finally:snapshot.close()
    destination=ArtifactStore(stage/'store')
    for identity,(kind,envelope) in _closure(store,roots).items():
        _audio_check(store,kind,identity,envelope)
        files={e['name']:(store._destination(kind,identity)/e['name']).read_bytes() for e in envelope.payload.get('files',[])}
        destination.publish(envelope,files)
    manifest=_write_manifest(stage,projects)
    _verify_snapshot(stage)
    if os.name!='nt':raise NotImplementedError('No-clobber publication requires Windows')
    os.rename(stage,target)
    return manifest


def restore_workspace(backup,target):
    """Restore into a fresh runtime, with no inherited running processes/leases."""
    backup=Path(backup).absolute()
    manifest=_verify_snapshot(backup)
    if Path(target).absolute().is_relative_to(backup):raise ValueError('Restore target must be outside the backup')
    target,stage=_new_stage(target)
    for entry in manifest['files']:
        destination=stage/entry['path'];destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(backup/entry['path'],destination)
    shutil.copyfile(backup/'backup.json',stage/'backup.json')
    _verify_snapshot(stage)
    c=_connection(stage/'state.sqlite')
    try:
        tables=_check_database(c)
        if 'jobs' in tables:
            for row in c.execute('SELECT runs.snapshot FROM jobs JOIN runs ON jobs.current_run=runs.run_id').fetchall():
                run=Run(**strict_json(row[0]))
                if run.state=='running' or run.state=='interrupted' and not run.exit_confirmed:
                    restored=replace(run,state='interrupted',stage='interrupted',exit_confirmed=True)
                    c.execute('UPDATE runs SET snapshot=? WHERE run_id=?',(canonical_json(asdict(restored)),run.run_id))
                    c.execute('UPDATE jobs SET active=0 WHERE job_id=?',(run.job_id,))
                    if 'job_events' in tables:
                        c.execute('INSERT INTO job_events(job_id,run_id,snapshot) VALUES (?,?,?)',(run.job_id,run.run_id,canonical_json(public_run(restored))))
                elif run.state=='queued':
                    c.execute('UPDATE job_outbox SET delivered=0,generation=generation+1 WHERE run_id=?',(run.run_id,))
            c.commit()
    finally:c.close()
    # backup.json described the source snapshot; restoration intentionally fences
    # transient runs. Keep that provenance and write a new checksum manifest.
    os.rename(stage/'backup.json',stage/'source-backup.json')
    ArtifactStore._write(stage/'restore.json',canonical_json({'schema_version':1,'source_scope':'workspace','projects':manifest['projects'],
        'database':_entry(stage/'state.sqlite','state.sqlite'),'running_policy':'interrupted-confirmed-in-new-runtime'}))
    if os.name!='nt':raise NotImplementedError('No-clobber publication requires Windows')
    os.rename(stage,target)
    return target

if __name__=='__main__':
    import argparse
    from mdac2.infrastructure.project_repository import ProjectRepository
    parser=argparse.ArgumentParser(description='Backup ALL projects, or restore into a new runtime directory')
    commands=parser.add_subparsers(dest='command',required=True)
    backup=commands.add_parser('backup');backup.add_argument('--runtime',required=True);backup.add_argument('--target',required=True)
    restore=commands.add_parser('restore');restore.add_argument('--backup',required=True);restore.add_argument('--target',required=True)
    args=parser.parse_args()
    if args.command=='backup':
        root=Path(args.runtime)
        if not (root/'state.sqlite').is_file() or not (root/'store').is_dir():parser.error('Existing runtime required')
        report=backup_workspace(ProjectRepository(root/'state.sqlite'),ArtifactStore(root/'store'),args.target)
        print(f"Verified backup: {len(report['projects'])} projects, {len(report['files'])} files")
    else:
        restore_workspace(args.backup,args.target)
        print('Verified restore created in new directory; original retained')
