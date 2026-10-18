"""Copy old automatic projects into chart; retain all original files/databases."""
import argparse
import json
from pathlib import Path
import shutil
import sqlite3
import sys
from uuid import uuid4

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from mdac2.application.studio_workflow import StudioWorkflow
from mdac2.infrastructure.job_backend import JobBackend


def migrate(source,target):
    source=Path(source).resolve();target=Path(target).resolve()
    if source==target:raise ValueError('Distinct source and destination required')
    repository=JobBackend(target/'state.sqlite');studio=StudioWorkflow(ROOT,target,repository)
    old=sqlite3.connect(source/'state.sqlite');old.row_factory=sqlite3.Row
    try:
        if not old.execute("SELECT 1 FROM sqlite_master WHERE name='studio_projects'").fetchone():return []
        projects=[json.loads(row[0]) for row in old.execute('SELECT data FROM studio_projects')]
        if any(p.get('active_attempt') for p in projects):raise ValueError('Stop active tasks before migration')
        prefix=str(source/'studio').replace('\\','/')
        def relocate(value):
            if isinstance(value,dict):return {k:relocate(v) for k,v in value.items()}
            if isinstance(value,list):return [relocate(v) for v in value]
            if isinstance(value,str):
                normalized=value.replace('\\','/')
                if normalized==prefix or normalized.startswith(prefix+'/'):return str(studio.directory/normalized[len(prefix):].lstrip('/'))
            return value
        cache=source/'studio/cache'
        if cache.exists():
            for folder in cache.iterdir():
                dest=studio.directory/'cache'/folder.name
                if folder.is_dir() and not dest.exists():shutil.copytree(folder,dest)
        imported=[]
        for project in projects:
            pid=project['project_id'];dest=(studio.directory/pid).resolve()
            if not dest.is_relative_to(studio.directory.resolve()):raise ValueError('Invalid project location')
            with repository._connection() as c:
                if c.execute('SELECT 1 FROM studio_projects WHERE project_id=?',(pid,)).fetchone():continue
            if dest.exists():raise FileExistsError(dest)
            stage=studio.directory/('migration-'+str(uuid4()));shutil.copytree(source/'studio'/pid,stage)
            for path in stage.glob('*/*.json'):
                if path.name in ('request.json','result.json'):
                    path.write_text(json.dumps(relocate(json.loads(path.read_text(encoding='utf-8'))),ensure_ascii=False,indent=2),encoding='utf-8')
            stage.rename(dest)
            row=dict(old.execute('SELECT * FROM projects WHERE project_id=?',(pid,)).fetchone())
            with repository._connection() as c:
                c.execute('BEGIN IMMEDIATE')
                c.execute(f"INSERT INTO projects ({','.join(row)}) VALUES ({','.join('?' for _ in row)})",tuple(row.values()))
                c.execute('INSERT INTO studio_projects VALUES (?,?)',(pid,json.dumps(relocate(project))))
                for attempt in old.execute('SELECT attempt_id,data FROM studio_attempts WHERE project_id=?',(pid,)):
                    c.execute('INSERT INTO studio_attempts VALUES (?,?,?)',(attempt['attempt_id'],pid,json.dumps(relocate(json.loads(attempt['data'])))))
                c.commit()
            imported.append(pid)
        for name in ('llm-settings.json','preferences.json'):
            if (source/name).exists() and not (target/name).exists():shutil.copyfile(source/name,target/name)
        return imported
    finally:old.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--target',type=Path,default=ROOT/'chart');a=p.parse_args()
    print(json.dumps(dict(imported=migrate(a.source,a.target)),ensure_ascii=False))
