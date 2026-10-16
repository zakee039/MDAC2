"""Online WAL snapshot, fresh-directory restore and semantic export equivalence."""
from pathlib import Path
from fractions import Fraction as F
import io
import wave
import pytest
from mdac2.server import build_app
from mdac2.application.backup import backup_workspace,restore_workspace
from mdac2.application.manual_workflow import timing_data,ManualWorkflow
from mdac2.application.export_builder import build_export
from mdac2.adapters.audio_io import AudioDecoder
from mdac2.domain.chart import ChartRevision,ChartEvent
from mdac2.domain.timing import TimingMap,Anchor
from mdac2.domain.timing_edits import TimingRevision
from mdac2.infrastructure.artifact_store import ArtifactStore
from mdac2.infrastructure.job_backend import JobBackend
from mdac2.domain.artifact import strict_json,canonical_json
from fastapi.testclient import TestClient


@pytest.fixture
def source(tmp_path):
    root=Path(__file__).parents[2]
    app=build_app(root,'t'*40,runtime_root=tmp_path/'original')
    api=TestClient(app,base_url='http://127.0.0.1:8000',client=('127.0.0.1',123),headers={'Authorization':'Bearer '+'t'*40})
    repo=app.state.repository;store=app.state.store
    with repo._connection() as c:c.execute('PRAGMA journal_mode=WAL')
    project=repo.create_project('Backup synthetic').project_id
    audio=io.BytesIO()
    with wave.open(audio,'wb') as w:
        w.setparams((2,2,48000,0,'NONE','none'));w.writeframes(bytes(48000*4*2))
    assert api.post(f'/api/v1/projects/{project}/audio',content=audio.getvalue(),headers={'Content-Type':'audio/wav'}).status_code==201
    chart=ChartRevision('c','t',(ChartEvent('tap',F(1),('ground',)),ChartEvent('hold',F(2),('air',),semantic='hold',duration_q=F(1))))
    timing=TimingRevision('t',TimingMap((Anchor(0,0),Anchor(4,2)),2))
    app.state.manual_workflow.save(project,expected_revision=None,command_id='cmd',chart=chart,timing=timing)
    binary=root/'tools/.deps/ffmpeg/extracted/ffmpeg-9.0.1-essentials_build/bin'
    decoder=AudioDecoder(binary/'ffmpeg.exe',binary/'ffprobe.exe')
    return repo,store,project,chart,timing,decoder,api


def test_consistent_snapshot_restore_and_same_semantic_export(source,tmp_path):
    repo,store,project,chart,timing,decoder,_=source
    manifest=backup_workspace(repo,store,tmp_path/'backup')
    assert manifest['scope']=='workspace'
    assert manifest['projects'][0]['head_revision']=='c'
    repo.create_project('created after snapshot')
    target=restore_workspace(tmp_path/'backup',tmp_path/'restored')
    restored=JobBackend(target/'state.sqlite');restored_store=ArtifactStore(target/'store')
    assert ManualWorkflow(restored,restored_store).load(project)==(chart,timing)
    with restored._connection() as c:assert c.execute('SELECT count(*) FROM projects').fetchone()[0]==1
    kwargs=dict(project_id=project,expected_revision='c',timing=timing.mapping,timing_revision='t',title='Backup synthetic',artist='MDAC2')
    before=build_export(repo,store,decoder,**kwargs)
    after=build_export(restored,restored_store,decoder,**kwargs)
    assert before.payload['logical_count']==after.payload['logical_count']==2
    assert before.payload['physical_count']==after.payload['physical_count']==2
    bms_before=next(r for r in before.payload['resources'] if r['name']=='map2.bms')
    bms_after=next(r for r in after.payload['resources'] if r['name']=='map2.bms')
    assert bms_before==bms_after
    assert repo.get_project(project).head_revision=='c'
    with pytest.raises(FileExistsError):restore_workspace(tmp_path/'backup',target)
    with pytest.raises(FileExistsError):backup_workspace(repo,store,tmp_path/'backup')


def test_corrupt_same_size_file_refused_without_visible_restore(source,tmp_path):
    repo,store,*_=source
    backup_workspace(repo,store,tmp_path/'backup')
    wav=next((tmp_path/'backup').rglob('audio.wav'))
    raw=bytearray(wav.read_bytes());raw[-1]^=1;wav.write_bytes(raw)
    with pytest.raises(ValueError,match='hash/length'):restore_workspace(tmp_path/'backup',tmp_path/'restored')
    assert not (tmp_path/'restored').exists()
    # Original silent audio was legitimate and still intact.
    original=next(store.artifacts.rglob('audio.wav'));assert original.read_bytes()!=bytes(raw)


def test_missing_dependency_and_copy_failure_never_publish(source,tmp_path,monkeypatch):
    repo,store,*_=source
    monkeypatch.setattr(ArtifactStore,'publish',lambda *a,**k:(_ for _ in ()).throw(OSError('disk full')))
    with pytest.raises(OSError):backup_workspace(repo,store,tmp_path/'backup')
    assert not (tmp_path/'backup').exists()
    monkeypatch.undo()
    with repo._connection() as c:c.execute("UPDATE revision_timing SET artifact_hash=?",('a'*64,))
    with pytest.raises(ValueError,match='Missing artifact'):backup_workspace(repo,store,tmp_path/'missing')
    assert not (tmp_path/'missing').exists()


def test_restore_fences_source_running_attempt_without_touching_source(source,tmp_path):
    repo,store,project,chart,timing,decoder,api=source
    job=api.post(f'/api/v1/projects/{project}/exports',json={'expected_revision':'c'}).json()
    import time
    repo.claim(job['job_id'],job['run_id'],'original-worker',now=time.time(),lease_seconds=30)
    backup_workspace(repo,store,tmp_path/'backup')
    target=restore_workspace(tmp_path/'backup',tmp_path/'restored')
    other=JobBackend(target/'state.sqlite')
    run=other.get_job(job['job_id']).runs[-1]
    assert run.state=='interrupted' and run.exit_confirmed
    assert repo.get_job(job['job_id']).state=='running'
    retry=other.retry(job['job_id'],run.run_id)
    assert retry.run_id!=run.run_id and retry.state=='queued'


def test_manifest_traversal_rejected(source,tmp_path):
    repo,store,*_=source
    backup_workspace(repo,store,tmp_path/'backup')
    path=tmp_path/'backup/backup.json'
    data=strict_json(path.read_bytes());data['files'][0]['path']='../outside'
    path.write_bytes(canonical_json(data))
    with pytest.raises(ValueError):restore_workspace(tmp_path/'backup',tmp_path/'restored')
    assert not (tmp_path/'restored').exists()

