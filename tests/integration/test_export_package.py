from fractions import Fraction as F
import io
import wave
import zipfile
from pathlib import Path
import pytest
from mdac2.adapters.audio_io import AudioDecoder
from mdac2.application.export_builder import build_export
from mdac2.domain.artifact import ArtifactEnvelope
from mdac2.domain.chart import ChartRevision,ChartEvent
from mdac2.domain.timing import TimingMap,Anchor
from mdac2.infrastructure.artifact_store import ArtifactStore
from mdac2.infrastructure.project_repository import ProjectRepository,RevisionConflict


def _package_inputs(tmp_path,seconds):
    binary=Path(__file__).parents[2]/"tools/.deps/ffmpeg/extracted/ffmpeg-9.0.1-essentials_build/bin"
    decoder=AudioDecoder(binary/"ffmpeg.exe",binary/"ffprobe.exe")
    repo=ProjectRepository(tmp_path/"state.sqlite");store=ArtifactStore(tmp_path/"store")
    project=repo.create_project("synthetic")
    stream=io.BytesIO()
    with wave.open(stream,"wb") as output:
        output.setparams((2,2,48000,0,"NONE","none"));output.writeframes(bytes(48000*4*seconds))
    audio=decoder.import_audio(stream.getvalue(),store)
    with repo._connection() as connection:
        connection.execute("CREATE TABLE project_audio(project_id TEXT PRIMARY KEY, artifact_hash TEXT)")
        connection.execute("INSERT INTO project_audio VALUES (?,?)",(project.project_id,audio.artifact_hash))
    chart=ChartRevision("c","t",(ChartEvent("e",F(1),("air",)),))
    envelope=ArtifactEnvelope.create(artifact_type="chart",producer={"name":"manual","version":"1","model_hash":None,"dependencies":{}},options={},payload=chart.to_dict())
    store.publish(envelope);repo.commit_revision(project.project_id,expected_revision=None,artifact=envelope,command_id="cmd",command_type="replace_chart",command_payload={})
    # Two beats per second keeps the anchor count pointed at the audio duration.
    kwargs=dict(project_id=project.project_id,expected_revision="c",timing=TimingMap((Anchor(0,0),Anchor(2*seconds,seconds)),seconds),timing_revision="t",title="Synthetic",artist="MDAC2")
    return repo,store,decoder,kwargs


@pytest.fixture
def package_inputs(tmp_path):
    return _package_inputs(tmp_path,2)


@pytest.fixture
def long_package_inputs(tmp_path):
    return _package_inputs(tmp_path,25)


def ogg_seconds(decoder,data):
    decoded,_=decoder.decode(data,target_rate=48000)
    canonical=decoded.payload["canonical"]
    return canonical["sample_count"]/canonical["sample_rate"]


def test_real_package_resources_and_atomic_manifest(package_inputs):
    repo,store,decoder,kwargs=package_inputs
    result=build_export(repo,store,decoder,**kwargs)
    verified=store.get("export",result.artifact_hash)
    assert verified.payload["game_verified"] is False
    assert verified.payload["logical_count"]==1
    with zipfile.ZipFile(store._destination("export",result.artifact_hash)/"chart.mdm") as archive:
        assert set(archive.namelist())=={"music.ogg","demo.ogg","cover.png","map2.bms","info.json"}
        assert archive.read("cover.png").startswith(b"\x89PNG")
        # A clip shorter than the 20 s preview cap keeps its whole length.
        assert abs(ogg_seconds(decoder,archive.read("demo.ogg"))-2)<0.05


def test_demo_preview_is_capped_at_twenty_seconds(long_package_inputs):
    repo,store,decoder,kwargs=long_package_inputs
    result=build_export(repo,store,decoder,**kwargs)
    with zipfile.ZipFile(store._destination("export",result.artifact_hash)/"chart.mdm") as archive:
        music,demo=ogg_seconds(decoder,archive.read("music.ogg")),ogg_seconds(decoder,archive.read("demo.ogg"))
    assert abs(music-25)<0.05
    assert abs(demo-20)<0.05


def test_stale_head_and_audio_mismatch_rejected(package_inputs):
    repo,store,decoder,kwargs=package_inputs
    with pytest.raises(RevisionConflict):build_export(repo,store,decoder,**{**kwargs,"expected_revision":"old"})
    with pytest.raises(ValueError,match="duration mismatch"):
        build_export(repo,store,decoder,**{**kwargs,"timing":TimingMap((Anchor(0,0),Anchor(8,4)),4)})
    assert not (store.artifacts/"export").exists()


def test_failure_never_publishes_export(package_inputs,monkeypatch):
    repo,store,decoder,kwargs=package_inputs
    monkeypatch.setattr(decoder,"_run",lambda *a,**k:(_ for _ in ()).throw(OSError("disk full")))
    with pytest.raises(OSError):build_export(repo,store,decoder,**kwargs)
    assert not (store.artifacts/"export").exists()
    assert repo.get_project(kwargs["project_id"]).head_revision=="c"
