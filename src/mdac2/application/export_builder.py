"""Validated snapshot package construction, with explicit unverified-game status."""
from fractions import Fraction
import hashlib
import io
import struct
import wave
import zipfile
import zlib

from mdac2.adapters.bms_timing import write_timed_bms, read_timed_bms
from mdac2.adapters.bms_reader import semantics
from mdac2.domain.artifact import ArtifactEnvelope, canonical_json, strict_json
from mdac2.domain.chart import ChartRevision
from mdac2.infrastructure.project_repository import RevisionConflict


def digest(data):
    return hashlib.sha256(data).hexdigest()


DEMO_SECONDS = 20  # Preview cap; clips shorter than this keep their full length.


def default_cover():
    # A declared placeholder, not an inferred user cover or external resource.
    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind+data))
    width = 256
    pixels = (b"\0" + b"\x35\x42\x70"*width)*width
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width,width,8,2,0,0,0)) + chunk(b"IDAT",zlib.compress(pixels)) + chunk(b"IEND", b"")


def encode_ogg(decoder, pcm, rate, channels):
    data = decoder._run(decoder.ffmpeg, ["-v", "fatal", "-nostdin", "-threads", "1", "-protocol_whitelist", "pipe",
        "-f", "s16le", "-ar", str(rate), "-ac", str(channels), "-i", "pipe:0", "-map_metadata", "-1",
        "-c:a", "libvorbis", "-q:a", "5", "-f", "ogg", "pipe:1"], pcm)
    decoded, _ = decoder.decode(data, target_rate=rate)
    actual = decoded.payload["canonical"]
    error = actual["sample_count"] - len(pcm)//(channels*2)
    if len(actual["channel_layout"]) != channels or abs(error) > 1024:
        raise ValueError("Encoded audio failed channel/length verification")
    return data, error


def build_export(repository, store, decoder, *, project_id, expected_revision, timing,
                 timing_revision, title, artist, historical=False):
    if type(historical) is not bool:
        raise ValueError("Explicit historical bool required")
    project = repository.get_project(project_id)
    if project.archived or (not historical and project.head_revision != expected_revision):
        raise RevisionConflict("Export base is not current head")
    reference = repository.get_revision(project_id, expected_revision)
    chart_artifact = store.get("chart", reference.artifact_hash)
    chart = ChartRevision.from_json(canonical_json(chart_artifact.payload))
    with repository._connection() as connection:
        row = connection.execute("SELECT artifact_hash FROM project_audio WHERE project_id=?", (project_id,)).fetchone()
    if row is None:
        raise ValueError("Project audio missing")
    audio = store.get("audio", row[0])
    wav_bytes = (store._destination("audio", row[0])/"audio.wav").read_bytes()
    with wave.open(io.BytesIO(wav_bytes), "rb") as source:
        rate, channels, count = source.getframerate(), source.getnchannels(), source.getnframes()
        if source.getsampwidth() != 2:
            raise ValueError("Canonical PCM16 required")
        pcm = source.readframes(count)
    if timing.audio_duration != Fraction(count, rate):
        raise ValueError("Timing/audio duration mismatch")
    plan = write_timed_bms(chart, timing, timing_revision=timing_revision, title=title, artist=artist)
    samples = plan.audio_padding_seconds * rate
    if samples.denominator != 1:
        raise ValueError("Audio padding is not sample-exact")
    padded = bytes(int(samples)*channels*2)+pcm
    if len(padded) > decoder.limits.max_pcm_bytes or len(padded)//(channels*2) > decoder.limits.max_seconds*rate:
        raise ValueError("Export audio exceeds limit")
    restored, seconds = read_timed_bms(plan.text, timing_revision=timing_revision, q_offset=plan.q_offset)
    if semantics(restored.chart) != semantics(chart):
        raise ValueError("BMS semantic roundtrip failed")
    for event in chart.events:
        for q in (event.q, event.end_q):
            if seconds(q) != timing.t(q)+plan.audio_padding_seconds:
                raise ValueError("BMS timing roundtrip failed")
    music, music_error = encode_ogg(decoder, padded, rate, channels)
    # Demo preview: first 20 s of the canonical audio, or the whole clip when it is shorter.
    demo_pcm = pcm[:DEMO_SECONDS*rate*channels*2]
    demo, demo_error = encode_ogg(decoder, demo_pcm, rate, channels)
    resources = {"music.ogg": music, "demo.ogg": demo, "cover.png": default_cover(),
        "map2.bms": plan.text.encode("utf-8"), "info.json": canonical_json({"name":title,"author":artist,
        "bpm":str(restored.bpm),"scene":"scene_01","levelDesigner":"MDAC2","difficulty1":"0",
        "difficulty2":"1","difficulty3":"0","difficulty4":"0","unlockLevel":0})}
    stream=io.BytesIO()
    with zipfile.ZipFile(stream,"w",compression=zipfile.ZIP_DEFLATED) as archive:
        for name,data in sorted(resources.items()):
            entry=zipfile.ZipInfo(name,(1980,1,1,0,0,0)); entry.compress_type=zipfile.ZIP_DEFLATED
            archive.writestr(entry,data)
    package=stream.getvalue()
    with zipfile.ZipFile(io.BytesIO(package)) as archive:
        if set(archive.namelist()) != set(resources) or archive.testzip() is not None:
            raise ValueError("Invalid package resources")
        for name,data in resources.items():
            if archive.read(name)!=data: raise ValueError("Package resource mismatch")
    timing_artifact=ArtifactEnvelope.create(artifact_type="timing",producer={"name":"manual","version":"1","model_hash":None,"dependencies":{}},
        options={},payload={"revision_id":timing_revision,"mapping":timing.to_dict()},inputs=[audio.artifact_hash])
    with repository._connection() as connection:
        if connection.execute("SELECT 1 FROM sqlite_master WHERE name='revision_timing'").fetchone():
            bound=connection.execute("SELECT artifact_hash FROM revision_timing WHERE revision_id=?",(expected_revision,)).fetchone()
            if bound is None:raise ValueError("Authoritative timing binding missing")
            timing_artifact=store.get("timing",bound[0])
            if timing_artifact.payload["mapping"]!=timing.to_dict() or timing_artifact.payload["revision_id"]!=timing_revision:
                raise ValueError("Export timing differs from authoritative mapping")
    store.publish(timing_artifact)
    report={"project_id":project_id,"chart_revision":chart.revision_id,"timing_revision":timing_revision,
        "historical":historical,"profile":plan.profile_id,"game_verified":False,
        "q_offset":str(plan.q_offset),"audio_padding_seconds":str(plan.audio_padding_seconds),"audio_padding_samples":int(samples),
        "audio_sample_rate":rate,"cover_policy":"generated-solid-v1","demo_policy":"first-20s-canonical-v2",
        "codec_length_error_samples":{"music":music_error,"demo":demo_error},
        "logical_count":restored.logical_count,"physical_count":restored.physical_count,
        "resources":[{"name":n,"sha256":digest(d),"byte_count":len(d)} for n,d in sorted(resources.items())],
        "files":[{"name":"chart.mdm","sha256":digest(package),"byte_count":len(package)}]}
    envelope=ArtifactEnvelope.create(artifact_type="export",producer={"name":"mdac2.export","version":"1","model_hash":None,"dependencies":decoder.producer["dependencies"]},
        options={"profile":plan.profile_id,"title":title,"artist":artist,"historical":historical},
        payload=report,inputs=[chart_artifact.artifact_hash,audio.artifact_hash,timing_artifact.artifact_hash])
    # A concurrent change leaves at most an orphan immutable artifact, never a current export.
    if not historical and repository.get_project(project_id).head_revision != expected_revision:
        raise RevisionConflict("Head changed during export")
    store.publish(envelope,{"chart.mdm":package})
    if not historical and repository.get_project(project_id).head_revision != expected_revision:
        raise RevisionConflict("Head changed during export publication")
    return envelope
