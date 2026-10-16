"""Bounded initial audio upload and authenticated ID-only WAV Range serving."""
import re
import tempfile
import wave
import struct

from fastapi import Request
from starlette.concurrency import run_in_threadpool
from starlette.responses import FileResponse

from mdac2.api.security import error, identifier
from mdac2.infrastructure.project_repository import RepositoryError, RevisionConflict


def attach_assets(app, repository, store, decoder):
    with repository._connection() as connection:
        connection.execute("BEGIN IMMEDIATE")
        if not connection.execute("SELECT 1 FROM sqlite_master WHERE name='asset_schema'").fetchone():
            if connection.execute("SELECT 1 FROM sqlite_master WHERE name='project_audio'").fetchone():
                raise RepositoryError("Unversioned audio table")
            connection.execute("CREATE TABLE asset_schema(version INTEGER NOT NULL CHECK(version=1))")
            connection.execute("INSERT INTO asset_schema VALUES(1)")
            connection.execute("CREATE TABLE project_audio(project_id TEXT PRIMARY KEY REFERENCES projects(project_id), artifact_hash TEXT NOT NULL)")
        elif [tuple(row) for row in connection.execute("SELECT version FROM asset_schema")] != [(1,)]:
            raise RepositoryError("Unknown audio schema")
        connection.commit()

    def check_empty(connection, project_id):
        project = connection.execute("SELECT head_revision,archived FROM projects WHERE project_id=?", (project_id,)).fetchone()
        if project is None:
            raise KeyError(project_id)
        if project[0] is not None or project[1] or connection.execute("SELECT 1 FROM project_audio WHERE project_id=?", (project_id,)).fetchone():
            raise RevisionConflict("Audio replacement requires an explicit workflow")

    @app.post("/api/v1/projects/{project_id}/audio", status_code=201)
    async def upload(project_id: str, request: Request):
        identifier(project_id)
        if request.headers.get("content-type", "").split(";")[0] not in {"application/octet-stream", "audio/wav", "audio/mpeg", "audio/flac", "audio/ogg", "audio/mp4"}:
            return error(415, "UNSUPPORTED_MEDIA", "Raw audio upload required")
        with repository._connection() as connection:
            check_empty(connection, project_id)
        declared = request.headers.get("content-length")
        if declared is not None and (not declared.isdigit() or int(declared) > decoder.limits.max_bytes):
            return error(413, "UPLOAD_TOO_LARGE", "Audio upload exceeds byte limit")
        # TemporaryFile is owned by this request and always closed, including disconnect.
        with tempfile.TemporaryFile(mode="w+b") as temporary:
            count = 0
            async for chunk in request.stream():
                count += len(chunk)
                if count > decoder.limits.max_bytes:
                    return error(413, "UPLOAD_TOO_LARGE", "Audio upload exceeds byte limit")
                await run_in_threadpool(temporary.write, chunk)
            if declared is not None and count != int(declared):
                raise ValueError("Content length mismatch")
            temporary.seek(0)
            data = await run_in_threadpool(temporary.read)
        envelope = await run_in_threadpool(decoder.import_audio, data, store)
        def register():
            with repository._connection() as connection:
                connection.execute("BEGIN IMMEDIATE")
                check_empty(connection, project_id)
                connection.execute("INSERT INTO project_audio VALUES (?,?)", (project_id, envelope.artifact_hash))
                connection.commit()
        await run_in_threadpool(register)
        return {"asset_id": envelope.artifact_hash, "canonical": envelope.payload["canonical"],
                "audio_url": f"/api/v1/assets/{envelope.artifact_hash}/audio"}

    @app.get("/api/v1/projects/{project_id}/audio")
    def project_audio(project_id: str):
        identifier(project_id)
        with repository._connection() as connection:
            row = connection.execute("SELECT artifact_hash FROM project_audio WHERE project_id=?", (project_id,)).fetchone()
        if row is None:
            raise KeyError(project_id)
        envelope = store.get("audio", row[0])
        return {"asset_id": row[0], "canonical": envelope.payload["canonical"], "audio_url": f"/api/v1/assets/{row[0]}/audio"}

    @app.get("/api/v1/assets/{asset_id}/audio")
    def audio(asset_id: str):
        if not re.fullmatch(r"[0-9a-f]{64}", asset_id):
            raise ValueError("Invalid audio ID")
        with repository._connection() as connection:
            if not connection.execute("SELECT 1 FROM project_audio WHERE artifact_hash=?", (asset_id,)).fetchone():
                raise KeyError(asset_id)
        envelope = store.get("audio", asset_id)
        wav = next(entry for entry in envelope.payload["files"] if entry["name"] == "audio.wav")
        return FileResponse(store._destination("audio", asset_id) / "audio.wav", media_type="audio/wav",
                            headers={"ETag": '"' + wav["sha256"] + '"'})

    @app.get("/api/v1/projects/{project_id}/peaks")
    def peaks(project_id: str):
        data = project_audio(project_id)
        with wave.open(str(store._destination("audio", data["asset_id"]) / "audio.wav"), "rb") as source:
            channels, frames = source.getnchannels(), source.getnframes()
            stride = max(1, (frames + 1999) // 2000)
            values = [[] for _ in range(channels)]
            for start in range(0, frames, stride):
                raw = source.readframes(min(stride, frames-start))
                samples = [sample[0] for sample in struct.iter_unpack("<h", raw)]
                for channel in range(channels):
                    values[channel].append(max((abs(v) for v in samples[channel::channels]), default=0)/32768)
            return {"peaks": values, "duration": frames/source.getframerate()}
