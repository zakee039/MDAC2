"""Windows same-volume immutable directory publication.

Manifest binds optional flat sidecars through payload.files entries (name,
sha256, byte_count). Files are flushed/fsynced and handles closed before rename.
Windows rename refuses existing destinations. No promise of power-loss durability
for directory metadata. Staging is retained on failure/race for reconciliation;
it is never served by get(). No database registration occurs here.
"""

import hashlib
import os
from pathlib import Path
import re
import stat
from uuid import uuid4

from mdac2.domain.artifact import ArtifactEnvelope


class StoreError(ValueError):
    pass


def _name(value):
    if type(value) is not str or not re.fullmatch(r"[a-z0-9][a-z0-9_.-]{0,95}", value):
        raise StoreError("Unsafe path component")
    if value.endswith(".") or value.split(".")[0].upper() in {
            "CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(10)), *(f"LPT{i}" for i in range(10))}:
        raise StoreError("Reserved Windows path component")
    return value


def _plain(path):
    info = path.lstat()
    if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
        raise StoreError("Reparse points are not allowed in the store")


def _file_manifest(envelope):
    payload = envelope.payload
    entries = payload.get("files", []) if type(payload) is dict else []
    if type(entries) is not list:
        raise StoreError("payload.files must be a list")
    result = {}
    for entry in entries:
        if type(entry) is not dict or set(entry) != {"name", "sha256", "byte_count"}:
            raise StoreError("Invalid sidecar manifest entry")
        name = _name(entry["name"])
        if name == "manifest.json" or name in result:
            raise StoreError("Reserved/duplicate sidecar")
        if type(entry["sha256"]) is not str or not re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]):
            raise StoreError("Invalid sidecar SHA-256")
        if type(entry["byte_count"]) is not int or entry["byte_count"] < 0:
            raise StoreError("Invalid sidecar length")
        result[name] = entry
    return result


class ArtifactStore:
    def __init__(self, root):
        self.root = Path(root).absolute()
        # Refuse existing reparse ancestors, rather than resolving through them.
        for path in reversed((self.root, *self.root.parents)):
            if path.exists():
                _plain(path)
        self.root.mkdir(parents=True, exist_ok=True)
        self.staging = self.root / "staging"
        self.artifacts = self.root / "artifacts"
        for path in (self.staging, self.artifacts):
            path.mkdir(exist_ok=True)
            _plain(path)

    def _destination(self, artifact_type, artifact_hash):
        _name(artifact_type)
        if type(artifact_hash) is not str or not re.fullmatch(r"[0-9a-f]{64}", artifact_hash):
            raise StoreError("Invalid artifact ID")
        for path in (self.root, self.staging, self.artifacts):
            _plain(path)
        parent = self.artifacts / artifact_type
        if parent.exists():
            _plain(parent)
        return parent / artifact_hash

    def _verify(self, directory, artifact_type, artifact_hash):
        _plain(directory)
        for path in directory.iterdir():
            _plain(path)
            if not path.is_file():
                raise StoreError("Unexpected directory or special file")
        envelope = ArtifactEnvelope.from_json((directory / "manifest.json").read_bytes())
        if envelope.artifact_hash != artifact_hash or envelope.to_dict()["artifact_type"] != artifact_type:
            raise StoreError("Manifest identity does not match destination")
        files = _file_manifest(envelope)
        if {path.name for path in directory.iterdir()} != {"manifest.json", *files}:
            raise StoreError("Unexpected or missing artifact files")
        for name, entry in files.items():
            path = directory / name
            if path.stat().st_size != entry["byte_count"]:
                raise StoreError("Sidecar size mismatch")
            with path.open("rb") as handle:
                digest = hashlib.file_digest(handle, "sha256").hexdigest()
            if digest != entry["sha256"]:
                raise StoreError("Sidecar content mismatch")
        return envelope

    def get(self, artifact_type, artifact_hash):
        return self._verify(self._destination(artifact_type, artifact_hash), artifact_type, artifact_hash)

    @staticmethod
    def _write(path, data):
        with path.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())

    def publish(self, envelope: ArtifactEnvelope, files=None):
        if os.name != "nt":
            raise NotImplementedError("No-clobber directory rename is validated only on Windows")
        files = {} if files is None else dict(files)
        expected = _file_manifest(envelope)
        if set(files) != set(expected) or any(type(data) is not bytes for data in files.values()):
            raise StoreError("Sidecar inputs must exactly match manifest and contain bytes")
        for name, data in files.items():
            if len(data) != expected[name]["byte_count"] or hashlib.sha256(data).hexdigest() != expected[name]["sha256"]:
                raise StoreError("Sidecar input does not match content identity")
        artifact_type = envelope.to_dict()["artifact_type"]
        destination = self._destination(artifact_type, envelope.artifact_hash)
        if destination.exists():
            self.get(artifact_type, envelope.artifact_hash)
            return destination
        destination.parent.mkdir(exist_ok=True)
        _plain(destination.parent)
        if self.staging.stat().st_dev != destination.parent.stat().st_dev:
            raise StoreError("Staging must be on the destination volume")
        stage = self.staging / str(uuid4())
        stage.mkdir()
        for name, data in files.items():
            self._write(stage / name, data)
        self._write(stage / "manifest.json", envelope.to_json())
        self._verify(stage, artifact_type, envelope.artifact_hash)
        try:
            os.rename(stage, destination)
        except OSError:
            if not destination.exists():
                raise
            # Concurrent winner must be deeply valid; never overwrite or repair it.
            self.get(artifact_type, envelope.artifact_hash)
        return destination
