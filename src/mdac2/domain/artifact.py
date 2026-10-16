"""Artifact schema 1 and mdac-json-v1 (not RFC 8785/JCS).

Canonical bytes: UTF-8, sorted string keys, compact separators, no ASCII escaping,
finite native JSON types only. Floats retain Python JSON spelling (1 != 1.0),
Unicode is not normalized. Arrays retain order, including inputs. Schema/producer
changes require explicit migration; payload's type-specific schema belongs to its
own contract. Hashes are integrity checks, not signatures or proof of quality.
"""

from dataclasses import dataclass
import hashlib
import json
import math
import re


class ArtifactError(ValueError):
    """Invalid JSON, unsupported contract, or inconsistent identity."""


def _json_value(value):
    if value is None or type(value) in (bool, int, str):
        return
    if type(value) is float and math.isfinite(value):
        return
    if type(value) is list:
        for item in value:
            _json_value(item)
        return
    if type(value) is dict and all(type(key) is str for key in value):
        for item in value.values():
            _json_value(item)
        return
    raise ArtifactError("Only finite, native JSON values and string keys are allowed")


def canonical_json(value) -> bytes:
    try:
        _json_value(value)
        return json.dumps(value, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False, allow_nan=False).encode("utf-8")
    except (ValueError, TypeError, UnicodeError, RecursionError) as exc:
        raise ArtifactError("Invalid canonical JSON") from exc


def content_hash(value) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ArtifactError(f"Duplicate JSON field: {key}")
        result[key] = value
    return result


def _constant(value):
    raise ArtifactError(f"Non-finite JSON number: {value}")


def strict_json(data: str | bytes):
    if type(data) not in (str, bytes):
        raise ArtifactError("JSON input must be str or UTF-8 bytes")
    try:
        if type(data) is bytes:
            data = data.decode("utf-8")
        value = json.loads(data, object_pairs_hook=_pairs, parse_constant=_constant)
        canonical_json(value)  # also rejects overflow such as 1e999 and surrogates
        return value
    except (ValueError, TypeError, UnicodeError, RecursionError) as exc:
        raise ArtifactError("Invalid strict JSON") from exc


def _fields(value, expected):
    if type(value) is not dict or set(value) != set(expected):
        raise ArtifactError(f"Expected exactly fields: {sorted(expected)}")


def _text(value):
    if type(value) is not str or not value.strip():
        raise ArtifactError("Expected non-empty string")


def _hash(value):
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ArtifactError("Expected lowercase SHA-256 hex")


_FIELDS = {"artifact_type", "schema_version", "request_key", "payload_hash",
           "artifact_hash", "inputs", "producer", "options_hash", "payload"}
_REQUEST_FIELDS = ("artifact_type", "schema_version", "inputs", "producer", "options_hash")


def _request(document):
    return {key: document[key] for key in _REQUEST_FIELDS}


def _validate(document):
    _fields(document, _FIELDS)
    _text(document["artifact_type"])
    if type(document["schema_version"]) is not int or document["schema_version"] != 1:
        raise ArtifactError("Unsupported schema_version; explicit migration required")
    for key in ("request_key", "payload_hash", "artifact_hash", "options_hash"):
        _hash(document[key])
    if type(document["inputs"]) is not list:
        raise ArtifactError("inputs must be an ordered list of artifact hashes")
    for item in document["inputs"]:
        _hash(item)
    producer = document["producer"]
    _fields(producer, {"name", "version", "model_hash", "dependencies"})
    _text(producer["name"])
    _text(producer["version"])
    if producer["model_hash"] is not None:
        _hash(producer["model_hash"])
    if type(producer["dependencies"]) is not dict:
        raise ArtifactError("dependencies must map names to exact version identities")
    for key, value in producer["dependencies"].items():
        _text(key)
        _text(value)
    if document["request_key"] != content_hash(_request(document)):
        raise ArtifactError("request_key mismatch")
    if document["payload_hash"] != content_hash(document["payload"]):
        raise ArtifactError("payload_hash mismatch")
    if document["artifact_hash"] != content_hash(
            {key: value for key, value in document.items() if key != "artifact_hash"}):
        raise ArtifactError("artifact_hash mismatch")


@dataclass(frozen=True, slots=True)
class ArtifactEnvelope:
    """Immutable canonical snapshot; accessors return detached mutable JSON."""

    _encoded: bytes

    def __post_init__(self):
        if type(self._encoded) is not bytes:
            raise ArtifactError("Envelope storage must be bytes")
        document = strict_json(self._encoded)
        _validate(document)
        object.__setattr__(self, "_encoded", canonical_json(document))

    @classmethod
    def create(cls, *, artifact_type, producer, options, payload, inputs=None):
        document = {
            "artifact_type": artifact_type, "schema_version": 1,
            "inputs": [] if inputs is None else inputs,
            "producer": producer, "options_hash": content_hash(options),
            "payload": payload, "payload_hash": content_hash(payload),
        }
        document["request_key"] = content_hash(_request(document))
        document["artifact_hash"] = content_hash(document)
        return cls(canonical_json(document))

    @classmethod
    def from_json(cls, data: str | bytes):
        return cls(canonical_json(strict_json(data)))

    def to_json(self) -> bytes:
        return self._encoded

    def to_dict(self) -> dict:
        return strict_json(self._encoded)

    @property
    def artifact_hash(self) -> str:
        return self.to_dict()["artifact_hash"]

    @property
    def request_key(self) -> str:
        return self.to_dict()["request_key"]

    @property
    def payload(self):
        return self.to_dict()["payload"]


@dataclass(frozen=True, slots=True)
class RunRecord:
    """Execution metadata references an artifact; never contributes to its hash."""

    run_id: str
    artifact_hash: str
    started_at: str
    machine_path: str
    log: str

    def __post_init__(self):
        _text(self.run_id)
        _hash(self.artifact_hash)
        _text(self.started_at)
        for value in (self.machine_path, self.log):
            if type(value) is not str:
                raise ArtifactError("Run metadata must be text")
