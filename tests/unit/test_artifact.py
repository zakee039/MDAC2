from dataclasses import FrozenInstanceError
import json

import pytest

from mdac2.domain.artifact import (
    ArtifactEnvelope, ArtifactError, RunRecord, canonical_json, content_hash, strict_json,
)


def make(**overrides):
    args = dict(artifact_type="fixture", producer={"name": "test", "version": "1",
                "model_hash": None, "dependencies": {}}, options={"seed": 1},
                payload={"notes": [1, 2]}, inputs=[])
    args.update(overrides)
    return ArtifactEnvelope.create(**args)


def test_canonical_order_and_unicode():
    assert canonical_json({"中": 1, "a": [True, None]}) == '{"a":[true,null],"中":1}'.encode()
    assert content_hash({"x": 1, "y": 2}) == content_hash({"y": 2, "x": 1})
    assert content_hash(1) != content_hash(1.0)


def test_result_identity_separate_from_request():
    a, b = make(), make(payload={"notes": [2, 1]})
    assert a.request_key == b.request_key
    assert a.artifact_hash != b.artifact_hash
    assert make(options={"seed": 2}).request_key != a.request_key
    assert make(inputs=["a" * 64]).request_key != a.request_key
    assert make(producer={"name": "test", "version": "2", "model_hash": None,
                          "dependencies": {}}).request_key != a.request_key


def test_roundtrip_and_defensive_copy():
    payload = {"notes": [1, 2]}
    a = make(payload=payload)
    before = a.to_json()
    payload["notes"].append(3)
    a.payload["notes"].append(4)
    a.to_dict()["producer"]["version"] = "999"
    assert a.to_json() == before
    assert ArtifactEnvelope.from_json(json.dumps(a.to_dict(), indent=2)) == a
    with pytest.raises(FrozenInstanceError):
        a._encoded = b"{}"


def test_run_metadata_cannot_change_content_identity():
    a = make()
    first = RunRecord("run-1", a.artifact_hash, "2026-09-15T00:00:00Z", "a", "ok")
    second = RunRecord("run-2", a.artifact_hash, "2026-09-15T01:00:00Z", "b", "retry")
    assert first != second
    assert first.artifact_hash == second.artifact_hash == make().artifact_hash


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf"),
                                  {1: "bad"}, (1, 2), {1, 2}, b"x", "\ud800"])
def test_invalid_native_values_rejected(bad):
    with pytest.raises(ArtifactError):
        make(payload={"nested": [bad]})


@pytest.mark.parametrize("bad", ['{"a":1,"a":2}', '{"x":{"a":1,"a":2}}',
                                  'NaN', 'Infinity', '-Infinity', '1e999',
                                  '"\\ud800"', '{', b"\xff", b'\xff\xfe{\x00}\x00'])
def test_strict_json_failure_faces(bad):
    with pytest.raises(ArtifactError):
        strict_json(bad)


@pytest.mark.parametrize("field,value", [
    ("payload", {"notes": [9]}), ("payload_hash", "0" * 64),
    ("artifact_hash", "0" * 64), ("request_key", "0" * 64),
    ("options_hash", "0" * 64), ("inputs", ["b" * 64]),
    ("schema_version", 2), ("schema_version", True), ("artifact_type", ""),
    ("inputs", ["BAD"]), ("inputs", {}),
])
def test_tampering_or_invalid_fields_rejected(field, value):
    data = make().to_dict()
    data[field] = value
    with pytest.raises(ArtifactError):
        ArtifactEnvelope.from_json(json.dumps(data))


@pytest.mark.parametrize("target", ["envelope", "producer"])
def test_unknown_fields_require_migration(target):
    data = make().to_dict()
    obj = data if target == "envelope" else data["producer"]
    obj["unexpected"] = "no silent ignore"
    with pytest.raises(ArtifactError):
        ArtifactEnvelope.from_json(json.dumps(data))


@pytest.mark.parametrize("field", sorted(make().to_dict()))
def test_missing_envelope_fields(field):
    data = make().to_dict()
    del data[field]
    with pytest.raises(ArtifactError):
        ArtifactEnvelope.from_json(json.dumps(data))


def test_cyclic_input_rejected():
    data = []
    data.append(data)
    with pytest.raises(ArtifactError):
        canonical_json(data)
