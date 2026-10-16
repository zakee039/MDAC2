from dataclasses import FrozenInstanceError, replace
from fractions import Fraction as F
import json

import pytest

from mdac2.domain.chart import ChartError, ChartEvent, ChartRevision, new_event_id


def event(**overrides):
    args = dict(event_id="stable-1", q=F(1, 3), lane_mask=("ground",))
    args.update(overrides)
    return ChartEvent(**args)


def test_simultaneous_events_and_fraction_roundtrip():
    events = (event(), event(event_id="stable-2", lane_mask=("air",)),
              event(event_id="stable-3", semantic="hold", duration_q=F(2, 3), locked=True),
              event(event_id="stable-4", semantic="gemini", lane_mask=("ground", "air")))
    chart = ChartRevision("r1", "t1", events, user_patch_ids=("patch-1",))
    loaded = ChartRevision.from_json(chart.to_json())
    assert loaded == chart
    assert len(loaded.events) == 4
    assert loaded.events[2].end_q == 1
    assert loaded.events[3].lane_mask == ("ground", "air")


def test_edit_does_not_generate_identity_or_mutate_base():
    original = event()
    updated = replace(original, q=2, lane_mask=("air",))
    assert original.event_id == updated.event_id
    assert original.q == F(1, 3)
    with pytest.raises(FrozenInstanceError):
        original.q = 3
    assert new_event_id() != new_event_id()


def test_duplicate_id_rejected_not_duplicate_q():
    with pytest.raises(ChartError):
        ChartRevision("r", "t", (event(), event(q=4)))


@pytest.mark.parametrize("overrides", [dict(event_id=""), dict(lane_mask=["ground"]),
                                        dict(accent=1), dict(locked="yes"), dict(semantic="unknown"),
                                        dict(source_kind="candidate"), dict(source_kind="fake"),
                                        dict(source_id="unexpected")])
def test_bad_structural_event(overrides):
    with pytest.raises(ChartError):
        event(**overrides)


def test_source_and_empty_chart_are_representable():
    assert event(source_kind="candidate", source_id="c1").source_id == "c1"
    chart = ChartRevision("empty", "t", ())
    assert ChartRevision.from_json(chart.to_json()) == chart


@pytest.mark.parametrize("field", list(event().to_dict()))
def test_missing_event_fields(field):
    data = event().to_dict()
    del data[field]
    with pytest.raises(ChartError):
        ChartEvent.from_dict(data)


def test_unknown_fields_rejected():
    data = ChartRevision("r", "t", (event(),)).to_dict()
    with pytest.raises(ChartError):
        ChartRevision.from_json(json.dumps({**data, "extra": 1}))
    data["events"][0]["extra"] = 1
    with pytest.raises(ChartError):
        ChartRevision.from_json(json.dumps(data))


def test_serialization_is_detached():
    chart = ChartRevision("r", "t", (event(),))
    data = chart.to_dict()
    data["events"][0]["lane_mask"].append("air")
    assert chart.events[0].lane_mask == ("ground",)
