from dataclasses import replace

import pytest

from mdac2.domain.chart import ChartEvent, ChartRevision
from mdac2.domain.timing import Anchor, TimingMap
from mdac2.domain.validation import validate_chart


TIMING = TimingMap((Anchor(-1, 0), Anchor(7, 4)), 4)


def report(*events, **kwargs):
    return validate_chart(ChartRevision("r", "t", tuple(events)), TIMING, timing_revision="t", **kwargs)


def note(**kwargs):
    args = dict(event_id="e1", q=0, lane_mask=("ground",))
    args.update(kwargs)
    return ChartEvent(**args)


def codes(result):
    return {issue.code for issue in result.issues if issue.severity == "error"}


def test_valid_does_not_claim_game_acceptance():
    result = report(note(q=-1), note(event_id="e2", q=7))
    assert not result.has_errors
    assert result.issues[0].code == "GAME_PROFILE_PENDING"


@pytest.mark.parametrize("semantic,duration", [("hold", 0), ("hold", -1), ("tap", 1), ("gemini", 2)])
def test_duration_error(semantic, duration):
    assert "INVALID_DURATION" in codes(report(note(semantic=semantic, duration_q=duration)))


@pytest.mark.parametrize("lanes", [(), ("sky",), ("ground", "ground"), ("ground", "third")])
def test_lane_error(lanes):
    assert "INVALID_LANE_MASK" in codes(report(note(lane_mask=lanes)))


def test_gemini_and_dual_hold_are_not_conflated():
    assert "GEMINI_INCOMPLETE" in codes(report(note(semantic="gemini")))
    assert not report(note(semantic="gemini", lane_mask=("ground", "air"))).has_errors
    assert not report(note(semantic="hold", duration_q=2, lane_mask=("ground", "air"))).has_errors


@pytest.mark.parametrize("q,duration", [(-2, 0), (8, 0), (6, 2)])
def test_head_and_tail_bounds(q, duration):
    assert "TIME_OUT_OF_RANGE" in codes(report(note(q=q, duration_q=duration,
                                                   semantic="hold" if duration else "tap")))


def test_empty_is_explicit_and_distinct():
    assert codes(report()) == {"EMPTY_CHART"}
    allowed = report(allow_empty=True)
    assert not allowed.has_errors
    assert "EMPTY_CHART_ALLOWED" in {issue.code for issue in allowed.issues}


def test_revision_mismatch_does_not_use_wrong_mapping():
    chart = ChartRevision("r", "other", (note(q=999),))
    result = validate_chart(chart, TIMING, timing_revision="t")
    assert codes(result) == {"TIMING_REVISION_MISMATCH"}


def test_validation_never_repairs_or_mutates():
    chart = ChartRevision("r", "t", (note(semantic="hold", duration_q=-4, locked=True),))
    before = chart.to_json()
    first = validate_chart(chart, TIMING, timing_revision="t")
    second = validate_chart(chart, TIMING, timing_revision="t")
    assert first == second
    assert chart.to_json() == before
    assert len(chart.events) == 1


def test_simultaneous_events_not_guessed_illegal():
    a = note()
    assert not report(a, replace(a, event_id="e2")).has_errors
