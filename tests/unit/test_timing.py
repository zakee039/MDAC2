from fractions import Fraction as F

import pytest

from mdac2.domain.timing import Anchor, MeterSegment, TimingError, TimingMap, quarter_bpm


def fixed(**kwargs):
    options = dict(anchors=(Anchor(0, 0), Anchor(8, 4)), audio_duration=4)
    options.update(kwargs)
    return TimingMap(**options)


def test_fixed_exact_roundtrip():
    m = fixed()
    for step in range(801):
        q = F(step, 100)
        assert m.t(q) == q / 2
        assert m.q(m.t(q)) == q
        assert m.local_bpm(q) == 120
    assert TimingMap.from_dict(m.to_dict()) == m


def test_piecewise_and_boundary_tempo():
    m = fixed(anchors=(Anchor(0, 0), Anchor(4, 2), Anchor(8, 6)), audio_duration=6)
    assert m.local_bpm(3) == 120
    assert m.local_bpm(4) == 60  # right-continuous at interior anchor
    assert m.t(6) == 4
    for step in range(601):
        t = F(step, 100)
        assert m.t(m.q(t)) == t


def test_compound_meter_and_pickup():
    m = TimingMap((Anchor(-1, 0), Anchor(5, 3)), 3, F(3, 8), (MeterSegment(0, 6, 8),))
    assert quarter_bpm(80, m.pulse_unit) == 120
    assert m.meter_at(-1).bar_quarters == 3
    assert m.t(0) == F(1, 2)
    assert m.q(0) == -1


def test_meter_changes():
    m = fixed(meter_segments=(MeterSegment(0, 4, 4), MeterSegment(4, 6, 8)))
    assert m.meter_at(F(399, 100)).numerator == 4
    assert m.meter_at(4).bar_quarters == 3


def test_bounded_extrapolation_both_directions():
    m = TimingMap((Anchor(0, 1), Anchor(4, 3)), 5, extrapolation_seconds=F(1, 2))
    assert m.time_bounds == (F(1, 2), F(7, 2))
    assert m.q(F(1, 2)) == -1
    assert m.t(5) == F(7, 2)
    for t in (0, 4, -1, 6):
        with pytest.raises(TimingError):
            m.q(t)
    with pytest.raises(TimingError):
        m.t(6)
    clipped = TimingMap((Anchor(0, 1), Anchor(4, 3)), 4, extrapolation_seconds=100)
    assert clipped.time_bounds == (0, 4)


@pytest.mark.parametrize("anchors", [(), (Anchor(0, 0),), (Anchor(0, 0), Anchor(0, 1)),
                                     (Anchor(0, 0), Anchor(1, 0)), (Anchor(1, 0), Anchor(0, 1)),
                                     (Anchor(0, 2), Anchor(1, 1))])
def test_degenerate_anchor_sets(anchors):
    with pytest.raises(TimingError):
        fixed(anchors=anchors)


@pytest.mark.parametrize("kwargs", [dict(audio_duration=3), dict(audio_duration=-1),
                                    dict(pulse_unit=0), dict(extrapolation_seconds=-1),
                                    dict(schema_version=True), dict(schema_version=2),
                                    dict(meter_segments=()),
                                    dict(meter_segments=(MeterSegment(1, 4, 4),)),
                                    dict(meter_segments=(MeterSegment(0, 4, 4), MeterSegment(3, 3, 4)))])
def test_invalid_map_fields(kwargs):
    with pytest.raises(TimingError):
        fixed(**kwargs)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), True, "1"])
def test_invalid_numbers(value):
    with pytest.raises(TimingError):
        Anchor(value, 0)
    with pytest.raises(TimingError):
        fixed().q(value)


@pytest.mark.parametrize("n,d", [(0, 4), (True, 4), (4, 0), (4, 3), (4, True)])
def test_bad_meter(n, d):
    with pytest.raises(TimingError):
        MeterSegment(0, n, d)


def test_negative_seconds_and_default_no_extrapolation():
    with pytest.raises(TimingError):
        Anchor(0, -1)
    for q in (-1, 9):
        with pytest.raises(TimingError):
            fixed().t(q)


def test_strict_serialization():
    data = fixed().to_dict()
    with pytest.raises(TimingError):
        TimingMap.from_dict({**data, "unknown": 1})
    data["anchors"][0]["q"] = "0/2"
    with pytest.raises(TimingError):
        TimingMap.from_dict(data)


def test_decimal_input_preserves_decimal_intent():
    assert Anchor(.1, .2).q == F(1, 10)
