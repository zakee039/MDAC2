from fractions import Fraction as F
import pytest
from mdac2.adapters.bms_writer import write_bms
from mdac2.domain.chart import ChartEvent as E, ChartRevision as C
from mdac2.domain.timing import TimingMap, Anchor, MeterSegment


def timing():
    return TimingMap((Anchor(0, 0), Anchor(16, 8)), 8)


def test_golden_entities_hold_and_music():
    chart = C("c", "t", (E("g", 0, ("ground",)), E("a", 1, ("air",)),
        E("x", 2, ("ground", "air"), "gemini"), E("h", 4, ("ground",), "hold", duration_q=F(2))))
    text = write_bms(chart, timing(), timing_revision="t")
    assert "#00001:10" in text
    assert "#00013:00010E00" in text
    assert "#00014:010E" in text
    assert "#00154:0F0F" in text
    assert "#WAV10 music.ogg" in text


def test_fractional_positions_not_rounded():
    text = write_bms(C("c", "t", (E("a", F(1, 3), ("air",)),)), timing(), timing_revision="t")
    data = next(line.split(":")[1] for line in text.splitlines() if line.startswith("#00013:"))
    assert len(data) == 24 and data[2:4] == "01"


@pytest.mark.parametrize("events", [(), (E("a", 0, ("ground",)), E("b", 0, ("ground",))),
    (E("h", 1, ("air",), "hold", duration_q=F(4)), E("a", 2, ("air",))),
    (E("a", F(1, 20000), ("air",)),), (E("a", 0, ("air",), accent=True),)])
def test_invalid_unrepresentable_and_collisions_rejected(events):
    with pytest.raises(ValueError):
        write_bms(C("c", "t", events), timing(), timing_revision="t")


def test_metadata_timing_and_resource_refusals():
    chart = C("c", "t", (E("a", 1, ("air",)),))
    for kwargs in ({"timing_revision": "old"}, {"timing_revision": "t", "title": "X\n#BPM 999"},
                   {"timing_revision": "t", "music": "../music.ogg"}):
        with pytest.raises(ValueError): write_bms(chart, timing(), **kwargs)
    with pytest.raises(ValueError):
        write_bms(chart, TimingMap((Anchor(0, 0), Anchor(4, 2), Anchor(8, 5)), 8), timing_revision="t")
