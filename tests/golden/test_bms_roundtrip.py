from pathlib import Path
from fractions import Fraction as F
import pytest
from mdac2.adapters.bms_reader import read_bms, semantics
from mdac2.adapters.bms_writer import write_bms
from mdac2.domain.chart import ChartEvent as E, ChartRevision as C
from mdac2.domain.timing import TimingMap, Anchor


def test_independent_golden_counts_and_channel_semantics():
    result = read_bms(Path(__file__).with_name("basic.bms").read_text(), timing_revision="t")
    assert (result.logical_count, result.physical_count, result.marker_count) == (4, 5, 6)
    assert result.chart.events[0].lane_mask == ("ground",)
    assert result.chart.events[1].lane_mask == ("air",)
    assert result.chart.events[-1].duration_q == 2


def test_roundtrip_semantics_not_ids_or_text():
    timing = TimingMap((Anchor(0, 0), Anchor(16, 8)), 8)
    chart = C("c", "t", (E("x", F(1, 3), ("ground", "air"), "gemini"),
        E("g", 2, ("ground",)), E("a", 2, ("air",)), E("h", 4, ("air",), "hold", duration_q=F(5, 3))))
    text = write_bms(chart, timing, timing_revision="t")
    restored = read_bms(text, timing_revision="t")
    assert semantics(restored.chart) == semantics(chart)
    assert restored.chart.events[0].event_id != "x"
    assert restored.bpm == 120


@pytest.mark.parametrize("transform", [lambda s:s.replace("#00013:00010E00", "#00013:00010000"),
    lambda s:s.replace("#00154:0F000F00", "#00154:0F"), lambda s:s+"#00099:01\n",
    lambda s:s+"#00014:01\n", lambda s:s.replace("#BPM 120", "#BPM NaN"),
    lambda s:s.replace("music.ogg", "../music.ogg"), lambda s:s.replace("#00014:01000E00", "#00014:0H000E00"),
    lambda s:s+"#BPM01 150\n", lambda s:s.replace("#00001:10", "#00001:0010")])
def test_bad_or_unsupported_never_silently_discarded(transform):
    source = Path(__file__).with_name("basic.bms").read_text()
    with pytest.raises(ValueError): read_bms(transform(source), timing_revision="t")
