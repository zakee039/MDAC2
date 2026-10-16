from fractions import Fraction as F
import pytest
from mdac2.adapters.bms_timing import write_timed_bms, read_timed_bms
from mdac2.adapters.bms_reader import semantics
from mdac2.domain.chart import ChartEvent as E, ChartRevision as C
from mdac2.domain.timing import TimingMap as T, Anchor as A, MeterSegment as M


@pytest.mark.parametrize("timing,events", [
    (T((A(0,0),A(4,2),A(12,10)),10),(E("h",2,("air",),"hold",duration_q=F(6)),)),
    (T((A(0,0),A(12,6)),6,meter_segments=(M(0,6,8),)),(E("a",F(7,2),("ground",)),)),
    (T((A(-1,0),A(15,8)),8),(E("a",-1,("air",)),E("b",0,("ground",)))),
    (T((A(0,1),A(12,7)),7),(E("a",1,("air",)),))])
def test_time_and_semantic_roundtrip_with_explicit_padding(timing, events):
    chart=C("c","t",events)
    output=write_timed_bms(chart,timing,timing_revision="t")
    result,seconds=read_timed_bms(output.text,timing_revision="t",q_offset=output.q_offset)
    assert semantics(result.chart)==semantics(chart)
    for event in events:
        assert seconds(event.q)==timing.t(event.q)+output.audio_padding_seconds
        assert seconds(event.end_q)==timing.t(event.end_q)+output.audio_padding_seconds
    if "#BPM01" in output.text:
        assert output.text.index("#00108:") < output.text.index("#00053:")


def test_unsupported_combinations_and_unrepresentable_bpm_rejected():
    chart=C("c","t",(E("a",1,("air",)),))
    cases=[T((A(0,0),A(4,2),A(12,10)),10,meter_segments=(M(0,6,8),)),
        T((A(0,0),A(12,6)),6,meter_segments=(M(0,4,4),M(4,6,8))),
        T((A(0,0),A(12,7)),7)]
    for timing in cases:
        with pytest.raises(ValueError):write_timed_bms(chart,timing,timing_revision="t")
