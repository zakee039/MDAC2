from fractions import Fraction as F
import pytest
from mdac2.domain.chart import ChartEvent
from mdac2.schemas.game_profiles import BASIC


def test_pinned_channel_and_object_mapping():
    assert BASIC.objects(ChartEvent("g", 0, ("ground",))) == (("14", "01"),)
    assert BASIC.objects(ChartEvent("a", 0, ("air",))) == (("13", "01"),)
    assert BASIC.objects(ChartEvent("h", 0, ("air",), "hold", duration_q=F(1))) == (("53", "0F"),)
    assert BASIC.objects(ChartEvent("x", 0, ("ground", "air"), "gemini")) == (("13", "0E"), ("14", "0E"))
    assert BASIC.game_verified is False


@pytest.mark.parametrize("event", [ChartEvent("a", 0, ("ground",), accent=True),
    ChartEvent("b", 0, ("ground", "air"), "hold", duration_q=F(1)),
    ChartEvent("c", 0, ("ground",), "gemini"), ChartEvent("d", 0, ("air",), "hold")])
def test_unsupported_is_not_silently_remapped(event):
    with pytest.raises(ValueError):
        BASIC.objects(event)
