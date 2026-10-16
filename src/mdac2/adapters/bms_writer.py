"""Exact, bounded basic BMS encoding. No file writes or silent quantization."""
from fractions import Fraction
from math import lcm
import re

from mdac2.domain.chart import ChartRevision
from mdac2.domain.timing import TimingMap
from mdac2.domain.validation import validate_chart
from mdac2.schemas.game_profiles import BASIC


def decimal(value):
    value = Fraction(value)
    denominator = value.denominator
    for prime in (2, 5):
        while denominator % prime == 0:
            denominator //= prime
    if denominator != 1:
        raise ValueError("BMS decimal cannot represent value exactly")
    from decimal import Decimal, localcontext
    with localcontext() as context:
        context.prec = max(50, len(str(abs(value.numerator))) + len(str(value.denominator)) + 10)
        return format(Decimal(value.numerator) / Decimal(value.denominator), "f")


def data_lines(cells, profile=BASIC):
    grouped = {}
    for measure, fraction, channel, value in cells:
        if not 0 <= measure <= profile.max_measure or not 0 <= fraction < 1:
            raise ValueError("BMS measure out of bounds")
        entries = grouped.setdefault((measure, channel), {})
        if fraction in entries:
            raise ValueError("BMS cell collision; no event was overwritten")
        entries[fraction] = value
    lines = []
    for (measure, channel), entries in sorted(grouped.items()):
        count = 1
        for fraction in entries:
            count = lcm(count, fraction.denominator)
            if count > profile.max_cells:
                raise ValueError("Exact BMS grid exceeds profile limit")
        slots = ["00"] * count
        for fraction, value in entries.items():
            slots[int(fraction * count)] = value
        lines.append(f"#{measure:03d}{channel}:" + "".join(slots))
    return lines


def physical_events(chart, timing, profile=BASIC):
    report = validate_chart(chart, timing, timing_revision=chart.timing_revision)
    if report.has_errors:
        raise ValueError("Invalid chart: " + ",".join(i.code for i in report.issues if i.severity == "error"))
    occupied = {"ground": [], "air": []}
    output = []
    for event in chart.events:
        objects = profile.objects(event)
        for lane in event.lane_mask:
            for start, end in occupied[lane]:
                if max(start, event.q) <= min(end, event.end_q):
                    raise ValueError("Overlapping lane occupancy is unsupported")
            occupied[lane].append((event.q, event.end_q))
        for channel, value in objects:
            output.append((event.q, channel, value))
            if event.semantic == "hold":
                output.append((event.end_q, channel, value))
    return output


def headers(bpm, title, artist, music):
    for value in (title, artist):
        if type(value) is not str or not value.strip() or len(value) > 256 or any(ord(c) < 32 for c in value):
            raise ValueError("Invalid BMS metadata")
    if music != "music.ogg":
        raise ValueError("Profile requires music.ogg resource")
    return ["#PLAYER 3", "#GENRE scene_01", f"#TITLE {title}", f"#ARTIST {artist}",
            "#LEVELDESIGN MDAC2", f"#BPM {decimal(bpm)}", "#PLAYLEVEL 1", "#RANK 1", "#LNTYPE 1",
            "#WAV01 Small", "#WAV0E Gemini", "#WAV0F Hold", "#WAV10 music.ogg"]


def write_bms(chart: ChartRevision, timing: TimingMap, *, timing_revision: str,
              title="MDAC2", artist="MDAC2", music="music.ogg", profile=BASIC):
    if chart.timing_revision != timing_revision:
        raise ValueError("Timing revision mismatch")
    if timing.bar_origin != 0 or any((m.numerator, m.denominator) != (4, 4) for m in timing.meter_segments):
        raise ValueError("Fixed writer supports zero-origin 4/4 only")
    first, second = timing.anchors[:2]
    slope = (second.t - first.t) / (second.q - first.q)
    if first.t != first.q * slope or any(a.t != a.q * slope for a in timing.anchors):
        raise ValueError("Nonzero origin or tempo change requires timing profile")
    cells = [(0, Fraction(0), "01", "10")]
    for q, channel, value in physical_events(chart, timing, profile):
        measure = int(q // 4)
        cells.append((measure, q / 4 - measure, channel, value))
    return "\n".join(headers(60 / slope, title, artist, music) + data_lines(cells, profile)) + "\n"
