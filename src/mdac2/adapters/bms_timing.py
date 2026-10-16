"""Conservative timing profile for the pinned CustomAlbums loader.

Fixed tempo supports one meter and explicit audio pre-padding. Variable tempo
supports zero-origin 4/4 only. Combined variable-meter/tempo is rejected because
the pinned loader's time conversion cannot establish the required equivalence.
"""
from dataclasses import dataclass
from fractions import Fraction as F
from math import ceil

from mdac2.adapters.bms_writer import headers, data_lines, decimal, physical_events
from mdac2.adapters.bms_reader import parse, decode_events
from mdac2.schemas.game_profiles import BASIC


@dataclass(frozen=True)
class TimingExport:
    text: str
    timing_revision: str
    q_offset: F
    audio_padding_seconds: F
    profile_id: str


def _base36(number):
    alphabet = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    if not 1 <= number < 36 * 36:
        raise ValueError("Too many BPM definitions")
    return alphabet[number // 36] + alphabet[number % 36]


def write_timed_bms(chart, timing, *, timing_revision, title="MDAC2", artist="MDAC2", profile=BASIC):
    if chart.timing_revision != timing_revision:
        raise ValueError("Timing revision mismatch")
    if len(timing.meter_segments) != 1 or timing.bar_origin != 0:
        raise ValueError("Meter changes/nonzero bar origin are not verified in this profile")
    slopes = [(b.t-a.t)/(b.q-a.q) for a,b in zip(timing.anchors, timing.anchors[1:])]
    fixed = len(set(slopes)) == 1
    meter = timing.meter_segments[0]
    quarters = meter.bar_quarters
    if not fixed and (quarters != 4 or timing.anchors[0].q != 0 or timing.anchors[0].t != 0):
        raise ValueError("Variable BPM requires zero-origin 4/4; combined timing unsupported")
    offset, padding = F(0), F(0)
    if fixed:
        first = timing.anchors[0]
        needed = max(F(0), -first.q, first.t/slopes[0]-first.q)
        offset = ceil(needed / quarters) * quarters
        padding = (first.q + offset) * slopes[0] - first.t
        if padding < 0 or padding > 60:
            raise ValueError("Audio padding outside profile limit")
    cells = [(0, F(0), "01", "10")]
    for q, channel, value in physical_events(chart, timing, profile):
        position = (q + offset) / quarters
        measure = int(position // 1)
        cells.append((measure, position-measure, channel, value))
    output = headers(60/slopes[0], title, artist, "music.ogg")
    # All time definitions precede all notes: the target loader converts on read.
    if not fixed:
        changes = []
        for index, slope in enumerate(slopes[1:], 1):
            if slope == slopes[index-1]:
                continue
            key = _base36(index)
            output.append(f"#BPM{key} {decimal(60/slope)}")
            position = timing.anchors[index].q / 4
            measure = int(position // 1)
            changes.append((measure, position-measure, "08", key))
        output.extend(data_lines(changes, profile))
    if quarters != 4:
        for measure in range(max(cell[0] for cell in cells)+1):
            output.append(f"#{measure:03d}02:{decimal(quarters/4)}")
    output.extend(data_lines(cells, profile))
    return TimingExport("\n".join(output)+"\n", timing_revision, offset, padding, profile.profile_id)


def read_timed_bms(text, *, timing_revision, q_offset=F(0), profile=BASIC):
    bpm, cells, head = parse(text, profile, extended=True)
    ratios = {int(q/4): F(value) for q, channel, value in cells if channel == "02"}
    changes = {F(0): bpm}
    for q, channel, value in cells:
        if channel == "08":
            key = "BPM"+value
            try:
                speed = F(head[key])
            except (KeyError, ValueError, ZeroDivisionError) as exc:
                raise ValueError("Missing/invalid BPM reference") from exc
            if speed <= 0 or q in changes:
                raise ValueError("Invalid/duplicate BPM change")
            changes[q] = speed
    if ratios and len(changes)>1:
        raise ValueError("Combined BPM/meter conversion unsupported")
    def musical(raw):
        measure = int(raw // 4)
        return sum((4*ratios.get(i,F(1)) for i in range(measure)),F(0)) + (raw-measure*4)*ratios.get(measure,F(1))
    notes = [(musical(q)-q_offset, c, v) for q,c,v in cells if c not in ("02", "08")]
    result = decode_events(bpm, notes, timing_revision)
    points = sorted(changes.items())
    def seconds(q):
        shifted = q + q_offset
        elapsed = F(0)
        for index, (start, speed) in enumerate(points):
            if shifted <= start:
                break
            end = points[index+1][0] if index+1<len(points) else shifted
            elapsed += (min(end, shifted)-start)*60/speed
        return elapsed
    return result, seconds
