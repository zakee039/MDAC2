"""Strict basic-profile reader; unsupported BMS is never silently discarded."""
from dataclasses import dataclass
from fractions import Fraction
import re
from uuid import uuid4

from mdac2.domain.chart import ChartEvent, ChartRevision
from mdac2.schemas.game_profiles import BASIC


@dataclass(frozen=True)
class ReadResult:
    chart: ChartRevision
    bpm: Fraction
    logical_count: int
    physical_count: int
    marker_count: int


def parse(text, profile=BASIC, *, extended=False):
    if type(text) is not str or len(text) > 8 * 1024 * 1024:
        raise ValueError("Invalid/oversized BMS")
    headers, cells, measures = {}, [], set()
    for line in text.lstrip("\ufeff").splitlines():
        line = line.strip()
        if not line or line.startswith("*"):
            continue
        match = re.fullmatch(r"#(\d{3})(\d{2}):(.+)", line)
        if match:
            measure, channel, data = int(match[1]), match[2], match[3]
            if measure > profile.max_measure or (measure, channel) in measures:
                raise ValueError("Duplicate/out-of-range measure channel")
            measures.add((measure, channel))
            if channel == "02" and extended:
                if not re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", data) or Fraction(data) <= 0:
                    raise ValueError("Invalid measure ratio")
                cells.append((Fraction(measure * 4), channel, data))
                continue
            if channel not in (("01", "13", "14", "53", "54", "08") if extended else ("01", "13", "14", "53", "54")):
                raise ValueError("Unsupported BMS channel")
            if len(data) % 2 or len(data) // 2 > profile.max_cells or re.fullmatch("[0-9A-Z]+", data) is None:
                raise ValueError("Invalid BMS cells")
            for index in range(len(data) // 2):
                value = data[index * 2:index * 2 + 2]
                if value != "00":
                    cells.append((Fraction(measure * 4) + Fraction(index * 8, len(data)), channel, value))
        else:
            match = re.fullmatch(r"#([A-Z0-9]+)\s+(.+)", line)
            if not match or match[1] in headers:
                raise ValueError("Malformed/duplicate BMS header")
            if match[1] not in {"PLAYER", "GENRE", "TITLE", "ARTIST", "LEVELDESIGN", "BPM", "PLAYLEVEL", "RANK", "LNTYPE", "WAV01", "WAV0E", "WAV0F", "WAV10"} and not (extended and re.fullmatch(r"BPM[0-9A-Z]{2}", match[1])):
                raise ValueError("Unsupported BMS header")
            headers[match[1]] = match[2]
    required = {"PLAYER": "3", "GENRE": "scene_01", "LNTYPE": "1", "WAV01": "Small", "WAV0E": "Gemini", "WAV0F": "Hold", "WAV10": "music.ogg"}
    if any(headers.get(key) != value for key, value in required.items()):
        raise ValueError("Missing/incompatible profile or resource headers")
    try:
        if not re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", headers.get("BPM", "")):
            raise ValueError("Invalid BPM")
        bpm = Fraction(headers["BPM"])
        if bpm <= 0:
            raise ValueError("Invalid BPM")
    except (ValueError, KeyError, ZeroDivisionError) as exc:
        raise ValueError("Invalid BPM") from exc
    if [(q, v) for q, c, v in cells if c == "01"] != [(Fraction(0), "10")]:
        raise ValueError("Exactly one music start at zero required")
    result = bpm, [cell for cell in cells if cell[1] != "01"]
    return (*result, headers) if extended else result


def read_bms(text, *, timing_revision, profile=BASIC):
    bpm, cells = parse(text, profile)
    return decode_events(bpm, cells, timing_revision)


def decode_events(bpm, cells, timing_revision):
    events, geminis, holds = [], {}, {"53": [], "54": []}
    physical = 0
    for q, channel, value in sorted(cells):
        lane = "air" if channel in ("13", "53") else "ground"
        if channel in holds:
            if value != "0F":
                raise ValueError("Unsupported object on Hold channel")
            holds[channel].append(q)
        elif value == "01":
            events.append(ChartEvent(str(uuid4()), q, (lane,)))
            physical += 1
        elif value == "0E":
            geminis.setdefault(q, []).append(lane)
            physical += 1
        else:
            raise ValueError("Unsupported object")
    for q, lanes in geminis.items():
        if sorted(lanes) != ["air", "ground"]:
            raise ValueError("Unpaired Gemini")
        events.append(ChartEvent(str(uuid4()), q, ("ground", "air"), "gemini"))
    for channel, marks in holds.items():
        if len(marks) % 2:
            raise ValueError("Unclosed Hold")
        for start, end in zip(marks[::2], marks[1::2]):
            if end <= start:
                raise ValueError("Invalid Hold length")
            events.append(ChartEvent(str(uuid4()), start, ("air" if channel == "53" else "ground",), "hold", duration_q=end-start))
            physical += 1
    # BMS cannot recover original stable IDs/provenance. Workflow retains source IR.
    events.sort(key=lambda e: (e.q, e.semantic, e.lane_mask))
    if not events:
        raise ValueError("Empty chart")
    for index, event in enumerate(events):
        for other in events[:index]:
            if set(event.lane_mask) & set(other.lane_mask) and max(event.q, other.q) <= min(event.end_q, other.end_q):
                raise ValueError("Unsupported lane overlap")
    chart = ChartRevision(str(uuid4()), timing_revision, tuple(events))
    return ReadResult(chart, bpm, len(events), physical, len(cells))


def semantics(chart):
    return sorted((e.q, e.duration_q, tuple(sorted(e.lane_mask)), e.semantic, e.accent) for e in chart.events)
