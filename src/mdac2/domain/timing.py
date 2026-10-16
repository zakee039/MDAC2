"""Schema 1 timing: immutable, bounded, rational piecewise-linear mapping."""

from bisect import bisect_right
from dataclasses import dataclass
from fractions import Fraction
import math


class TimingError(ValueError):
    """Invalid timing contract or query outside its supported audio range."""


def rational(value) -> Fraction:
    if type(value) in (int, Fraction):
        return Fraction(value)
    if type(value) is float and math.isfinite(value):
        return Fraction(str(value))
    raise TimingError("Expected a finite numeric coordinate (bool is not numeric)")


def _decode_fraction(value):
    if type(value) is not str:
        raise TimingError("Serialized fractions must be strings")
    try:
        result = Fraction(value)
    except (ValueError, ZeroDivisionError) as exc:
        raise TimingError("Invalid fraction") from exc
    if str(result) != value:
        raise TimingError("Noncanonical fraction spelling")
    return result


def _fields(value, names):
    if type(value) is not dict or set(value) != set(names):
        raise TimingError("Missing/unknown timing fields; migration required")


@dataclass(frozen=True, slots=True)
class Anchor:
    q: Fraction
    t: Fraction

    def __post_init__(self):
        object.__setattr__(self, "q", rational(self.q))
        object.__setattr__(self, "t", rational(self.t))
        if self.t < 0:
            raise TimingError("Audio seconds cannot be negative")


@dataclass(frozen=True, slots=True)
class MeterSegment:
    start_q: Fraction
    numerator: int
    denominator: int

    def __post_init__(self):
        object.__setattr__(self, "start_q", rational(self.start_q))
        if type(self.numerator) is not int or self.numerator <= 0:
            raise TimingError("Meter numerator must be positive integer")
        d = self.denominator
        if type(d) is not int or d <= 0 or d & (d - 1):
            raise TimingError("Meter denominator must be a positive power of two")

    @property
    def bar_quarters(self):
        return Fraction(self.numerator * 4, self.denominator)


def quarter_bpm(pulse_bpm, pulse_unit):
    bpm, unit = rational(pulse_bpm), rational(pulse_unit)
    if bpm <= 0 or unit <= 0:
        raise TimingError("Pulse tempo and whole-note unit must be positive")
    return bpm * unit * 4


@dataclass(frozen=True, slots=True)
class TimingMap:
    anchors: tuple[Anchor, ...]
    audio_duration: Fraction
    pulse_unit: Fraction = Fraction(1, 4)
    meter_segments: tuple[MeterSegment, ...] = (MeterSegment(0, 4, 4),)
    bar_origin: Fraction = Fraction(0)
    extrapolation_seconds: Fraction = Fraction(0)
    schema_version: int = 1

    def __post_init__(self):
        for name in ("audio_duration", "pulse_unit", "bar_origin", "extrapolation_seconds"):
            object.__setattr__(self, name, rational(getattr(self, name)))
        if self.audio_duration < 0 or self.pulse_unit <= 0 or self.extrapolation_seconds < 0:
            raise TimingError("Invalid duration/pulse/extrapolation limit")
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise TimingError("Unsupported timing schema")
        if type(self.anchors) is not tuple or len(self.anchors) < 2:
            raise TimingError("At least two immutable anchors required")
        if any(type(anchor) is not Anchor for anchor in self.anchors):
            raise TimingError("Invalid anchor type")
        for left, right in zip(self.anchors, self.anchors[1:]):
            if right.q <= left.q or right.t <= left.t:
                raise TimingError("Anchors must be strictly increasing in q and t")
        if self.anchors[-1].t > self.audio_duration:
            raise TimingError("Anchor exceeds audio duration")
        if type(self.meter_segments) is not tuple or not self.meter_segments:
            raise TimingError("Non-empty immutable meter segments required")
        if any(type(meter) is not MeterSegment for meter in self.meter_segments):
            raise TimingError("Invalid meter segment")
        if self.meter_segments[0].start_q != self.bar_origin:
            raise TimingError("First meter starts at bar_origin; pickup uses this meter backwards")
        for left, right in zip(self.meter_segments, self.meter_segments[1:]):
            delta = right.start_q - left.start_q
            if delta <= 0 or (delta / left.bar_quarters).denominator != 1:
                raise TimingError("Meter change must be on an increasing bar boundary")

    @property
    def time_bounds(self):
        return (max(Fraction(0), self.anchors[0].t - self.extrapolation_seconds),
                min(self.audio_duration, self.anchors[-1].t + self.extrapolation_seconds))

    def _segment(self, value, axis):
        coordinates = [getattr(anchor, axis) for anchor in self.anchors]
        index = max(0, min(len(self.anchors) - 2, bisect_right(coordinates, value) - 1))
        return self.anchors[index], self.anchors[index + 1]

    def _check_time(self, value):
        lo, hi = self.time_bounds
        if not lo <= value <= hi:
            raise TimingError("Query exceeds audio duration or extrapolation policy")

    def t(self, q) -> Fraction:
        q = rational(q)
        left, right = self._segment(q, "q")
        value = left.t + (q - left.q) * (right.t - left.t) / (right.q - left.q)
        self._check_time(value)
        return value

    def q(self, t) -> Fraction:
        t = rational(t)
        self._check_time(t)
        left, right = self._segment(t, "t")
        return left.q + (t - left.t) * (right.q - left.q) / (right.t - left.t)

    def local_bpm(self, q) -> Fraction:
        q = rational(q)
        self.t(q)
        left, right = self._segment(q, "q")
        return 60 * (right.q - left.q) / (right.t - left.t)

    def meter_at(self, q) -> MeterSegment:
        q = rational(q)
        index = max(0, bisect_right([m.start_q for m in self.meter_segments], q) - 1)
        return self.meter_segments[index]

    def to_dict(self):
        return {
            "schema_version": self.schema_version,
            "anchors": [{"q": str(a.q), "t": str(a.t)} for a in self.anchors],
            "audio_duration": str(self.audio_duration), "pulse_unit": str(self.pulse_unit),
            "bar_origin": str(self.bar_origin),
            "extrapolation_seconds": str(self.extrapolation_seconds),
            "meter_segments": [{"start_q": str(m.start_q), "numerator": m.numerator,
                                "denominator": m.denominator} for m in self.meter_segments],
        }

    @classmethod
    def from_dict(cls, data):
        _fields(data, {"schema_version", "anchors", "audio_duration", "pulse_unit",
                       "bar_origin", "extrapolation_seconds", "meter_segments"})
        if type(data["anchors"]) is not list or type(data["meter_segments"]) is not list:
            raise TimingError("Serialized anchors/meters must be lists")
        anchors, meters = [], []
        for item in data["anchors"]:
            _fields(item, {"q", "t"})
            anchors.append(Anchor(_decode_fraction(item["q"]), _decode_fraction(item["t"])))
        for item in data["meter_segments"]:
            _fields(item, {"start_q", "numerator", "denominator"})
            meters.append(MeterSegment(_decode_fraction(item["start_q"]), item["numerator"], item["denominator"]))
        return cls(tuple(anchors), _decode_fraction(data["audio_duration"]),
                   _decode_fraction(data["pulse_unit"]), tuple(meters),
                   _decode_fraction(data["bar_origin"]), _decode_fraction(data["extrapolation_seconds"]),
                   data["schema_version"])
