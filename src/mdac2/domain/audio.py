"""Audio schema 1: sample-boundary coordinates, exact rational transforms.

Crop counts use source samples; padding counts use target samples. Retained
intervals include both boundaries for duration mapping; sample indices themselves
remain half-open. Padding has no original sample. Resampling uses an exact rate
ratio and ceil output length, so a final fractional tail has no source inverse.
Integer quantization is explicit half-up with <= 1/2 target-sample error.
"""

from dataclasses import asdict, dataclass
from fractions import Fraction
import math
import re


class AudioError(ValueError):
    """Invalid audio identity, transform, or coordinate."""


def _integer(value, minimum=0):
    if type(value) is not int or value < minimum:
        raise AudioError(f"Expected integer >= {minimum}")


def _text(value):
    if type(value) is not str or not value.strip():
        raise AudioError("Expected non-empty string")


def _hash(value):
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise AudioError("Expected SHA-256 hex")


def _coordinate(value):
    if type(value) not in (int, Fraction):
        raise AudioError("Coordinates must be int or Fraction, not floats")
    return Fraction(value)


def _decode(cls, data):
    from dataclasses import fields

    if type(data) is not dict or set(data) != {field.name for field in fields(cls)}:
        raise AudioError("Missing or unknown audio contract fields")
    return cls(**data)


@dataclass(frozen=True, slots=True)
class AudioAsset:
    original_sha256: str
    byte_count: int
    container: str
    schema_version: int = 1

    def __post_init__(self):
        _hash(self.original_sha256)
        _integer(self.byte_count)
        _text(self.container)
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise AudioError("Unsupported audio schema")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        return _decode(cls, data)


@dataclass(frozen=True, slots=True)
class CanonicalAudio:
    pcm_sha256: str
    sample_rate: int
    channel_layout: tuple[str, ...]
    sample_count: int
    pcm_format: str
    decoder_version: str
    normalization_options_hash: str
    schema_version: int = 1

    def __post_init__(self):
        _hash(self.pcm_sha256)
        _hash(self.normalization_options_hash)
        _integer(self.sample_rate, 1)
        _integer(self.sample_count)
        if type(self.channel_layout) is not tuple or not self.channel_layout:
            raise AudioError("channel_layout must be non-empty immutable tuple")
        for channel in self.channel_layout:
            _text(channel)
        if len(set(self.channel_layout)) != len(self.channel_layout):
            raise AudioError("Duplicate channels")
        _text(self.pcm_format)
        _text(self.decoder_version)
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise AudioError("Unsupported canonical schema")

    @property
    def duration(self):
        return Fraction(self.sample_count, self.sample_rate)

    def to_dict(self):
        result = asdict(self)
        result["channel_layout"] = list(self.channel_layout)
        return result

    @classmethod
    def from_dict(cls, data):
        if type(data) is not dict or type(data.get("channel_layout")) is not list:
            raise AudioError("Serialized channel_layout must be a list")
        return _decode(cls, {**data, "channel_layout": tuple(data["channel_layout"])})


@dataclass(frozen=True, slots=True)
class AudioTransform:
    source_sample_count: int
    source_rate: int
    target_rate: int
    crop_head: int = 0
    crop_tail: int = 0
    pad_head: int = 0
    pad_tail: int = 0
    gain: float = 1.0
    schema_version: int = 1

    def __post_init__(self):
        for value in (self.source_sample_count, self.crop_head, self.crop_tail,
                      self.pad_head, self.pad_tail):
            _integer(value)
        _integer(self.source_rate, 1)
        _integer(self.target_rate, 1)
        if self.crop_head + self.crop_tail > self.source_sample_count:
            raise AudioError("Crop exceeds source")
        if type(self.gain) not in (int, float) or not math.isfinite(self.gain) or self.gain < 0:
            raise AudioError("gain must be finite and nonnegative")
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise AudioError("Unsupported transform schema")

    @property
    def retained_count(self):
        return self.source_sample_count - self.crop_head - self.crop_tail

    @property
    def exact_target_end(self):
        return self.pad_head + Fraction(self.retained_count * self.target_rate, self.source_rate)

    @property
    def target_sample_count(self):
        return math.ceil(self.exact_target_end) + self.pad_tail

    def forward(self, sample):
        sample = _coordinate(sample)
        if not self.crop_head <= sample <= self.source_sample_count - self.crop_tail:
            raise AudioError("Source coordinate outside retained interval")
        return self.pad_head + (sample - self.crop_head) * Fraction(self.target_rate, self.source_rate)

    def inverse(self, sample):
        sample = _coordinate(sample)
        if not self.pad_head <= sample <= self.exact_target_end:
            raise AudioError("Padding/rounded tail/out-of-range has no source inverse")
        return self.crop_head + (sample - self.pad_head) * Fraction(self.source_rate, self.target_rate)

    def quantized_forward(self, sample):
        exact = self.forward(sample)
        rounded = math.floor(exact + Fraction(1, 2))
        return rounded, Fraction(rounded) - exact

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        return _decode(cls, data)


@dataclass(frozen=True, slots=True)
class TransformChain:
    steps: tuple[AudioTransform, ...]

    def __post_init__(self):
        if type(self.steps) is not tuple or not self.steps:
            raise AudioError("Non-empty immutable transform chain required")
        if any(type(step) is not AudioTransform for step in self.steps):
            raise AudioError("Invalid chain element")
        for previous, current in zip(self.steps, self.steps[1:]):
            if (previous.target_rate != current.source_rate or
                    previous.target_sample_count != current.source_sample_count):
                raise AudioError("Adjacent transform sample counts/rates do not match")

    def forward(self, sample):
        for step in self.steps:
            sample = step.forward(sample)
        return sample

    def inverse(self, sample):
        for step in reversed(self.steps):
            sample = step.inverse(sample)
        return sample
