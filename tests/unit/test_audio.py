from fractions import Fraction

import pytest

from mdac2.domain.audio import AudioAsset, AudioError, AudioTransform, CanonicalAudio, TransformChain


def test_asset_and_canonical_roundtrip():
    a = AudioAsset("a" * 64, 100, "wav")
    assert AudioAsset.from_dict(a.to_dict()) == a
    c = CanonicalAudio("b" * 64, 48000, ("left", "right"), 96000,
                       "f32le-interleaved-v1", "ffmpeg:fixture", "c" * 64)
    assert c.duration == 2
    assert CanonicalAudio.from_dict(c.to_dict()) == c
    copied = c.to_dict()
    copied["channel_layout"].append("other")
    assert c.channel_layout == ("left", "right")


@pytest.mark.parametrize("count", [0, 1, 100, 48000])
def test_identity_and_end_boundary(count):
    transform = AudioTransform(count, 48000, 48000)
    assert transform.target_sample_count == count
    for sample in (0, Fraction(count, 2), count):
        assert transform.inverse(transform.forward(sample)) == sample


def test_crop_padding_gain_and_chain():
    a = AudioTransform(100, 10, 10, crop_head=10, crop_tail=20, pad_head=5, pad_tail=7, gain=.5)
    b = AudioTransform(82, 10, 10, crop_head=2, pad_head=3)
    chain = TransformChain((a, b))
    assert a.forward(10) == 5
    assert a.forward(80) == 75
    assert a.target_sample_count == 82
    for sample in range(10, 81):
        assert chain.inverse(chain.forward(sample)) == sample
    assert AudioTransform.from_dict(a.to_dict()) == a


@pytest.mark.parametrize("sample", [-1, 0, 9, 81, 101])
def test_cropped_source_rejected(sample):
    with pytest.raises(AudioError):
        AudioTransform(100, 10, 10, crop_head=10, crop_tail=20).forward(sample)


@pytest.mark.parametrize("sample", [-1, 0, 4, 76, 82, 83])
def test_padding_inverse_rejected(sample):
    with pytest.raises(AudioError):
        AudioTransform(100, 10, 10, crop_head=10, crop_tail=20, pad_head=5, pad_tail=7).inverse(sample)


def test_rational_resampling_and_explicit_rounding_error():
    transform = AudioTransform(441, 44100, 48000)
    assert transform.target_sample_count == 480
    for sample in range(442):
        exact = transform.forward(sample)
        assert transform.inverse(exact) == sample
        rounded, error = transform.quantized_forward(sample)
        assert Fraction(rounded) - exact == error
        assert abs(error) <= Fraction(1, 2)
    assert AudioTransform(3, 2, 1).quantized_forward(1) == (1, Fraction(1, 2))
    assert transform.inverse(transform.quantized_forward(1)[0]) != 1


def test_fractional_tail_is_not_original_audio():
    transform = AudioTransform(1, 3, 2)
    assert transform.target_sample_count == 1
    assert transform.inverse(Fraction(2, 3)) == 1
    with pytest.raises(AudioError):
        transform.inverse(1)


@pytest.mark.parametrize("kwargs", [
    {"source_rate": 0}, {"target_rate": -1}, {"source_sample_count": True},
    {"crop_head": -1}, {"crop_tail": 101}, {"pad_head": 1.5},
    {"gain": float("nan")}, {"gain": float("inf")}, {"gain": True},
    {"schema_version": 2},
])
def test_invalid_transform(kwargs):
    args = dict(source_sample_count=100, source_rate=10, target_rate=10)
    args.update(kwargs)
    with pytest.raises(AudioError):
        AudioTransform(**args)


@pytest.mark.parametrize("sample", [True, .5, float("nan"), "1"])
def test_exact_coordinate_types(sample):
    with pytest.raises(AudioError):
        AudioTransform(10, 10, 10).forward(sample)


def test_chain_refuses_mismatched_dimensions():
    with pytest.raises(AudioError):
        TransformChain((AudioTransform(10, 10, 10), AudioTransform(11, 10, 10)))
    with pytest.raises(AudioError):
        TransformChain((AudioTransform(10, 10, 10), AudioTransform(10, 11, 10)))


def test_empty_retained_audio_has_only_one_boundary():
    transform = AudioTransform(10, 10, 10, crop_head=10, pad_head=3)
    assert transform.forward(10) == 3
    assert transform.inverse(3) == 10
    assert transform.target_sample_count == 3


def test_unknown_or_missing_fields():
    data = AudioAsset("a" * 64, 0, "wav").to_dict()
    with pytest.raises(AudioError):
        AudioAsset.from_dict({**data, "path": "private"})
    del data["schema_version"]
    with pytest.raises(AudioError):
        AudioAsset.from_dict(data)
