import hashlib
import io
from pathlib import Path
import struct
import subprocess
import wave

import pytest

from mdac2.adapters.audio_io import AudioDecoder, AudioLimits, DecodeError
from mdac2.domain.audio import AudioTransform
from mdac2.infrastructure.artifact_store import ArtifactStore


BIN = Path(__file__).parents[2] / "tools/.deps/ffmpeg/extracted/ffmpeg-9.0.1-essentials_build/bin"


@pytest.fixture(scope="module")
def decoder():
    # Intentionally fail if tools are absent: this suite claims real decoder evidence.
    return AudioDecoder(BIN / "ffmpeg.exe", BIN / "ffprobe.exe")


def wav(rate=48000, frames=4800, channels=2):
    data = io.BytesIO()
    samples = [(10000 if i == 100 else 0, -5000 if i == 200 else 0) for i in range(frames)]
    pcm = b"".join(struct.pack("<" + "h" * channels, *pair[:channels]) for pair in samples)
    with wave.open(data, "wb") as output:
        output.setparams((channels, 2, rate, 0, "NONE", "none"))
        output.writeframes(pcm)
    return data.getvalue(), pcm


def test_stereo_identity_and_atomic_roundtrip(decoder, tmp_path):
    data, pcm = wav()
    store = ArtifactStore(tmp_path)
    envelope = decoder.import_audio(data, store)
    payload = envelope.payload
    assert payload["asset"]["original_sha256"] == hashlib.sha256(data).hexdigest()
    assert payload["canonical"]["pcm_sha256"] == hashlib.sha256(pcm).hexdigest()
    assert payload["canonical"]["channel_layout"] == ["left", "right"]
    assert store.get("audio", envelope.artifact_hash) == envelope
    assert decoder.import_audio(data, store) == envelope


def test_crop_pad_exact_origin(decoder):
    data, original = wav()
    envelope, files = decoder.decode(data, crop_head=80, crop_tail=20, pad_head=30, pad_tail=10)
    transform = AudioTransform.from_dict(envelope.payload["transform"])
    assert transform.forward(100) == 50
    with wave.open(io.BytesIO(files["audio.wav"]), "rb") as reader:
        pcm = reader.readframes(reader.getnframes())
    assert struct.unpack_from("<h", pcm, 50 * 4)[0] == 10000
    assert pcm == bytes(30 * 4) + original[80 * 4:-20 * 4] + bytes(10 * 4)


def test_resample_mono_ceil_profile(decoder):
    data, _ = wav(rate=44100, frames=4410, channels=1)
    envelope, _ = decoder.decode(data)
    assert envelope.payload["canonical"]["sample_count"] == 4800
    assert envelope.payload["canonical"]["channel_layout"] == ["mono"]


@pytest.mark.parametrize("data", [b"not audio", b"", b"#EXTM3U\nhttps://example.com/private\n"])
def test_bad_files_and_playlist_rejected(decoder, tmp_path, data):
    store = ArtifactStore(tmp_path)
    with pytest.raises(DecodeError):
        decoder.import_audio(data, store)
    assert list(store.artifacts.iterdir()) == []


def test_byte_duration_gain_and_truncation_limits(decoder, tmp_path):
    short = AudioDecoder(decoder.ffmpeg, decoder.ffprobe, limits=AudioLimits(max_bytes=20))
    with pytest.raises(DecodeError):
        short.decode(wav()[0])
    duration = AudioDecoder(decoder.ffmpeg, decoder.ffprobe, limits=AudioLimits(max_seconds=1))
    with pytest.raises(DecodeError):
        duration.decode(wav(frames=96000)[0])
    with pytest.raises(DecodeError):
        decoder.decode(wav()[0], gain=10)
    with pytest.raises(DecodeError):
        decoder.decode(wav()[0][:-51])


def test_missing_tool_and_timeout(decoder, tmp_path, monkeypatch):
    with pytest.raises(DecodeError):
        AudioDecoder(tmp_path / "missing.exe", decoder.ffprobe)
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired("decoder", 1)
    monkeypatch.setattr(subprocess, "run", timeout)
    with pytest.raises(DecodeError, match="timed out"):
        decoder.decode(wav()[0])


@pytest.mark.parametrize("codec,container", [("flac", "flac"), ("libmp3lame", "mp3")])
def test_real_compressed_audio(decoder, codec, container):
    data = decoder._run(decoder.ffmpeg, ["-v", "fatal", "-i", "pipe:0", "-c:a", codec, "-f", container, "pipe:1"], wav()[0])
    envelope, _ = decoder.decode(data)
    assert envelope.payload["canonical"]["sample_count"] > 0
    assert envelope.payload["canonical"]["channel_layout"] == ["left", "right"]
