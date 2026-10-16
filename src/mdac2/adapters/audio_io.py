"""Bounded FFmpeg decoding to PCM16 WAV; no automatic tool/model download.

Input is a byte snapshot passed via pipe-only protocols, not a URL or filename
interpreted by FFmpeg. Native decoded samples define the transform source.
"""
from dataclasses import dataclass
import hashlib
import io
import math
from pathlib import Path
import struct
import subprocess
import wave

from mdac2.domain.artifact import ArtifactEnvelope, content_hash, strict_json
from mdac2.domain.audio import AudioAsset, AudioError, AudioTransform, CanonicalAudio


class DecodeError(AudioError):
    pass


@dataclass(frozen=True)
class AudioLimits:
    max_bytes: int = 64 * 1024 * 1024
    max_pcm_bytes: int = 128 * 1024 * 1024
    max_seconds: int = 600
    timeout_seconds: int = 30

    def __post_init__(self):
        if any(type(value) is not int or value <= 0 for value in (
                self.max_bytes, self.max_pcm_bytes, self.max_seconds, self.timeout_seconds)):
            raise DecodeError("Positive integer limits required")


class AudioDecoder:
    def __init__(self, ffmpeg, ffprobe, *, limits=AudioLimits()):
        self.ffmpeg, self.ffprobe = Path(ffmpeg).absolute(), Path(ffprobe).absolute()
        self.limits = limits
        if not self.ffmpeg.is_file() or not self.ffprobe.is_file():
            raise DecodeError("FFmpeg/ffprobe capability missing; configure local tools explicitly")
        self.version = self._run(self.ffmpeg, ["-version"]).decode("utf-8").splitlines()[0]
        self.producer = {"name": "mdac2.ffmpeg-audio", "version": "1", "model_hash": None,
            "dependencies": {"ffmpeg": self.version, "ffmpeg_sha256": self._digest(self.ffmpeg),
                             "ffprobe_sha256": self._digest(self.ffprobe)}}

    @staticmethod
    def _digest(path):
        with path.open("rb") as handle:
            return hashlib.file_digest(handle, "sha256").hexdigest()

    def _run(self, program, arguments, data=None):
        try:
            result = subprocess.run([str(program), *arguments], input=data, capture_output=True,
                timeout=self.limits.timeout_seconds, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise DecodeError("Decoder unavailable or timed out") from exc
        if result.returncode:
            raise DecodeError("Decoder rejected input")  # no paths/tags/stderr exposure
        return result.stdout

    def decode(self, data, *, target_rate=48000, crop_head=0, crop_tail=0, pad_head=0, pad_tail=0, gain=1.0):
        if type(data) is not bytes or not 0 < len(data) <= self.limits.max_bytes:
            raise DecodeError("Empty audio or input byte limit exceeded")
        if type(target_rate) is not int or not 8000 <= target_rate <= 192000:
            raise DecodeError("Unsupported target sample rate")
        probe_args = ["-v", "fatal", "-max_alloc", "67108864", "-protocol_whitelist", "pipe",
            "-format_whitelist", "wav,mp3,flac,ogg,aac,mov", "-probesize", str(min(len(data), 5000000)),
            "-analyzeduration", "5000000", "-select_streams", "a", "-show_entries",
            "stream=sample_rate,channels,channel_layout:format=format_name,duration", "-of", "json", "pipe:0"]
        probe = strict_json(self._run(self.ffprobe, probe_args, data))
        streams = probe.get("streams", [])
        if len(streams) != 1:
            raise DecodeError("Exactly one audio stream required")
        stream = streams[0]
        try:
            rate, channels = int(stream["sample_rate"]), int(stream["channels"])
            duration = float(probe.get("format", {}).get("duration", 0))
        except (ValueError, TypeError, KeyError) as exc:
            raise DecodeError("Invalid stream metadata") from exc
        if not 8000 <= rate <= 192000 or channels not in (1, 2):
            raise DecodeError("Only mono/stereo audio at 8–192 kHz supported")
        if not math.isfinite(duration) or duration < 0 or duration > self.limits.max_seconds:
            raise DecodeError("Audio duration limit exceeded")
        raw = self._run(self.ffmpeg, ["-v", "fatal", "-nostdin", "-xerror", "-err_detect", "explode", "-max_alloc", "67108864",
            "-threads", "1", "-protocol_whitelist", "pipe", "-format_whitelist", "wav,mp3,flac,ogg,aac,mov",
            "-i", "pipe:0", "-map", "0:a:0", "-vn", "-sn", "-dn", "-threads", "1",
            "-t", str(self.limits.max_seconds + 1 / rate), "-fs", str(self.limits.max_pcm_bytes + 1),
            "-c:a", "pcm_s16le", "-f", "s16le", "pipe:1"], data)
        frame_size = channels * 2
        if not raw or len(raw) % frame_size or len(raw) > self.limits.max_pcm_bytes:
            raise DecodeError("Invalid or oversized decoded PCM")
        source_count = len(raw) // frame_size
        if source_count > self.limits.max_seconds * rate:
            raise DecodeError("Decoded duration limit exceeded")
        transform = AudioTransform(source_count, rate, target_rate, crop_head, crop_tail, pad_head, pad_tail, gain)
        if transform.retained_count == 0 or transform.target_sample_count > self.limits.max_seconds * target_rate:
            raise DecodeError("Empty retained audio or normalized duration exceeds limit")
        if transform.target_sample_count * frame_size > self.limits.max_pcm_bytes:
            raise DecodeError("Normalized PCM byte limit exceeded")
        pcm = raw[crop_head * frame_size:(source_count - crop_tail) * frame_size]
        if target_rate != rate:
            pcm = self._run(self.ffmpeg, ["-v", "fatal", "-nostdin", "-threads", "1", "-filter_threads", "1",
                "-protocol_whitelist", "pipe", "-f", "s16le", "-ar", str(rate), "-ac", str(channels),
                "-i", "pipe:0", "-ar", str(target_rate), "-c:a", "pcm_s16le", "-f", "s16le", "pipe:1"], pcm)
        expected = math.ceil(transform.retained_count * target_rate / rate) * frame_size
        if len(pcm) != expected:
            raise DecodeError("Resampler length does not match the declared ceil profile")
        if gain != 1:
            values = []
            for (sample,) in struct.iter_unpack("<h", pcm):
                value = round(sample * gain)
                if not -32768 <= value <= 32767:
                    raise DecodeError("Gain would clip; explicit lower gain required")
                values.append(value)
            pcm = struct.pack("<" + "h" * len(values), *values)
        pcm = bytes(pad_head * frame_size) + pcm + bytes(pad_tail * frame_size)
        options = {"profile": "pcm16-preserve-channels-ceil-v1", "target_rate": target_rate,
                   "crop_head": crop_head, "crop_tail": crop_tail, "pad_head": pad_head, "pad_tail": pad_tail, "gain": gain}
        asset = AudioAsset(hashlib.sha256(data).hexdigest(), len(data), probe["format"]["format_name"])
        canonical = CanonicalAudio(hashlib.sha256(pcm).hexdigest(), target_rate,
            ("mono",) if channels == 1 else ("left", "right"), len(pcm) // frame_size,
            "s16le", self.version, content_hash(options))
        output = io.BytesIO()
        with wave.open(output, "wb") as writer:
            writer.setparams((channels, 2, target_rate, 0, "NONE", "not compressed"))
            writer.writeframes(pcm)
        files = {"original.bin": data, "audio.wav": output.getvalue()}
        envelope = ArtifactEnvelope.create(artifact_type="audio", producer=self.producer, options=options,
            payload={"asset": asset.to_dict(), "canonical": canonical.to_dict(), "transform": transform.to_dict(),
                     "files": [{"name": name, "sha256": hashlib.sha256(value).hexdigest(), "byte_count": len(value)} for name, value in files.items()]})
        return envelope, files

    def import_audio(self, data, store, **options):
        envelope, files = self.decode(data, **options)
        store.publish(envelope, files)
        return envelope
