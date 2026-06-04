"""Audio encoding helpers shared by the API layer."""

from __future__ import annotations

import io
import shutil
import subprocess

import numpy as np
import soundfile as sf

CONTENT_TYPES = {
    "wav": "audio/wav",
    "mp3": "audio/mpeg",
    "flac": "audio/flac",
    "ogg": "audio/ogg",
}


def encode(audio: np.ndarray, sample_rate: int, fmt: str = "wav") -> bytes:
    """Encode a float32 mono waveform to the requested container format."""
    fmt = fmt.lower()
    audio = np.clip(audio, -1.0, 1.0)

    if fmt == "wav":
        buf = io.BytesIO()
        sf.write(buf, audio, sample_rate, format="WAV", subtype="PCM_16")
        return buf.getvalue()

    if fmt in ("flac", "ogg"):
        buf = io.BytesIO()
        sf.write(buf, audio, sample_rate, format=fmt.upper())
        return buf.getvalue()

    if fmt == "mp3":
        return _encode_mp3(audio, sample_rate)

    raise ValueError(f"Unsupported audio format: {fmt!r}")


def _encode_mp3(audio: np.ndarray, sample_rate: int) -> bytes:
    """Encode to MP3 via ffmpeg (required on PATH for mp3 output)."""
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg is required for mp3 output but was not found on PATH.")
    pcm16 = (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2").tobytes()
    proc = subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-f", "s16le", "-ar", str(sample_rate), "-ac", "1", "-i", "pipe:0",
            "-f", "mp3", "-b:a", "128k", "pipe:1",
        ],
        input=pcm16,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    return proc.stdout
