"""Lightweight unit tests that do not require GPU or model downloads."""

from __future__ import annotations

import io

import numpy as np
import pytest
import soundfile as sf

from app.audio import encode
from app.config import Settings
from app.schemas import SpeechRequest
from app.tts.piper_engine import PiperEngine
from app.tts.registry import available_engines, create_engine
from app.tts.xtts_engine import XTTSEngine


def test_available_engines():
    assert set(available_engines()) == {"xtts", "piper"}


def test_registry_selects_engine():
    assert isinstance(create_engine(Settings(tts_engine="xtts")), XTTSEngine)
    assert isinstance(create_engine(Settings(tts_engine="piper")), PiperEngine)


def test_registry_rejects_unknown():
    with pytest.raises(ValueError):
        create_engine(Settings(tts_engine="nope"))


def test_engine_cloning_flags():
    assert XTTSEngine(Settings()).supports_cloning is True
    assert PiperEngine(Settings()).supports_cloning is False


def test_wav_encode_roundtrip():
    sr = 22050
    audio = (0.3 * np.sin(np.linspace(0, 200, sr))).astype(np.float32)
    data = encode(audio, sr, "wav")
    back, sr_back = sf.read(io.BytesIO(data))
    assert sr_back == sr
    assert len(back) == len(audio)
    assert np.max(np.abs(back)) < 1.0


def test_encode_rejects_unknown_format():
    with pytest.raises(ValueError):
        encode(np.zeros(10, dtype=np.float32), 16000, "xyz")


def test_speech_request_defaults():
    req = SpeechRequest(input="привет мир")
    assert req.response_format == "wav"
    assert req.speed == 1.0
