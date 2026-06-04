"""Engine registry — selects the active TTS engine from configuration."""

from __future__ import annotations

from ..config import Settings
from .base import TTSEngine
from .piper_engine import PiperEngine
from .xtts_engine import XTTSEngine

_ENGINES: dict[str, type[TTSEngine]] = {
    "xtts": XTTSEngine,
    "piper": PiperEngine,
}


def available_engines() -> list[str]:
    return sorted(_ENGINES)


def create_engine(settings: Settings) -> TTSEngine:
    key = settings.tts_engine.lower()
    try:
        engine_cls = _ENGINES[key]
    except KeyError as exc:
        raise ValueError(
            f"Unknown TTS engine {settings.tts_engine!r}; available: {available_engines()}"
        ) from exc
    return engine_cls(settings)
