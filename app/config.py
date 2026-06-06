"""Runtime configuration, loaded from environment variables / .env."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="", extra="ignore")

    # --- HTTP server ---
    host: str = "0.0.0.0"
    port: int = 8000
    # Optional bearer token. If set, every request must send `Authorization: Bearer <key>`.
    api_key: str | None = None

    # --- Storage ---
    models_dir: Path = Path("/opt/voice-ai/models")

    # --- Speech-to-text (faster-whisper) ---
    stt_model: str = "large-v3"
    stt_device: str = "cuda"
    stt_compute_type: str = "float16"  # float16 on GPU, int8 on CPU
    # Default transcription language. Empty string => autodetect.
    stt_language: str = "ru"
    stt_beam_size: int = 5
    # Keep the STT model resident in VRAM at startup.
    stt_preload: bool = True

    # --- Text-to-speech ---
    # Which engine to use: "xtts" (default, voice cloning) or "piper" (permissive license).
    tts_engine: str = "xtts"
    tts_language: str = "ru"
    tts_device: str = "cuda"
    tts_preload: bool = True

    # XTTS-v2 (coqui-tts). Weights are CPML licensed (non-commercial) and are
    # downloaded by the operator separately — see README.
    xtts_model: str = "tts_models/multilingual/multi-dataset/xtts_v2"
    # Default reference voice (6+ sec wav) used when a request omits a speaker sample.
    xtts_default_speaker_wav: Path | None = None
    # Fallback to one of XTTS's built-in studio speakers when a request provides
    # neither `speaker_wav` nor `voice` and no default wav is set. Lets
    # /v1/audio/speech work out of the box (e.g. from Swagger UI). Empty => off.
    xtts_default_voice: str = "Ana Florence"

    # Piper (MIT licensed). Point these at a downloaded Russian voice.
    piper_model_path: Path | None = None
    piper_config_path: Path | None = None

    @property
    def whisper_model_dir(self) -> Path:
        return self.models_dir / "faster-whisper"


@lru_cache
def get_settings() -> Settings:
    return Settings()
