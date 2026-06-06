"""Piper engine (MIT licensed, fully permissive — safe for commercial distribution).

Piper is a fast, lightweight neural TTS that runs well even on CPU. It uses
fixed per-voice models (no cloning). Download a Russian voice (e.g.
`ru_RU-irina-medium`) and point PIPER_MODEL_PATH / PIPER_CONFIG_PATH at it.
"""

from __future__ import annotations

import logging

import numpy as np

from ..config import Settings
from .base import SynthesisResult, TTSEngine

logger = logging.getLogger(__name__)


class PiperEngine(TTSEngine):
    supports_cloning = False

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._voice = None  # type: ignore[var-annotated]

    def load(self) -> None:
        if self._voice is not None:
            return
        if not self.settings.piper_model_path:
            raise RuntimeError("PIPER_MODEL_PATH is not set; download a Piper voice first.")

        from piper import PiperVoice, SynthesisConfig  # lazy import

        self._SynthesisConfig = SynthesisConfig

        model = str(self.settings.piper_model_path)
        config = str(self.settings.piper_config_path) if self.settings.piper_config_path else None
        logger.info("Loading Piper voice model=%s", model)
        use_cuda = self.settings.tts_device.startswith("cuda")
        self._voice = PiperVoice.load(model, config_path=config, use_cuda=use_cuda)

    def synthesize(
        self,
        text: str,
        *,
        language: str | None = None,
        voice: str | None = None,
        speaker_wav: str | None = None,
        speed: float = 1.0,
    ) -> SynthesisResult:
        if self._voice is None:
            self.load()
        assert self._voice is not None

        # Piper exposes multi-speaker models via integer speaker ids.
        speaker_id = int(voice) if voice and voice.isdigit() else None
        length_scale = 1.0 / speed if speed else 1.0

        # piper>=1.3 takes synthesis options via a SynthesisConfig object.
        syn_config = self._SynthesisConfig(speaker_id=speaker_id, length_scale=length_scale)

        chunks: list[np.ndarray] = []
        sample_rate = self._voice.config.sample_rate
        for audio_chunk in self._voice.synthesize(text, syn_config=syn_config):
            # piper>=1.3 yields AudioChunk objects with int16 PCM bytes.
            pcm = np.frombuffer(audio_chunk.audio_int16_bytes, dtype=np.int16)
            chunks.append(pcm.astype(np.float32) / 32768.0)
            sample_rate = audio_chunk.sample_rate

        audio = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
        return SynthesisResult(audio=audio, sample_rate=int(sample_rate))
