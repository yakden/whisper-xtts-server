"""XTTS-v2 engine (via the maintained `coqui-tts` package).

XTTS-v2 supports zero-shot voice cloning from a ~6 second reference clip and
multilingual synthesis including Russian.

NOTE ON LICENSING: the XTTS-v2 *weights* are released under the Coqui Public
Model License (CPML), which is non-commercial. The weights are downloaded by
the operator on first use. This server's own source code is MIT licensed and is
a separate work — see README and LICENSE.
"""

from __future__ import annotations

import logging
import os

import numpy as np

from ..config import Settings
from .base import SynthesisResult, TTSEngine

logger = logging.getLogger(__name__)


class XTTSEngine(TTSEngine):
    supports_cloning = True

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._tts = None  # type: ignore[var-annotated]

    def load(self) -> None:
        if self._tts is not None:
            return
        # Accept the model license non-interactively for unattended/server use.
        os.environ.setdefault("COQUI_TOS_AGREED", "1")
        # Route model downloads into our models dir.
        os.environ.setdefault("TTS_HOME", str(self.settings.models_dir / "coqui"))

        from TTS.api import TTS  # imported lazily: heavy (pulls in torch)

        logger.info("Loading XTTS-v2 model=%s device=%s", self.settings.xtts_model, self.settings.tts_device)
        self._tts = TTS(self.settings.xtts_model).to(self.settings.tts_device)

    def synthesize(
        self,
        text: str,
        *,
        language: str | None = None,
        voice: str | None = None,
        speaker_wav: str | None = None,
        speed: float = 1.0,
    ) -> SynthesisResult:
        if self._tts is None:
            self.load()
        assert self._tts is not None

        lang = language or self.settings.tts_language
        ref = speaker_wav

        kwargs: dict = {"text": text, "language": lang, "speed": speed}
        if ref:
            kwargs["speaker_wav"] = ref
        elif self.settings.xtts_default_speaker_wav:
            kwargs["speaker_wav"] = str(self.settings.xtts_default_speaker_wav)
        elif voice:
            # XTTS ships a set of built-in studio speakers selectable by name.
            kwargs["speaker"] = voice
        else:
            raise ValueError(
                "XTTS requires a reference voice: provide `speaker_wav`, set "
                "XTTS_DEFAULT_SPEAKER_WAV, or pass a built-in `voice` name."
            )

        wav = self._tts.tts(**kwargs)
        audio = np.asarray(wav, dtype=np.float32)
        sample_rate = int(self._tts.synthesizer.output_sample_rate)
        return SynthesisResult(audio=audio, sample_rate=sample_rate)
