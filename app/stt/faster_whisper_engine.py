"""faster-whisper (CTranslate2) speech-to-text engine."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from faster_whisper import WhisperModel

from ..config import Settings
from ..schemas import TranscriptionResponse, TranscriptionSegment

logger = logging.getLogger(__name__)


@dataclass
class WhisperEngine:
    """Thin wrapper around a single resident WhisperModel."""

    settings: Settings
    _model: WhisperModel | None = None

    def load(self) -> None:
        if self._model is not None:
            return
        download_root = self.settings.whisper_model_dir
        download_root.mkdir(parents=True, exist_ok=True)
        logger.info(
            "Loading faster-whisper model=%s device=%s compute_type=%s",
            self.settings.stt_model,
            self.settings.stt_device,
            self.settings.stt_compute_type,
        )
        self._model = WhisperModel(
            self.settings.stt_model,
            device=self.settings.stt_device,
            compute_type=self.settings.stt_compute_type,
            download_root=str(download_root),
        )

    @property
    def model(self) -> WhisperModel:
        if self._model is None:
            self.load()
        assert self._model is not None
        return self._model

    def transcribe(
        self,
        audio_path: str | Path,
        language: str | None = None,
        translate: bool = False,
        with_segments: bool = True,
    ) -> TranscriptionResponse:
        # An explicit empty language means "autodetect"; None falls back to the configured default.
        lang = language if language is not None else self.settings.stt_language
        lang = lang or None

        segments_iter, info = self.model.transcribe(
            str(audio_path),
            language=lang,
            beam_size=self.settings.stt_beam_size,
            task="translate" if translate else "transcribe",
            vad_filter=True,
        )

        segments: list[TranscriptionSegment] = []
        texts: list[str] = []
        for i, seg in enumerate(segments_iter):
            texts.append(seg.text)
            if with_segments:
                segments.append(
                    TranscriptionSegment(
                        id=i, start=round(seg.start, 3), end=round(seg.end, 3), text=seg.text.strip()
                    )
                )

        return TranscriptionResponse(
            text="".join(texts).strip(),
            language=info.language,
            duration=round(info.duration, 3),
            segments=segments if with_segments else None,
        )
