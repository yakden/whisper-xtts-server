"""Pluggable TTS engine interface.

Every engine returns mono 16-bit PCM as (samples: np.ndarray float32 in [-1, 1], sample_rate: int).
Encoding to wav/mp3 happens in the API layer so engines stay format-agnostic.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass

import numpy as np


@dataclass
class SynthesisResult:
    audio: np.ndarray  # float32, shape (n_samples,), range [-1, 1]
    sample_rate: int


class TTSEngine(abc.ABC):
    """Base class for all text-to-speech engines."""

    #: Whether this engine can clone an arbitrary voice from a reference sample.
    supports_cloning: bool = False

    @abc.abstractmethod
    def load(self) -> None:
        """Load model weights into memory. Safe to call repeatedly."""

    @abc.abstractmethod
    def synthesize(
        self,
        text: str,
        *,
        language: str | None = None,
        voice: str | None = None,
        speaker_wav: str | None = None,
        speed: float = 1.0,
    ) -> SynthesisResult:
        """Synthesize `text` to a waveform.

        Args:
            language: ISO code (e.g. "ru"); engines may ignore if monolingual.
            voice: engine-specific built-in voice/speaker id.
            speaker_wav: path to a reference clip for cloning (cloning engines only).
            speed: speaking-rate multiplier.
        """

    @property
    def name(self) -> str:
        return self.__class__.__name__
