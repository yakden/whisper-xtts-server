"""Request/response models for the REST API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class TranscriptionSegment(BaseModel):
    id: int
    start: float
    end: float
    text: str


class TranscriptionResponse(BaseModel):
    """OpenAI-compatible transcription payload (verbose subset)."""

    text: str
    language: str | None = None
    duration: float | None = None
    segments: list[TranscriptionSegment] | None = None


class SpeechRequest(BaseModel):
    """OpenAI-compatible /v1/audio/speech request body."""

    model: str = Field(default="tts-1", description="Ignored; kept for OpenAI compatibility.")
    input: str = Field(..., description="Text to synthesize.")
    voice: str | None = Field(
        default=None,
        description="Engine-specific voice id (Piper speaker) or built-in XTTS speaker name.",
    )
    response_format: str = Field(default="wav", description="wav or mp3.")
    speed: float = Field(default=1.0, ge=0.25, le=4.0)
    language: str | None = Field(default=None, description="Override the synthesis language.")


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    owned_by: str = "whisper-xtts-server"


class ModelList(BaseModel):
    object: str = "list"
    data: list[ModelInfo]


class HealthResponse(BaseModel):
    status: str
    stt_model: str
    tts_engine: str
    cuda_available: bool
    device: str
