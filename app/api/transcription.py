"""Speech-to-text endpoint (OpenAI-compatible)."""

from __future__ import annotations

import os
import tempfile

from fastapi import APIRouter, File, Form, Request, UploadFile

from ..schemas import TranscriptionResponse

router = APIRouter(tags=["stt"])


@router.post("/v1/audio/transcriptions", response_model=TranscriptionResponse)
async def transcribe(
    request: Request,
    file: UploadFile = File(..., description="Audio file (wav, mp3, m4a, ogg, ...)."),
    model: str = Form(default="whisper-1"),  # accepted for OpenAI compatibility, ignored
    language: str | None = Form(default=None),
    response_format: str = Form(default="json"),
    translate: bool = Form(default=False),
) -> TranscriptionResponse:
    engine = request.app.state.stt

    suffix = os.path.splitext(file.filename or "audio")[1] or ".bin"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        with_segments = response_format in ("verbose_json",)
        result = engine.transcribe(
            tmp_path,
            language=language,
            translate=translate,
            with_segments=with_segments,
        )
    finally:
        os.unlink(tmp_path)

    return result
