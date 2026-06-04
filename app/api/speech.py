"""Text-to-speech endpoints."""

from __future__ import annotations

import os
import tempfile

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import Response

from .. import audio
from ..schemas import SpeechRequest

router = APIRouter(tags=["tts"])


def _audio_response(data: bytes, fmt: str) -> Response:
    return Response(content=data, media_type=audio.CONTENT_TYPES.get(fmt, "application/octet-stream"))


@router.post("/v1/audio/speech")
async def create_speech(request: Request, body: SpeechRequest) -> Response:
    """OpenAI-compatible synthesis. Uses the configured default/built-in voice."""
    engine = request.app.state.tts
    try:
        result = engine.synthesize(
            body.input,
            language=body.language,
            voice=body.voice,
            speed=body.speed,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    data = audio.encode(result.audio, result.sample_rate, body.response_format)
    return _audio_response(data, body.response_format.lower())


@router.post("/tts/clone")
async def clone_voice(
    request: Request,
    text: str = Form(...),
    speaker_wav: UploadFile = File(..., description="Reference voice clip (6+ seconds)."),
    language: str | None = Form(default=None),
    response_format: str = Form(default="wav"),
    speed: float = Form(default=1.0),
) -> Response:
    """Synthesize `text` in the voice of the uploaded reference clip (cloning engines only)."""
    engine = request.app.state.tts
    if not getattr(engine, "supports_cloning", False):
        raise HTTPException(
            status_code=400,
            detail=f"Active TTS engine '{engine.name}' does not support voice cloning.",
        )

    suffix = os.path.splitext(speaker_wav.filename or "ref.wav")[1] or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await speaker_wav.read())
        ref_path = tmp.name

    try:
        result = engine.synthesize(text, language=language, speaker_wav=ref_path, speed=speed)
    finally:
        os.unlink(ref_path)

    data = audio.encode(result.audio, result.sample_rate, response_format)
    return _audio_response(data, response_format.lower())
