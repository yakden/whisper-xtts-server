"""FastAPI application entrypoint."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request

from . import __version__
from .api import speech, transcription
from .config import get_settings
from .schemas import HealthResponse, ModelInfo, ModelList
from .stt.faster_whisper_engine import WhisperEngine
from .tts.registry import available_engines, create_engine

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("whisper-xtts-server")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    app.state.settings = settings

    app.state.stt = WhisperEngine(settings=settings)
    app.state.tts = create_engine(settings)

    if settings.stt_preload:
        app.state.stt.load()
    if settings.tts_preload:
        app.state.tts.load()

    logger.info("Startup complete (stt=%s, tts=%s)", settings.stt_model, settings.tts_engine)
    yield


app = FastAPI(title="whisper-xtts-server", version=__version__, lifespan=lifespan)


@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    settings = get_settings()
    if settings.api_key and request.url.path not in ("/health", "/docs", "/openapi.json"):
        header = request.headers.get("authorization", "")
        token = header[7:] if header.lower().startswith("bearer ") else None
        if token != settings.api_key:
            raise HTTPException(status_code=401, detail="Invalid or missing API key.")
    return await call_next(request)


app.include_router(transcription.router)
app.include_router(speech.router)


@app.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    settings = request.app.state.settings
    try:
        import torch

        cuda = bool(torch.cuda.is_available())
    except Exception:
        cuda = False
    return HealthResponse(
        status="ok",
        stt_model=settings.stt_model,
        tts_engine=settings.tts_engine,
        cuda_available=cuda,
        device=settings.stt_device,
    )


@app.get("/v1/models", response_model=ModelList)
async def list_models(request: Request) -> ModelList:
    settings = request.app.state.settings
    data = [
        ModelInfo(id=f"whisper-{settings.stt_model}"),
        ModelInfo(id=f"tts-{settings.tts_engine}"),
    ]
    return ModelList(data=data)
