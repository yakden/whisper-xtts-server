# CUDA runtime base with cuDNN (required by CTranslate2 / faster-whisper on GPU).
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    MODELS_DIR=/opt/voice-ai/models \
    COQUI_TOS_AGREED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-venv ffmpeg git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install GPU torch first from the CUDA wheel index, then the rest.
COPY pyproject.toml ./
RUN pip3 install --upgrade pip \
    && pip3 install torch torchaudio --index-url https://download.pytorch.org/whl/cu124 \
    && pip3 install .

COPY app ./app

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
