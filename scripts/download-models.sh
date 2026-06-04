#!/usr/bin/env bash
# Pre-download models into MODELS_DIR so the first request is fast.
set -euo pipefail

MODELS_DIR="${MODELS_DIR:-/opt/voice-ai/models}"
STT_MODEL="${STT_MODEL:-large-v3}"
mkdir -p "$MODELS_DIR"

echo ">> faster-whisper: $STT_MODEL -> $MODELS_DIR/faster-whisper"
python3 - "$STT_MODEL" "$MODELS_DIR" <<'PY'
import sys
from faster_whisper import download_model
model, root = sys.argv[1], sys.argv[2]
download_model(model, output_dir=None, cache_dir=f"{root}/faster-whisper")
print("ok")
PY

echo ">> XTTS-v2 (CPML weights) -> $MODELS_DIR/coqui"
COQUI_TOS_AGREED=1 TTS_HOME="$MODELS_DIR/coqui" python3 - <<'PY'
from TTS.utils.manage import ModelManager
ModelManager().download_model("tts_models/multilingual/multi-dataset/xtts_v2")
print("ok")
PY

# Optional permissive Piper Russian voice (uncomment to fetch).
# PIPER_DIR="$MODELS_DIR/piper"; mkdir -p "$PIPER_DIR"
# base="https://huggingface.co/rhasspy/piper-voices/resolve/main/ru/ru_RU/irina/medium"
# curl -L "$base/ru_RU-irina-medium.onnx"      -o "$PIPER_DIR/ru_RU-irina-medium.onnx"
# curl -L "$base/ru_RU-irina-medium.onnx.json" -o "$PIPER_DIR/ru_RU-irina-medium.onnx.json"

echo "Done."
