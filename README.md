# Self-Hosted AI Platform (single NVIDIA Tesla T4)

A self-hosted, GPU-accelerated AI platform built around this Russian **speech-to-text / text-to-speech** server and extended into a full media-AI stack: voice cloning, talking-avatar / lip-sync, text-and-photo→video animation, multi-model chat with MCP tools, image object-removal, and a real-time video-analytics pipeline — all behind a single secure web entrypoint, managed from one control panel.

Everything runs on one **NVIDIA Tesla T4 (16 GB, Turing)**. Because a single T4 cannot hold every model at once, heavy services are started **on demand** and the control plane frees VRAM before launching the large generative jobs.

---

## Components

### Core API — `whisper-xtts-server` (this repo)
Russian STT + TTS REST API, OpenAI-compatible.

| Method | Path | Description |
| --- | --- | --- |
| `POST` | `/v1/audio/transcriptions` | Transcribe audio → text (faster-whisper `large-v3`, GPU). |
| `POST` | `/v1/audio/speech` | Synthesize text → audio (XTTS-v2 or Piper). |
| `POST` | `/tts/clone` | Synthesize text in the voice of an uploaded reference clip (XTTS zero-shot voice cloning). |
| `GET`  | `/v1/translate` | (optional) RU→target machine translation. |
| `GET`  | `/health`, `/v1/models` | Liveness + active models. |

- **STT**: faster-whisper `large-v3`, tuned for Russian, GPU.
- **TTS engines**: **XTTS-v2** (multilingual, zero-shot voice cloning from a ~6 s sample) and **Piper** (fast, permissive license, fixed voices). Engine selectable via `TTS_ENGINE`.
- **Auth**: optional `API_KEY` bearer token. Misconfigured-auth bug fixed (returns clean 401, exposes a bearer scheme so Swagger shows an Authorize button).

### Talking avatar — MuseTalk (`avatar-muse`)
Lip-syncs a base presenter video to synthesized speech. `POST /avatar` (video + text or audio) → mp4. Runs in a dedicated CUDA 11.8 / torch 2.0.1 container with the prebuilt `mmcv` op stack.

### Self-dub pipeline (`dub.sh` + webcam tool)
End-to-end dubbing: speech → STT → translate → **clone the speaker's own voice** in the target language → full lip-sync → mp4.
- CLI: `dub.sh <video> <speech.wav> [target_lang] [out.mp4]`.
- Web tool (webcam capture → near-live dubbed video) served behind the gateway.

### Text + photo → talking video — Wan 2.2-S2V (`animate`)
Upload **text + a photo**, get a video where the character is animated (head/hair/expression motion) and speaks the text — with **live progress and ETA** in the browser. Audio-driven portrait animation via Wan 2.2-S2V (GGUF Q4) on ComfyUI; ~13–14 GB VRAM, batch (offline) generation. The control plane evicts other GPU services for the duration of a job.

### Multi-model chat — Open WebUI + Ollama + MCP
Off-the-shelf web UI for chatting with local models (model selection, vision, voice in/out wired to the STT/TTS server). Models served by **Ollama** (`qwen2.5vl` vision, `llama3.2`, easily extended). **MCP** tool support via the `mcpo` gateway (MCP → OpenAPI).

### Image object removal — IOPaint
LaMa eraser + SAM2 interactive segmentation ("click → remove"), self-hosted web UI.

### Real-time video analytics — NVIDIA DeepStream
RTSP → person/face detection → tracking (NvDCF) → **face recognition against a known-face database** (ArcFace embeddings + FAISS) → ReID for occluded subjects → events/alerts. TensorRT INT8 inference; the detection+tracking stage benchmarks at ~700 FPS on the T4 (huge headroom for many camera streams).

---

## Management, automation, and security

- **Control plane** — a single web dashboard + JSON API to see every service's status and GPU/VRAM, and to start/stop/restart them. New services plug in via a small registry entry. Handles **GPU eviction** so heavy jobs (animate) get the whole T4 and lighter services resume afterward.
- **Single sign-on** — one login for the whole platform via a cookie-based SSO gateway on the parent domain; app-internal logins are disabled or trusted-header-authenticated so nothing prompts twice.
- **Public access** — everything is reverse-proxied by **nginx** under per-service subdomains with **Let's Encrypt HTTPS**; service ports stay bound to `127.0.0.1` and are reachable only through the gateway. The host firewall exposes only SSH + HTTP/HTTPS; internal SDN guests get NAT/forwarding.
- **Services** run as Docker (compose) projects and systemd units, restored automatically on boot.

> Secrets (API keys, passwords, tokens) live only in local `.env` / root-only env files and are never committed.

---

## Hardware notes (T4 / Turing)

The T4 has hardware FP16 + INT8 but **no native bf16 and no FP8**. Across the stack this means: prefer fp16 checkpoints, **GGUF** quantization for large diffusion/LLM models, and INT4/INT8 for VLMs; TensorRT INT8 for the analytics models. bf16 weights are converted to fp16 where needed. Large video models (Wan 2.2-S2V) run quantized (Q4) and are batch/offline — not real-time. The largest single job uses ~14 GB, so it runs exclusively while other GPU services are paused.

---

## Quick start (core API)

```bash
cp .env.example .env          # adjust; set API_KEY, TTS_ENGINE, etc.
docker compose up -d --build
curl localhost:8000/health
```

Requires an NVIDIA GPU with recent drivers and the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

Transcribe:
```bash
curl -s http://localhost:8000/v1/audio/transcriptions -F file=@sample.mp3 -F language=ru | jq .text
```

Synthesize (built-in voice):
```bash
curl -s http://localhost:8000/v1/audio/speech -H 'Content-Type: application/json' \
  -d '{"input":"Привет! Это синтез речи.","response_format":"wav"}' --output speech.wav
```

Clone a voice (XTTS):
```bash
curl -s http://localhost:8000/tts/clone -F text="Текст голосом из образца." \
  -F speaker_wav=@reference.wav --output cloned.wav
```

## Configuration
All core-API settings come from environment variables / `.env` (see [`.env.example`](.env.example)).
Switch the TTS engine with `TTS_ENGINE=xtts|piper`. XTTS adds a configurable built-in fallback voice so `/v1/audio/speech` works without a reference clip.

## Deploy on Proxmox (LXC + GPU sharing)
Driver install on the Proxmox host and GPU sharing into a container are covered in
[`deploy/proxmox/nvidia-host-setup.md`](deploy/proxmox/nvidia-host-setup.md).
Native install (venv + systemd) is provided by [`scripts/install-native.sh`](scripts/install-native.sh).

## Licensing
This project's **source code is MIT licensed** (see [LICENSE](LICENSE)).

Model weights are downloaded separately by the operator and carry their own licenses:
- **faster-whisper / Whisper**, **Piper** voices — MIT / permissive.
- **XTTS-v2** weights — Coqui Public Model License (non-commercial); use Piper for a commercially usable voice.
- **Wan 2.2** — Apache 2.0; **MuseTalk**, **Ollama** models, **DeepStream/TAO** models, **SAM2/LaMa** — each carry their own terms.

You are responsible for complying with the license of whichever model you deploy, and with applicable law when using face recognition / video analytics.

## Development
```bash
pip install -e '.[dev]'
pytest          # lightweight unit tests (no GPU/model downloads)
```
