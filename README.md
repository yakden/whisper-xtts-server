# whisper-xtts-server

Self-hosted **Russian speech-to-text and text-to-speech** REST API, built for a
single NVIDIA GPU (developed on a Tesla T4, 16 GB).

- **STT** — [`faster-whisper`](https://github.com/SYSTRAN/faster-whisper) `large-v3`, tuned for Russian, GPU-accelerated.
- **TTS** — pluggable engines. Default **XTTS-v2** (zero-shot voice cloning from a ~6 s sample); permissive-license **Piper** alternative.
- **OpenAI-compatible** endpoints (`/v1/audio/transcriptions`, `/v1/audio/speech`) — drop-in for existing OpenAI clients.

Both models stay resident in VRAM (~3 GB Whisper + ~3 GB XTTS ≈ 6 / 16 GB on a T4).

## Endpoints

| Method | Path | Description |
| --- | --- | --- |
| `POST` | `/v1/audio/transcriptions` | Transcribe audio → text (OpenAI-compatible, multipart `file`). |
| `POST` | `/v1/audio/speech` | Synthesize text → audio (OpenAI-compatible JSON body). |
| `POST` | `/tts/clone` | Synthesize text in the voice of an uploaded reference clip (XTTS only). |
| `GET`  | `/health` | Liveness + CUDA status. |
| `GET`  | `/v1/models` | List active models. |

Interactive docs at `/docs`.

## Quick start (Docker)

```bash
git clone https://github.com/yakden/whisper-xtts-server
cd whisper-xtts-server
cp .env.example .env
docker compose up -d --build
curl localhost:8000/health
```

Requires an NVIDIA GPU with recent drivers and the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

## Usage

Transcribe:

```bash
curl -s http://localhost:8000/v1/audio/transcriptions \
  -F file=@sample.mp3 -F language=ru | jq .text
```

Synthesize (default/built-in voice):

```bash
curl -s http://localhost:8000/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"input":"Привет! Это синтез речи.","response_format":"wav"}' \
  --output speech.wav
```

Clone a voice (XTTS):

```bash
curl -s http://localhost:8000/tts/clone \
  -F text="Текст, озвученный голосом из образца." \
  -F speaker_wav=@reference.wav --output cloned.wav
```

## Configuration

All settings come from environment variables / `.env` — see [`.env.example`](.env.example).
Switch the TTS engine with `TTS_ENGINE=xtts|piper`.

## Deploy on Proxmox (LXC + GPU sharing)

A full guide for installing the driver on a Proxmox host and sharing the GPU
into an LXC container (lighter than a passthrough VM) is in
[`deploy/proxmox/nvidia-host-setup.md`](deploy/proxmox/nvidia-host-setup.md).
Native install (venv + systemd) is provided by [`scripts/install-native.sh`](scripts/install-native.sh).

## Licensing

This project's **source code is MIT licensed** (see [LICENSE](LICENSE)).

Model weights are downloaded separately by the operator and carry their own licenses:

- **XTTS-v2** weights — Coqui Public Model License (**CPML, non-commercial**). If you
  need a commercially usable voice out of the box, switch to Piper.
- **Piper** voices — MIT / permissive.
- **faster-whisper / Whisper** — MIT.

You are responsible for complying with the license of whichever model you deploy.

## Development

```bash
pip install -e '.[dev]'
pytest          # lightweight unit tests (no GPU/model downloads)
```
