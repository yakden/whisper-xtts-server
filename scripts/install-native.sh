#!/usr/bin/env bash
# Native install inside an LXC container (no Docker): venv + systemd service.
# Run as root inside the container. Assumes the NVIDIA userspace driver is already
# present (see deploy/proxmox/nvidia-host-setup.md).
set -euo pipefail

APP_DIR="${APP_DIR:-/opt/whisper-xtts-server}"
VENV="$APP_DIR/.venv"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo ">> Installing system packages"
apt-get update
apt-get install -y python3 python3-venv python3-pip ffmpeg git

echo ">> Syncing source to $APP_DIR"
mkdir -p "$APP_DIR"
cp -r "$REPO_DIR/app" "$REPO_DIR/pyproject.toml" "$APP_DIR/"
[ -f "$REPO_DIR/.env" ] && cp "$REPO_DIR/.env" "$APP_DIR/.env" || cp "$REPO_DIR/.env.example" "$APP_DIR/.env"

echo ">> Creating venv + installing deps (GPU torch from CUDA wheel index)"
python3 -m venv "$VENV"
"$VENV/bin/pip" install --upgrade pip
"$VENV/bin/pip" install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
"$VENV/bin/pip" install "$APP_DIR" 2>/dev/null || "$VENV/bin/pip" install -e "$APP_DIR"

echo ">> Installing systemd service"
cat >/etc/systemd/system/whisper-xtts.service <<EOF
[Unit]
Description=whisper-xtts-server (Russian STT + TTS REST API)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory=$APP_DIR
EnvironmentFile=$APP_DIR/.env
ExecStart=$VENV/bin/uvicorn app.main:app --host 0.0.0.0 --port \${PORT:-8000}
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable --now whisper-xtts.service
echo ">> Done. Check: systemctl status whisper-xtts && curl localhost:8000/health"
