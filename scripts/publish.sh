#!/usr/bin/env bash
# One-shot: initialize git, make the first commit (authored solely by you),
# create a PUBLIC GitHub repo via the API, and push.
#
# Requires: git, curl. Run from the repo root.
#
# Provide a GitHub token with `repo` scope:
#   GITHUB_TOKEN=ghp_xxx GITHUB_USER=yakden bash scripts/publish.sh
set -euo pipefail

REPO_NAME="${REPO_NAME:-whisper-xtts-server}"
GITHUB_USER="${GITHUB_USER:-yakden}"
AUTHOR_NAME="${AUTHOR_NAME:-yakden}"
AUTHOR_EMAIL="${AUTHOR_EMAIL:-yakden@gmail.com}"
: "${GITHUB_TOKEN:?Set GITHUB_TOKEN (a GitHub personal access token with 'repo' scope)}"

# Identity for this repo only — no global config changes, no co-authors.
git init -q
git config user.name  "$AUTHOR_NAME"
git config user.email "$AUTHOR_EMAIL"
git add -A
git commit -q -m "Initial commit: Russian STT (faster-whisper) + TTS (XTTS-v2/Piper) REST API"

echo ">> Creating public GitHub repo $GITHUB_USER/$REPO_NAME"
curl -fsSL -X POST https://api.github.com/user/repos \
  -H "Authorization: Bearer $GITHUB_TOKEN" \
  -H "Accept: application/vnd.github+json" \
  -d "{\"name\":\"$REPO_NAME\",\"private\":false,\"description\":\"Russian speech-to-text (faster-whisper) and text-to-speech (XTTS-v2/Piper) REST API\"}" \
  >/dev/null

git branch -M main
git remote add origin "https://${GITHUB_USER}:${GITHUB_TOKEN}@github.com/${GITHUB_USER}/${REPO_NAME}.git"
git push -u origin main
# Strip the token back out of the stored remote URL.
git remote set-url origin "https://github.com/${GITHUB_USER}/${REPO_NAME}.git"

echo ">> Published: https://github.com/${GITHUB_USER}/${REPO_NAME}"
