#!/usr/bin/env bash
set -euo pipefail

# Calque deploy script — runs on the GPU host (WSL2) as the deploy user.
# Pulls latest, builds the orchestrator, refreshes the worker venv,
# restarts both services, and health-checks.

APP=/opt/calque/app
BIN=/opt/calque/bin
PORT="${PORT:-3001}"

cd "$APP"
git config --global --add safe.directory "$APP" 2>/dev/null || true
git remote set-url origin https://github.com/cano-lab/Calque01.git

echo "==> Pulling latest..."
git fetch origin main
git reset --hard origin/main

echo "==> Building orchestrator (release)..."
cargo build --release -p calque-orchestrator

echo "==> Syncing worker venv..."
cd "$APP/worker"
if [[ ! -d .venv ]]; then python3 -m venv .venv; fi
./.venv/bin/pip install --quiet --upgrade pip
./.venv/bin/pip install --quiet -r requirements.txt
cd "$APP"

echo "==> Backing up + swapping binary..."
[[ -f "$BIN/calque-orchestrator" ]] && cp "$BIN/calque-orchestrator" "$BIN/calque-orchestrator.backup" || true
cp target/release/calque-orchestrator "$BIN/calque-orchestrator"

echo "==> Restarting services..."
sudo systemctl restart calque-worker
sudo systemctl restart calque-orchestrator

echo "==> Health check on :$PORT ..."
for i in $(seq 1 10); do
  if curl -fsS --max-time 5 "http://localhost:${PORT}/health" >/dev/null 2>&1; then
    echo "Health check passed."
    exit 0
  fi
  echo "  attempt $i/10 failed; retrying in 3s..."
  sleep 3
done

echo "ERROR: health check failed — rolling back binary." >&2
if [[ -f "$BIN/calque-orchestrator.backup" ]]; then
  cp "$BIN/calque-orchestrator.backup" "$BIN/calque-orchestrator"
  sudo systemctl restart calque-orchestrator
fi
exit 1
