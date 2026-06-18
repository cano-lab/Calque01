#!/usr/bin/env bash
set -euo pipefail

# One-time idempotent provisioning for the Calque GPU host (WSL2 Ubuntu).
# Run as root inside WSL:  sudo bash scripts/setup-vps.sh
#
# Assumes: recent NVIDIA driver on the Windows side (provides CUDA into WSL),
# Rust toolchain available to the deploy user, python3 + venv installed.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

getent group calque >/dev/null 2>&1 || groupadd --system calque
id -u calque >/dev/null 2>&1 || useradd --system --home-dir /opt/calque --shell /usr/sbin/nologin --gid calque calque
id -u deploy  >/dev/null 2>&1 || useradd --create-home --shell /bin/bash deploy

mkdir -p /opt/calque/{app,bin}
chown deploy:calque /opt/calque/app /opt/calque/bin
chmod 755 /opt/calque /opt/calque/app /opt/calque/bin

apt-get update
apt-get install -y build-essential pkg-config libssl-dev curl git python3 python3-venv python3-pip

# Populate app dir from this repo if empty
if [[ -z "$(ls -A /opt/calque/app 2>/dev/null || true)" ]]; then
  cp -a "${REPO_ROOT}/." /opt/calque/app
  chown -R deploy:calque /opt/calque/app
fi

# Install systemd units
cp "${REPO_ROOT}/scripts/calque-worker.service"       /etc/systemd/system/calque-worker.service
cp "${REPO_ROOT}/scripts/calque-orchestrator.service" /etc/systemd/system/calque-orchestrator.service

# Passwordless restart for the deploy user
cat > /etc/sudoers.d/calque-deploy <<'EOF'
deploy ALL=(root) NOPASSWD: /bin/systemctl restart calque-worker
deploy ALL=(root) NOPASSWD: /bin/systemctl restart calque-orchestrator
deploy ALL=(root) NOPASSWD: /bin/systemctl start calque-worker
deploy ALL=(root) NOPASSWD: /bin/systemctl start calque-orchestrator
deploy ALL=(root) NOPASSWD: /bin/systemctl stop calque-worker
deploy ALL=(root) NOPASSWD: /bin/systemctl stop calque-orchestrator
EOF
chmod 440 /etc/sudoers.d/calque-deploy
visudo -c

systemctl daemon-reload
systemctl enable calque-worker calque-orchestrator

echo
echo "Provisioning complete. Next:"
echo "  1. cp .env.example /opt/calque/.env  and edit it"
echo "  2. As deploy user: bash /opt/calque/app/scripts/deploy.sh   (builds + starts)"
echo "  3. Point cloudflared at http://localhost:3001 (studio.clawpen.ca)"
