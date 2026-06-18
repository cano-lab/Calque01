# Calque

Style laid over structure. Calque turns 3D / architectural renders into
artistic styles — watercolor, lithograph, anime — while preserving the
underlying drawing. A companion tool to [Almanach](../Almanach01); built for
students.

A *calque* is the transparent overlay an architect traces over a drawing. That
is exactly what this tool does: an artistic layer placed over a preserved
structure, with control.

## Architecture

```
student browser
   → clawpen.ca / studio.clawpen.ca   (DNS → Cloudflare Tunnel)
   → cloudflared on the GPU host (outbound only — no open ports)
   → orchestrator  (Rust/Axum, :3001)  serves the web UI, proxies /api/*
   → worker        (Python/FastAPI, :8001)  image effects + GPU generation
```

- `orchestrator/` — Rust/Axum front door. Serves `static-site/`, exposes
  `/health`, proxies `/api/*` to the worker.
- `worker/` — Python/FastAPI. CPU post-processing effects now; diffusers /
  ControlNet GPU generation (`/generate`) is stubbed for later.
- `static-site/` — single-page web UI (replaces the old tkinter desktop app).
- `scripts/` — WSL2 provisioning, systemd units, deploy + health-check.
- `.github/workflows/deploy.yml` — push to `main` → self-hosted runner deploys.

## Develop (Windows + WSL2)

```bash
# worker
cd worker && python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn calque.server:app --host 127.0.0.1 --port 8001

# orchestrator (another shell)
cd orchestrator && cargo run     # serves http://localhost:3001
```

Open http://localhost:3001 — the UI proxies to the worker.

## Deploy (GPU host)

```bash
sudo bash scripts/setup-vps.sh      # one-time provisioning
cp .env.example /opt/calque/.env    # edit
bash /opt/calque/app/scripts/deploy.sh
# then point cloudflared at http://localhost:3001
```

## Status

- ✅ CPU effects (watercolor / lithograph / anime) served over the web
- 🚧 GPU diffusion + ControlNet generation (`/generate`) — see `PLAN.md`
- 🚧 Auth / student accounts (borrow Almanach's pattern when needed)

## License

MIT
