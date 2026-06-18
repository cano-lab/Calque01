# Calque Worker (Python)

Image-processing + GPU generation service. The Rust orchestrator proxies
`/api/*` requests here.

## Run (WSL2, dev)

```bash
cd worker
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn calque.server:app --host 127.0.0.1 --port 8001
```

## Endpoints

- `GET  /health` — liveness
- `POST /effect/{watercolor|lithograph|anime}` — multipart `image=@file`, returns PNG
- `POST /generate` — GPU diffusion path (stub, returns 501 until wired)

The CPU effects need no GPU. For `/generate`, uncomment the torch/diffusers
lines in `requirements.txt` and install a CUDA build of torch on the GPU host.
