"""
Calque GPU/image worker — FastAPI service.

Wraps the existing post-processing effects (CPU, OpenCV/PIL) behind HTTP so the
Rust orchestrator can proxy to it. The diffusers/ControlNet generation path
(GPU) hangs off /generate and is stubbed until the pipeline is wired in.

Run (inside WSL2, venv active):
    uvicorn calque.server:app --host 127.0.0.1 --port 8001
"""

from __future__ import annotations

import io
import threading
import uuid
from typing import Optional

import requests
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from PIL import Image

from .postprocess.true_watercolor import TrueWatercolorEffect
from .postprocess.style_fx import LithographEffect, AnimeEffect
from .generator import ComfyClient
from .build import GenParams, build_workflow, edge_strength_map

app = FastAPI(title="Calque Worker", version="0.1.0")

# In production the orchestrator fronts the worker; CORS here lets the UI talk
# to the worker directly during development too.
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

comfy = ComfyClient()

# In-memory job registry. job_id -> {status, prompt_id, error, image (bytes)}.
_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()


@app.get("/health")
def health() -> dict:
    """Liveness probe — the orchestrator and deploy script poll this."""
    return {"status": "ok", "service": "calque-worker"}


def _read_image(upload: UploadFile) -> Image.Image:
    try:
        data = upload.file.read()
        return Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as exc:  # noqa: BLE001 - surface a clean 400
        raise HTTPException(status_code=400, detail=f"Invalid image: {exc}") from exc


def _png_response(img: Image.Image) -> Response:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return Response(content=buf.getvalue(), media_type="image/png")


@app.post("/effect/{style}")
def apply_effect(
    style: str,
    image: UploadFile = File(...),
    # Watercolor params
    paper_transparency: float = Form(0.5),
    wash_intensity: float = Form(0.7),
    dark_edges: float = Form(0.6),
    flow_amount: float = Form(0.5),
    brush_visibility: float = Form(0.3),
    # Lithograph params
    contrast: float = Form(1.5),
    threshold: float = Form(128),
    halftone_size: int = Form(4),
    grain_amount: float = Form(0.1),
    # Anime params
    edge_thickness: int = Form(2),
    edge_darkness: float = Form(0.8),
    color_levels: int = Form(8),
    saturation: float = Form(1.2),
) -> Response:
    """Apply a CPU post-processing style to an uploaded image, return PNG."""
    img = _read_image(image)

    if style == "watercolor":
        effect = TrueWatercolorEffect(
            paper_transparency=paper_transparency,
            wash_intensity=wash_intensity,
            dark_edges=dark_edges,
            flow_amount=flow_amount,
            brush_visibility=brush_visibility,
        )
    elif style == "lithograph":
        effect = LithographEffect(
            contrast=contrast,
            threshold=threshold,
            halftone_size=halftone_size,
            grain_amount=grain_amount,
        )
    elif style == "anime":
        effect = AnimeEffect(
            edge_thickness=edge_thickness,
            edge_darkness=edge_darkness,
            color_levels=color_levels,
            saturation=saturation,
        )
    else:
        raise HTTPException(status_code=404, detail=f"Unknown style '{style}'")

    return _png_response(effect.apply(img))


# ----------------------------------------------------------------------------
# Generation (ComfyUI-backed). The UI populates dropdowns from /models, submits
# to /generate, and polls /jobs/{id}.
# ----------------------------------------------------------------------------

def _combo(node: str, field: str) -> list:
    """Pull a COMBO option list (model names, samplers, ...) from ComfyUI."""
    try:
        r = requests.get(f"{comfy.base}/object_info/{node}", timeout=10).json()
        return r[node]["input"]["required"][field][0]
    except Exception:
        return []


@app.get("/models")
def models() -> dict:
    """Live, swappable model + option lists straight from ComfyUI."""
    if not comfy.is_up():
        raise HTTPException(503, "ComfyUI not reachable")
    return {
        "checkpoints": _combo("CheckpointLoaderSimple", "ckpt_name"),
        "loras": _combo("LoraLoader", "lora_name"),
        "controlnets": _combo("ControlNetLoader", "control_net_name"),
        "samplers": _combo("KSampler", "sampler_name"),
        "schedulers": _combo("KSampler", "scheduler"),
    }


def _run_job(job_id: str, workflow: dict) -> None:
    try:
        res = comfy.run(workflow)
        with _jobs_lock:
            if res.images:
                _jobs[job_id].update(status="done", image=res.images[0],
                                     prompt_id=res.prompt_id)
            else:
                _jobs[job_id].update(status="error", error="no image produced")
    except Exception as exc:  # noqa: BLE001
        with _jobs_lock:
            _jobs[job_id].update(status="error", error=str(exc))


@app.post("/generate")
async def generate(
    params: str = Form(...),                       # JSON-encoded GenParams
    image: Optional[UploadFile] = File(None),      # required for mode=image
) -> dict:
    """Submit a generation job; returns {job_id} immediately."""
    import json
    if not comfy.is_up():
        raise HTTPException(503, "ComfyUI not reachable")
    p = GenParams.from_dict(json.loads(params))

    render_name = smap_name = None
    if p.mode == "image":
        if image is None:
            raise HTTPException(400, "mode=image requires an image upload")
        data = image.file.read()
        img = Image.open(io.BytesIO(data)).convert("RGB")
        # snap to VAE-safe dims and remember output size
        w, h = img.size
        img = img.resize((w - w % 8, h - h % 8))
        p.width, p.height = img.size
        render_name = comfy.upload_image(_to_png(img), "calque_render.png")
        smap_name = comfy.upload_image(
            _to_png(edge_strength_map(img)), "calque_smap.png")

    workflow = build_workflow(p, render_name=render_name, smap_name=smap_name)

    job_id = uuid.uuid4().hex
    with _jobs_lock:
        _jobs[job_id] = {"status": "running", "prompt_id": "", "error": None, "image": None}
    threading.Thread(target=_run_job, args=(job_id, workflow), daemon=True).start()
    return {"job_id": job_id}


@app.get("/jobs/{job_id}")
def job_status(job_id: str) -> dict:
    with _jobs_lock:
        job = _jobs.get(job_id)
    if not job:
        raise HTTPException(404, "unknown job")
    return {"status": job["status"], "error": job["error"],
            "has_image": job["image"] is not None}


@app.get("/jobs/{job_id}/image")
def job_image(job_id: str) -> Response:
    with _jobs_lock:
        job = _jobs.get(job_id)
    if not job:
        raise HTTPException(404, "unknown job")
    if job["status"] != "done" or job["image"] is None:
        raise HTTPException(409, f"job not ready (status={job['status']})")
    return Response(content=job["image"], media_type="image/png")


def _to_png(img: Image.Image) -> bytes:
    b = io.BytesIO(); img.save(b, format="PNG"); return b.getvalue()
