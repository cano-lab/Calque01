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
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from PIL import Image

from .postprocess.true_watercolor import TrueWatercolorEffect
from .postprocess.style_fx import LithographEffect, AnimeEffect

app = FastAPI(title="Calque Worker", version="0.1.0")


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


@app.post("/generate")
def generate() -> dict:
    """GPU diffusion/ControlNet generation. Not yet wired in — see PLAN.md."""
    raise HTTPException(
        status_code=501,
        detail="Generation pipeline not implemented yet (GPU path). See PLAN.md.",
    )
