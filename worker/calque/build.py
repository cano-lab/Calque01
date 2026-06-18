"""
Programmatic ComfyUI graph builder.

The UI exposes every knob and lets models be swapped, so the workflow can't be a
fixed template — LoRA may be on/off, each ControlNet on/off, and the input is
either an uploaded image (structure-controlled img2img) or nothing (text2img).
build_workflow() assembles the right graph from a params dict.

Node ids are stable strings; links are [node_id, output_index] per ComfyUI's
API format.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import cv2
import numpy as np
from PIL import Image

DEPTH_CN = "control_v11f1p_sd15_depth_fp16.safetensors"
LINEART_CN = "control_v11p_sd15_lineart_fp16.safetensors"


@dataclass
class GenParams:
    mode: str = "image"            # "image" | "text"
    ckpt: str = "counterfeitV30_v30.safetensors"
    lora: Optional[str] = None
    lora_strength: float = 1.0
    positive: str = ""
    negative: str = "lowres, blurry, watermark, text"
    steps: int = 28
    cfg: float = 7.0
    sampler: str = "dpmpp_2m"
    scheduler: str = "karras"
    seed: int = 0
    denoise: float = 0.8          # img2img strength (ignored for text2img)
    width: int = 768              # text2img / output sizing
    height: int = 512
    depth_strength: float = 0.5   # 0 disables
    lineart_strength: float = 0.4 # 0 disables

    @classmethod
    def from_dict(cls, d: dict) -> "GenParams":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})


def edge_strength_map(img: Image.Image) -> Image.Image:
    """Auto precision map: anchor near strong edges, free on flat surfaces."""
    g = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2GRAY)
    e = cv2.dilate(cv2.Canny(g, 60, 160), np.ones((3, 3), np.uint8))
    soft = 255 - cv2.GaussianBlur(e, (0, 0), 2.0)
    return Image.fromarray(soft).convert("RGB")


def build_workflow(
    p: GenParams,
    render_name: Optional[str] = None,
    smap_name: Optional[str] = None,
) -> dict[str, Any]:
    g: dict[str, Any] = {}

    g["ckpt"] = {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": p.ckpt}}
    model: list = ["ckpt", 0]
    clip: list = ["ckpt", 1]
    vae: list = ["ckpt", 2]

    if p.lora:
        g["lora"] = {"class_type": "LoraLoader", "inputs": {
            "model": ["ckpt", 0], "clip": ["ckpt", 1], "lora_name": p.lora,
            "strength_model": p.lora_strength, "strength_clip": p.lora_strength}}
        model, clip = ["lora", 0], ["lora", 1]

    g["pos"] = {"class_type": "CLIPTextEncode", "inputs": {"text": p.positive, "clip": clip}}
    g["neg"] = {"class_type": "CLIPTextEncode", "inputs": {"text": p.negative, "clip": clip}}
    pos, neg = ["pos", 0], ["neg", 0]

    sampler_model = model

    if p.mode == "image":
        if not render_name:
            raise ValueError("image mode requires an uploaded render")
        g["render"] = {"class_type": "LoadImage", "inputs": {"image": render_name}}
        res = max(p.width, p.height)

        if p.depth_strength > 0:
            g["depth_pre"] = {"class_type": "DepthAnythingV2Preprocessor",
                              "inputs": {"image": ["render", 0], "resolution": res}}
            g["cn_depth"] = {"class_type": "ControlNetLoader",
                             "inputs": {"control_net_name": DEPTH_CN}}
            g["apply_depth"] = {"class_type": "ControlNetApplyAdvanced", "inputs": {
                "positive": pos, "negative": neg, "control_net": ["cn_depth", 0],
                "image": ["depth_pre", 0], "strength": p.depth_strength,
                "start_percent": 0.0, "end_percent": 1.0}}
            pos, neg = ["apply_depth", 0], ["apply_depth", 1]

        if p.lineart_strength > 0:
            g["lineart_pre"] = {"class_type": "LineArtPreprocessor",
                                "inputs": {"image": ["render", 0], "resolution": res,
                                           "coarse": "disable"}}
            g["cn_lineart"] = {"class_type": "ControlNetLoader",
                               "inputs": {"control_net_name": LINEART_CN}}
            g["apply_lineart"] = {"class_type": "ControlNetApplyAdvanced", "inputs": {
                "positive": pos, "negative": neg, "control_net": ["cn_lineart", 0],
                "image": ["lineart_pre", 0], "strength": p.lineart_strength,
                "start_percent": 0.0, "end_percent": 1.0}}
            pos, neg = ["apply_lineart", 0], ["apply_lineart", 1]

        g["diff"] = {"class_type": "DifferentialDiffusion", "inputs": {"model": model}}
        sampler_model = ["diff", 0]
        g["encode"] = {"class_type": "VAEEncode", "inputs": {"pixels": ["render", 0], "vae": vae}}
        latent: list = ["encode", 0]
        denoise = p.denoise

        if smap_name:
            g["smap"] = {"class_type": "LoadImage", "inputs": {"image": smap_name}}
            g["smask"] = {"class_type": "ImageToMask", "inputs": {"image": ["smap", 0], "channel": "red"}}
            g["setmask"] = {"class_type": "SetLatentNoiseMask",
                            "inputs": {"samples": ["encode", 0], "mask": ["smask", 0]}}
            latent = ["setmask", 0]
    else:  # text2img
        g["empty"] = {"class_type": "EmptyLatentImage",
                      "inputs": {"width": p.width, "height": p.height, "batch_size": 1}}
        latent = ["empty", 0]
        denoise = 1.0

    g["ksampler"] = {"class_type": "KSampler", "inputs": {
        "model": sampler_model, "seed": p.seed, "steps": p.steps, "cfg": p.cfg,
        "sampler_name": p.sampler, "scheduler": p.scheduler,
        "positive": pos, "negative": neg, "latent_image": latent, "denoise": denoise}}

    g["decode"] = {"class_type": "VAEDecode", "inputs": {"samples": ["ksampler", 0], "vae": vae}}
    g["save"] = {"class_type": "SaveImage", "inputs": {"images": ["decode", 0], "filename_prefix": "calque"}}
    return g
