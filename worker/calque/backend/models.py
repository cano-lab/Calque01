"""
Model Loading and Caching
Centralized model management with lazy loading
"""

import torch
from pathlib import Path
from typing import Optional, Dict, Any
from functools import lru_cache

from diffusers import (
    StableDiffusionXLPipeline,
    StableDiffusionXLControlNetPipeline,
    ControlNetModel,
    StableDiffusionUpscalePipeline,
)
from diffusers.schedulers import DPMSolverMultistepScheduler

from .memory import get_device, optimize_pipeline, print_vram_status


class ModelCache:
    """
    Cache for loaded models to avoid reloading.
    Models are loaded on first access and kept in VRAM.
    """

    def __init__(self, models_dir: Path = None):
        self.models_dir = models_dir or Path(r"F:\Software\models")
        self.huggingface_cache = self.models_dir / "huggingface" / "hub"
        self._cache: Dict[str, Any] = {}
        self.device = get_device()

    def _resolve_model_path(self, model_id: str) -> Path:
        """Resolve a model ID to its local path."""
        # Check if it's already a local path
        if Path(model_id).exists():
            return Path(model_id)

        # Check huggingface cache
        for path in self.huggingface_cache.rglob(model_id.replace("--", "--").replace("/", "--")):
            if path.is_dir() and (path / "model_index.json").exists():
                return path

        # Check snapshots directory
        snapshots = self.huggingface_cache / f"models--{model_id.replace('/', '--')}" / "snapshots"
        if snapshots.exists():
            for snapshot in snapshots.iterdir():
                if snapshot.is_dir():
                    return snapshot

        # Return as-is (will download from HF)
        return Path(model_id)

    def get_controlnet(self, model_id: str, force_reload: bool = False) -> ControlNetModel:
        """Load or retrieve a ControlNet model."""
        cache_key = f"controlnet_{model_id}"

        if cache_key in self._cache and not force_reload:
            return self._cache[cache_key]

        print(f"Loading ControlNet: {model_id}")
        print_vram_status("before")

        controlnet = ControlNetModel.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            use_safetensors=True,
        ).to(self.device)

        print_vram_status("after")
        self._cache[cache_key] = controlnet
        return controlnet

    def get_sdxl_pipeline(
        self,
        controlnet: Optional[ControlNetModel] = None,
        force_reload: bool = False
    ) -> StableDiffusionXLControlNetPipeline:
        """
        Load SDXL pipeline with optional ControlNet.
        Uses the inpainting model as base for better quality.
        """
        cache_key = f"sdxl_{id(controlnet)}"

        if cache_key in self._cache and not force_reload:
            return self._cache[cache_key]

        # Use the SDXL inpainting model as base (you have this installed)
        model_id = "stabilityai/stable-diffusion-xl-base-1.0"
        local_path = self._resolve_model_path(model_id)

        print(f"Loading SDXL pipeline from: {local_path}")
        print_vram_status("before")

        if controlnet:
            pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
                str(local_path),
                controlnet=controlnet,
                torch_dtype=torch.float16,
                use_safetensors=True,
            )
        else:
            pipe = StableDiffusionXLPipeline.from_pretrained(
                str(local_path),
                torch_dtype=torch.float16,
                use_safetensors=True,
            )

        # Use DPM++ scheduler for quality
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config
        )

        # Optimize for memory
        pipe = optimize_pipeline(pipe, self.device)

        print_vram_status("after")
        self._cache[cache_key] = pipe
        return pipe

    def unload_model(self, model_type: str = None):
        """
        Unload models from VRAM.

        Args:
            model_type: "controlnet", "pipeline", or None for all
        """
        keys_to_remove = []
        for key in self._cache:
            if model_type is None or key.startswith(model_type):
                keys_to_remove.append(key)

        for key in keys_to_remove:
            del self._cache[key]

        from .memory import clear_vram
        clear_vram(aggressive=True)
        print(f"Unloaded {len(keys_to_remove)} model(s)")


# Global model cache instance
_model_cache: Optional[ModelCache] = None


def get_model_cache() -> ModelCache:
    """Get the global model cache instance."""
    global _model_cache
    if _model_cache is None:
        _model_cache = ModelCache()
    return _model_cache


def load_controlnet(model_id: str = "lllyasviel/control_v11f1p_sd15_depth") -> ControlNetModel:
    """Convenience function to load a ControlNet."""
    return get_model_cache().get_controlnet(model_id)


def load_sdxl_pipeline(
    controlnet: Optional[ControlNetModel] = None
) -> StableDiffusionXLControlNetPipeline:
    """Convenience function to load SDXL pipeline."""
    return get_model_cache().get_sdxl_pipeline(controlnet)


# Available ControlNet models (you have these installed)
AVAILABLE_CONTROLNETS = {
    "depth": "lllyasviel/control_v11f1p_sd15_depth",
    "canny": "lllyasviel/control_v11p_sd15_canny",
    "seg": "lllyasviel/control_v11p_sd15_seg",
    "hed": "lllyasviel/control_v11p_sd15_hed",
    "lineart": "lllyasviel/control_v11p_sd15_lineart",
    "normal": "lllyasviel/control_v11p_sd15_normalbae",
    "tile": "lllyasviel/control_v11f1e_sd15_tile",
    "mlsd": "lllyasviel/control_v11p_sd15_mlsd",
}
