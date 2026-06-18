"""
VRAM Memory Management
Optimizations for running on limited GPU memory
"""

import torch
import gc
from typing import Optional


def get_device() -> str:
    """Get the best available device."""
    if torch.cuda.is_available():
        return "cuda"
    # Add MPS (Apple Silicon) support if needed
    # if torch.backends.mps.is_available():
    #     return "mps"
    return "cpu"


def clear_vram(aggressive: bool = False):
    """
    Clear VRAM cache.

    Args:
        aggressive: If True, also run Python garbage collection
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if aggressive:
            torch.cuda.synchronize()

    if aggressive:
        gc.collect()


def get_vram_usage() -> dict:
    """Get current VRAM usage stats."""
    if not torch.cuda.is_available():
        return {"device": "cpu", "used": 0, "total": 0, "free": 0}

    used = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3

    return {
        "device": "cuda",
        "used_gb": round(used, 2),
        "reserved_gb": round(reserved, 2),
        "total_gb": round(total, 2),
        "free_gb": round(total - reserved, 2),
    }


def optimize_pipeline(pipe, device: str = "cuda", enable_slicing: bool = True):
    """
    Apply memory optimizations to a Diffusers pipeline.

    Args:
        pipe: Diffusers pipeline to optimize
        device: Target device
        enable_slicing: Enable attention slicing

    Returns:
        The optimized pipeline
    """
    pipe.to(device)

    # Try xformers for memory efficient attention
    try:
        import xformers
        pipe.enable_xformers_memory_efficient_attention()
        print("    [OK] xformers memory efficient attention enabled")
    except (ImportError, Exception) as e:
        print(f"    [WARN] xformers not available: {e}")

    # Attention slicing for memory efficiency
    if enable_slicing:
        pipe.enable_attention_slicing(slice_size="auto")
        print("    [OK] attention slicing enabled")

    # VAE optimizations
    if hasattr(pipe, 'vae') and pipe.vae is not None:
        try:
            pipe.vae.enable_slicing()
            pipe.vae.enable_tiling()
            print("    [OK] VAE slicing and tiling enabled")
        except Exception as e:
            print(f"    [WARN] VAE optimization failed: {e}")

    # Clear cache
    clear_vram()

    return pipe


def print_vram_status(label: str = ""):
    """Print current VRAM status."""
    if not torch.cuda.is_available():
        return

    stats = get_vram_usage()
    if label:
        print(f"  VRAM [{label}]: {stats['used_gb']}GB / {stats['total_gb']}GB")
