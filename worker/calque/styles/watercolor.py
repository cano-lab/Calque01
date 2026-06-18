"""
Watercolor Style Pipeline
Transforms architectural renders into watercolor paintings
"""

import torch
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Optional, Callable

from diffusers import (
    StableDiffusionXLControlNetPipeline,
    ControlNetModel,
    DPMSolverMultistepScheduler,
)
from diffusers.utils import load_image

from ..backend.models import get_model_cache, AVAILABLE_CONTROLNETS
from ..backend.memory import clear_vram, print_vram_status
from ..postprocess.style_fx import WatercolorEffect, apply_watercolor
from ..postprocess.core import sharpen, adjust_saturation, resize


class WatercolorPipeline:
    """
    Watercolor enhancement pipeline for architectural images.

    Uses ControlNet for structure preservation while applying
    watercolor styling through prompts and post-processing.
    """

    def __init__(
        self,
        model_id: str = "stabilityai/stable-diffusion-xl-base-1.0",
        controlnet_id: str = None,
        device: str = "cuda",
    ):
        """
        Initialize watercolor pipeline.

        Args:
            model_id: Base SDXL model ID
            controlnet_id: ControlNet model (default: depth)
            device: Device to run on
        """
        self.device = device
        self.cache = get_model_cache()

        # Default to depth controlnet for structure
        if controlnet_id is None:
            controlnet_id = AVAILABLE_CONTROLNETS["depth"]

        print("Loading Watercolor Pipeline...")
        print_vram_status("start")

        # Load ControlNet
        self.controlnet = self.cache.get_controlnet(controlnet_id)

        # Load SDXL pipeline with ControlNet
        self.pipe = self.cache.get_sdxl_pipeline(self.controlnet)

        # Configure scheduler for quality
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            self.pipe.scheduler.config
        )

        print_vram_status("loaded")
        print("Watercolor Pipeline ready!")

    def enhance(
        self,
        image_path: str,
        depth_path: Optional[str] = None,
        output_path: Optional[str] = None,
        prompt: str = "watercolor painting, architectural sketch, soft edges, artistic brush strokes, painted on paper",
        negative_prompt: str = "photorealistic, sharp edges, digital art, 3D render, hyperrealistic, detailed",
        strength: float = 0.5,
        controlnet_scale: float = 0.8,
        guidance_scale: float = 7.5,
        num_steps: int = 25,
        seed: Optional[int] = None,
        width: int = 1024,
        height: int = 1024,
        postprocess: bool = True,
        watercolor_intensity: float = 0.7,
        progress_callback: Optional[Callable] = None,
    ) -> Image.Image:
        """
        Enhance an image with watercolor style.

        Args:
            image_path: Path to input image
            depth_path: Optional path to depth map (auto-generated if None)
            output_path: Optional path to save result
            prompt: Positive prompt for generation
            negative_prompt: Negative prompt
            strength: How much to transform (0-1, higher = more stylized)
            controlnet_scale: ControlNet influence (0-2, higher = more structure)
            guidance_scale: CFG scale for prompt adherence
            num_steps: Number of diffusion steps
            seed: Random seed for reproducibility
            width: Output width
            height: Output height
            postprocess: Apply watercolor post-processing
            watercolor_intensity: Post-processing effect strength
            progress_callback: Optional callback for progress updates

        Returns:
            Enhanced PIL Image
        """
        # Load input image
        print(f"Loading input: {image_path}")
        input_image = load_image(image_path).convert("RGB")
        input_image = resize(input_image, (width, height))

        # Load or generate depth control
        if depth_path and Path(depth_path).exists():
            print(f"Loading depth: {depth_path}")
            control_image = load_image(depth_path).convert("RGB")
        else:
            print("No depth map provided, using input as control")
            control_image = input_image

        control_image = resize(control_image, (width, height))

        # Set up generator
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        clear_vram()
        print("Generating watercolor style...")

        # Run diffusion
        result = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=input_image,
            control_image=control_image,
            strength=strength,
            controlnet_conditioning_scale=controlnet_scale,
            guidance_scale=guidance_scale,
            num_inference_steps=num_steps,
            generator=generator,
            callback_on_step_end=lambda step, timestep, kwargs: self._progress(
                step, num_steps, progress_callback
            ) if progress_callback else None,
        ).images[0]

        print("Generation complete!")

        # Apply watercolor post-processing
        if postprocess:
            print("Applying watercolor post-processing...")
            result = apply_watercolor(result, intensity=watercolor_intensity)

            # Subtle sharpen to bring back some detail
            result = sharpen(result, amount=0.2, radius=0.8)

        # Save if path provided
        if output_path:
            result.save(output_path)
            print(f"Saved to: {output_path}")

        return result

    def _progress(self, step: int, total: int, callback: Callable):
        """Internal progress callback."""
        if callback:
            callback(step + 1, total, "Watercolor")

    def enhance_selective(
        self,
        image_path: str,
        depth_path: str,
        building_prompt: str = "watercolor painting, architectural rendering, building",
        environment_prompt: str = "watercolor painting, soft sky, artistic background",
        building_strength: float = 0.3,  # Preserve building
        environment_strength: float = 0.7,  # Creative freedom
        building_control: float = 1.0,  # Strict structure
        environment_control: float = 0.4,  # Loose structure
        depth_threshold: float = 0.7,
        **kwargs
    ) -> Image.Image:
        """
        Selective enhancement: strict on building, creative on environment.

        Uses depth to separate foreground (building) from background (sky/environment).

        Args:
            image_path: Path to input image
            depth_path: Path to depth map
            building_prompt: Prompt for building area
            environment_prompt: Prompt for environment area
            building_strength: Transformation strength for building
            environment_strength: Transformation strength for environment
            building_control: ControlNet strength for building
            environment_control: ControlNet strength for environment
            depth_threshold: Depth threshold for separation (0-1)
            **kwargs: Additional arguments for enhance()

        Returns:
            Enhanced PIL Image with selective styling
        """
        print("Running selective watercolor enhancement...")

        # Load images
        input_image = load_image(image_path).convert("RGB")
        depth_image = load_image(depth_path).convert("L")

        width, height = input_image.size
        input_image = resize(input_image, (1024, 1024))
        depth_image = resize(depth_image, (1024, 1024))

        # Create building mask from depth
        depth_arr = np.array(depth_image).astype(float) / 255.0
        building_mask = (depth_arr > depth_threshold).astype(float)

        # Feather mask edges
        from scipy.ndimage import gaussian_filter
        building_mask = gaussian_filter(building_mask, sigma=10)
        building_mask = np.clip(building_mask, 0, 1)

        # Generate building version (strict)
        print("  Pass 1: Building (strict preservation)...")
        result_building = self.enhance(
            image_path=image_path,
            depth_path=depth_path,
            prompt=building_prompt,
            strength=building_strength,
            controlnet_scale=building_control,
            **kwargs
        )

        # Generate environment version (creative)
        print("  Pass 2: Environment (creative freedom)...")
        result_env = self.enhance(
            image_path=image_path,
            depth_path=depth_path,
            prompt=environment_prompt,
            strength=environment_strength,
            controlnet_scale=environment_control,
            seed=kwargs.get('seed', 0) + 1000 if kwargs.get('seed') else None,
            **{k: v for k, v in kwargs.items() if k != 'seed'}
        )

        # Blend based on depth mask
        print("  Blending results...")
        building_arr = np.array(result_building).astype(float)
        env_arr = np.array(result_env).astype(float)
        mask_3ch = building_mask[:, :, np.newaxis]

        blended = building_arr * mask_3ch + env_arr * (1 - mask_3ch)
        blended = np.clip(blended, 0, 255).astype(np.uint8)

        return Image.fromarray(blended)


def quick_watercolor(
    image_path: str,
    output_path: str = None,
    depth_path: str = None,
    intensity: float = 0.7,
    style: str = "painting"
) -> Image.Image:
    """
    Quick one-function watercolor enhancement.

    Args:
        image_path: Input image path
        output_path: Output path (optional)
        depth_path: Depth map path (optional)
        intensity: Effect strength (0-1)
        style: Style variant - "painting", "sketch", or "washed"

    Returns:
        Enhanced PIL Image
    """
    # Style presets
    prompts = {
        "painting": "watercolor painting, architectural, artistic brush strokes, soft edges",
        "sketch": "watercolor sketch, architectural drawing, ink and wash, line art",
        "washed": "light watercolor wash, subtle, soft colors, gentle painting",
    }

    pipeline = WatercolorPipeline()

    result = pipeline.enhance(
        image_path=image_path,
        depth_path=depth_path,
        prompt=prompts.get(style, prompts["painting"]),
        strength=0.4 + intensity * 0.3,
        controlnet_scale=0.7 + intensity * 0.3,
        watercolor_intensity=intensity,
    )

    if output_path:
        result.save(output_path)

    return result


# Example usage
if __name__ == "__main__":
    # Test the pipeline
    import sys

    if len(sys.argv) > 1:
        input_path = sys.argv[1]
        output = input_path.rsplit('.', 1)[0] + "_watercolor.png"
    else:
        input_path = "test_image.png"
        output = "test_watercolor.png"

    print(f"Processing: {input_path}")
    result = quick_watercolor(input_path, output, intensity=0.7)
    print(f"Done! Saved to: {output}")
