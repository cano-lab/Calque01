"""
Enhanced Watercolor Effect - Simplified and robust
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from typing import Optional


class EnhancedWatercolorEffect:
    """
    Enhanced watercolor painting effect.
    Adds visible brush strokes, color bleeding, and paper texture.
    """

    def __init__(
        self,
        brush_stroke_intensity: float = 0.7,
        brush_size: int = 8,
        color_bleed: float = 0.6,
        wetness: float = 0.5,
        paper_texture: float = 0.4,
        pigment_granulation: float = 0.3
    ):
        self.brush_stroke_intensity = max(0, min(1, brush_stroke_intensity))
        self.brush_size = max(1, min(20, int(brush_size)))
        self.color_bleed = max(0, min(1, color_bleed))
        self.wetness = max(0, min(1, wetness))
        self.paper_texture = max(0, min(1, paper_texture))
        self.pigment_granulation = max(0, min(1, pigment_granulation))

    def apply(self, img: Image.Image) -> Image.Image:
        """Apply enhanced watercolor effect."""
        result = img.copy()

        # Step 1: Color bleeding (wet-on-wet effect)
        if self.color_bleed > 0:
            result = self._apply_color_bleeding(result)

        # Step 2: Brush strokes
        if self.brush_stroke_intensity > 0:
            result = self._apply_brush_strokes(result)

        # Step 3: Pigment granulation
        if self.pigment_granulation > 0:
            result = self._apply_granulation(result)

        # Step 4: Paper texture
        if self.paper_texture > 0:
            result = self._apply_paper_texture(result)

        # Step 5: Soften edges
        result = self._soften_edges(result)

        return result

    def _apply_brush_strokes(self, img: Image.Image) -> Image.Image:
        """Add visible brush stroke texture."""
        arr = np.array(img).astype(float)
        h, w = arr.shape[:2]

        # Create stroke pattern
        stroke_pattern = np.random.randn(h, w) * self.brush_stroke_intensity * 15

        # Apply Gaussian blur for stroke look
        sigma = max(1, self.brush_size / 2)
        kernel_size = int(sigma * 3) * 2 + 1
        blurred = cv2.GaussianBlur(stroke_pattern, (kernel_size, kernel_size), sigma)

        # Apply to each channel
        for c in range(3):
            arr[:, :, c] += blurred * self.brush_stroke_intensity * 0.25

        arr = np.clip(arr, 0, 255).astype(np.uint8)
        return Image.fromarray(arr)

    def _apply_color_bleeding(self, img: Image.Image) -> Image.Image:
        """Apply wet-on-wet color bleeding effect."""
        # Large area bleeding
        if self.wetness > 0.2:
            bleed_size = int(3 + self.wetness * 12)
            img = img.filter(ImageFilter.GaussianBlur(radius=bleed_size * 0.4))

        # Local color mixing with bilateral filter
        arr = np.array(img).astype(float)
        arr_bgr = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_RGB2BGR)

        d = min(15, int(5 + self.color_bleed * 10))
        sigma_color = int(50 + self.color_bleed * 100)
        sigma_space = int(50 + self.color_bleed * 100)

        blended = cv2.bilateralFilter(arr_bgr, d, sigma_color, sigma_space)
        blended_rgb = cv2.cvtColor(blended, cv2.COLOR_BGR2RGB).astype(float)

        # Create organic bleed mask
        h, w = arr.shape[:2]
        noise = np.random.randn(h, w) * 0.3 + 0.5
        kernel_size = 41
        mask = cv2.GaussianBlur(noise, (kernel_size, kernel_size), 0)
        mask = np.clip(mask, 0, 1)
        # Expand to 3 channels
        mask = mask[:, :, np.newaxis]

        # Blend
        bleed_amount = self.color_bleed * 0.6
        result_arr = arr * (1 - mask * bleed_amount) + blended_rgb * mask * bleed_amount

        result_arr = np.clip(result_arr, 0, 255).astype(np.uint8)
        return Image.fromarray(result_arr)

    def _apply_granulation(self, img: Image.Image) -> Image.Image:
        """Add pigment granulation effect."""
        arr = np.array(img).astype(float)
        h, w = arr.shape[:2]

        # Fine grain
        fine_grain = np.random.randn(h, w) * self.pigment_granulation * 12

        # Coarser pigment clumps
        coarse_h, coarse_w = max(1, h // 4), max(1, w // 4)
        coarse = np.random.randn(coarse_h, coarse_w) * self.pigment_granulation * 15
        coarse = cv2.resize(coarse, (w, h), interpolation=cv2.INTER_CUBIC)

        total_grain = fine_grain + coarse

        # More grain in dark areas - expand to match channels
        luminance = np.mean(arr, axis=2) / 255.0
        grain_amount = (1 - luminance) * self.pigment_granulation

        # Expand grain_amount to 3 channels
        if grain_amount.ndim == 2:
            grain_amount = grain_amount[:, :, np.newaxis]

        # Ensure total_grain can broadcast
        if total_grain.ndim == 2:
            total_grain = total_grain[:, :, np.newaxis]

        total_grain = total_grain * grain_amount

        # Add to each channel
        arr = arr + total_grain
        arr = np.clip(arr, 0, 255).astype(np.uint8)

        return Image.fromarray(arr)

    def _apply_paper_texture(self, img: Image.Image) -> Image.Image:
        """Apply watercolor paper texture."""
        arr = np.array(img).astype(float) / 255.0
        h, w = arr.shape[:2]

        # Create paper fiber texture
        texture = np.random.randn(h, w) * 0.1

        # Add vertical fiber pattern
        fibers_v = np.random.randn(h, max(1, w // 10)) * 0.12
        fibers_v = np.repeat(fibers_v, 10, axis=1)
        if fibers_v.shape[1] != w:
            fibers_v = fibers_v[:, :w]
        texture = texture * 0.5 + fibers_v * 0.5

        # Blur for organic feel
        kernel_size = 5
        texture = cv2.GaussianBlur(texture, (kernel_size, kernel_size), 0)

        # Normalize
        texture = (texture - texture.min()) / (texture.max() - texture.min() + 1e-6)

        # Expand texture to match arr channels
        texture = texture[:, :, np.newaxis]

        # Apply using multiply blend
        arr = arr * (1 - texture * self.paper_texture * 0.25)
        arr = arr + texture * self.paper_texture * 0.08

        arr = np.clip(arr, 0, 1)
        return Image.fromarray((arr * 255).astype(np.uint8))

    def _soften_edges(self, img: Image.Image) -> Image.Image:
        """Soften hard edges."""
        arr = np.array(img).astype(float)
        arr_bgr = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_RGB2BGR)

        # Gentle bilateral filtering
        softened = cv2.bilateralFilter(arr_bgr, 7, 70, 70)

        # Find edges to preserve
        gray = cv2.cvtColor(arr_bgr, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 40, 120)
        edges = edges.astype(float) / 255.0

        soft_bgr = cv2.cvtColor(softened, cv2.COLOR_BGR2RGB).astype(float)

        # Ensure edge mask has same shape as arr
        edge_preserve = edges[:, :, np.newaxis]

        # Blend - preserve edges, soften elsewhere
        result = arr * edge_preserve + soft_bgr * (1 - edge_preserve)
        result = np.clip(result, 0, 255).astype(np.uint8)

        return Image.fromarray(result)


def apply_enhanced_watercolor(
    img: Image.Image,
    intensity: float = 1.0,
    brush_strokes: float = 0.7,
    color_bleed: float = 0.6,
    paper_texture: float = 0.4
) -> Image.Image:
    """Quick enhanced watercolor application."""
    effect = EnhancedWatercolorEffect(
        brush_stroke_intensity=brush_strokes * intensity,
        brush_size=int(8),
        color_bleed=color_bleed * intensity,
        wetness=0.5 * intensity,
        paper_texture=paper_texture * intensity,
        pigment_granulation=0.3 * intensity
    )
    return effect.apply(img)
