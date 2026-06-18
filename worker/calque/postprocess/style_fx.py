"""
Style-Specific Post-Processing Effects
Watercolor, Lithograph, Anime effects
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from typing import Optional, Tuple
from pathlib import Path

from .core import pil_to_cv2, cv2_to_pil, bilateral_filter, blur_edges


class WatercolorEffect:
    """
    Watercolor painting effect.
    Simulates wet-on-wet watercolor with paper texture.
    """

    def __init__(
        self,
        paper_texture_path: Optional[str] = None,
        edge_softening: float = 3.0,
        color_bleed: float = 0.3,
        texture_intensity: float = 0.15
    ):
        """
        Initialize watercolor effect.

        Args:
            paper_texture_path: Path to paper texture image
            edge_softening: Amount of edge softening (0-10)
            color_bleed: Color bleeding/bleeding effect (0-1)
            texture_intensity: Paper texture overlay intensity (0-1)
        """
        self.edge_softening = edge_softening
        self.color_bleed = color_bleed
        self.texture_intensity = texture_intensity

        # Load paper texture if provided
        self.paper_texture = None
        if paper_texture_path and Path(paper_texture_path).exists():
            self.paper_texture = Image.open(paper_texture_path).convert('L')

    def apply(self, img: Image.Image) -> Image.Image:
        """
        Apply watercolor effect to an image.

        Args:
            img: Input PIL Image

        Returns:
            Watercolor-styled PIL Image
        """
        result = img.copy()

        # Step 1: Edge softening with bilateral filter
        if self.edge_softening > 0:
            result = bilateral_filter(
                result,
                d=9,
                sigma_color=75 + self.edge_softening * 10,
                sigma_space=75 + self.edge_softening * 10
            )

        # Step 2: Color bleeding effect (subtle blur + multiply blend)
        if self.color_bleed > 0:
            # Create blurred version for "wet" effect
            wet_layer = result.filter(
                ImageFilter.GaussianBlur(radius=self.color_bleed * 3)
            )
            # Blend using multiply for watercolor look
            arr_result = np.array(result).astype(float) / 255
            arr_wet = np.array(wet_layer).astype(float) / 255
            blended = arr_result * arr_wet * 1.5
            blended = np.clip(blended, 0, 1)
            result = Image.fromarray((blended * 255).astype(np.uint8))

        # Step 3: Paper texture overlay
        if self.paper_texture and self.texture_intensity > 0:
            texture = self.paper_texture.resize(result.size, Image.Resampling.LANCZOS)
            # Convert to grayscale and invert for paper texture
            texture_arr = np.array(texture).astype(float) / 255
            # Multiply blend for texture
            result_arr = np.array(result).astype(float) / 255
            textured = result_arr * (1 - texture_arr * self.texture_intensity)
            textured = np.clip(textured, 0, 1)
            result = Image.fromarray((textured * 255).astype(np.uint8))

        # Step 4: Slightly reduce saturation for watercolor look
        from .core import adjust_saturation
        result = adjust_saturation(result, 0.9)

        return result


class LithographEffect:
    """
    Lithograph/Printmaking effect.
    High contrast, halftone simulation, grain.
    """

    def __init__(
        self,
        contrast: float = 1.5,
        threshold: float = 128,
        halftone_size: int = 4,
        grain_amount: float = 0.1
    ):
        """
        Initialize lithograph effect.

        Args:
            contrast: Contrast boost (1-3)
            threshold: Binarization threshold (0-255)
            halftone_size: Halftone dot size (0-10, 0 = none)
            grain_amount: Film grain amount (0-1)
        """
        self.contrast = contrast
        self.threshold = threshold
        self.halftone_size = halftone_size
        self.grain_amount = grain_amount

    def apply(self, img: Image.Image) -> Image.Image:
        """
        Apply lithograph effect to an image.

        Args:
            img: Input PIL Image

        Returns:
            Lithograph-styled PIL Image
        """
        # Convert to grayscale first
        gray = img.convert('L')

        # Step 1: Boost contrast
        if self.contrast != 1.0:
            arr = np.array(gray).astype(float)
            arr = (arr - 128) * self.contrast + 128
            arr = np.clip(arr, 0, 255).astype(np.uint8)
            gray = Image.fromarray(arr)

        # Step 2: Halftone effect (if enabled)
        if self.halftone_size > 0:
            gray = self._apply_halftone(gray, self.halftone_size)

        # Step 3: Add grain
        if self.grain_amount > 0:
            gray = self._add_grain(gray, self.grain_amount)

        # Convert back to RGB for consistency
        return gray.convert('RGB')

    def _apply_halftone(self, img: Image.Image, dot_size: int) -> Image.Image:
        """Apply halftone dot pattern."""
        arr = np.array(img).astype(float)
        h, w = arr.shape

        # Create halftone pattern
        result = np.zeros_like(arr)

        for y in range(0, h, dot_size):
            for x in range(0, w, dot_size):
                # Get average brightness in this block
                block = arr[y:y+dot_size, x:x+dot_size]
                if block.size == 0:
                    continue
                brightness = np.mean(block)

                # Create dot based on brightness
                center_y, center_x = y + dot_size // 2, x + dot_size // 2
                radius = (brightness / 255) * (dot_size / 2)

                # Draw dot
                yy, xx = np.ogrid[:dot_size, :dot_size]
                yy = yy + y - center_y
                xx = xx + x - center_x
                mask = (xx**2 + yy**2) <= radius**2

                # Apply to block area
                block_y_end = min(y + dot_size, h)
                block_x_end = min(x + dot_size, w)
                mask_h, mask_w = mask.shape[:2]
                result[y:block_y_end, x:block_x_end] = np.maximum(
                    result[y:block_y_end, x:block_x_end],
                    np.where(mask[:block_y_end-y, :block_x_end-x], 255, 0)
                )

        return Image.fromarray(result.astype(np.uint8))

    def _add_grain(self, img: Image.Image, amount: float) -> Image.Image:
        """Add film grain noise."""
        arr = np.array(img).astype(float)
        noise = np.random.randn(*arr.shape) * 255 * amount
        noisy = arr + noise
        noisy = np.clip(noisy, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy)


class AnimeEffect:
    """
    Anime/Cel-shaded effect.
    Flat colors, dark edges, posterized.
    """

    def __init__(
        self,
        edge_thickness: int = 2,
        edge_darkness: float = 0.8,
        color_levels: int = 8,
        saturation: float = 1.2
    ):
        """
        Initialize anime effect.

        Args:
            edge_thickness: Outline thickness (0-5)
            edge_darkness: How dark edges are (0-1)
            color_levels: Number of color levels for posterization
            saturation: Color saturation boost (0.5-2)
        """
        self.edge_thickness = edge_thickness
        self.edge_darkness = edge_darkness
        self.color_levels = color_levels
        self.saturation = saturation

    def apply(self, img: Image.Image) -> Image.Image:
        """
        Apply anime effect to an image.

        Args:
            img: Input PIL Image

        Returns:
            Anime-styled PIL Image
        """
        # Step 1: Posterize colors (reduce palette)
        if self.color_levels > 0:
            img = self._posterize(img, self.color_levels)

        # Step 2: Boost saturation
        from .core import adjust_saturation
        img = adjust_saturation(img, self.saturation)

        # Step 3: Add cel edges
        if self.edge_thickness > 0:
            img = self._add_cel_edges(img)

        return img

    def _posterize(self, img: Image.Image, levels: int) -> Image.Image:
        """Reduce color palette for flat look."""
        arr = np.array(img).astype(float)
        step = 255 / (levels - 1)
        posterized = np.floor(arr / step) * step
        posterized = np.clip(posterized, 0, 255).astype(np.uint8)
        return Image.fromarray(posterized)

    def _add_cel_edges(self, img: Image.Image) -> Image.Image:
        """Add dark cel-shaded edges."""
        # Convert to grayscale for edge detection
        gray = pil_to_cv2(img.convert('L'))

        # Detect edges using Canny
        edges = cv2.Canny(gray, 50, 150)

        # Dilate edges for thickness
        if self.edge_thickness > 1:
            kernel = np.ones((self.edge_thickness, self.edge_thickness), np.uint8)
            edges = cv2.dilate(edges, kernel, iterations=1)

        # Create dark edge overlay
        edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

        # Blend: edges darken the image
        arr_img = np.array(img).astype(float)
        arr_edges = edges_bgr.astype(float)

        # Where edges exist, darken
        edge_mask = (edges > 0).astype(float)[:, :, np.newaxis]
        darkened = arr_img * (1 - edge_mask * self.edge_darkness * 0.5)

        result = np.clip(darkened, 0, 255).astype(np.uint8)
        return Image.fromarray(result)


def apply_watercolor(
    img: Image.Image,
    intensity: float = 1.0,
    paper_texture: Optional[str] = None
) -> Image.Image:
    """
    Quick watercolor effect application.

    Args:
        img: Input image
        intensity: Effect strength (0-1)
        paper_texture: Optional path to paper texture

    Returns:
        Watercolor-styled image
    """
    effect = WatercolorEffect(
        paper_texture_path=paper_texture,
        edge_softening=3.0 * intensity,
        color_bleed=0.3 * intensity,
        texture_intensity=0.15 * intensity
    )
    return effect.apply(img)


def apply_lithograph(
    img: Image.Image,
    intensity: float = 1.0
) -> Image.Image:
    """
    Quick lithograph effect application.

    Args:
        img: Input image
        intensity: Effect strength (0-1)

    Returns:
        Lithograph-styled image
    """
    effect = LithographEffect(
        contrast=1.0 + 0.5 * intensity,
        threshold=128,
        halftone_size=int(4 * intensity),
        grain_amount=0.1 * intensity
    )
    return effect.apply(img)


def apply_anime(
    img: Image.Image,
    intensity: float = 1.0
) -> Image.Image:
    """
    Quick anime effect application.

    Args:
        img: Input image
        intensity: Effect strength (0-1)

    Returns:
        Anime-styled image
    """
    effect = AnimeEffect(
        edge_thickness=int(2 * intensity),
        edge_darkness=0.8,
        color_levels=int(8 * intensity) if intensity > 0 else 0,
        saturation=1.0 + 0.2 * intensity
    )
    return effect.apply(img)
