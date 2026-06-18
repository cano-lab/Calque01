"""
True Watercolor Effect
Simulates real watercolor: paper showing through, color pooling, dark edges, flowing paint
"""

import cv2
import numpy as np
from PIL import Image, ImageFilter
from typing import Optional


class TrueWatercolorEffect:
    """
    Authentic watercolor simulation.
    Simulates wet-on-wet, wet-on-dry, paper transparency, and pigment flow.
    """

    def __init__(
        self,
        paper_transparency: float = 0.5,    # How much paper shows through
        wash_intensity: float = 0.7,         # Color pooling/gradual washes
        dark_edges: float = 0.6,              # Darker dried edges
        flow_amount: float = 0.5,             # Flowing/running paint
        brush_visibility: float = 0.3         # Subtle brush marks
    ):
        self.paper_transparency = max(0, min(1, paper_transparency))
        self.wash_intensity = max(0, min(1, wash_intensity))
        self.dark_edges = max(0, min(1, dark_edges))
        self.flow_amount = max(0, min(1, flow_amount))
        self.brush_visibility = max(0, min(1, brush_visibility))

    def apply(self, img: Image.Image) -> Image.Image:
        """Apply authentic watercolor effect."""
        arr = np.array(img).astype(float) / 255.0
        h, w = arr.shape[:2]

        # Step 1: Create paper layer
        paper = self._create_paper_layer(h, w)

        # Step 2: Create paint washes with pooling
        washes = self._create_paint_washes(arr)

        # Step 3: Apply flowing paint simulation
        if self.flow_amount > 0:
            washes = self._apply_paint_flow(washes)

        # Step 4: Add dark dried edges (wet edge technique)
        if self.dark_edges > 0:
            washes = self._add_dark_edges(washes, arr)

        # Step 5: Blend paint with paper (transparency)
        result = self._blend_with_paper(washes, paper)

        # Step 6: Add subtle brush strokes
        if self.brush_visibility > 0:
            result = self._add_brush_strokes(result)

        result = np.clip(result, 0, 1)
        return Image.fromarray((result * 255).astype(np.uint8))

    def _create_paper_layer(self, h, w):
        """Create watercolor paper texture."""
        # Base paper - warm white
        paper = np.ones((h, w, 3)) * [0.98, 0.96, 0.94]

        # Add paper fiber texture
        fibers = np.random.randn(h, w) * 0.02
        fibers = cv2.GaussianBlur(fibers, (3, 3), 0)

        # Add vertical paper grain
        grain = np.random.randn(h, w // 8) * 0.015
        grain = np.repeat(grain, 8, axis=1)
        if grain.shape[1] != w:
            grain = grain[:, :w]

        texture = fibers + grain
        texture = texture[:, :, np.newaxis]

        # Texture affects brightness
        paper = paper * (1 + texture)

        # Add subtle cockling (paper warping from water)
        cockling = cv2.GaussianBlur(
            np.random.randn(h, w) * 0.01, (25, 25), 0
        )[:, :, np.newaxis]
        paper = paper + cockling

        return np.clip(paper, 0, 1)

    def _create_paint_washes(self, arr):
        """Create color washes with pooling effect."""
        # Convert to LAB for better color manipulation
        arr_bgr = (arr * 255).astype(np.uint8)
        arr_bgr = cv2.cvtColor((arr[:, :, ::-1] * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        lab = cv2.cvtColor(arr_bgr, cv2.COLOR_BGR2LAB).astype(float) / 255.0

        # Create wash patterns - areas where paint pools
        l, a, b = lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]

        # Pooling happens in darker/mid-tone areas
        # Create pool mask based on luminance
        pool_mask = 1.0 - l  # Darker areas pool more
        pool_mask = cv2.GaussianBlur(pool_mask, (15, 15), 0)
        pool_mask = np.clip(pool_mask * self.wash_intensity, 0, 1)

        # Expand pooling slightly (paint spreads)
        if self.wash_intensity > 0.3:
            kernel_size = int(5 + self.wash_intensity * 15)
            pool_mask = cv2.GaussianBlur(pool_mask, (kernel_size, kernel_size), 0)

        # Apply pooling - colors intensify and spread in pooled areas
        pool_factor = 1 + pool_mask * 0.4 * self.wash_intensity

        # Increase color saturation in pooled areas
        a = a * pool_factor
        b = b * pool_factor

        # Reduce luminance slightly in pooled areas (paint is darker when wet)
        l = l * (1 - pool_mask * 0.1 * self.wash_intensity)

        # Soften the transitions (watercolor bleeds)
        l = cv2.bilateralFilter(
            (l * 255).astype(np.uint8), 9, 80, 80
        ).astype(float) / 255.0

        lab_new = np.stack([l, a, b], axis=2)
        lab_new = np.clip(lab_new, 0, 1)

        # Convert back to RGB
        result_bgr = cv2.cvtColor(
            (lab_new * 255).astype(np.uint8), cv2.COLOR_LAB2BGR
        ).astype(float) / 255.0

        return result_bgr[:, :, ::-1]  # BGR to RGB

    def _apply_paint_flow(self, arr):
        """Simulate paint flowing on wet paper."""
        # Use morphological operations to simulate flow
        result = arr.copy()

        for c in range(3):
            channel = result[:, :, c]

            # Paint flows downward due to gravity
            # Create flow kernels
            flow_kernel = np.array([
                [0.1, 0.2, 0.1],
                [0.2, 0.0, 0.2],
                [0.1, 0.3, 0.1]
            ]) / 2.0

            # Apply flow multiple times for gradual effect
            for _ in range(int(self.flow_amount * 5)):
                channel = cv2.filter2D(channel, -1, flow_kernel)

            # Color diffusion (paint spreads)
            if self.flow_amount > 0.3:
                diffusion_size = int(3 + self.flow_amount * 5)
                channel = cv2.GaussianBlur(channel, (diffusion_size, diffusion_size), 0)

            result[:, :, c] = channel

        return result

    def _add_dark_edges(self, washes, original):
        """Add darker dried edges (wet edge technique)."""
        # Find edges
        gray = cv2.cvtColor(
            (original * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY
        )

        # Detect edges at multiple scales
        edges_small = cv2.Canny(gray, 30, 80)
        edges_medium = cv2.Canny(gray, 50, 120)
        edges_large = cv2.Canny(gray, 70, 160)

        # Blend edge scales
        edges = (edges_small * 0.5 + edges_medium * 0.3 + edges_large * 0.2)
        edges = edges.astype(float) / 255.0

        # Dilate edges slightly (paint pools at edges)
        if self.dark_edges > 0.5:
            kernel = np.ones((3, 3), np.uint8)
            edges = cv2.dilate(edges, kernel, iterations=1)

        # Blur edges for natural look
        edges = cv2.GaussianBlur(edges, (5, 5), 0)

        # Darken the washes at edge locations
        edge_darkening = edges[:, :, np.newaxis] * self.dark_edges * 0.4
        result = washes * (1 - edge_darkening)

        return result

    def _blend_with_paper(self, paint, paper):
        """Blend paint layer with paper, showing paper through."""
        # Paper shows through more in light areas
        paint_luminance = np.mean(paint, axis=2)
        transparency = self.paper_transparency * (0.3 + paint_luminance * 0.7)

        # Add variation for realism
        noise = np.random.randn(*paint_luminance.shape) * 0.1
        transparency = np.clip(transparency + noise * 0.2, 0, 1)
        transparency = transparency[:, :, np.newaxis]

        # Blend: paper shows through transparent areas
        result = paint * (1 - transparency) + paper * transparency

        return result

    def _add_brush_strokes(self, arr):
        """Add subtle directional brush strokes."""
        h, w = arr.shape[:2]

        # Create stroke texture
        strokes = np.random.randn(h, w) * 0.02 * self.brush_visibility

        # Directional blur for stroke direction (slightly diagonal)
        kernel = np.zeros((5, 5))
        for i in range(5):
            kernel[i, max(0, i-1):i+2] = 1.0
        kernel = kernel / kernel.sum()

        strokes = cv2.filter2D(strokes, -1, kernel)

        # Apply strokes
        result = arr + strokes[:, :, np.newaxis]

        return result


def apply_true_watercolor(
    img: Image.Image,
    intensity: float = 1.0,
    transparency: float = 0.5,
    washes: float = 0.7,
    edges: float = 0.6,
    flow: float = 0.5
) -> Image.Image:
    """Quick true watercolor application."""
    effect = TrueWatercolorEffect(
        paper_transparency=transparency * intensity,
        wash_intensity=washes * intensity,
        dark_edges=edges * intensity,
        flow_amount=flow * intensity,
        brush_visibility=0.3 * intensity
    )
    return effect.apply(img)
