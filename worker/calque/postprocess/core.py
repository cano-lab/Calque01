"""
Core Post-Processing Operations
Basic image processing operations for all styles
"""

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from typing import Tuple, Optional, Union


def load_image(path: str) -> Image.Image:
    """Load image from path."""
    return Image.open(path).convert("RGB")


def pil_to_cv2(img: Image.Image) -> np.ndarray:
    """Convert PIL Image to OpenCV format (BGR)."""
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def cv2_to_pil(img: np.ndarray) -> Image.Image:
    """Convert OpenCV format (BGR) to PIL Image."""
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def sharpen(
    img: Image.Image,
    amount: float = 0.3,
    radius: float = 1.0
) -> Image.Image:
    """
    Apply sharpening to an image.

    Args:
        img: Input PIL Image
        amount: Sharpening strength (0-1)
        radius: Sharpening radius

    Returns:
        Sharpened PIL Image
    """
    # Convert to numpy for unsharp mask
    arr = np.array(img).astype(float)

    # Gaussian blur for unsharp mask
    blurred = cv2.GaussianBlur(arr, (0, 0), radius)
    sharpened = arr + (arr - blurred) * amount

    # Clip and convert back
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
    return Image.fromarray(sharpened)


def denoise(
    img: Image.Image,
    strength: float = 10,
    h_color: float = 10
) -> Image.Image:
    """
    Apply fast photo denoising.

    Args:
        img: Input PIL Image
        strength: Filter strength (higher = more blur)
        h_color: Color filter strength

    Returns:
        Denoised PIL Image
    """
    arr = pil_to_cv2(img)
    denoised = cv2.fastNlMeansDenoisingColored(
        arr,
        None,
        h_color,
        strength,
        7,
        21
    )
    return cv2_to_pil(denoised)


def adjust_brightness_contrast(
    img: Image.Image,
    brightness: float = 1.0,
    contrast: float = 1.0
) -> Image.Image:
    """
    Adjust brightness and contrast.

    Args:
        img: Input PIL Image
        brightness: Brightness multiplier (1.0 = no change)
        contrast: Contrast multiplier (1.0 = no change)

    Returns:
        Adjusted PIL Image
    """
    arr = np.array(img).astype(float)

    # Apply contrast first
    if contrast != 1.0:
        arr = (arr - 128) * contrast + 128

    # Then brightness
    if brightness != 1.0:
        arr = arr * brightness

    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def adjust_saturation(
    img: Image.Image,
    saturation: float = 1.0
) -> Image.Image:
    """
    Adjust color saturation.

    Args:
        img: Input PIL Image
        saturation: Saturation multiplier (1.0 = no change)

    Returns:
        Adjusted PIL Image
    """
    enhancer = ImageEnhance.Color(img)
    return enhancer.enhance(saturation)


def adjust_gamma(
    img: Image.Image,
    gamma: float = 1.0
) -> Image.Image:
    """
    Apply gamma correction.

    Args:
        img: Input PIL Image
        gamma: Gamma value (<1 brightens, >1 darkens)

    Returns:
        Adjusted PIL Image
    """
    arr = np.array(img).astype(float) / 255.0
    arr = np.power(arr, gamma)
    arr = (arr * 255).astype(np.uint8)
    return Image.fromarray(arr)


def bilateral_filter(
    img: Image.Image,
    d: int = 9,
    sigma_color: float = 75,
    sigma_space: float = 75
) -> Image.Image:
    """
    Apply bilateral filter for edge-preserving smoothing.
    Great for watercolor effect!

    Args:
        img: Input PIL Image
        d: Diameter of pixel neighborhood
        sigma_color: Filter sigma in color space
        sigma_space: Filter sigma in coordinate space

    Returns:
        Smoothed PIL Image
    """
    arr = pil_to_cv2(img)
    smoothed = cv2.bilateralFilter(arr, d, sigma_color, sigma_space)
    return cv2_to_pil(smoothed)


def blur_edges(
    img: Image.Image,
    radius: float = 2.0
) -> Image.Image:
    """
    Soften edges while keeping overall image sharp.
    Good for watercolor/painting effects.

    Args:
        img: Input PIL Image
        radius: Blur radius

    Returns:
        PIL Image with softened edges
    """
    return img.filter(ImageFilter.GaussianBlur(radius=radius))


def enhance_details(
    img: Image.Image,
    amount: float = 0.5
) -> Image.Image:
    """
    Enhance fine details using unsharp masking.

    Args:
        img: Input PIL Image
        amount: Enhancement amount (0-1)

    Returns:
        Enhanced PIL Image
    """
    arr = np.array(img).astype(float)

    # Create Gaussian-blurred version
    kernel_size = int(5 * amount) + 1
    blurred = cv2.GaussianBlur(arr, (kernel_size, kernel_size), 0)

    # Unsharp mask
    enhanced = arr + (arr - blurred) * (1 + amount * 2)
    enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)

    return Image.fromarray(enhanced)


def resize(
    img: Image.Image,
    size: Tuple[int, int],
    resample: int = Image.Resampling.LANCZOS
) -> Image.Image:
    """
    Resize image to specified dimensions.

    Args:
        img: Input PIL Image
        size: (width, height) tuple
        resample: Resampling method

    Returns:
        Resized PIL Image
    """
    return img.resize(size, resample)


def blend_images(
    img1: Image.Image,
    img2: Image.Image,
    alpha: float = 0.5,
    mask: Optional[Image.Image] = None
) -> Image.Image:
    """
    Blend two images together.

    Args:
        img1: First PIL Image
        img2: Second PIL Image (must be same size as img1)
        alpha: Blend factor (0 = img1, 1 = img2)
        mask: Optional mask for selective blending

    Returns:
        Blended PIL Image
    """
    if img1.size != img2.size:
        img2 = img2.resize(img1.size, Image.Resampling.LANCZOS)

    if mask is None:
        # Simple alpha blend
        result = Image.blend(img1, img2, alpha)
    else:
        # Mask-based blend
        if mask.size != img1.size:
            mask = mask.resize(img1.size, Image.Resampling.LANCZOS)

        # Convert mask to grayscale if needed
        if mask.mode != 'L':
            mask = mask.convert('L')

        result = Image.composite(img2, img1, mask)

    return result
