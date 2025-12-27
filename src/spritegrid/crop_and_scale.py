"""
Crop and scale module for processing AI-generated images into pixel art.

Provides:
- remove_background(): AI-powered background removal using rembg
- crop_to_content(): Crop to non-transparent content
- scale_nearest(): Scale with point sampling for pixel art
- process_sprite(): One-shot pipeline combining all steps
"""

from typing import Optional, Tuple, Union
from PIL import Image
import numpy as np


def remove_background(image: Image.Image) -> Image.Image:
    """
    Remove background from image using AI (rembg/U2-Net).

    Args:
        image: PIL Image (any mode)

    Returns:
        RGBA image with background removed (transparent)
    """
    from rembg import remove
    return remove(image.convert("RGBA"))


def detect_bounds(image: Image.Image, alpha_threshold: int = 10) -> Optional[Tuple[int, int, int, int]]:
    """
    Find bounding box of non-transparent content.

    Args:
        image: RGBA image
        alpha_threshold: Alpha values > this are considered content

    Returns:
        (left, top, right, bottom) or None if no content
    """
    if image.mode != "RGBA":
        image = image.convert("RGBA")

    alpha = np.array(image)[:, :, 3]
    coords = np.where(alpha > alpha_threshold)

    if len(coords[0]) == 0:
        return None

    top, bottom = int(coords[0].min()), int(coords[0].max()) + 1
    left, right = int(coords[1].min()), int(coords[1].max()) + 1
    return (left, top, right, bottom)


def crop_to_content(
    image: Image.Image,
    padding: int = 0,
    alpha_threshold: int = 10,
) -> Image.Image:
    """
    Crop image to its non-transparent content.

    Args:
        image: RGBA image
        padding: Pixels to add around content
        alpha_threshold: Alpha threshold for content detection

    Returns:
        Cropped image
    """
    bounds = detect_bounds(image, alpha_threshold)
    if bounds is None:
        return image

    left, top, right, bottom = bounds
    w, h = image.size

    # Apply padding within bounds
    left = max(0, left - padding)
    top = max(0, top - padding)
    right = min(w, right + padding)
    bottom = min(h, bottom + padding)

    return image.crop((left, top, right, bottom))


def scale_nearest(
    image: Image.Image,
    size: Union[int, Tuple[int, int]],
    square: bool = True,
) -> Image.Image:
    """
    Scale image using NEAREST (point) resampling for pixel art.

    Args:
        image: PIL Image
        size: Target size (int for square, or (w, h) tuple)
        square: If True and size is int, force square output

    Returns:
        Scaled image
    """
    if isinstance(size, int):
        if square:
            target = (size, size)
        else:
            # Fit within size x size maintaining aspect
            w, h = image.size
            ratio = min(size / w, size / h)
            target = (max(1, int(w * ratio)), max(1, int(h * ratio)))
    else:
        target = size

    return image.resize(target, Image.Resampling.NEAREST)


def center_on_canvas(
    image: Image.Image,
    size: Union[int, Tuple[int, int]],
    fill: Tuple[int, ...] = (0, 0, 0, 0),
) -> Image.Image:
    """
    Center image on a canvas of given size.

    Args:
        image: PIL Image to center
        size: Canvas size (int for square, or (w, h) tuple)
        fill: Background color (default transparent)

    Returns:
        Image centered on canvas
    """
    if isinstance(size, int):
        size = (size, size)

    canvas = Image.new("RGBA", size, fill)
    x = (size[0] - image.width) // 2
    y = (size[1] - image.height) // 2

    if image.mode == "RGBA":
        canvas.paste(image, (x, y), image)
    else:
        canvas.paste(image, (x, y))

    return canvas


def process_sprite(
    image: Image.Image,
    size: int = 32,
    remove_bg: bool = True,
    padding: int = 0,
) -> Image.Image:
    """
    Full pipeline: remove background, crop, scale to pixel art size.

    This is the main function for converting AI-generated images to pixel art.

    Args:
        image: Source image (any format)
        size: Target pixel art size (e.g., 16, 32, 64)
        remove_bg: Use AI background removal (requires rembg)
        padding: Padding around content before scaling

    Returns:
        RGBA image at target size

    Example:
        >>> img = Image.open("ai_generated.png")
        >>> sprite = process_sprite(img, size=32)
        >>> sprite.save("sprite_32x32.png")
    """
    # Step 1: Remove background (optional)
    if remove_bg:
        image = remove_background(image)
    elif image.mode != "RGBA":
        image = image.convert("RGBA")

    # Step 2: Crop to content
    cropped = crop_to_content(image, padding=padding)

    # Step 3: Scale to fit in target size (maintain aspect)
    w, h = cropped.size
    ratio = min(size / w, size / h)
    new_size = (max(1, int(w * ratio)), max(1, int(h * ratio)))
    scaled = scale_nearest(cropped, new_size)

    # Step 4: Center on canvas
    result = center_on_canvas(scaled, size)

    return result


# Aliases for backwards compatibility
crop_and_scale = process_sprite
crop_and_scale_centered = process_sprite


def batch_process(
    images: list,
    size: int = 32,
    **kwargs,
) -> list:
    """Process multiple images."""
    return [process_sprite(img, size=size, **kwargs) for img in images]
