"""
spritegrid - Convert AI-generated images to clean pixel art sprites.

Main function:
    process_sprite(image, size=32, remove_bg=True) -> RGBA sprite
"""

from spritegrid.crop_and_scale import (
    process_sprite,
    remove_background,
    crop_to_content,
    scale_nearest,
    center_on_canvas,
    detect_bounds,
    batch_process,
    # Backwards compatibility aliases
    crop_and_scale,
    crop_and_scale_centered,
)

__all__ = [
    "process_sprite",
    "remove_background",
    "crop_to_content",
    "scale_nearest",
    "center_on_canvas",
    "detect_bounds",
    "batch_process",
    # Aliases
    "crop_and_scale",
    "crop_and_scale_centered",
]
