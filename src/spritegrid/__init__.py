"""
spritegrid - Detect grids in AI pixel art and process AI-generated images into sprites.

Main functions:
    - Grid detection: Use the main CLI (spritegrid) to detect and downsample pixel art grids
    - Sprite processing: Use process_sprite() or crop_and_scale() to convert AI images to pixel art

Sprite processing pipeline:
    1. Remove background (optional, using rembg/U2-Net)
    2. Crop to non-transparent content
    3. Scale to target size with nearest-neighbor sampling
    4. Optionally center on canvas

Example:
    from spritegrid import process_sprite
    from PIL import Image

    img = Image.open("ai_generated.png")
    sprite = process_sprite(img, size=32, remove_bg=True)
    sprite.save("sprite_32x32.png")
"""

from spritegrid.crop_and_scale import (
    process_sprite,
    remove_background,
    crop_to_content,
    crop_and_scale,
    crop_and_scale_centered,
    batch_process,
)

__all__ = [
    "process_sprite",
    "remove_background",
    "crop_to_content",
    "crop_and_scale",
    "crop_and_scale_centered",
    "batch_process",
]
