"""
ComfyUI custom node for SpriteGrid.

Detects pixel grids in AI-generated pixel art and downsamples to clean sprites.
"""

import numpy as np
import torch
from PIL import Image

from ..detection import detect_grid
from ..main import create_downsampled_image
from ..crop_and_scale import crop_to_content
from ..segmentation import make_background_transparent


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert ComfyUI IMAGE tensor to PIL Image."""
    img = tensor.squeeze(0).cpu().numpy()
    img = (img * 255).clip(0, 255).astype(np.uint8)
    if img.shape[-1] == 4:
        return Image.fromarray(img, mode="RGBA")
    return Image.fromarray(img, mode="RGB")


def pil_to_tensor(pil_img: Image.Image) -> torch.Tensor:
    """Convert PIL Image to ComfyUI IMAGE tensor."""
    if pil_img.mode != "RGBA":
        pil_img = pil_img.convert("RGBA")
    img = np.array(pil_img).astype(np.float32) / 255.0
    return torch.from_numpy(img).unsqueeze(0)


class SpriteGrid:
    """
    Detect pixel grid in AI-generated pixel art and downsample to clean sprites.

    Mirrors the spritegrid CLI: detects the underlying pixel grid, downsamples
    using median color sampling, optionally removes background and crops.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "optional": {
                "min_grid": ("INT", {
                    "default": 4,
                    "min": 1,
                    "max": 32,
                    "step": 1,
                }),
                "quantize": ("INT", {
                    "default": 8,
                    "min": 4,
                    "max": 8,
                    "step": 1,
                }),
                "remove_background": (["none", "before", "after"], {"default": "none"}),
                "crop": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process"
    CATEGORY = "image/sprite"

    def process(self, image, min_grid=4, quantize=8, remove_background="none", crop=False):
        pil_img = tensor_to_pil(image)

        # Remove background before detection if requested
        if remove_background == "before":
            pil_img, _ = make_background_transparent(pil_img, debug=False)

        # Detect grid
        grid_w, grid_h = detect_grid(pil_img, min_grid_size=min_grid)

        if grid_w <= 0 or grid_h <= 0:
            # No grid detected - image is likely already clean pixel art
            return (pil_to_tensor(pil_img),)

        # Calculate cells
        num_cells_w = max(1, round(pil_img.width / grid_w))
        num_cells_h = max(1, round(pil_img.height / grid_h))

        # Idempotence check: if output matches input dimensions, image is already clean
        if num_cells_w == pil_img.width and num_cells_h == pil_img.height:
            result = pil_img
        else:
            # Downsample
            result = create_downsampled_image(
                pil_img,
                grid_w,
                grid_h,
                num_cells_w,
                num_cells_h,
                bit=quantize,
            )

        # Remove background after if requested
        if remove_background == "after":
            result, _ = make_background_transparent(result, debug=False)

        # Crop if requested
        if crop and result.mode == "RGBA":
            result = crop_to_content(result)

        return (pil_to_tensor(result),)


NODE_CLASS_MAPPINGS = {
    "SpriteGrid": SpriteGrid,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SpriteGrid": "SpriteGrid",
}
