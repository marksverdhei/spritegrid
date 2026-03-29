"""Tests for spritegrid/comfyui/nodes.py — tensor/PIL converters and SpriteGrid node metadata."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from spritegrid.comfyui.nodes import (
    NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS,
    SpriteGrid,
    pil_to_tensor,
    tensor_to_pil,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rgb_tensor(h: int, w: int, r: float = 0.5, g: float = 0.3, b: float = 0.2) -> torch.Tensor:
    """Create a (1, h, w, 3) float32 tensor with uniform colour."""
    t = torch.zeros(1, h, w, 3)
    t[0, :, :, 0] = r
    t[0, :, :, 1] = g
    t[0, :, :, 2] = b
    return t


def _rgba_tensor(h: int, w: int, a: float = 1.0) -> torch.Tensor:
    """Create a (1, h, w, 4) float32 tensor with full alpha."""
    t = torch.zeros(1, h, w, 4)
    t[0, :, :, 3] = a
    return t


# ---------------------------------------------------------------------------
# tensor_to_pil
# ---------------------------------------------------------------------------

class TestTensorToPil:
    def test_rgb_input_returns_rgb_image(self):
        t = _rgb_tensor(8, 8)
        img = tensor_to_pil(t)
        assert img.mode == "RGB"

    def test_rgba_input_returns_rgba_image(self):
        t = _rgba_tensor(8, 8)
        img = tensor_to_pil(t)
        assert img.mode == "RGBA"

    def test_output_size_matches_tensor(self):
        t = _rgb_tensor(16, 32)
        img = tensor_to_pil(t)
        assert img.size == (32, 16)  # PIL size is (width, height)

    def test_pixel_values_scaled_to_0_255(self):
        # Tensor value 1.0 → pixel 255, 0.0 → pixel 0
        t = torch.ones(1, 4, 4, 3)  # all 1.0
        img = tensor_to_pil(t)
        arr = np.array(img)
        assert arr.max() == 255

    def test_zero_tensor_gives_black_image(self):
        t = torch.zeros(1, 4, 4, 3)
        img = tensor_to_pil(t)
        arr = np.array(img)
        assert arr.max() == 0

    def test_returns_pil_image_instance(self):
        t = _rgb_tensor(4, 4)
        img = tensor_to_pil(t)
        assert isinstance(img, Image.Image)

    def test_batch_dim_squeezed(self):
        # Shape (1, h, w, c) should not crash — batch dim is squeezed
        t = _rgb_tensor(4, 4)
        assert t.shape[0] == 1
        img = tensor_to_pil(t)
        assert img.size == (4, 4)

    def test_rgba_alpha_channel_preserved(self):
        t = _rgba_tensor(4, 4, a=0.5)  # alpha=0.5 → pixel ~127
        img = tensor_to_pil(t)
        arr = np.array(img)
        alpha = arr[:, :, 3]
        # 0.5 * 255 clipped to int is either 127 or 128
        assert 126 <= alpha.mean() <= 129


# ---------------------------------------------------------------------------
# pil_to_tensor
# ---------------------------------------------------------------------------

class TestPilToTensor:
    def test_returns_float32_tensor(self):
        img = Image.new("RGB", (8, 8), (128, 64, 32))
        t = pil_to_tensor(img)
        assert t.dtype == torch.float32

    def test_output_shape_is_1_h_w_4(self):
        img = Image.new("RGB", (6, 10), (0, 0, 0))
        t = pil_to_tensor(img)
        assert t.shape == (1, 10, 6, 4)

    def test_rgb_image_converted_to_rgba(self):
        img = Image.new("RGB", (4, 4), (255, 0, 0))
        t = pil_to_tensor(img)
        # RGBA → 4 channels
        assert t.shape[-1] == 4

    def test_values_in_zero_one_range(self):
        img = Image.new("RGB", (4, 4), (200, 100, 50))
        t = pil_to_tensor(img)
        assert t.min().item() >= 0.0
        assert t.max().item() <= 1.0

    def test_white_image_values_near_one(self):
        img = Image.new("RGB", (4, 4), (255, 255, 255))
        t = pil_to_tensor(img)
        # RGB channels of white are all 1.0
        assert torch.allclose(t[0, :, :, :3], torch.ones(4, 4, 3), atol=0.01)

    def test_black_image_rgb_values_zero(self):
        img = Image.new("RGB", (4, 4), (0, 0, 0))
        t = pil_to_tensor(img)
        assert torch.allclose(t[0, :, :, :3], torch.zeros(4, 4, 3), atol=0.01)

    def test_rgba_image_preserved(self):
        img = Image.new("RGBA", (4, 4), (255, 0, 0, 128))
        t = pil_to_tensor(img)
        assert t.shape[-1] == 4
        # Alpha channel ~128/255 ≈ 0.502
        assert abs(t[0, 0, 0, 3].item() - 128 / 255) < 0.01

    def test_size_h_w_order(self):
        img = Image.new("RGB", (12, 8))  # width=12, height=8
        t = pil_to_tensor(img)
        assert t.shape == (1, 8, 12, 4)


# ---------------------------------------------------------------------------
# SpriteGrid node — metadata
# ---------------------------------------------------------------------------

class TestSpriteGridMetadata:
    def test_return_types_is_tuple_with_image(self):
        assert SpriteGrid.RETURN_TYPES == ("IMAGE",)

    def test_function_is_process(self):
        assert SpriteGrid.FUNCTION == "process"

    def test_category_contains_image(self):
        assert "image" in SpriteGrid.CATEGORY.lower()

    def test_input_types_has_required_image(self):
        types = SpriteGrid.INPUT_TYPES()
        assert "required" in types
        assert "image" in types["required"]

    def test_input_types_has_optional_fields(self):
        types = SpriteGrid.INPUT_TYPES()
        assert "optional" in types
        optional = types["optional"]
        assert "min_grid" in optional
        assert "quantize" in optional
        assert "remove_background" in optional
        assert "crop" in optional

    def test_min_grid_default_is_4(self):
        types = SpriteGrid.INPUT_TYPES()
        min_grid = types["optional"]["min_grid"]
        assert min_grid[1]["default"] == 4

    def test_quantize_default_is_8(self):
        types = SpriteGrid.INPUT_TYPES()
        quantize = types["optional"]["quantize"]
        assert quantize[1]["default"] == 8

    def test_remove_background_choices(self):
        types = SpriteGrid.INPUT_TYPES()
        choices = types["optional"]["remove_background"][0]
        assert "none" in choices
        assert "before" in choices
        assert "after" in choices

    def test_crop_default_is_false(self):
        types = SpriteGrid.INPUT_TYPES()
        crop = types["optional"]["crop"]
        assert crop[1]["default"] is False


# ---------------------------------------------------------------------------
# Module-level mappings
# ---------------------------------------------------------------------------

class TestNodeMappings:
    def test_spritegrid_in_class_mappings(self):
        assert "SpriteGrid" in NODE_CLASS_MAPPINGS

    def test_class_mappings_points_to_class(self):
        assert NODE_CLASS_MAPPINGS["SpriteGrid"] is SpriteGrid

    def test_spritegrid_in_display_name_mappings(self):
        assert "SpriteGrid" in NODE_DISPLAY_NAME_MAPPINGS

    def test_display_name_is_string(self):
        assert isinstance(NODE_DISPLAY_NAME_MAPPINGS["SpriteGrid"], str)
