"""Tests for the before/after comparison feature."""

import pytest
from PIL import Image

from spritegrid.main import create_comparison_image


def _solid(width: int, height: int, color=(255, 0, 0)) -> Image.Image:
    return Image.new("RGB", (width, height), color=color)


class TestCreateComparisonImage:
    def test_output_width_is_double_plus_divider(self):
        before = _solid(64, 32)
        after = _solid(8, 4)
        comp = create_comparison_image(before, after, divider_width=2)
        assert comp.width == 64 * 2 + 2

    def test_output_height_includes_label_bar(self):
        before = _solid(64, 32)
        after = _solid(8, 4)
        comp = create_comparison_image(before, after, label_height=16)
        assert comp.height == 32 + 16

    def test_after_panel_is_scaled_to_before_size(self):
        """The after image should be upscaled to the before's dimensions."""
        before = _solid(64, 32, color=(255, 0, 0))
        after = _solid(8, 4, color=(0, 0, 255))
        comp = create_comparison_image(before, after, label_height=0, divider_width=0)
        # Right half should be blue (upscaled after image)
        right_pixel = comp.convert("RGBA").getpixel((64, 0))
        assert right_pixel[2] == 255, "Right panel should be blue (after image)"

    def test_before_panel_appears_on_left(self):
        before = _solid(64, 32, color=(255, 0, 0))
        after = _solid(8, 4, color=(0, 0, 255))
        comp = create_comparison_image(before, after, label_height=0, divider_width=0)
        # Left half should be red (before image)
        left_pixel = comp.convert("RGBA").getpixel((0, 0))
        assert left_pixel[0] == 255, "Left panel should be red (before image)"
        assert left_pixel[2] == 0, "Left panel should not be blue"

    def test_square_before_image(self):
        before = _solid(32, 32)
        after = _solid(16, 16)
        comp = create_comparison_image(before, after, label_height=16, divider_width=2)
        assert comp.size == (32 * 2 + 2, 32 + 16)

    def test_returns_rgba_image(self):
        before = _solid(32, 16)
        after = _solid(4, 2)
        comp = create_comparison_image(before, after)
        assert comp.mode == "RGBA"

    def test_custom_divider_width(self):
        before = _solid(64, 32)
        after = _solid(8, 4)
        for dw in [0, 1, 4, 10]:
            comp = create_comparison_image(before, after, divider_width=dw)
            assert comp.width == 64 * 2 + dw

    def test_custom_label_height(self):
        before = _solid(64, 32)
        after = _solid(8, 4)
        for lh in [0, 8, 24]:
            comp = create_comparison_image(before, after, label_height=lh)
            assert comp.height == 32 + lh

    def test_rgba_input_preserved(self):
        before = Image.new("RGBA", (32, 16), (255, 0, 0, 128))
        after = Image.new("RGBA", (4, 2), (0, 255, 0, 200))
        comp = create_comparison_image(before, after, label_height=0)
        assert comp.mode == "RGBA"
