"""Tests for spritegrid.main draw_grid_overlay and create_downsampled_image.

Covers grid line drawing, downsampling with median kernels, and edge cases.
Uses small synthetic images — no file I/O or GPU.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from spritegrid.main import draw_grid_overlay, create_downsampled_image


# ---------------------------------------------------------------------------
# draw_grid_overlay
# ---------------------------------------------------------------------------

class TestDrawGridOverlay:
    def _solid(self, w=100, h=100, color=(128, 128, 128)):
        return Image.new("RGB", (w, h), color)

    def test_returns_image(self):
        result = draw_grid_overlay(self._solid(), 10, 10)
        assert isinstance(result, Image.Image)

    def test_size_preserved(self):
        img = self._solid(200, 150)
        result = draw_grid_overlay(img, 20, 15)
        assert result.size == (200, 150)

    def test_original_not_modified(self):
        img = self._solid()
        original_data = np.array(img).copy()
        draw_grid_overlay(img, 10, 10)
        np.testing.assert_array_equal(np.array(img), original_data)

    def test_grid_lines_drawn(self):
        img = self._solid(100, 100, (255, 255, 255))
        result = draw_grid_overlay(img, 50, 50, color="red")
        # At x=50, there should be a red pixel
        pixel = result.getpixel((50, 25))
        assert pixel[0] > 200  # red channel high
        assert pixel[1] < 50   # green channel low

    def test_invalid_grid_returns_copy(self):
        img = self._solid()
        result = draw_grid_overlay(img, 0, 10)
        assert result.size == img.size

    def test_negative_grid_returns_copy(self):
        img = self._solid()
        result = draw_grid_overlay(img, -5, 10)
        assert result.size == img.size

    def test_rgba_input_converted(self):
        img = Image.new("RGBA", (100, 100), (128, 128, 128, 255))
        result = draw_grid_overlay(img, 10, 10)
        assert result.mode == "RGB"

    def test_custom_color(self):
        img = self._solid(100, 100, (0, 0, 0))
        result = draw_grid_overlay(img, 50, 50, color="blue")
        pixel = result.getpixel((50, 25))
        assert pixel[2] > 200  # blue channel high

    def test_line_width(self):
        img = self._solid(100, 100, (255, 255, 255))
        result = draw_grid_overlay(img, 50, 50, color="red", line_width=3)
        # Wider line should affect adjacent pixels too
        pixel_center = result.getpixel((50, 25))
        pixel_adj = result.getpixel((51, 25))
        assert pixel_center[0] > 200
        assert pixel_adj[0] > 200  # adjacent pixel also red due to width=3


# ---------------------------------------------------------------------------
# create_downsampled_image
# ---------------------------------------------------------------------------

class TestCreateDownsampledImage:
    def _uniform(self, w=100, h=100, color=(200, 100, 50)):
        return Image.new("RGB", (w, h), color)

    def test_returns_image(self):
        result = create_downsampled_image(
            self._uniform(), grid_w=10, grid_h=10,
            num_cells_w=10, num_cells_h=10
        )
        assert isinstance(result, Image.Image)

    def test_output_size_matches_cells(self):
        result = create_downsampled_image(
            self._uniform(100, 80), grid_w=10, grid_h=10,
            num_cells_w=10, num_cells_h=8
        )
        assert result.size == (10, 8)

    def test_uniform_image_preserves_color(self):
        img = self._uniform(100, 100, (200, 100, 50))
        result = create_downsampled_image(
            img, grid_w=10, grid_h=10,
            num_cells_w=10, num_cells_h=10, bit=8
        )
        pixel = result.getpixel((5, 5))
        # Should be close to original (quantization may shift slightly)
        assert abs(pixel[0] - 200) < 5
        assert abs(pixel[1] - 100) < 5
        assert abs(pixel[2] - 50) < 5

    def test_even_kernel_raises(self):
        with pytest.raises(ValueError, match="odd"):
            create_downsampled_image(
                self._uniform(), grid_w=10, grid_h=10,
                num_cells_w=10, num_cells_h=10, kernel_size=(2, 2)
            )

    def test_zero_grid_raises(self):
        with pytest.raises(ValueError):
            create_downsampled_image(
                self._uniform(), grid_w=0, grid_h=10,
                num_cells_w=10, num_cells_h=10
            )

    def test_kernel_larger_than_grid_raises(self):
        with pytest.raises(ValueError, match="cannot be larger"):
            create_downsampled_image(
                self._uniform(), grid_w=2, grid_h=2,
                num_cells_w=50, num_cells_h=50, kernel_size=(5, 5)
            )

    def test_bit_quantization_reduces_colors(self):
        # Create image with many colors
        arr = np.random.default_rng(42).integers(0, 255, (100, 100, 3), dtype=np.uint8)
        img = Image.fromarray(arr)
        result = create_downsampled_image(
            img, grid_w=10, grid_h=10,
            num_cells_w=10, num_cells_h=10, bit=2
        )
        # With 2-bit quantization, max value per channel is 3
        result_arr = np.array(result)
        # Values should be quantized to steps of 85 (255/3)
        unique_values = np.unique(result_arr)
        assert len(unique_values) <= 4 * 3  # at most 4 levels per channel

    def test_naive_median_type(self):
        result = create_downsampled_image(
            self._uniform(), grid_w=10, grid_h=10,
            num_cells_w=10, num_cells_h=10, median_type="naive"
        )
        assert result.size == (10, 10)

    def test_geometric_median_type(self):
        result = create_downsampled_image(
            self._uniform(), grid_w=10, grid_h=10,
            num_cells_w=10, num_cells_h=10, median_type="geometric"
        )
        assert result.size == (10, 10)
