"""Tests for spritegrid.utils — convert_image_to_ascii, naive_median,
geometric_median, crop_to_content."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from spritegrid.utils import (
    convert_image_to_ascii,
    convert_image_to_halfblock,
    crop_to_content,
    geometric_median,
    naive_median,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rgba(w: int, h: int, color=(255, 0, 0, 255)) -> Image.Image:
    arr = np.full((h, w, 4), color, dtype=np.uint8)
    return Image.fromarray(arr)


def _transparent(w: int, h: int) -> Image.Image:
    arr = np.zeros((h, w, 4), dtype=np.uint8)
    return Image.fromarray(arr)


# ---------------------------------------------------------------------------
# naive_median
# ---------------------------------------------------------------------------

class TestNaiveMedian:
    def test_single_point(self):
        X = np.array([[3.0, 7.0]])
        result = naive_median(X)
        np.testing.assert_allclose(result, [3.0, 7.0])

    def test_two_points(self):
        X = np.array([[0.0, 0.0], [2.0, 4.0]])
        result = naive_median(X)
        np.testing.assert_allclose(result, [1.0, 2.0])

    def test_symmetric_returns_center(self):
        X = np.array([[-1.0], [0.0], [1.0]])
        result = naive_median(X)
        np.testing.assert_allclose(result, [0.0])

    def test_1d_array(self):
        X = np.array([[1.0], [3.0], [5.0], [7.0]])
        result = naive_median(X)
        np.testing.assert_allclose(result, [4.0])

    def test_returns_ndarray(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        assert isinstance(naive_median(X), np.ndarray)


# ---------------------------------------------------------------------------
# geometric_median
# ---------------------------------------------------------------------------

class TestGeometricMedian:
    def test_single_point_returns_itself(self):
        X = np.array([[5.0, 3.0]])
        result = geometric_median(X)
        np.testing.assert_allclose(result, [5.0, 3.0], atol=1e-4)

    def test_symmetric_points_converge_to_center(self):
        X = np.array([
            [-1.0, 0.0], [1.0, 0.0],
            [0.0, -1.0], [0.0, 1.0],
        ])
        result = geometric_median(X)
        np.testing.assert_allclose(result, [0.0, 0.0], atol=0.01)

    def test_result_close_to_true_median(self):
        rng = np.random.default_rng(42)
        X = rng.uniform(-10, 10, (50, 2))
        gm = geometric_median(X)
        nm = naive_median(X)
        # Geometric median should be reasonably close to naive median
        assert np.linalg.norm(gm - nm) < 5.0

    def test_collinear_points(self):
        X = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [10.0, 0.0]])
        result = geometric_median(X)
        # Geometric median is robust to outliers; should be between 1 and 3
        assert 1.0 <= result[0] <= 3.0

    def test_returns_ndarray(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        result = geometric_median(X)
        assert isinstance(result, np.ndarray)


# ---------------------------------------------------------------------------
# convert_image_to_ascii
# ---------------------------------------------------------------------------

class TestConvertImageToAscii:
    def test_single_opaque_pixel_contains_ansi(self):
        img = _rgba(1, 1, (255, 0, 0, 255))
        result = convert_image_to_ascii(img)
        assert "\x1b[48;2;" in result

    def test_single_transparent_pixel_is_space(self):
        img = _transparent(1, 1)
        result = convert_image_to_ascii(img)
        # Should be a space (no ANSI codes) + newline
        assert "\x1b" not in result
        assert " " in result

    def test_output_has_newlines_per_row(self):
        img = _rgba(3, 2)
        result = convert_image_to_ascii(img)
        assert result.count("\n") == 2

    def test_ascii_space_width_one(self):
        img = _rgba(2, 1)
        result_1 = convert_image_to_ascii(img, ascii_space_width=1)
        result_2 = convert_image_to_ascii(img, ascii_space_width=2)
        # Wider spacing means longer string
        assert len(result_2) > len(result_1)

    def test_rgb_color_encoded_in_output(self):
        img = _rgba(1, 1, (10, 20, 30, 255))
        result = convert_image_to_ascii(img)
        assert "10;20;30" in result

    def test_zero_space_width_raises(self):
        img = _rgba(1, 1)
        with pytest.raises(AssertionError):
            convert_image_to_ascii(img, ascii_space_width=0)

    def test_returns_string(self):
        img = _rgba(2, 2)
        assert isinstance(convert_image_to_ascii(img), str)


# ---------------------------------------------------------------------------
# convert_image_to_halfblock
# ---------------------------------------------------------------------------

def _rgba_rows(colors):
    """Build a 1-px-wide image, one row per color in *colors* (RGBA tuples)."""
    arr = np.array([[c] for c in colors], dtype=np.uint8)
    return Image.fromarray(arr)


class TestConvertImageToHalfblock:
    def test_returns_string(self):
        assert isinstance(convert_image_to_halfblock(_rgba(2, 2)), str)

    def test_opaque_cell_has_both_fg_and_bg(self):
        result = convert_image_to_halfblock(_rgba(1, 2, (255, 0, 0, 255)))
        assert "\x1b[38;2;" in result  # foreground (top pixel)
        assert "48;2;" in result       # background (bottom pixel)
        assert "▀" in result

    def test_top_is_fg_bottom_is_bg(self):
        # top row red, bottom row blue -> fg red, bg blue
        img = _rgba_rows([(255, 0, 0, 255), (0, 0, 255, 255)])
        result = convert_image_to_halfblock(img)
        assert "38;2;255;0;0" in result
        assert "48;2;0;0;255" in result

    def test_one_cell_row_per_two_pixel_rows(self):
        # 4 px tall -> 2 half-block rows -> 2 newlines
        assert convert_image_to_halfblock(_rgba(2, 4)).count("\n") == 2

    def test_odd_height_preserves_last_row(self):
        # 3 px tall -> one complete pair plus one top-only row.
        result = convert_image_to_halfblock(_rgba(2, 3, (10, 20, 30, 255)))
        assert result.count("\n") == 2
        assert "0;38;2;10;20;30m▀" in result

    def test_single_row_is_not_dropped(self):
        result = convert_image_to_halfblock(_rgba(1, 1, (10, 20, 30, 255)))
        assert result.count("\n") == 1
        assert "0;38;2;10;20;30m▀" in result

    def test_transparent_pair_is_uncolored_space(self):
        img = _transparent(1, 2)
        result = convert_image_to_halfblock(img)
        assert "38;2;" not in result and "48;2;" not in result
        assert " " in result

    def test_transparent_bottom_uses_default_background(self):
        img = _rgba_rows([(10, 20, 30, 255), (200, 210, 220, 0)])
        result = convert_image_to_halfblock(img)
        assert "0;38;2;10;20;30m▀" in result
        assert "48;2;" not in result
        assert "200;210;220" not in result

    def test_transparent_top_uses_lower_halfblock_and_default_background(self):
        img = _rgba_rows([(200, 210, 220, 0), (10, 20, 30, 255)])
        result = convert_image_to_halfblock(img)
        assert "0;38;2;10;20;30m▄" in result
        assert "48;2;" not in result
        assert "200;210;220" not in result


# ---------------------------------------------------------------------------
# crop_to_content
# ---------------------------------------------------------------------------

class TestCropToContent:
    def test_already_tight_image_unchanged_size(self):
        img = _rgba(4, 4, (255, 0, 0, 255))
        cropped = crop_to_content(img)
        assert cropped.size == (4, 4)

    def test_fully_transparent_returns_original(self):
        img = _transparent(8, 8)
        result = crop_to_content(img)
        assert result.size == img.size

    def test_crops_transparent_border(self):
        arr = np.zeros((10, 10, 4), dtype=np.uint8)
        arr[3:7, 2:8] = [255, 0, 0, 255]
        img = Image.fromarray(arr)
        cropped = crop_to_content(img)
        assert cropped.width == 6
        assert cropped.height == 4

    def test_non_rgba_image_returned_unchanged(self):
        arr = np.full((5, 5, 3), 128, dtype=np.uint8)
        img = Image.fromarray(arr)
        result = crop_to_content(img)
        assert result is img

    def test_single_pixel_content(self):
        arr = np.zeros((10, 10, 4), dtype=np.uint8)
        arr[5, 5] = [255, 255, 255, 255]
        img = Image.fromarray(arr)
        cropped = crop_to_content(img)
        assert cropped.size == (1, 1)

    def test_output_mode_is_rgba(self):
        img = _rgba(4, 4, (0, 255, 0, 255))
        result = crop_to_content(img)
        assert result.mode == "RGBA"
