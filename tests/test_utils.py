"""Tests for spritegrid.utils — naive_median, geometric_median, convert_image_to_ascii."""

import numpy as np
import pytest
from PIL import Image

from spritegrid.utils import naive_median, geometric_median, convert_image_to_ascii


# ---------------------------------------------------------------------------
# naive_median
# ---------------------------------------------------------------------------

class TestNaiveMedian:
    def test_single_point_returns_itself(self):
        X = np.array([[3.0, 7.0]])
        result = naive_median(X)
        np.testing.assert_allclose(result, [3.0, 7.0])

    def test_odd_count_returns_middle_value(self):
        X = np.array([[1.0], [2.0], [3.0]])
        result = naive_median(X)
        np.testing.assert_allclose(result, [2.0])

    def test_even_count_returns_average_of_middle(self):
        X = np.array([[1.0], [2.0], [3.0], [4.0]])
        result = naive_median(X)
        np.testing.assert_allclose(result, [2.5])

    def test_two_dimensional(self):
        X = np.array([[0.0, 0.0], [2.0, 4.0], [4.0, 8.0]])
        result = naive_median(X)
        np.testing.assert_allclose(result, [2.0, 4.0])

    def test_returns_numpy_array(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = naive_median(X)
        assert isinstance(result, np.ndarray)

    def test_symmetric_distribution_returns_center(self):
        X = np.array([[-5.0], [-1.0], [0.0], [1.0], [5.0]])
        result = naive_median(X)
        np.testing.assert_allclose(result, [0.0])


# ---------------------------------------------------------------------------
# geometric_median
# ---------------------------------------------------------------------------

class TestGeometricMedian:
    def test_single_point_returns_itself(self):
        X = np.array([[3.0, 7.0]])
        result = geometric_median(X)
        np.testing.assert_allclose(result, [3.0, 7.0], atol=1e-4)

    def test_symmetric_1d_returns_center(self):
        X = np.array([[-1.0], [0.0], [1.0]])
        result = geometric_median(X)
        np.testing.assert_allclose(result, [0.0], atol=1e-4)

    def test_all_identical_points(self):
        """When all points are identical, geometric median should be that point."""
        X = np.array([[2.0, 3.0], [2.0, 3.0], [2.0, 3.0]])
        result = geometric_median(X)
        np.testing.assert_allclose(result, [2.0, 3.0], atol=1e-4)

    def test_returns_numpy_array(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        result = geometric_median(X)
        assert isinstance(result, np.ndarray)

    def test_geometric_median_close_to_naive_for_symmetric_data(self):
        """For symmetric data, geometric and naive medians should be close."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((100, 2))
        gm = geometric_median(X)
        nm = naive_median(X)
        # They won't be identical, but should be in the same ballpark
        assert np.linalg.norm(gm - nm) < 1.0

    def test_convergence_for_collinear_points(self):
        """Geometric median should lie between the extremes for collinear data."""
        X = np.array([[0.0], [1.0], [2.0], [3.0], [4.0]])
        result = geometric_median(X)
        assert 0.0 <= result[0] <= 4.0


# ---------------------------------------------------------------------------
# convert_image_to_ascii
# ---------------------------------------------------------------------------

class TestConvertImageToAscii:
    def _solid_rgba(self, w, h, r, g, b, a=255):
        img = Image.new("RGBA", (w, h), (r, g, b, a))
        return img

    def test_returns_string(self):
        img = self._solid_rgba(2, 2, 255, 0, 0)
        result = convert_image_to_ascii(img)
        assert isinstance(result, str)

    def test_fully_transparent_produces_spaces(self):
        img = Image.new("RGBA", (3, 2), (0, 0, 0, 0))
        result = convert_image_to_ascii(img, ascii_space_width=1)
        # Each pixel → space, each row ends with newline
        assert result == "   \n   \n"

    def test_opaque_pixel_has_ansi_escape(self):
        img = self._solid_rgba(1, 1, 255, 128, 0)
        result = convert_image_to_ascii(img, ascii_space_width=1)
        assert "\x1b[48;2;255;128;0m" in result
        assert "\x1b[0m" in result

    def test_width_parameter_multiplies_spaces(self):
        img = self._solid_rgba(1, 1, 255, 0, 0)
        narrow = convert_image_to_ascii(img, ascii_space_width=1)
        wide = convert_image_to_ascii(img, ascii_space_width=3)
        # Wide output should be longer (3 spaces per pixel vs 1)
        assert len(wide) > len(narrow)

    def test_newline_after_each_row(self):
        img = self._solid_rgba(2, 3, 255, 0, 0)
        result = convert_image_to_ascii(img)
        assert result.count("\n") == 3

    def test_rgb_image_without_alpha_treated_as_opaque(self):
        """RGB images (no alpha channel) should render all pixels as opaque."""
        img = Image.new("RGB", (1, 1), (0, 255, 0))
        img = img.convert("RGBA")  # explicit: utils expects RGBA
        result = convert_image_to_ascii(img, ascii_space_width=1)
        assert "\x1b[48;2;0;255;0m" in result

    def test_ascii_space_width_assertion_on_zero(self):
        img = self._solid_rgba(1, 1, 0, 0, 0)
        with pytest.raises(AssertionError):
            convert_image_to_ascii(img, ascii_space_width=0)

    def test_ascii_space_width_assertion_on_none(self):
        img = self._solid_rgba(1, 1, 0, 0, 0)
        with pytest.raises(AssertionError):
            convert_image_to_ascii(img, ascii_space_width=None)
