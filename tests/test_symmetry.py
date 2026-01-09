import numpy as np
import pytest
from PIL import Image

from spritegrid.utils import enforce_symmetry


class TestEnforceSymmetry:
    """Tests for the enforce_symmetry function."""

    def test_already_symmetric_unchanged(self):
        """Symmetric image should remain unchanged."""
        # Create a symmetric 4x4 RGBA image
        arr = np.array([
            [[255, 0, 0, 255], [0, 255, 0, 255], [0, 255, 0, 255], [255, 0, 0, 255]],
            [[0, 0, 255, 255], [255, 255, 0, 255], [255, 255, 0, 255], [0, 0, 255, 255]],
            [[0, 0, 255, 255], [255, 255, 0, 255], [255, 255, 0, 255], [0, 0, 255, 255]],
            [[255, 0, 0, 255], [0, 255, 0, 255], [0, 255, 0, 255], [255, 0, 0, 255]],
        ], dtype=np.uint8)
        img = Image.fromarray(arr, 'RGBA')

        result = enforce_symmetry(img)
        result_arr = np.array(result)

        # Should be identical
        assert np.array_equal(arr, result_arr)

    def test_asymmetric_becomes_symmetric(self):
        """Asymmetric image should become symmetric after enforcement."""
        # Create asymmetric image: left side has strong colors, right has weak
        arr = np.array([
            [[255, 0, 0, 255], [0, 0, 0, 255], [128, 128, 128, 255], [100, 100, 100, 255]],
            [[0, 255, 0, 255], [255, 255, 255, 255], [200, 200, 200, 255], [150, 150, 150, 255]],
        ], dtype=np.uint8)
        img = Image.fromarray(arr, 'RGBA')

        result = enforce_symmetry(img)
        result_arr = np.array(result)

        # Check result is symmetric
        h, w = result_arr.shape[:2]
        for y in range(h):
            for x in range(w // 2):
                assert np.array_equal(
                    result_arr[y, x],
                    result_arr[y, w - 1 - x]
                ), f"Pixel ({x}, {y}) not symmetric with ({w-1-x}, {y})"

    def test_transparent_pixels_low_confidence(self):
        """Transparent pixels should have low confidence and defer to opaque."""
        # Left side transparent, right side opaque red
        arr = np.array([
            [[0, 0, 0, 0], [0, 0, 0, 0], [255, 0, 0, 255], [255, 0, 0, 255]],
            [[0, 0, 0, 0], [0, 0, 0, 0], [255, 0, 0, 255], [255, 0, 0, 255]],
        ], dtype=np.uint8)
        img = Image.fromarray(arr, 'RGBA')

        result = enforce_symmetry(img)
        result_arr = np.array(result)

        # Right side (opaque) should win, so left becomes red too
        # Check symmetry
        h, w = result_arr.shape[:2]
        for y in range(h):
            for x in range(w // 2):
                assert np.array_equal(
                    result_arr[y, x],
                    result_arr[y, w - 1 - x]
                )

    def test_high_saturation_wins(self):
        """Higher saturation pixel should win over gray."""
        # Left: saturated red, Right: gray
        arr = np.array([
            [[255, 0, 0, 255], [128, 128, 128, 255]],
        ], dtype=np.uint8)
        img = Image.fromarray(arr, 'RGBA')

        result = enforce_symmetry(img)
        result_arr = np.array(result)

        # Red (higher saturation) should win
        assert result_arr[0, 0, 0] == 255  # Red channel
        assert result_arr[0, 1, 0] == 255  # Mirror should also be red

    def test_extreme_brightness_wins(self):
        """Very dark or very bright pixels should win over middle gray."""
        # Left: black (extreme), Right: middle gray
        arr = np.array([
            [[0, 0, 0, 255], [128, 128, 128, 255]],
        ], dtype=np.uint8)
        img = Image.fromarray(arr, 'RGBA')

        result = enforce_symmetry(img)
        result_arr = np.array(result)

        # Black (further from middle gray) should win
        assert result_arr[0, 0, 0] == 0
        assert result_arr[0, 1, 0] == 0

    def test_odd_width_center_pixel_unchanged(self):
        """Center pixel in odd-width image should remain unchanged."""
        arr = np.array([
            [[255, 0, 0, 255], [0, 255, 0, 255], [0, 0, 255, 255]],
        ], dtype=np.uint8)
        img = Image.fromarray(arr, 'RGBA')

        result = enforce_symmetry(img)
        result_arr = np.array(result)

        # Center pixel (green) should be unchanged
        assert np.array_equal(result_arr[0, 1], [0, 255, 0, 255])

    def test_rgb_mode_works(self):
        """Should work with RGB images (no alpha)."""
        arr = np.array([
            [[255, 0, 0], [128, 128, 128]],
        ], dtype=np.uint8)
        img = Image.fromarray(arr, 'RGB')

        result = enforce_symmetry(img)
        result_arr = np.array(result)

        # Should be symmetric
        assert np.array_equal(result_arr[0, 0], result_arr[0, 1])

    def test_preserves_image_dimensions(self):
        """Output should have same dimensions as input."""
        arr = np.zeros((10, 20, 4), dtype=np.uint8)
        img = Image.fromarray(arr, 'RGBA')

        result = enforce_symmetry(img)

        assert result.width == 20
        assert result.height == 10


class TestEnforceSymmetryIntegration:
    """Integration tests for symmetry with realistic pixel art patterns."""

    def test_eye_pattern_symmetric(self):
        """Eye-like pattern should become symmetric."""
        # Simulate a face with slightly asymmetric eyes
        # Left eye darker, right eye slightly lighter (artifact)
        arr = np.zeros((5, 6, 4), dtype=np.uint8)
        # Background
        arr[:, :, :3] = 200
        arr[:, :, 3] = 255
        # Left eye (black)
        arr[1:3, 1, :3] = 0
        # Right eye (dark gray - artifact)
        arr[1:3, 4, :3] = 30

        img = Image.fromarray(arr, 'RGBA')
        result = enforce_symmetry(img)
        result_arr = np.array(result)

        # Check eyes are now symmetric
        assert np.array_equal(result_arr[1, 1], result_arr[1, 4])
        assert np.array_equal(result_arr[2, 1], result_arr[2, 4])
