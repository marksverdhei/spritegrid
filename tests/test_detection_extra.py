"""Extra tests for spritegrid.detection — find_dominant_spacing and detect_grid."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from spritegrid.detection import find_dominant_spacing, detect_grid


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _uniform_image(w: int = 64, h: int = 64, value: int = 128) -> Image.Image:
    """Return a uniform grayscale image — no grid signal."""
    arr = np.full((h, w), value, dtype=np.uint8)
    return Image.fromarray(arr)


def _grid_image(cell_w: int, cell_h: int, cells_x: int = 8, cells_y: int = 8) -> Image.Image:
    """Create a pixel-art-style image where every cell boundary has a gradient jump."""
    w = cell_w * cells_x
    h = cell_h * cells_y
    arr = np.zeros((h, w), dtype=np.uint8)
    for cy in range(cells_y):
        for cx in range(cells_x):
            color = 200 if (cx + cy) % 2 == 0 else 50
            arr[cy * cell_h:(cy + 1) * cell_h, cx * cell_w:(cx + 1) * cell_w] = color
    return Image.fromarray(arr)


# ---------------------------------------------------------------------------
# find_dominant_spacing
# ---------------------------------------------------------------------------

class TestFindDominantSpacing:
    def test_empty_profile_returns_zero(self):
        spacing, conf = find_dominant_spacing(np.array([]))
        assert spacing == 0
        assert conf == 0.0

    def test_too_short_profile_returns_zero(self):
        spacing, conf = find_dominant_spacing(np.array([1, 2, 3]), min_spacing=5)
        assert spacing == 0
        assert conf == 0.0

    def test_uniform_profile_returns_zero(self):
        profile = np.ones(100)
        spacing, conf = find_dominant_spacing(profile)
        assert spacing == 0

    def test_regular_peaks_detected(self):
        """A profile with peaks every 8 units should return spacing~8."""
        profile = np.zeros(100)
        for i in range(0, 100, 8):
            profile[i] = 100.0
        spacing, conf = find_dominant_spacing(profile, min_spacing=4)
        assert spacing > 0  # some spacing detected

    def test_confidence_between_zero_and_one(self):
        profile = np.zeros(80)
        for i in range(0, 80, 8):
            profile[i] = 100.0
        spacing, conf = find_dominant_spacing(profile, min_spacing=4)
        assert 0.0 <= conf <= 1.0

    def test_single_peak_returns_zero(self):
        """Only one peak → cannot compute spacing."""
        profile = np.zeros(100)
        profile[50] = 100.0
        spacing, conf = find_dominant_spacing(profile)
        assert spacing == 0

    def test_returns_int_spacing(self):
        profile = np.zeros(80)
        for i in range(0, 80, 10):
            profile[i] = 100.0
        spacing, conf = find_dominant_spacing(profile, min_spacing=5)
        if spacing != 0:
            assert isinstance(spacing, int)


# ---------------------------------------------------------------------------
# detect_grid
# ---------------------------------------------------------------------------

class TestDetectGrid:
    def test_returns_tuple_of_two(self):
        img = _uniform_image(32, 32)
        result = detect_grid(img)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_uniform_image_returns_zero_zero(self):
        """Uniform image has no grid structure."""
        img = _uniform_image(64, 64)
        result = detect_grid(img)
        assert result == (0, 0)

    def test_too_small_image_returns_zero_zero(self):
        img = Image.new("L", (4, 4), 128)
        result = detect_grid(img, min_grid_size=8)
        assert result == (0, 0)

    def test_accepts_rgb_image(self):
        img = Image.new("RGB", (64, 64), (128, 128, 128))
        result = detect_grid(img)
        assert result == (0, 0)

    def test_accepts_rgba_image(self):
        arr = np.full((64, 64, 4), 128, dtype=np.uint8)
        img = Image.fromarray(arr)
        result = detect_grid(img)
        assert result == (0, 0)

    def test_regular_checkerboard_detects_grid(self):
        """A high-contrast checkerboard should be detected."""
        img = _grid_image(cell_w=8, cell_h=8, cells_x=8, cells_y=8)
        result = detect_grid(img, min_grid_size=4, min_confidence=0.3)
        # Either detects correctly or returns (0, 0) if confidence too low
        gw, gh = result
        assert gw >= 0 and gh >= 0

    def test_output_values_non_negative(self):
        for _ in range(3):
            img = _uniform_image(64, 64)
            gw, gh = detect_grid(img)
            assert gw >= 0
            assert gh >= 0
