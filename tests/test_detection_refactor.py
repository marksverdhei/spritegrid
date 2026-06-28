"""Regression + decomposition tests for the detection refactor.

The grid-detection internals were split into compute_gradient_profiles() +
detect_grid_from_profiles() so profiles can be aggregated across animation
frames. These tests lock the existing single-image behaviour and verify the
decomposition is exact.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from spritegrid.detection import (
    compute_gradient_profiles,
    detect_grid,
    detect_grid_from_profiles,
    detect_grid_with_offset,
)


def _make_grid_image(cell_size: int, cells_w: int, cells_h: int,
                     offset_x: int = 0, offset_y: int = 0) -> Image.Image:
    w = cells_w * cell_size + offset_x
    h = cells_h * cell_size + offset_y
    arr = np.full((h, w, 3), 128, dtype=np.uint8)
    for x in range(offset_x, w, cell_size):
        arr[:, x, :] = 0
    for y in range(offset_y, h, cell_size):
        arr[y, :, :] = 0
    return Image.fromarray(arr)


class TestComputeGradientProfiles:
    def test_profile_lengths_match_dimensions(self):
        img = _make_grid_image(8, 10, 6)  # 80 x 48
        profile_h, profile_v = compute_gradient_profiles(img)
        assert len(profile_h) == img.width
        assert len(profile_v) == img.height

    def test_handles_rgba_and_l_modes(self):
        rgba = _make_grid_image(8, 6, 6).convert("RGBA")
        gray = _make_grid_image(8, 6, 6).convert("L")
        for img in (rgba, gray):
            ph, pv = compute_gradient_profiles(img)
            assert len(ph) == img.width and len(pv) == img.height


class TestDecompositionIsExact:
    def test_from_profiles_equals_with_offset(self):
        for cell in (4, 6, 8, 10):
            img = _make_grid_image(cell, 9, 9)
            ph, pv = compute_gradient_profiles(img)
            assert detect_grid_from_profiles(ph, pv) == detect_grid_with_offset(img)

    def test_too_small_profiles_return_zeros(self):
        assert detect_grid_from_profiles(np.zeros(5), np.zeros(5), min_grid_size=4) == (
            0, 0, 0, 0
        )


class TestBackwardCompatibility:
    def test_known_grid_recovered(self):
        img = _make_grid_image(8, 10, 10)
        assert detect_grid_with_offset(img)[:2] == (8, 8)

    def test_detect_grid_wrapper_matches(self):
        img = _make_grid_image(8, 10, 10)
        gw, gh, _ox, _oy = detect_grid_with_offset(img)
        assert detect_grid(img) == (gw, gh)

    def test_uniform_image_returns_zeros(self):
        img = Image.fromarray(np.full((64, 64, 3), 128, dtype=np.uint8))
        assert detect_grid_with_offset(img) == (0, 0, 0, 0)

    def test_tiny_image_returns_zeros(self):
        img = Image.fromarray(np.full((5, 5, 3), 128, dtype=np.uint8))
        assert detect_grid_with_offset(img, min_grid_size=4) == (0, 0, 0, 0)
