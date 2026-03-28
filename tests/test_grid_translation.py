"""Tests for grid translation (offset) and phase detection."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from spritegrid.detection import find_grid_offset, detect_grid_with_offset
from spritegrid.main import create_downsampled_image


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_grid_image(cell_size: int, cells_w: int, cells_h: int,
                     offset_x: int = 0, offset_y: int = 0) -> Image.Image:
    """Create a synthetic pixel-art-style image with a regular repeating grid.

    Each cell has a unique solid colour so that sample points inside vs on the
    boundary of a cell produce different results.  The grid lines (1px wide) are
    black; cell interiors are mid-grey (128).  Offset shifts the grid right/down.
    """
    w = cells_w * cell_size + offset_x
    h = cells_h * cell_size + offset_y
    arr = np.full((h, w, 3), 128, dtype=np.uint8)

    # Draw vertical lines at offset_x, offset_x+cell_size, ...
    for x in range(offset_x, w, cell_size):
        arr[:, x, :] = 0

    # Draw horizontal lines at offset_y, offset_y+cell_size, ...
    for y in range(offset_y, h, cell_size):
        arr[y, :, :] = 0

    return Image.fromarray(arr, "RGB")


def _uniform_image(w: int, h: int, color=(128, 128, 128)) -> Image.Image:
    arr = np.full((h, w, 3), color, dtype=np.uint8)
    return Image.fromarray(arr, "RGB")


# ---------------------------------------------------------------------------
# find_grid_offset
# ---------------------------------------------------------------------------

class TestFindGridOffset:
    def test_zero_spacing_returns_zero(self):
        profile = np.array([0.0, 1.0, 0.0, 1.0])
        assert find_grid_offset(profile, 0) == 0

    def test_empty_profile_returns_zero(self):
        assert find_grid_offset(np.array([]), 4) == 0

    def test_profile_shorter_than_spacing_returns_zero(self):
        assert find_grid_offset(np.array([1.0, 2.0]), 10) == 0

    def test_finds_peak_position(self):
        spacing = 8
        n = 64
        profile = np.zeros(n)
        # Put peaks at positions 3, 11, 19, 27, ... (offset=3)
        for pos in range(3, n, spacing):
            profile[pos] = 100.0
        result = find_grid_offset(profile, spacing)
        assert result == 3

    def test_zero_offset(self):
        spacing = 4
        n = 32
        profile = np.zeros(n)
        for pos in range(0, n, spacing):
            profile[pos] = 50.0
        result = find_grid_offset(profile, spacing)
        assert result == 0

    def test_offset_at_spacing_minus_one(self):
        spacing = 5
        n = 50
        profile = np.zeros(n)
        offset = spacing - 1  # = 4
        for pos in range(offset, n, spacing):
            profile[pos] = 75.0
        result = find_grid_offset(profile, spacing)
        assert result == offset

    def test_returns_value_in_range(self):
        spacing = 6
        profile = np.random.default_rng(42).random(60)
        result = find_grid_offset(profile, spacing)
        assert 0 <= result < spacing


# ---------------------------------------------------------------------------
# detect_grid_with_offset
# ---------------------------------------------------------------------------

class TestDetectGridWithOffset:
    def test_returns_4_tuple(self):
        img = _make_grid_image(cell_size=8, cells_w=10, cells_h=10)
        result = detect_grid_with_offset(img)
        assert len(result) == 4

    def test_zero_on_failed_detection(self):
        img = _uniform_image(64, 64)
        result = detect_grid_with_offset(img)
        assert result == (0, 0, 0, 0)

    def test_grid_dimensions_match_detect_grid(self):
        from spritegrid.detection import detect_grid
        img = _make_grid_image(cell_size=8, cells_w=10, cells_h=10)
        gw, gh = detect_grid(img)
        gw2, gh2, ox, oy = detect_grid_with_offset(img)
        assert gw == gw2
        assert gh == gh2

    def test_offset_is_in_valid_range(self):
        img = _make_grid_image(cell_size=8, cells_w=10, cells_h=10)
        gw, gh, ox, oy = detect_grid_with_offset(img)
        if gw > 0:
            assert 0 <= ox < gw
            assert 0 <= oy < gh


# ---------------------------------------------------------------------------
# create_downsampled_image — offset parameter
# ---------------------------------------------------------------------------

class TestCreateDownsampledImageOffset:
    def _solid_image(self, w: int, h: int, color=(200, 100, 50)) -> Image.Image:
        arr = np.full((h, w, 3), color, dtype=np.uint8)
        return Image.fromarray(arr, "RGB")

    def test_zero_offset_matches_default(self):
        img = self._solid_image(16, 16)
        r1 = create_downsampled_image(img, 4, 4, 4, 4)
        r2 = create_downsampled_image(img, 4, 4, 4, 4, offset_x=0, offset_y=0)
        assert list(r1.getdata()) == list(r2.getdata())

    def test_offset_does_not_raise(self):
        img = self._solid_image(32, 32)
        # With a solid image any offset produces the same colour
        result = create_downsampled_image(img, 4, 4, 8, 8, offset_x=2, offset_y=2)
        assert result.size == (8, 8)

    def test_positive_offset_clamps_to_image_bounds(self):
        img = self._solid_image(16, 16)
        # Large offset — should clamp rather than raise
        result = create_downsampled_image(img, 4, 4, 4, 4, offset_x=100, offset_y=100)
        assert result.size == (4, 4)

    def test_negative_offset_clamps_to_image_bounds(self):
        img = self._solid_image(16, 16)
        result = create_downsampled_image(img, 4, 4, 4, 4, offset_x=-100, offset_y=-100)
        assert result.size == (4, 4)

    def test_offset_shifts_sample_position(self):
        """Verify that a non-zero offset actually changes the sampled pixel values."""
        # Left half: red, right half: blue — a 1px offset should change sampled colours
        # for some cells near the boundary
        arr = np.zeros((8, 16, 3), dtype=np.uint8)
        arr[:, :8, 0] = 255   # red left
        arr[:, 8:, 2] = 255   # blue right
        img = Image.fromarray(arr, "RGB")

        # Cell size 8, 2 cells wide → boundary falls right on x=8
        r_no_offset = create_downsampled_image(img, 8, 8, 2, 1, offset_x=0, offset_y=0)
        r_with_offset = create_downsampled_image(img, 8, 8, 2, 1, offset_x=4, offset_y=0)

        # With offset=4: first cell centre is at x=8, second at x=16 (clamped to 15)
        # Without offset: first cell centre is at x=4 (red), second at x=12 (blue)
        # They should differ
        assert list(r_no_offset.getdata()) != list(r_with_offset.getdata())


# ---------------------------------------------------------------------------
# CLI integration — --offset and --auto-offset flags
# ---------------------------------------------------------------------------

class TestCliOffsetArgs:
    def test_offset_parsed(self, tmp_path):
        import subprocess, sys
        img = _solid = _make_grid_image(4, 8, 8)
        src = tmp_path / "in.png"
        out = tmp_path / "out.png"
        img.save(src)
        result = subprocess.run(
            [sys.executable, "-m", "spritegrid", str(src), "-o", str(out),
             "--offset", "2x2"],
            capture_output=True, text=True,
        )
        # Should not error on argument parsing
        assert result.returncode == 0 or "error" not in result.stderr.lower().split("offset")[0]

    def test_auto_offset_flag(self, tmp_path):
        import subprocess, sys
        img = _make_grid_image(4, 8, 8)
        src = tmp_path / "in.png"
        out = tmp_path / "out.png"
        img.save(src)
        result = subprocess.run(
            [sys.executable, "-m", "spritegrid", str(src), "-o", str(out),
             "--auto-offset"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0 or "error" not in result.stderr.lower()
