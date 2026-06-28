"""Cross-frame grid detection (detect_grid_across_frames).

The grid lives at fixed pixel positions across frames while content moves, so
summing per-frame gradient profiles reinforces the grid and washes out moving
content. These tests use frames with a *fixed* grid and *moving* content.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from spritegrid.detection import detect_grid_across_frames, detect_grid_with_offset


def _frame(cell: int = 8, cells: int = 10, seed: int = 0,
           block_cells: int = 2) -> Image.Image:
    """A fixed grid with a grid-aligned moving block.

    Real pixel-art animation moves the sprite by whole pixels, i.e. whole cells
    of the upscaled grid, so the moving content's edges land on grid lines. The
    grid sits at fixed positions in every frame; only the block moves.
    """
    w = h = cell * cells
    arr = np.full((h, w, 3), 128, dtype=np.uint8)
    for x in range(0, w, cell):
        arr[:, x, :] = 0   # fixed vertical grid lines
    for y in range(0, h, cell):
        arr[y, :, :] = 0   # fixed horizontal grid lines
    b = block_cells * cell
    bx = ((seed * cell) % (w - b)) // cell * cell
    by = ((seed * 2 * cell) % (h - b)) // cell * cell
    arr[by:by + b, bx:bx + b] = [200, 60, 60]
    return Image.fromarray(arr)


class TestDetectAcrossFrames:
    def test_recovers_shared_grid(self):
        frames = [_frame(8, 10, s) for s in range(8)]
        gw, gh, _ox, _oy = detect_grid_across_frames(frames)
        assert (gw, gh) == (8, 8)

    def test_returns_four_tuple(self):
        frames = [_frame(8, 10, s) for s in range(3)]
        assert len(detect_grid_across_frames(frames)) == 4

    def test_aggregation_across_many_frames(self):
        # Larger moving block across many frames; the shared grid still wins.
        frames = [_frame(8, 12, s, block_cells=3) for s in range(12)]
        assert detect_grid_across_frames(frames)[:2] == (8, 8)

    def test_empty_returns_zeros(self):
        assert detect_grid_across_frames([]) == (0, 0, 0, 0)

    def test_single_frame_matches_still(self):
        f = _frame(8, 10, 0)
        assert detect_grid_across_frames([f]) == detect_grid_with_offset(f)

    def test_mismatched_sizes_raise(self):
        with pytest.raises(ValueError):
            detect_grid_across_frames([_frame(8, 10), _frame(8, 8)])

    def test_offset_in_valid_range(self):
        frames = [_frame(8, 10, s) for s in range(5)]
        gw, gh, ox, oy = detect_grid_across_frames(frames)
        if gw > 0:
            assert 0 <= ox < gw
            assert 0 <= oy < gh
