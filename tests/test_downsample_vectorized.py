"""The vectorized fast path in create_downsampled_image (naive median, bit=8, RGB/RGBA)
must be BIT-EXACT with the straightforward per-cell reference loop, including at the image
edges (where the reference kernel clips instead of duplicating) and with float grid sizes
and negative offsets."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from spritegrid.main import create_downsampled_image


def _reference(img: Image.Image, grid_w, grid_h, cw, ch, ox, oy, kw=3, kh=3):
    """Independent, obviously-correct per-cell naive-median downsample (mirrors the
    original loop): non-duplicating edge slice, np.median, int() truncation."""
    arr = np.array(img)
    H, W = arr.shape[:2]
    C = arr.shape[2]
    hw, hh = kw // 2, kh // 2
    out = np.zeros((ch, cw, C), np.uint8)
    for y in range(ch):
        for x in range(cw):
            cx = min(max(0, int(x * grid_w + grid_w / 2) + ox), W - 1)
            cy = min(max(0, int(y * grid_h + grid_h / 2) + oy), H - 1)
            x0, x1 = max(0, cx - hw), min(W, cx + hw + 1)
            y0, y1 = max(0, cy - hh), min(H, cy + hh + 1)
            px = arr[y0:y1, x0:x1].reshape(-1, C)
            m = np.median(px, axis=0)
            out[y, x] = tuple(int(v) for v in m)
    return out


def _random_image(rng, W, H, mode):
    C = 4 if mode == "RGBA" else 3
    return Image.fromarray(rng.integers(0, 256, (H, W, C), dtype=np.uint8))


@pytest.mark.parametrize("seed", range(30))
def test_fast_path_bit_exact_vs_reference(seed):
    rng = np.random.default_rng(seed)
    mode = "RGBA" if seed % 2 else "RGB"
    W = int(rng.integers(5, 64))
    H = int(rng.integers(5, 64))
    img = _random_image(rng, W, H, mode)
    grid_w = int(rng.integers(3, min(W, 20) + 1))
    grid_h = int(rng.integers(3, min(H, 20) + 1))
    ox = int(rng.integers(-4, 5))
    oy = int(rng.integers(-4, 5))
    cw = max(1, round(W / grid_w))
    ch = max(1, round(H / grid_h))

    got = np.array(create_downsampled_image(img, grid_w, grid_h, cw, ch,
                                            offset_x=ox, offset_y=oy))
    ref = _reference(img, grid_w, grid_h, cw, ch, ox, oy)
    assert np.array_equal(got, ref), f"mismatch seed={seed} mode={mode}"


def test_more_cells_than_pixels_stresses_edges():
    # cw/ch == W/H forces nearly every cell's kernel to clip -> exercises the edge recompute.
    rng = np.random.default_rng(123)
    img = _random_image(rng, 16, 12, "RGBA")
    got = np.array(create_downsampled_image(img, 3, 3, 16, 12))
    ref = _reference(img, 3, 3, 16, 12, 0, 0)
    assert np.array_equal(got, ref)


def test_float_grid_and_negative_offset():
    # Our video pipeline passes fractional grid cells and shifted offsets.
    rng = np.random.default_rng(7)
    img = _random_image(rng, 60, 40, "RGB")
    got = np.array(create_downsampled_image(img, 5.5, 4.5, 24, 18,
                                            offset_x=-3, offset_y=2))
    ref = _reference(img, 5.5, 4.5, 24, 18, -3, 2)
    assert np.array_equal(got, ref)


def test_non_square_kernel_uses_reference_loop_unchanged():
    # (3,5) kernel still goes through the fast path (naive, bit8) -> must match reference.
    rng = np.random.default_rng(9)
    img = _random_image(rng, 40, 40, "RGB")
    got = np.array(create_downsampled_image(img, 6, 6, 6, 6, kernel_size=(3, 5)))
    ref = _reference(img, 6, 6, 6, 6, 0, 0, kw=3, kh=5)
    assert np.array_equal(got, ref)
