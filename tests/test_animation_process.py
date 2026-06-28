"""process_frames / process_animation orchestration + temporal stability."""

from __future__ import annotations

import numpy as np
from PIL import Image

from spritegrid.animation import (
    _shared_crop,
    load_frames,
    process_animation,
    process_frames,
    save_frames,
)


def _grid_frame(cell: int = 8, cells: int = 10, seed: int = 0,
                block_cells: int = 2) -> Image.Image:
    """Fixed full-contrast grid with a grid-aligned moving block (RGBA)."""
    w = h = cell * cells
    arr = np.full((h, w, 3), 128, dtype=np.uint8)
    for x in range(0, w, cell):
        arr[:, x, :] = 0
    for y in range(0, h, cell):
        arr[y, :, :] = 0
    b = block_cells * cell
    bx = ((seed * cell) % (w - b)) // cell * cell
    by = ((seed * 2 * cell) % (h - b)) // cell * cell
    arr[by:by + b, bx:bx + b] = [200, 60, 60]
    return Image.fromarray(arr).convert("RGBA")


class TestProcessFrames:
    def test_uniform_output_size(self):
        frames = [_grid_frame(8, 10, s) for s in range(6)]
        out = process_frames(frames, min_grid=4, verbose=False)
        assert len(out) == 6
        assert len({f.size for f in out}) == 1
        assert out[0].size == (10, 10)

    def test_no_grid_passthrough(self):
        frames = [Image.new("RGBA", (40, 40), (100, 150, 200, 255)) for _ in range(4)]
        out = process_frames(frames, min_grid=4, verbose=False)
        assert [f.size for f in out] == [(40, 40)] * 4

    def test_res_constant_across_frames(self):
        frames = [_grid_frame(8, 10, s) for s in range(4)]
        out = process_frames(frames, min_grid=4, res=(32, 32), verbose=False)
        assert all(f.size == (32, 32) for f in out)

    def test_empty(self):
        assert process_frames([], verbose=False) == []


class TestSharedCrop:
    def test_union_box_gives_constant_size(self):
        # Content moves frame to frame; the union box keeps every output the
        # same size (per-frame cropping would make the sprite jitter).
        frames = []
        for s in range(5):
            a = np.zeros((20, 20, 4), dtype=np.uint8)
            a[s:s + 5, s:s + 5] = [255, 0, 0, 255]  # moving opaque block
            frames.append(Image.fromarray(a))
        out = _shared_crop(frames)
        assert len({f.size for f in out}) == 1
        assert out[0].size == (9, 9)  # union spans x/y 0..8 inclusive

    def test_all_transparent_left_unchanged(self):
        frames = [Image.new("RGBA", (12, 12), (0, 0, 0, 0)) for _ in range(3)]
        out = _shared_crop(frames)
        assert [f.size for f in out] == [(12, 12)] * 3


class TestProcessAnimation:
    def test_gif_end_to_end(self, tmp_path):
        frames = [_grid_frame(8, 10, s) for s in range(6)]
        src = str(tmp_path / "in.gif")
        save_frames(frames, src, durations=[100] * 6)
        out_path = str(tmp_path / "out.gif")

        out = process_animation(src, out_path, min_grid=4, auto_offset=True)
        assert len(out) == 6

        reload, _meta = load_frames(out_path)
        assert len(reload) == 6
        assert len({f.size for f in reload}) == 1
        assert reload[0].size == (10, 10)

    def test_single_frame_degrades_to_still(self, tmp_path):
        src = str(tmp_path / "s.png")
        _grid_frame(8, 10, 0).save(src)
        out = process_animation(src, str(tmp_path / "o.png"), min_grid=4)
        assert len(out) == 1
        assert out[0].size == (10, 10)

    def test_fps_sets_output_duration(self, tmp_path):
        frames = [_grid_frame(8, 10, s) for s in range(4)]
        src = str(tmp_path / "in.gif")
        save_frames(frames, src)
        out_path = str(tmp_path / "out.gif")
        process_animation(src, out_path, min_grid=4, fps=20)  # 20fps -> 50ms
        assert abs(Image.open(out_path).info.get("duration", 0) - 50) <= 10
