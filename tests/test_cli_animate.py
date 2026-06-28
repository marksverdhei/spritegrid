"""CLI auto-routing of animations through the `spritegrid` command.

Markus chose auto-routing (no separate command): `spritegrid` detects a
multi-frame input and runs the animation pipeline; stills are unaffected.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest
from PIL import Image, ImageSequence

from spritegrid import cli as cli_mod
from spritegrid.animation import is_animated_source, save_frames


def _grid_frames(n: int = 6, cell: int = 8, cells: int = 10) -> list[Image.Image]:
    out = []
    for s in range(n):
        w = h = cell * cells
        arr = np.full((h, w, 3), 128, dtype=np.uint8)
        for x in range(0, w, cell):
            arr[:, x, :] = 0
        for y in range(0, h, cell):
            arr[y, :, :] = 0
        b = 2 * cell
        bx = ((s * cell) % (w - b)) // cell * cell  # grid-aligned moving block
        by = ((s * 2 * cell) % (h - b)) // cell * cell
        arr[by:by + b, bx:bx + b] = [200, 60, 60]
        out.append(Image.fromarray(arr))
    return out


def _run_cli(argv: list[str]) -> None:
    old = sys.argv
    sys.argv = ["spritegrid"] + argv
    try:
        cli_mod.cli()
    finally:
        sys.argv = old


class TestAutoRoute:
    def test_gif_routes_to_animation(self, tmp_path):
        src = tmp_path / "in.gif"
        save_frames(_grid_frames(6), str(src), durations=[100] * 6)
        out = tmp_path / "out.gif"

        _run_cli([str(src), "-o", str(out), "--auto-offset"])

        assert out.exists()
        frames = list(ImageSequence.Iterator(Image.open(out)))
        assert len(frames) == 6
        assert len({f.size for f in frames}) == 1

    def test_still_png_uses_single_image_path(self, tmp_path):
        src = tmp_path / "s.png"
        _grid_frames(1)[0].save(src)
        out = tmp_path / "out.png"

        _run_cli([str(src), "-o", str(out)])

        assert out.exists()
        assert Image.open(out).size == (10, 10)  # single downsampled sprite


class TestTimingFlags:
    def test_fps_and_duration_are_mutually_exclusive(self, tmp_path):
        src = tmp_path / "in.gif"
        save_frames(_grid_frames(4), str(src))
        with pytest.raises(SystemExit):
            _run_cli([str(src), "-o", str(tmp_path / "o.gif"),
                      "--fps", "10", "--duration", "100"])


class TestClassifier:
    def test_directory_is_animated(self, tmp_path):
        d = tmp_path / "frames"
        d.mkdir()
        for i, f in enumerate(_grid_frames(3)):
            f.save(d / f"frame{i}.png")
        assert is_animated_source(str(d)) is True
