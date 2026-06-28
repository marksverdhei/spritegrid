"""load_frames / save_frames round-trips for GIF, APNG, frame dirs, and video."""

from __future__ import annotations

import os

import numpy as np
import pytest
from PIL import Image

from spritegrid.animation import is_animated_source, load_frames, save_frames


def _frames(n: int = 5, size: int = 16) -> list[Image.Image]:
    out = []
    for s in range(n):
        arr = np.zeros((size, size, 4), dtype=np.uint8)
        arr[..., 3] = 255
        arr[:, :, 0] = (s * 40) % 256  # distinct red per frame (ordering probe)
        out.append(Image.fromarray(arr))
    return out


class TestGif:
    def test_roundtrip(self, tmp_path):
        p = str(tmp_path / "a.gif")
        save_frames(_frames(5), p, durations=[100] * 5, loop=0)
        frames, meta = load_frames(p)
        assert len(frames) == 5
        assert meta["kind"] == "gif"
        assert len({f.size for f in frames}) == 1

    def test_duration_preserved_within_rounding(self, tmp_path):
        p = str(tmp_path / "a.gif")
        save_frames(_frames(4), p, durations=120)
        # GIF stores centiseconds, so allow +-10ms.
        assert abs(Image.open(p).info.get("duration", 0) - 120) <= 10

    def test_has_global_palette(self, tmp_path):
        p = str(tmp_path / "a.gif")
        save_frames(_frames(5), p)
        assert Image.open(p).getpalette() is not None


class TestApng:
    def test_roundtrip(self, tmp_path):
        p = str(tmp_path / "a.png")
        save_frames(_frames(5), p, durations=120)
        frames, meta = load_frames(p)
        assert len(frames) == 5
        assert meta["kind"] == "apng"


class TestFrameDir:
    def test_natural_sort_roundtrip(self, tmp_path):
        d = tmp_path / "frames"
        d.mkdir()
        for i, f in enumerate(_frames(12)):
            f.save(d / f"frame{i}.png")  # un-padded: lexical sort would misorder
        frames, meta = load_frames(str(d))
        assert len(frames) == 12
        assert meta["kind"] == "frame_dir"
        # frame10 must load at index 10 (natural sort), not index 2 (lexical).
        assert int(np.array(frames[10])[0, 0, 0]) == (10 * 40) % 256

    def test_directory_output(self, tmp_path):
        out = tmp_path / "out"
        save_frames(_frames(5), str(out))
        files = sorted(os.listdir(out))
        assert len(files) == 5
        assert files[0] == "frame_0000.png"


class TestVideo:
    def test_mp4_roundtrip(self, tmp_path):
        cv2 = pytest.importorskip("cv2")
        p = str(tmp_path / "v.mp4")
        probe = cv2.VideoWriter(p, cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (16, 16))
        ok = probe.isOpened()
        probe.release()
        if not ok:
            pytest.skip("No usable mp4 writer in this OpenCV build")
        save_frames(_frames(6, 16), p, durations=[100] * 6)
        frames, meta = load_frames(p)
        assert meta["kind"] == "video"
        assert len(frames) >= 5  # codecs may drop/add a frame; tolerate
        assert frames[0].size == (16, 16)


class TestClassification:
    def test_single_image_is_one_frame(self, tmp_path):
        p = tmp_path / "s.png"
        _frames(1)[0].save(p)
        frames, meta = load_frames(str(p))
        assert len(frames) == 1
        assert meta["kind"] == "single"

    def test_is_animated_source(self, tmp_path):
        gif = str(tmp_path / "a.gif")
        save_frames(_frames(4), gif)
        assert is_animated_source(gif) is True

        still = tmp_path / "s.png"
        _frames(1)[0].save(still)
        assert is_animated_source(str(still)) is False

        assert is_animated_source(str(tmp_path)) is True  # a directory
