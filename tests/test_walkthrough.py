"""Trace integrity and CLI equivalence for optional Manim walkthroughs."""

from __future__ import annotations

import builtins

import numpy as np
import pytest
from PIL import Image

from spritegrid.detection import analyze_grid
from spritegrid.main import main
from spritegrid.walkthrough import (
    WalkthroughTrace,
    _require_manim,
    record_grid_discovery,
    record_sampling,
)


def _grid_image(cell: int = 6, cells: int = 6) -> Image.Image:
    data = np.zeros((cell * cells, cell * cells, 3), dtype=np.uint8)
    for y in range(cells):
        for x in range(cells):
            data[y * cell : (y + 1) * cell, x * cell : (x + 1) * cell] = (
                220 if (x + y) % 2 else 25,
                40 + x * 20,
                40 + y * 20,
            )
    return Image.fromarray(data)


def test_trace_snapshots_do_not_alias_pipeline_images():
    image = Image.new("RGB", (4, 4), "red")
    trace = WalkthroughTrace("memory")
    trace.add("load", "load", image)

    image.paste("blue", (0, 0, 4, 4))

    assert trace.steps[0].image.getpixel((0, 0)) == (255, 0, 0)


def test_grid_and_sampling_trace_exact_detector_data():
    source = _grid_image()
    output = source.resize((6, 6), Image.Resampling.NEAREST)
    analysis = analyze_grid(source)
    trace = WalkthroughTrace("memory")

    record_grid_discovery(trace, source, analysis, 0, 0)
    record_sampling(trace, source, output, 6, 6, 1, 2)

    grid, sampling = trace.steps
    assert grid.details["analysis"] is analysis
    assert grid.details["analysis"].result == analysis.result
    assert (grid.details["applied_offset_x"], grid.details["applied_offset_y"]) == (
        0,
        0,
    )
    assert sampling.details["representative_center"] == (22, 23)
    assert sampling.details["kernel_box"] == (21, 22, 24, 25)
    assert sampling.details["sampled_color"] == output.getpixel((3, 3))


def test_walkthrough_output_is_pixel_identical(monkeypatch, tmp_path):
    source = tmp_path / "source.png"
    ordinary = tmp_path / "ordinary.png"
    illustrated = tmp_path / "illustrated.png"
    video = tmp_path / "walkthrough.mp4"
    _grid_image().save(source)

    captured = {}

    def fake_render(trace, output_path):
        captured["trace"] = trace
        captured["path"] = output_path

    monkeypatch.setattr("spritegrid.walkthrough.render_walkthrough", fake_render)

    main(str(source), output_file=str(ordinary), res=(48, 48))
    main(
        str(source),
        output_file=str(illustrated),
        res=(48, 48),
        walkthrough_video=str(video),
    )

    with Image.open(ordinary) as plain, Image.open(illustrated) as with_walkthrough:
        assert plain.mode == with_walkthrough.mode
        assert plain.size == with_walkthrough.size
        np.testing.assert_array_equal(np.asarray(plain), np.asarray(with_walkthrough))

    assert captured["path"] == str(video)
    assert [step.kind for step in captured["trace"].steps] == [
        "load",
        "grid",
        "sampling",
        "resize",
        "final",
    ]


def test_missing_manim_error_explains_optional_extra(monkeypatch):
    real_import = builtins.__import__

    def without_manim(name, *args, **kwargs):
        if name == "manim":
            raise ImportError("not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_manim)

    with pytest.raises(RuntimeError, match=r"spritegrid\[walkthrough\]"):
        _require_manim()
