"""Tests for the SpriteGridAnimation ComfyUI batch node.

Skipped automatically when torch isn't installed (ComfyUI provides it), exactly
like test_comfyui_nodes.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from spritegrid.comfyui.nodes import (  # noqa: E402
    NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS,
    SpriteGridAnimation,
    pils_to_tensor_batch,
    tensor_batch_to_pils,
)


def _batch(n: int = 6, cell: int = 8, cells: int = 10) -> "torch.Tensor":
    frames = []
    for s in range(n):
        w = h = cell * cells
        arr = np.full((h, w, 3), 200, dtype=np.float32)
        arr[::cell, :, :] = 40
        arr[:, ::cell, :] = 40
        rng = np.random.RandomState(s)
        x = rng.randint(0, w - 18)
        y = rng.randint(0, h - 18)
        arr[y:y + 17, x:x + 17] = [220, 30, 30]
        frames.append(arr / 255.0)
    return torch.from_numpy(np.stack(frames))  # (n, h, w, 3)


class TestTensorBatchHelpers:
    def test_batch_to_pils_count_and_size(self):
        pils = tensor_batch_to_pils(_batch(5))
        assert len(pils) == 5
        assert pils[0].size == (80, 80)

    def test_single_image_tensor_treated_as_one_frame(self):
        assert len(tensor_batch_to_pils(torch.zeros(8, 8, 3))) == 1

    def test_pils_to_batch_shape(self):
        pils = [Image.new("RGBA", (10, 10)) for _ in range(4)]
        t = pils_to_tensor_batch(pils)
        assert tuple(t.shape) == (4, 10, 10, 4)


class TestSpriteGridAnimationNode:
    def test_batch_downsampled_and_stable(self):
        (out,) = SpriteGridAnimation().process(_batch(6), min_grid=4, auto_offset=True)
        assert out.shape[0] == 6
        assert tuple(out.shape[1:3]) == (10, 10)

    def test_registered(self):
        assert NODE_CLASS_MAPPINGS["SpriteGridAnimation"] is SpriteGridAnimation
        assert "SpriteGridAnimation" in NODE_DISPLAY_NAME_MAPPINGS

    def test_metadata(self):
        types = SpriteGridAnimation.INPUT_TYPES()
        assert "images" in types["required"]
        assert SpriteGridAnimation.RETURN_TYPES == ("IMAGE",)
        assert "image" in SpriteGridAnimation.CATEGORY.lower()
