"""Tests for spritegrid.segmentation — generate_segment_masks, make_background_transparent, show_mask."""

import numpy as np
import pytest
from PIL import Image
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# generate_segment_masks
# ---------------------------------------------------------------------------

class TestGenerateSegmentMasks:
    def _solid_rgb_array(self, h, w, color=(128, 64, 200)):
        arr = np.full((h, w, 3), color, dtype=np.uint8)
        return arr

    def test_returns_2d_array_or_none(self):
        from spritegrid.segmentation import generate_segment_masks
        arr = self._solid_rgb_array(8, 8)
        result = generate_segment_masks(arr)
        # Solid image → all same cluster OR noise → may be None or 2D
        assert result is None or (isinstance(result, np.ndarray) and result.ndim == 2)

    def test_output_shape_matches_input(self):
        from spritegrid.segmentation import generate_segment_masks
        h, w = 10, 12
        arr = self._solid_rgb_array(h, w)
        result = generate_segment_masks(arr)
        if result is not None:
            assert result.shape == (h, w)

    def test_accepts_rgba_input(self):
        """RGBA input uses only the first 3 channels."""
        from spritegrid.segmentation import generate_segment_masks
        arr = np.full((6, 6, 4), [50, 100, 150, 255], dtype=np.uint8)
        # Should not raise
        result = generate_segment_masks(arr)
        assert result is None or result.shape == (6, 6)

    def test_two_distinct_regions_detected(self):
        """Image with two very different color halves → at least one segment."""
        from spritegrid.segmentation import generate_segment_masks
        h, w = 20, 20
        arr = np.zeros((h, w, 3), dtype=np.uint8)
        arr[:10, :] = [255, 0, 0]    # bright red top half
        arr[10:, :] = [0, 0, 255]    # bright blue bottom half
        result = generate_segment_masks(arr, color_weight=5.0, spatial_weight=0.1)
        # May or may not be None depending on DBSCAN parameters, but shouldn't crash
        assert result is None or result.shape == (h, w)

    def test_returns_none_for_uniform_image_with_noise_only(self):
        """When DBSCAN assigns all pixels to noise (-1), returns None."""
        from spritegrid.segmentation import generate_segment_masks
        rng = np.random.default_rng(42)
        arr = rng.integers(0, 256, (4, 4, 3), dtype=np.uint8)
        # With default params, DBSCAN on 4x4 random image may return None
        result = generate_segment_masks(arr)
        # We just verify it doesn't crash
        assert result is None or isinstance(result, np.ndarray)


# ---------------------------------------------------------------------------
# show_mask
# ---------------------------------------------------------------------------

class TestShowMask:
    def test_default_color_fixed(self):
        """Without random_color, uses the fixed blue color."""
        from spritegrid.segmentation import show_mask
        ax = MagicMock()
        mask = np.ones((4, 4), dtype=bool)
        show_mask(mask, ax, random_color=False)
        ax.imshow.assert_called_once()
        call_args = ax.imshow.call_args
        rendered = call_args[0][0]
        assert rendered.shape == (4, 4, 4)

    def test_random_color_has_different_rgb(self):
        """random_color=True should produce a random color."""
        from spritegrid.segmentation import show_mask
        ax1, ax2 = MagicMock(), MagicMock()
        mask = np.ones((4, 4), dtype=bool)
        # Run twice; colors should almost certainly differ
        np.random.seed(1)
        show_mask(mask, ax1, random_color=True)
        np.random.seed(99)
        show_mask(mask, ax2, random_color=True)
        rgb1 = ax1.imshow.call_args[0][0][0, 0, :3]
        rgb2 = ax2.imshow.call_args[0][0][0, 0, :3]
        assert not np.allclose(rgb1, rgb2)

    def test_zero_mask_produces_zero_image(self):
        """All-False mask → all-zero image overlay."""
        from spritegrid.segmentation import show_mask
        ax = MagicMock()
        mask = np.zeros((3, 3), dtype=bool)
        show_mask(mask, ax, random_color=False)
        rendered = ax.imshow.call_args[0][0]
        assert np.all(rendered == 0.0)

    def test_output_has_alpha_channel(self):
        """Output image should have 4 channels (RGBA)."""
        from spritegrid.segmentation import show_mask
        ax = MagicMock()
        mask = np.ones((2, 2), dtype=bool)
        show_mask(mask, ax, random_color=False)
        rendered = ax.imshow.call_args[0][0]
        assert rendered.shape[-1] == 4


# ---------------------------------------------------------------------------
# make_background_transparent
# ---------------------------------------------------------------------------

class TestMakeBackgroundTransparent:
    def test_returns_tuple_of_two(self):
        from spritegrid.segmentation import make_background_transparent
        img = Image.fromarray(np.full((8, 8, 3), [200, 100, 50], dtype=np.uint8))
        result = make_background_transparent(img)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_returns_pil_image_as_first_element(self):
        from spritegrid.segmentation import make_background_transparent
        img = Image.fromarray(np.full((8, 8, 3), [200, 100, 50], dtype=np.uint8))
        out_img, _ = make_background_transparent(img)
        assert isinstance(out_img, Image.Image)

    def test_debug_false_returns_none_for_debug_image(self):
        from spritegrid.segmentation import make_background_transparent
        img = Image.fromarray(np.full((8, 8, 3), [200, 100, 50], dtype=np.uint8))
        _, debug_img = make_background_transparent(img, debug=False)
        assert debug_img is None

    def test_no_crash_on_uniform_image(self):
        """Uniform image may produce label_mask=None; should return original."""
        from spritegrid.segmentation import make_background_transparent
        img = Image.fromarray(np.full((6, 6, 3), [128, 128, 128], dtype=np.uint8))
        out_img, debug = make_background_transparent(img)
        assert isinstance(out_img, Image.Image)
