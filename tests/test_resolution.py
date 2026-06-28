"""Tests for --res and --aspectratio features."""

import pytest
from PIL import Image

from spritegrid.main import apply_resolution, apply_aspect_ratio
from spritegrid.cli import parse_aspect_ratio, parse_size
import argparse


# ---------------------------------------------------------------------------
# parse_aspect_ratio
# ---------------------------------------------------------------------------

class TestParseAspectRatio:
    def test_standard_ratio(self):
        assert parse_aspect_ratio("4:3") == (4, 3)

    def test_widescreen(self):
        assert parse_aspect_ratio("16:9") == (16, 9)

    def test_square(self):
        assert parse_aspect_ratio("1:1") == (1, 1)

    def test_missing_colon_raises(self):
        with pytest.raises(argparse.ArgumentTypeError):
            parse_aspect_ratio("43")

    def test_non_integer_raises(self):
        with pytest.raises(argparse.ArgumentTypeError):
            parse_aspect_ratio("4.5:3")

    def test_zero_value_raises(self):
        with pytest.raises(argparse.ArgumentTypeError):
            parse_aspect_ratio("0:3")

    def test_negative_value_raises(self):
        with pytest.raises(argparse.ArgumentTypeError):
            parse_aspect_ratio("-1:3")


# ---------------------------------------------------------------------------
# apply_resolution
# ---------------------------------------------------------------------------

class TestApplyResolution:
    def test_upscale(self):
        img = Image.new("RGB", (8, 4))
        result = apply_resolution(img, (16, 8))
        assert result.size == (16, 8)

    def test_downscale(self):
        img = Image.new("RGB", (64, 32))
        result = apply_resolution(img, (16, 8))
        assert result.size == (16, 8)

    def test_identity_returns_original(self):
        img = Image.new("RGB", (32, 32))
        result = apply_resolution(img, (32, 32))
        assert result is img  # same object — no copy made

    def test_non_square_target(self):
        img = Image.new("RGB", (32, 32))
        result = apply_resolution(img, (64, 32))
        assert result.size == (64, 32)

    def test_preserves_pixel_values(self):
        """NEAREST resampling should preserve the exact color of the upscaled pixels."""
        img = Image.new("RGB", (1, 1), color=(255, 0, 0))
        result = apply_resolution(img, (4, 4))
        assert result.getpixel((0, 0)) == (255, 0, 0)
        assert result.getpixel((3, 3)) == (255, 0, 0)


# ---------------------------------------------------------------------------
# apply_aspect_ratio
# ---------------------------------------------------------------------------

class TestApplyAspectRatio:
    def test_4_3_on_square(self):
        """4:3 crop on 32x32 should give 32x24."""
        img = Image.new("RGB", (32, 32))
        result = apply_aspect_ratio(img, (4, 3))
        assert result.size == (32, 24)
        assert result.width * 3 == result.height * 4

    def test_16_9_on_square(self):
        """16:9 crop on 32x32 → 32x18."""
        img = Image.new("RGB", (32, 32))
        result = apply_aspect_ratio(img, (16, 9))
        assert result.height == 18

    def test_portrait_crop(self):
        """3:4 on 32x32 should give 24x32."""
        img = Image.new("RGB", (32, 32))
        result = apply_aspect_ratio(img, (3, 4))
        assert result.size == (24, 32)

    def test_already_correct_ratio(self):
        """No crop needed when image already has the right ratio."""
        img = Image.new("RGB", (16, 12))
        result = apply_aspect_ratio(img, (4, 3))
        assert result.size == (16, 12)

    def test_returns_original_when_no_crop_needed(self):
        img = Image.new("RGB", (16, 12))
        result = apply_aspect_ratio(img, (4, 3))
        assert result is img

    def test_crop_is_centered(self):
        """Center crop: pixels at the sides should be removed, not from one end."""
        # 4:1 wide image, crop to 1:1 — should remove equal amounts from left and right
        img = Image.new("RGB", (40, 10), color=(0, 0, 0))
        # Mark left and right edges red
        for y in range(10):
            img.putpixel((0, y), (255, 0, 0))
            img.putpixel((39, y), (255, 0, 0))
        result = apply_aspect_ratio(img, (1, 1))
        # Corners should not be red (they were removed in the crop)
        assert result.getpixel((0, 0)) != (255, 0, 0)
        assert result.getpixel((result.width - 1, 0)) != (255, 0, 0)
