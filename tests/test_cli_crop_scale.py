"""Tests for the crop-and-scale CLI."""
import pytest
import tempfile
import os
from pathlib import Path
from PIL import Image
import numpy as np


def create_test_image(path, size=(200, 200), content_rect=(50, 50, 150, 150)):
    """Create a test image file."""
    img_array = np.zeros((size[1], size[0], 4), dtype=np.uint8)
    left, top, right, bottom = content_rect
    img_array[top:bottom, left:right] = [255, 100, 100, 255]
    img = Image.fromarray(img_array, "RGBA")
    img.save(path)
    return path


def test_crop_scale_cli_basic(monkeypatch):
    """Test basic CLI functionality."""
    from spritegrid.cli import crop_scale_cli

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test input image
        input_path = os.path.join(tmpdir, "input.png")
        output_path = os.path.join(tmpdir, "output.png")
        create_test_image(input_path)

        # Mock command line arguments
        test_args = [
            "spritegrid-crop",
            input_path,
            "-o",
            output_path,
            "-s",
            "32",
        ]
        monkeypatch.setattr("sys.argv", test_args)

        # Run CLI
        crop_scale_cli()

        # Check output exists
        assert os.path.exists(output_path)

        # Verify output
        output = Image.open(output_path)
        assert output.width <= 32
        assert output.height <= 32


def test_crop_scale_cli_with_padding(monkeypatch):
    """Test CLI with padding option."""
    from spritegrid.cli import crop_scale_cli

    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, "input.png")
        output_path = os.path.join(tmpdir, "output.png")
        create_test_image(input_path)

        test_args = [
            "spritegrid-crop",
            input_path,
            "-o",
            output_path,
            "-s",
            "32",
            "-p",
            "5",
        ]
        monkeypatch.setattr("sys.argv", test_args)

        crop_scale_cli()

        assert os.path.exists(output_path)


def test_crop_scale_cli_centered(monkeypatch):
    """Test CLI with --center option."""
    from spritegrid.cli import crop_scale_cli

    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, "input.png")
        output_path = os.path.join(tmpdir, "output.png")
        create_test_image(input_path)

        test_args = [
            "spritegrid-crop",
            input_path,
            "-o",
            output_path,
            "-s",
            "64",
            "--center",
        ]
        monkeypatch.setattr("sys.argv", test_args)

        crop_scale_cli()

        assert os.path.exists(output_path)

        # With --center, output should be exactly the target size
        output = Image.open(output_path)
        assert output.size == (64, 64)


def test_crop_scale_cli_non_square_size(monkeypatch):
    """Test CLI with WxH size format."""
    from spritegrid.cli import crop_scale_cli

    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, "input.png")
        output_path = os.path.join(tmpdir, "output.png")
        create_test_image(input_path)

        test_args = [
            "spritegrid-crop",
            input_path,
            "-o",
            output_path,
            "-s",
            "64x32",
        ]
        monkeypatch.setattr("sys.argv", test_args)

        crop_scale_cli()

        assert os.path.exists(output_path)

        output = Image.open(output_path)
        assert output.width <= 64
        assert output.height <= 32


def test_crop_scale_cli_no_aspect(monkeypatch):
    """Test CLI with --no-aspect option."""
    from spritegrid.cli import crop_scale_cli

    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, "input.png")
        output_path = os.path.join(tmpdir, "output.png")
        create_test_image(input_path, size=(200, 100))

        test_args = [
            "spritegrid-crop",
            input_path,
            "-o",
            output_path,
            "-s",
            "32",
            "--no-aspect",
        ]
        monkeypatch.setattr("sys.argv", test_args)

        crop_scale_cli()

        assert os.path.exists(output_path)

        # With --no-aspect, should be exactly 32x32 even though input is 2:1
        output = Image.open(output_path)
        assert output.size == (32, 32)


def test_parse_size():
    """Test size string parsing."""
    from spritegrid.cli import parse_size

    # Test single number
    assert parse_size("32") == (32, 32)
    assert parse_size("64") == (64, 64)

    # Test WxH format
    assert parse_size("32x48") == (32, 48)
    assert parse_size("64X32") == (64, 32)  # Case insensitive

    # Test invalid formats
    with pytest.raises(Exception):
        parse_size("invalid")

    with pytest.raises(Exception):
        parse_size("32x48x64")


def test_background_color_parsing(monkeypatch):
    """Test background color parameter parsing."""
    from spritegrid.cli import crop_scale_cli

    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, "input.png")
        output_path = os.path.join(tmpdir, "output.png")
        create_test_image(input_path)

        test_args = [
            "spritegrid-crop",
            input_path,
            "-o",
            output_path,
            "--background-color",
            "255,255,255",
        ]
        monkeypatch.setattr("sys.argv", test_args)

        # Should not crash (even though this feature is not fully implemented yet)
        crop_scale_cli()

        assert os.path.exists(output_path)
