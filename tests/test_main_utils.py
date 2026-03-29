"""Tests for spritegrid.main utility functions: load_image, handle_txt, handle_png, create_downsampled_image."""

import os
import sys
from io import BytesIO
from unittest.mock import patch, MagicMock

import numpy as np
import pytest
from PIL import Image


# ---------------------------------------------------------------------------
# load_image
# ---------------------------------------------------------------------------

class TestLoadImage:
    def test_load_local_file(self, tmp_path):
        from spritegrid.main import load_image
        img_path = str(tmp_path / "test.png")
        Image.new("RGB", (4, 4), (255, 0, 0)).save(img_path)
        result = load_image(img_path)
        assert isinstance(result, Image.Image)
        assert result.size == (4, 4)

    def test_missing_local_file_returns_none(self, tmp_path, capsys):
        from spritegrid.main import load_image
        result = load_image(str(tmp_path / "nonexistent.png"))
        assert result is None

    def test_url_loads_image(self):
        from spritegrid.main import load_image
        mock_response = MagicMock()
        mock_response.headers = {"content-type": "image/png"}
        # Create a fake PNG in memory
        buf = BytesIO()
        Image.new("RGB", (2, 2), (0, 255, 0)).save(buf, format="PNG")
        mock_response.content = buf.getvalue()
        mock_response.raise_for_status = MagicMock()

        with patch("requests.get", return_value=mock_response):
            result = load_image("http://example.com/img.png")
        assert isinstance(result, Image.Image)

    def test_url_wrong_content_type_returns_none(self, capsys):
        from spritegrid.main import load_image
        mock_response = MagicMock()
        mock_response.headers = {"content-type": "text/html"}
        mock_response.raise_for_status = MagicMock()

        with patch("requests.get", return_value=mock_response):
            result = load_image("http://example.com/page.html")
        assert result is None


# ---------------------------------------------------------------------------
# handle_txt
# ---------------------------------------------------------------------------

class TestHandleTxt:
    def test_writes_content_to_file(self, tmp_path):
        from spritegrid.main import handle_txt
        path = str(tmp_path / "output.txt")
        handle_txt("hello world", path)
        assert open(path).read() == "hello world"

    def test_non_txt_extension_warns(self, tmp_path, capsys):
        from spritegrid.main import handle_txt
        path = str(tmp_path / "output.ansi")
        handle_txt("content", path)
        out = capsys.readouterr().out
        assert "Warning" in out

    def test_txt_extension_no_warning(self, tmp_path, capsys):
        from spritegrid.main import handle_txt
        path = str(tmp_path / "output.txt")
        handle_txt("content", path)
        out = capsys.readouterr().out
        assert "Warning" not in out

    def test_creates_file_if_not_exists(self, tmp_path):
        from spritegrid.main import handle_txt
        path = str(tmp_path / "new.txt")
        assert not os.path.exists(path)
        handle_txt("data", path)
        assert os.path.exists(path)


# ---------------------------------------------------------------------------
# handle_png
# ---------------------------------------------------------------------------

class TestHandlePng:
    def test_saves_image_to_file(self, tmp_path):
        from spritegrid.main import handle_png
        path = str(tmp_path / "out.png")
        img = Image.new("RGB", (4, 4), (100, 200, 50))
        handle_png(img, path)
        assert os.path.exists(path)
        loaded = Image.open(path)
        assert loaded.size == (4, 4)

    def test_prints_success_message(self, tmp_path, capsys):
        from spritegrid.main import handle_png
        path = str(tmp_path / "out.png")
        handle_png(Image.new("RGB", (2, 2)), path)
        out = capsys.readouterr().out
        assert "Success" in out or "success" in out.lower() or str(path) in out

    def test_invalid_path_does_not_raise(self, capsys):
        from spritegrid.main import handle_png
        img = Image.new("RGB", (2, 2))
        # Non-existent directory → IOError caught
        handle_png(img, "/nonexistent/dir/out.png")
        err = capsys.readouterr().err
        assert "Error" in err


# ---------------------------------------------------------------------------
# create_downsampled_image
# ---------------------------------------------------------------------------

class TestCreateDownsampledImage:
    def _checkerboard(self, grid_w, grid_h, cells_w, cells_h):
        """Create a simple test image with grid_w*cells_w × grid_h*cells_h pixels."""
        w = grid_w * cells_w
        h = grid_h * cells_h
        arr = np.zeros((h, w, 3), dtype=np.uint8)
        return Image.fromarray(arr, "RGB")

    def test_output_size_matches_num_cells(self):
        from spritegrid.main import create_downsampled_image
        img = self._checkerboard(4, 4, 3, 2)
        result = create_downsampled_image(img, 4, 4, 3, 2, kernel_size=(1, 1))
        assert result.size == (3, 2)

    def test_even_kernel_raises_value_error(self):
        from spritegrid.main import create_downsampled_image
        img = self._checkerboard(4, 4, 3, 3)
        with pytest.raises(ValueError, match="odd"):
            create_downsampled_image(img, 4, 4, 3, 3, kernel_size=(2, 3))

    def test_kernel_larger_than_grid_raises(self):
        from spritegrid.main import create_downsampled_image
        img = self._checkerboard(3, 3, 2, 2)
        with pytest.raises(ValueError, match="[Kk]ernel"):
            create_downsampled_image(img, 3, 3, 2, 2, kernel_size=(5, 5))

    def test_zero_grid_width_raises(self):
        from spritegrid.main import create_downsampled_image
        img = self._checkerboard(4, 4, 2, 2)
        with pytest.raises(ValueError):
            create_downsampled_image(img, 0, 4, 2, 2, kernel_size=(1, 1))

    def test_returns_pil_image(self):
        from spritegrid.main import create_downsampled_image
        img = self._checkerboard(4, 4, 4, 4)
        result = create_downsampled_image(img, 4, 4, 4, 4, kernel_size=(1, 1))
        assert isinstance(result, Image.Image)

    def test_geometric_median_type(self):
        from spritegrid.main import create_downsampled_image
        img = self._checkerboard(4, 4, 3, 3)
        # Should not raise with median_type="geometric"
        result = create_downsampled_image(img, 4, 4, 3, 3, kernel_size=(1, 1), median_type="geometric")
        assert result.size == (3, 3)
