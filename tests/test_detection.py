import os
import sys
import numpy as np
from PIL import Image
import spritegrid.detection as detection_module
from spritegrid.detection import detect_grid
from spritegrid.detection import find_dominant_spacing

# Import the functions to be tested


def create_test_image(img_w, img_h, grid_w, grid_h, color=False, noise_level=0):
    """Creates a PIL Image with a grid pattern."""
    if grid_w <= 0 or grid_h <= 0:  # Create uniform image if grid size is invalid
        array = np.ones((img_h, img_w), dtype=np.uint8) * 128
    else:
        # Create a checkerboard pattern
        array = np.zeros((img_h, img_w), dtype=np.uint8)
        for r in range(0, img_h, grid_h):
            for c in range(0, img_w, grid_w):
                row_block = (r // grid_h) % 2
                col_block = (c // grid_w) % 2
                if row_block == col_block:
                    array[r : min(r + grid_h, img_h), c : min(c + grid_w, img_w)] = (
                        200  # Light gray
                    )
                else:
                    array[r : min(r + grid_h, img_h), c : min(c + grid_w, img_w)] = (
                        50  # Dark gray
                    )

    # Add noise if requested
    if noise_level > 0:
        noise = np.random.randint(
            -noise_level, noise_level + 1, size=array.shape, dtype=np.int16
        )
        array = np.clip(array.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    if color:
        # Stack grayscale array to create RGB
        array_rgb = np.stack([array] * 3, axis=-1)
        return Image.fromarray(array_rgb, "RGB")
    else:
        return Image.fromarray(array, "L")


def test_detect_grid():
    """Test with image dimensions barely meeting the minimum requirement."""
    img_w, img_h = 25, 25
    grid_w, grid_h = 5, 5
    image = create_test_image(img_w, img_h, grid_w, grid_h)
    # image.show()
    image.save("test_image.png")

    detected_w, detected_h = detect_grid(image, min_grid_size=1)

    assert (detected_w, detected_h) == (grid_w, grid_h)

# tests/test_detection.py



def test_find_dominant_spacing_none_profile_returns_zero():
    assert find_dominant_spacing(None) == 0


def test_find_dominant_spacing_short_profile_returns_zero():
    # Length less than min_spacing * 2 = 6 by default
    short_profile = np.arange(5)
    assert find_dominant_spacing(short_profile) == 0


def test_find_dominant_spacing_flat_profile_returns_zero():
    # Flat profile yields no peaks
    flat_profile = np.zeros(100)
    assert find_dominant_spacing(flat_profile) == 0


def test_find_dominant_spacing_single_peak_returns_zero():
    # Only one significant peak
    profile = np.zeros(50)
    profile[25] = 10
    assert find_dominant_spacing(profile) == 0


def test_find_dominant_spacing_two_peaks_correct_spacing():
    # Two peaks spaced by 5
    profile = np.zeros(50)
    profile[10] = 20
    profile[15] = 15
    spacing = find_dominant_spacing(profile)
    assert spacing == 5


def test_find_dominant_spacing_multiple_even_peaks():
    # Peaks at [5,10,15,20] => spacing 5
    profile = np.zeros(100)
    for idx in [5, 10, 15, 20]:
        profile[idx] = 30
    assert find_dominant_spacing(profile) == 5


def test_find_dominant_spacing_most_common_among_varied():
    # Peaks at [5,10,15,22] => spacings [5,5,7] => mode 5
    profile = np.zeros(100)
    for idx in [5, 10, 15, 22]:
        profile[idx] = 50
    assert find_dominant_spacing(profile) == 5


def test_find_dominant_spacing_default_min_spacing_skips_close_peaks():
    # Peaks at [0,2,4,6]
    profile = np.zeros(20)
    for idx in [0, 2, 4, 6]:
        profile[idx] = 40
    # Default min_spacing=3: detected peaks at [0,4] => spacing 4
    assert find_dominant_spacing(profile) == 4


def test_find_dominant_spacing_custom_min_spacing_allows_close_peaks():
    # With min_spacing=2, peaks at [0,2,4,6] => spacing 2
    profile = np.zeros(20)
    for idx in [0, 2, 4, 6]:
        profile[idx] = 40
    assert find_dominant_spacing(profile, min_spacing=2) == 2


def test_find_dominant_spacing_low_prominence_ratio_detects_spacing():
    # Low threshold, should detect peaks spaced by 3
    profile = np.zeros(20)
    for idx in [0, 3, 6, 9]:
        profile[idx] = 10
    assert find_dominant_spacing(profile, prominence_ratio=0.01) == 3


def test_find_dominant_spacing_excessive_prominence_ratio_returns_zero():
    # Use prominence_ratio > 1 to set min_prominence above actual peak heights
    profile = np.zeros(20)
    for idx in [0, 3, 6, 9]:
        profile[idx] = 10
    # Prominence threshold = (max-min)*1.1 = 11, peaks of height 10 won't qualify
    assert find_dominant_spacing(profile, prominence_ratio=1.1) == 0
# tests/test_detection.py

# Ensure that 'src' directory is on sys.path so pytest can import 'spritegrid'
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))



def create_grid_image(width, height, spacing, mode="L"):  # noqa: E302
    """
    Create a synthetic grid image with vertical and horizontal lines every 'spacing' pixels.
    Lines are white (255), background is black (0).
    Returns a PIL Image of given mode.
    """
    arr = np.zeros((height, width), dtype=np.uint8)
    # Draw vertical lines
    for x in range(0, width, spacing):
        arr[:, x] = 255
    # Draw horizontal lines
    for y in range(0, height, spacing):
        arr[y, :] = 255
    img = Image.fromarray(arr, mode="L")
    if mode == "RGBA":
        return img.convert("RGBA")
    elif mode == "LA":
        return img.convert("LA")
    else:
        return img


def test_detect_grid_simple_pattern():
    # Create a 50x40 grid with spacing 10
    w, h, s = 50, 40, 10
    img = create_grid_image(w, h, s)
    grid_w, grid_h = detect_grid(img)
    assert (grid_w, grid_h) == (s, s)


def test_detect_grid_without_smoothing():
    # Same grid, disable smoothing
    w, h, s = 60, 48, 12
    img = create_grid_image(w, h, s)
    grid_w, grid_h = detect_grid(img, smoothing_sigma=0)
    assert (grid_w, grid_h) == (s, s)


def test_detect_grid_rgba_and_la_modes():
    # Test that RGBA and LA images are handled correctly
    w, h, s = 36, 36, 6
    for mode in ("RGBA", "LA"):
        img = create_grid_image(w, h, s, mode=mode)
        grid_w, grid_h = detect_grid(img)
        assert (grid_w, grid_h) == (s, s), f"Mode {mode}: expected {(s, s)}, got {(grid_w, grid_h)}"


def test_detect_grid_too_small_image():
    # Image smaller than 2 * min_grid_size in each dimension -> early exit
    img = Image.new("L", (5, 5), color=0)
    # min_grid_size default is 4 -> need at least 8px
    assert detect_grid(img) == (0, 0)
    # Even if spacing is small but insufficient dimension
    assert detect_grid(img, min_grid_size=1) == (0, 0)


def test_detect_grid_min_grid_size_larger_than_pattern():
    # Create grid with spacing 5, but require min_grid_size 7
    w, h, s = 50, 50, 5
    img = create_grid_image(w, h, s)
    # actual_min_spacing = 7, but real spacing is 5 -> detection fails
    assert detect_grid(img, min_grid_size=7) == (0, 0)


def test_detect_grid_importerror_branch(monkeypatch, capsys):
    # Simulate scipy missing by making gaussian_filter1d raise ImportError
    monkeypatch.setattr(detection_module, 'gaussian_filter1d', lambda *args, **kwargs: (_ for _ in ()).throw(ImportError()))
    img = create_grid_image(40, 40, 8)
    grid = detect_grid(img)
    captured = capsys.readouterr()
    assert grid == (0, 0)
    assert 'Please install' in captured.err


def test_detect_grid_generic_exception_branch(monkeypatch, capsys):
    # Simulate unexpected exception in find_dominant_spacing
    def bad_find(*args, **kwargs):
        raise ValueError("boom")
    monkeypatch.setattr(detection_module, 'find_dominant_spacing', bad_find)
    img = create_grid_image(40, 40, 8)
    grid = detect_grid(img)
    captured = capsys.readouterr()
    assert grid == (0, 0)
    # Capture either the printed message or traceback containing 'boom'
    err = captured.err.lower()
    assert 'unexpected error' in err or 'boom' in err