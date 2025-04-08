import pytest
import numpy as np
from PIL import Image

# Import the functions to be tested
from detection import find_dominant_spacing, detect_grid


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


def test_logo():
    image = Image.open("assets/image.png")
    grid = detect_grid(image, min_grid_size=1)

    assert grid != (0, 0), "Grid detection failed"

    w_eyes = 5
    w_mouth_min = 4
    w_mouth_max = 5

    w_pad_min = 2
    w_pad_max = 3
    w_mouth_max = 5
    w_min = w_eyes * 2 + w_mouth_min + w_pad_min * 2 + 2 * 1
    w_max = w_eyes * 2 + w_mouth_max + w_pad_max * 2 + 2 * 1
    detected_w, detected_h = grid

    assert detected_w >= w_min, (
        f"Detected width {detected_w} is less than minimum {w_min}"
    )
    assert detected_w <= w_max, (
        f"Detected width {detected_w} is greater than maximum {w_max}"
    )
    h = 1 + 3 + 1 + 6 + 1 + 5 + 1 + 2 + 1
    assert detected_h >= h - 2, f"Detected height {detected_h} is less than minimum {h}"
    assert detected_h <= h + 2, f"Detected height {detected_h} is less than minimum {h}"
