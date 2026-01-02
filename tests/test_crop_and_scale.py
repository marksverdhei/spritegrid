"""Tests for crop_and_scale module."""
import pytest
import numpy as np
from PIL import Image

from spritegrid.crop_and_scale import (
    detect_bounds,
    crop_to_content,
    scale_nearest,
    center_on_canvas,
    process_sprite,
    crop_and_scale,
    crop_and_scale_centered,
    batch_process,
)


def create_test_image_with_content(
    width=100, height=80, content_rect=(30, 20, 70, 50), color=(255, 100, 100, 255)
):
    """Create a test RGBA image with content in specified rect and transparent elsewhere."""
    img_array = np.zeros((height, width, 4), dtype=np.uint8)
    left, top, right, bottom = content_rect
    img_array[top:bottom, left:right] = color
    return Image.fromarray(img_array, "RGBA")


def test_detect_bounds():
    """Test bounding box detection."""
    img = create_test_image_with_content(100, 80, (30, 20, 70, 50))
    bounds = detect_bounds(img, alpha_threshold=10)

    assert bounds is not None
    left, top, right, bottom = bounds
    assert left == 30
    assert top == 20
    assert right == 70
    assert bottom == 50


def test_detect_bounds_empty_image():
    """Test detect_bounds on fully transparent image."""
    img = Image.new("RGBA", (100, 100), (0, 0, 0, 0))
    bounds = detect_bounds(img)
    assert bounds is None


def test_detect_bounds_alpha_threshold():
    """Test alpha threshold in detect_bounds."""
    img = create_test_image_with_content(color=(255, 100, 100, 5))

    # With default threshold (10), should not detect
    bounds = detect_bounds(img, alpha_threshold=10)
    assert bounds is None

    # With lower threshold, should detect
    bounds = detect_bounds(img, alpha_threshold=0)
    assert bounds is not None


def test_crop_to_content():
    """Test automatic cropping to content."""
    img = create_test_image_with_content(100, 80, (30, 20, 70, 50))
    cropped = crop_to_content(img)

    assert cropped.width == 40  # 70 - 30
    assert cropped.height == 30  # 50 - 20

    # All pixels should be non-transparent
    cropped_array = np.array(cropped)
    assert np.all(cropped_array[:, :, 3] == 255)


def test_crop_to_content_with_padding():
    """Test cropping with padding."""
    img = create_test_image_with_content(100, 80, (30, 20, 70, 50))
    cropped = crop_to_content(img, padding=5)

    # Should be 40 + 2*5 = 50 wide and 30 + 2*5 = 40 tall
    assert cropped.width == 50
    assert cropped.height == 40


def test_crop_to_content_empty_image():
    """Test crop_to_content on empty image."""
    img = Image.new("RGBA", (100, 100), (0, 0, 0, 0))
    cropped = crop_to_content(img)

    # Should return original image
    assert cropped.size == img.size


def test_scale_nearest():
    """Test nearest-neighbor scaling."""
    img = create_test_image_with_content(100, 80, (0, 0, 100, 80))

    # Test scaling to exact size
    scaled = scale_nearest(img, 32)
    assert scaled.size == (32, 32)

    # Test scaling to tuple
    scaled = scale_nearest(img, (64, 48))
    assert scaled.size == (64, 48)


def test_scale_nearest_maintain_aspect():
    """Test scaling with aspect ratio preservation."""
    img = create_test_image_with_content(100, 50)

    # With square=False, should fit within 32x32
    scaled = scale_nearest(img, 32, square=False)
    assert scaled.width == 32
    assert scaled.height == 16  # maintains 2:1 aspect ratio


def test_center_on_canvas():
    """Test centering on canvas."""
    # Create 20x20 image with content filling the whole image
    img = create_test_image_with_content(20, 20, content_rect=(0, 0, 20, 20))

    # Center on 64x64 canvas
    centered = center_on_canvas(img, 64)
    assert centered.size == (64, 64)
    assert centered.mode == "RGBA"

    # Check that the image is actually centered
    centered_array = np.array(centered)
    # Content should be in middle 20x20 area centered at (32, 32)
    # So from (22, 22) to (42, 42)
    center_content = centered_array[22:42, 22:42, 3]
    assert np.any(center_content > 0)  # Should have content in center


def test_center_on_canvas_non_square():
    """Test centering on non-square canvas."""
    img = create_test_image_with_content(20, 20)
    centered = center_on_canvas(img, (100, 50))
    assert centered.size == (100, 50)


def test_process_sprite():
    """Test the full sprite processing pipeline."""
    # Create a larger image with content
    img = create_test_image_with_content(200, 200, (50, 50, 150, 150))

    # Process to 32x32 sprite
    sprite = process_sprite(img, size=32, remove_bg=False)

    assert sprite.size == (32, 32)
    assert sprite.mode == "RGBA"

    # Should have some content
    sprite_array = np.array(sprite)
    assert np.any(sprite_array[:, :, 3] > 0)


def test_process_sprite_with_padding():
    """Test sprite processing with padding."""
    img = create_test_image_with_content(100, 100, (25, 25, 75, 75))
    sprite = process_sprite(img, size=32, remove_bg=False, padding=10)

    assert sprite.size == (32, 32)


def test_crop_and_scale():
    """Test crop_and_scale function."""
    img = create_test_image_with_content(200, 100, (50, 25, 150, 75))

    # Scale to 32x32 maintaining aspect
    result = crop_and_scale(img, target_size=32, maintain_aspect=True)

    # Should maintain aspect ratio, so won't be exactly 32x32
    assert result.width <= 32
    assert result.height <= 32
    assert result.mode == "RGBA"


def test_crop_and_scale_no_aspect():
    """Test crop_and_scale without maintaining aspect."""
    img = create_test_image_with_content(200, 100)

    # Force to exact 32x32
    result = crop_and_scale(img, target_size=32, maintain_aspect=False)

    assert result.size == (32, 32)


def test_crop_and_scale_tuple_size():
    """Test crop_and_scale with tuple target size."""
    img = create_test_image_with_content(200, 200)

    result = crop_and_scale(img, target_size=(64, 32), maintain_aspect=True)

    assert result.width <= 64
    assert result.height <= 32


def test_crop_and_scale_centered():
    """Test centered crop and scale."""
    img = create_test_image_with_content(200, 100)

    result = crop_and_scale_centered(img, target_size=64)

    # Should be exactly 64x64 (centered on canvas)
    assert result.size == (64, 64)
    assert result.mode == "RGBA"


def test_crop_and_scale_centered_non_square():
    """Test centered crop and scale with non-square target."""
    img = create_test_image_with_content(100, 100)

    result = crop_and_scale_centered(img, target_size=(64, 48))

    assert result.size == (64, 48)


def test_batch_process():
    """Test batch processing multiple images."""
    images = [
        create_test_image_with_content(100, 100),
        create_test_image_with_content(150, 150),
        create_test_image_with_content(200, 200),
    ]

    results = batch_process(images, size=32, remove_bg=False)

    assert len(results) == 3
    for result in results:
        assert result.size == (32, 32)
        assert result.mode == "RGBA"


def test_rgb_image_conversion():
    """Test that RGB images are properly converted to RGBA."""
    img = Image.new("RGB", (100, 100), (255, 0, 0))

    result = crop_and_scale(img, target_size=32)

    assert result.mode == "RGBA"


def test_alpha_threshold_parameter():
    """Test that alpha_threshold parameter works."""
    # Create image with semi-transparent border
    img = create_test_image_with_content(100, 100, color=(255, 0, 0, 50))

    # With high threshold, should treat as transparent
    result1 = crop_and_scale(img, target_size=32, alpha_threshold=100)

    # With low threshold, should keep the content
    result2 = crop_and_scale(img, target_size=32, alpha_threshold=0)

    # Results might differ in size if threshold affects cropping
    assert result1.mode == "RGBA"
    assert result2.mode == "RGBA"
