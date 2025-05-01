import numpy as np
from PIL import Image

# Import the function to be tested
from spritegrid.utils import crop_to_content


def test_remove_background():
    pass


def test_crop_to_content():
    """Test the automatic cropping of transparent images."""
    # Create a test image with transparent border
    width, height = 100, 80

    # Create a rectangle with non-transparent pixels in the middle
    content_top, content_left = 20, 30
    content_width, content_height = 40, 30
    content_bottom = content_top + content_height
    content_right = content_left + content_width

    # Fill the non-transparent area
    draw_area = np.zeros((height, width, 4), dtype=np.uint8)
    draw_area[content_top:content_bottom, content_left:content_right] = [
        255,
        100,
        100,
        255,
    ]  # Red with alpha=255

    # Convert numpy array to PIL Image
    test_image = Image.fromarray(draw_area, "RGBA")

    # Apply cropping
    cropped = crop_to_content(test_image)

    # Check dimensions
    assert cropped.width == content_width
    assert cropped.height == content_height

    # Check that the image actually contains the content
    cropped_array = np.array(cropped)
    assert np.all(cropped_array[:, :, 3] == 255)  # All pixels should be non-transparent
