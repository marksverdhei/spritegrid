# main.py

import argparse
import io
import sys
from typing import Optional

import requests
from PIL import Image, UnidentifiedImageError, ImageDraw

# Import the detection function from the detection module
from src import main


def load_image(image_source: str) -> Optional[Image.Image]:
    """
    Loads an image from a local file path or a URL.
    (Function remains the same)
    """
    # ... (code is identical to previous version) ...
    try:
        if image_source.startswith(('http://', 'https://')):
            response = requests.get(image_source, stream=True, timeout=15)
            response.raise_for_status()
            content_type = response.headers.get('content-type')
            if content_type and not content_type.startswith('image/'):
                 print(f"Error: URL content type ({content_type}) doesn't appear to be an image.", file=sys.stderr)
                 return None
            image_bytes = io.BytesIO(response.content)
            img = Image.open(image_bytes)
            return img
        else:
            img = Image.open(image_source)
            return img
    except FileNotFoundError:
        print(f"Error: Local file not found at '{image_source}'", file=sys.stderr)
        return None
    except requests.exceptions.RequestException as e:
        print(f"Error: Could not retrieve image from URL '{image_source}'. Reason: {e}", file=sys.stderr)
        return None
    except UnidentifiedImageError:
        print(f"Error: Cannot identify image file. It might be corrupted or an unsupported format: '{image_source}'", file=sys.stderr)
        return None
    except IOError as e:
        print(f"Error: An I/O error occurred while handling '{image_source}'. Reason: {e}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"An unexpected error occurred while loading '{image_source}': {e}", file=sys.stderr)
        return None


def draw_grid_overlay(image: Image.Image, grid_w: int, grid_h: int, color: str = "red", line_width: int = 1) -> Image.Image:
    """
    Draws the detected grid lines onto a copy of the original image (for debugging).
    (Function remains the same, but its primary use changes)
    """
    # ... (code is identical to previous version) ...
    img_copy = image.copy().convert("RGB")
    draw = ImageDraw.Draw(img_copy)
    img_width, img_height = img_copy.size
    if grid_w <= 0 or grid_h <= 0:
        print("Warning: Invalid grid dimensions provided for drawing overlay.", file=sys.stderr)
        return img_copy
    for x in range(grid_w, img_width, grid_w):
        draw.line([(x, 0), (x, img_height)], fill=color, width=line_width)
    for y in range(grid_h, img_height, grid_h):
        draw.line([(0, y), (img_width, y)], fill=color, width=line_width)
    # print(f"Debug grid overlay drawn with {grid_w}x{grid_h} cells.") # Make less verbose
    return img_copy


def create_downsampled_image(image: Image.Image, grid_w: int, grid_h: int, num_cells_w: int, num_cells_h: int) -> Image.Image:
    """
    Creates a new image by sampling the center pixel of each grid cell
    from the original image.

    Args:
        image: The original PIL Image object.
        grid_w: The detected width of a grid cell in the original image.
        grid_h: The detected height of a grid cell in the original image.
        num_cells_w: The number of grid cells horizontally.
        num_cells_h: The number of grid cells vertically.

    Returns:
        A new PIL Image object with dimensions (num_cells_w, num_cells_h).
    """
    print(f"Creating downsampled image ({num_cells_w}x{num_cells_h}) by center-pixel sampling...")
    # Ensure grid dimensions are valid
    if grid_w <= 0 or grid_h <= 0 or num_cells_w <= 0 or num_cells_h <= 0:
        print("Error: Invalid dimensions provided for downsampling.", file=sys.stderr)
        # Return a 1x1 pixel image in this case? Or raise error? Let's return small blank.
        return Image.new(image.mode, (1, 1))

    # Use a mode that supports transparency if the original has it (e.g., PNG)
    mode = image.mode if image.mode in ['RGB', 'RGBA', 'L'] else 'RGBA'
    if image.mode != mode:
        print(f"Info: Converting original image from {image.mode} to {mode} for processing.")
        original_image = image.convert(mode)
    else:
        original_image = image # No conversion needed

    new_img = Image.new(mode, (num_cells_w, num_cells_h))
    original_pixels = original_image.load()
    new_pixels = new_img.load()
    original_width, original_height = original_image.size

    for y_new in range(num_cells_h):
        for x_new in range(num_cells_w):
            # Calculate center coordinates in the original image
            center_x = min(int(x_new * grid_w + grid_w / 2), original_width - 1)
            center_y = min(int(y_new * grid_h + grid_h / 2), original_height - 1)

            # Get pixel value from original image's center
            pixel_value = original_pixels[center_x, center_y]

            # Set pixel value in the new image
            new_pixels[x_new, y_new] = pixel_value

    print("Downsampled image created.")
    return new_img

def handle_output(image_to_process: Image.Image, filename: Optional[str], show_flag: bool, is_debug: bool, default_title: str="Spritegrid Output"):
    """Helper function to save or show the processed image."""

    # Default action for debug mode if no other option is chosen
    if is_debug and not filename and not show_flag:
        print("Info: Debug mode active, defaulting to show the overlay image.")
        show_flag = True

    # Save the image if filename is provided
    if filename:
        try:
            image_to_process.save(filename)
            type_str = "Debug overlay" if is_debug else "Downsampled"
            print(f"Success: {type_str} image saved to '{filename}'")
        except IOError as e:
            print(f"Error: Could not save image to '{filename}'. Reason: {e}", file=sys.stderr)
        except ValueError as e: # Catch errors like unknown extension
             print(f"Error: Could not save image to '{filename}'. Is the file extension valid? Reason: {e}", file=sys.stderr)

    # Show the image if show_flag is true
    if show_flag:
        try:
            type_str = "Debug Overlay" if is_debug else "Downsampled Image"
            title = f"{default_title} - {type_str}"
            print(f"Displaying {type_str.lower()} (Press Ctrl+C in terminal if viewer doesn't close automatically)...")
            image_to_process.show(title=title)
        except Exception as e:
            print(f"Error: Could not display image using default viewer. Reason: {e}", file=sys.stderr)


def parse_args():
    """
    Main function to parse arguments, load image, detect grid, and generate output/debug image.
    """
    parser = argparse.ArgumentParser(
        description="Detect grid in AI pixel art & create downsampled image or debug overlay."
    )
    parser.add_argument(
        "image_source",
        type=str,
        help="Path to the local image file or URL of the image."
    )
    parser.add_argument(
        "--min-grid",
        type=int,
        default=4,
        help="Minimum expected grid dimension (width or height) for peak detection. (Default: 4)"
    )
    # --- Modified/New Arguments ---
    parser.add_argument(
        "-o", "--output",
        metavar="FILENAME",
        dest="output_file",
        type=str,
        help="Save the output image (downsampled by default, or debug overlay if -d is used) to FILENAME."
    )
    parser.add_argument(
        "-i", "--show",
        action="store_true",
        help="Display the output image (downsampled by default, or debug overlay if -d is used) using the default system viewer."
    )
    parser.add_argument(
        "-d", "--debug",
        action="store_true",
        help="Enable debug mode: output/show a grid overlay instead of the downsampled image. Defaults to showing if -o or -i are not specified."
    )
    # --- End Modified/New Arguments ---

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)