import io
import sys
from typing import Optional

import requests
from PIL import Image, ImageDraw

from spritegrid.segmentation import make_background_transparent

from .detection import detect_grid
from .utils import (
    convert_image_to_ascii,
    geometric_median,
    naive_median,
    crop_to_content,
)
import numpy as np


def load_image(image_source: str) -> Optional[Image.Image]:
    """
    Loads an image from a local file path or a URL.
    (Function remains the same)
    """
    # ... (code is identical to previous version) ...
    try:
        if image_source.startswith(("http://", "https://")):
            response = requests.get(image_source, stream=True, timeout=15)
            response.raise_for_status()
            content_type = response.headers.get("content-type")
            if content_type and not content_type.startswith("image/"):
                print(
                    f"Error: URL content type ({content_type}) doesn't appear to be an image.",
                    file=sys.stderr,
                )
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
    except Exception as e:
        print(
            f"An unexpected error occurred while loading '{image_source}': {e}",
            file=sys.stderr,
        )
        return None


def draw_grid_overlay(
    image: Image.Image,
    grid_w: int,
    grid_h: int,
    color: str = "red",
    line_width: int = 1,
) -> Image.Image:
    """
    Draws the detected grid lines onto a copy of the original image (for debugging).
    (Function remains the same, but its primary use changes)
    """
    # ... (code is identical to previous version) ...
    img_copy = image.copy().convert("RGB")
    draw = ImageDraw.Draw(img_copy)
    img_width, img_height = img_copy.size
    if grid_w <= 0 or grid_h <= 0:
        print(
            "Warning: Invalid grid dimensions provided for drawing overlay.",
            file=sys.stderr,
        )
        return img_copy
    for x in range(grid_w, img_width, grid_w):
        draw.line([(x, 0), (x, img_height)], fill=color, width=line_width)
    for y in range(grid_h, img_height, grid_h):
        draw.line([(0, y), (img_width, y)], fill=color, width=line_width)
    # print(f"Debug grid overlay drawn with {grid_w}x{grid_h} cells.") # Make less verbose
    return img_copy


def create_downsampled_image(
    image: Image.Image,
    grid_w: int,
    grid_h: int,
    num_cells_w: int,
    num_cells_h: int,
    bit: int = 8,
    kernel_size: tuple = (3, 3),
    median_type: str = "naive",
) -> Image.Image:
    """
    Creates a new image by sampling the geometric median pixel of each grid cell
    from the original image and quantizing the colors.

    Args:
        image: The original PIL Image object.
        grid_w: The detected width of a grid cell in the original image.
        grid_h: The detected height of a grid cell in the original image.
        num_cells_w: The number of grid cells horizontally.
        num_cells_h: The number of grid cells vertically.
        bit: Number of bits per color channel.
        kernel_size: Size of the kernel to sample from (width, height).

    Returns:
        A new PIL Image object with dimensions (num_cells_w, num_cells_h).
    """

    _median = naive_median if median_type == "naive" else geometric_median
    kernel_w, kernel_h = kernel_size

    # Validate kernel size is odd (to have a clear center)
    if kernel_w % 2 == 0 or kernel_h % 2 == 0:
        raise ValueError(
            f"Kernel dimensions must be odd numbers: {kernel_w=}, {kernel_h=}"
        )

    # Ensure grid dimensions are valid
    if grid_w <= 0 or grid_h <= 0 or num_cells_w <= 0 or num_cells_h <= 0:
        raise ValueError(
            f"Invalid grid dimensions or number of cells provided: {grid_w=}, {grid_h=}, {num_cells_w=}, {num_cells_h=}"
        )

    # Check if kernel size is compatible with grid size
    if kernel_w > grid_w or kernel_h > grid_h:
        raise ValueError(
            f"Kernel size ({kernel_w}x{kernel_h}) cannot be larger than grid cell size ({grid_w}x{grid_h})"
        )

    print(
        f"Creating downsampled image ({num_cells_w}x{num_cells_h}) using geometric median of {kernel_w}x{kernel_h} kernel..."
    )

    # Use a mode that supports transparency if the original has it (e.g., PNG)
    mode = image.mode if image.mode in ["RGB", "RGBA", "L"] else "RGBA"
    if image.mode != mode:
        print(
            f"Info: Converting original image from {image.mode} to {mode} for processing."
        )
        original_image = image.convert(mode)
    else:
        original_image = image  # No conversion needed

    new_img = Image.new(mode, (num_cells_w, num_cells_h))
    original_width, original_height = original_image.size

    # Convert PIL image to numpy array for easier processing
    original_array = np.array(original_image)

    max_value = 2**bit - 1

    def quantize(value):
        return round(value * max_value / 255) * 255 // max_value

    for y_new in range(num_cells_h):
        for x_new in range(num_cells_w):
            # Calculate center coordinates in the original image
            center_x = min(int(x_new * grid_w + grid_w / 2), original_width - 1)
            center_y = min(int(y_new * grid_h + grid_h / 2), original_height - 1)

            # Calculate kernel boundaries
            half_kernel_w = kernel_w // 2
            half_kernel_h = kernel_h // 2

            # Ensure kernel stays within image boundaries
            x_start = max(0, center_x - half_kernel_w)
            y_start = max(0, center_y - half_kernel_h)
            x_end = min(original_width, center_x + half_kernel_w + 1)
            y_end = min(original_height, center_y + half_kernel_h + 1)

            # Extract kernel pixels
            kernel_pixels = original_array[y_start:y_end, x_start:x_end]

            # Check if we have enough pixels to compute the geometric median
            if kernel_pixels.size == 0:
                raise ValueError(
                    f"Kernel at position ({x_new}, {y_new}) contains no pixels"
                )

            # Reshape to list of pixels for geometric median calculation
            pixels_list = kernel_pixels.reshape(-1, kernel_pixels.shape[-1])

            # Compute geometric median pixel
            median_pixel = _median(pixels_list)

            # Quantize the median pixel if needed
            if bit != 8:
                if mode == "RGB":
                    median_pixel = np.array(
                        [
                            quantize(median_pixel[0]),
                            quantize(median_pixel[1]),
                            quantize(median_pixel[2]),
                        ]
                    )
                elif mode == "RGBA":
                    median_pixel = np.array(
                        [
                            quantize(median_pixel[0]),
                            quantize(median_pixel[1]),
                            quantize(median_pixel[2]),
                            median_pixel[3],
                        ]
                    )  # keep alpha
                else:  # Grayscale
                    median_pixel = np.array([quantize(median_pixel[0])])

            # Convert the median pixel to the appropriate format and set it in the new image
            if mode == "L":
                new_img.putpixel((x_new, y_new), int(median_pixel[0]))
            else:
                new_img.putpixel((x_new, y_new), tuple(map(int, median_pixel)))

    print("Downsampled image created.")
    return new_img


def handle_output(
    image: Image.Image,
    save_path: Optional[str],
    show_flag: bool,
    is_debug: bool,
    default_title: str = "Spritegrid Output",
    ascii_space_width: Optional[int] = None,
):
    """Helper function to save or show the processed image."""
    if save_path is not None:
        if save_path.endswith(".txt") and ascii_space_width is None:
            ascii_space_width = 1

    show_stdout = (ascii_space_width is not None and save_path is None)
    show_image = (show_flag or (save_path is None and not show_stdout))

    # Default action for debug mode if no other option is chosen
    if is_debug and not save_path and not show_flag:
        print("Info: Debug mode active, defaulting to show the overlay image.")
        show_flag = True

    # Save the image if filename is provided
    if save_path:
        if save_path.endswith(".png"):
            handle_png(image, save_path)
        elif save_path.endswith(".txt"):
            image_string = convert_image_to_ascii(image, ascii_space_width)
            handle_txt(image_string, save_path)
    else:
        if show_image:
            type_str = "Debug Overlay" if is_debug else "Downsampled Image"
            title = f"{default_title} - {type_str}"
            handle_show_image(image, title)
        
        if show_stdout:
            print(convert_image_to_ascii(image, ascii_space_width))


def handle_show_image(image: Image.Image, title: str) -> None:
    try:

        print(
            f"Displaying {title} (Press Ctrl+C in terminal if viewer doesn't close automatically)..."
        )
        image.show(title=title)
    except Exception as e:
        print(
            f"Error: Could not display image using default viewer. Reason: {e}",
            file=sys.stderr,
        )

def handle_txt(image_string, save_path) -> None:
    if not save_path.endswith(".txt"):
        print(f"Warning: {save_path} does not end with .txt, saving as ascii+ansi codes")
    with open(save_path, "w+") as f:
        f.write(image_string)


def handle_png(image: Image.Image, save_path: str) -> None:
    try:
        image.save(save_path)
        print(f"Success: image saved to '{save_path}'")
    except IOError as e:
        print(
            f"Error: Could not save image to '{save_path}'. Reason: {e}",
            file=sys.stderr,
        )
    except ValueError as e:  # Catch errors like unknown extension
        print(
            f"Error: Could not save image to '{save_path}'. Is the file extension valid? Reason: {e}",
            file=sys.stderr,
        )

def main(
    image_source: str,
    min_grid: int = 4,
    output_file: Optional[str] = None,
    show: bool = False,
    debug: bool = False,
    quantize: int = 8,
    remove_background: Optional[str] = None,
    crop: bool = False,
    ascii_space_width: Optional[int] = None,
) -> None:
    """
    Main function to parse arguments, load image, detect grid, and generate output/debug image.
    """
    debug_image = None

    if remove_background == "default":
        remove_background = "after"

    # Info message if no primary output action selected (and not in debug mode)
    if not debug and not output_file and not show:
        print(
            "Info: No output option (-o or -i) selected for downsampled image. Only detection results will be printed.",
            file=sys.stderr,
        )

    print(f"Loading image from: {image_source}")
    image = load_image(image_source)

    if remove_background == "before":
        image = make_background_transparent(image, debug=False)[0]

    if image is None:
        sys.exit(1)

    print(
        f"Image loaded successfully ({image.width}x{image.height}, Mode: {image.mode})."
    )

    # Call the grid detection function from the detection module
    detected_w, detected_h = detect_grid(image, min_grid_size=min_grid)

    # Check the results returned by detect_grid
    if detected_w > 0 and detected_h > 0:
        print("\n--- Result ---")
        print(
            f"Detected Grid Dimensions (W x H): {detected_w} x {detected_h} pixels per grid cell"
        )

        # Estimate number of cells
        num_cells_w = round(image.width / detected_w)
        num_cells_h = round(image.height / detected_h)
        # Ensure at least 1 cell if rounding leads to 0
        num_cells_w = max(1, num_cells_w)
        num_cells_h = max(1, num_cells_h)

        print(f"Estimated Output Grid: {num_cells_w} x {num_cells_h} cells")
        est_width = num_cells_w * detected_w
        est_height = num_cells_h * detected_h
        if (
            abs(est_width - image.width) > detected_w / 2
            or abs(est_height - image.height) > detected_h / 2
        ):
            print(
                f"(Note: Estimated coverage based on cell count is {est_width}x{est_height}, original image is {image.width}x{image.height}. Check results.)"
            )

        if debug:
            print("\n--- Debug Mode ---")
            output_image = draw_grid_overlay(image, detected_w, detected_h)
        else:
            print("\n--- Generating Downsampled Image ---")
            output_image = create_downsampled_image(
                image,
                detected_w,
                detected_h,
                num_cells_w,
                num_cells_h,
                quantize,
            )

            if remove_background == "after":
                print("Removing background from the downsampled image...")
                # Call the background removal function
                output_image, debug_image = make_background_transparent(
                    output_image, debug=True
                )
                print("Background removed successfully.")

            # Apply automatic cropping if requested
            if crop and output_image.mode == "RGBA":
                print("Automatically cropping the image to non-transparent content...")
                output_image = crop_to_content(output_image)
                print(f"Image cropped to {output_image.width}x{output_image.height}")

        if debug:
            handle_output(
                debug_image,
                output_file,
                show,
                is_debug=True,
                default_title=f"{image_source} ({num_cells_w}x{num_cells_h})",
                ascii_space_width=ascii_space_width,
            )
        else:
            handle_output(
                output_image,
                output_file,
                show,
                is_debug=False,
                default_title=f"{image_source} ({num_cells_w}x{num_cells_h})",
                ascii_space_width=ascii_space_width,
            )

    else:
        print("\n--- Failure ---")
        print("Could not reliably determine grid dimensions.")
        sys.exit(1)  # Exit with error code if detection failed
