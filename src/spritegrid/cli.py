import argparse
import sys
from typing import Optional, Tuple

from .main import main


def parse_size(size_str: str) -> Tuple[int, int]:
    """Parse a size string like '32' or '32x48' into (width, height) tuple."""
    if "x" in size_str.lower():
        parts = size_str.lower().split("x")
        if len(parts) != 2:
            raise argparse.ArgumentTypeError(
                f"Invalid size format: {size_str}. Use 'WxH' (e.g., '32x48') or single number (e.g., '32')."
            )
        try:
            return (int(parts[0]), int(parts[1]))
        except ValueError:
            raise argparse.ArgumentTypeError(
                f"Invalid size format: {size_str}. Width and height must be integers."
            )
    else:
        try:
            size = int(size_str)
            return (size, size)
        except ValueError:
            raise argparse.ArgumentTypeError(
                f"Invalid size format: {size_str}. Use 'WxH' (e.g., '32x48') or single number (e.g., '32')."
            )


def parse_args() -> argparse.Namespace:
    """
    Main function to parse arguments, load image, detect grid, and generate output/debug image.
    """
    parser = argparse.ArgumentParser(
        description="Detect grid in AI pixel art & create downsampled image or debug overlay."
    )
    parser.add_argument(
        "image_source",
        type=str,
        help="Path to the local image file or URL of the image.",
    )

    parser.add_argument(
        "--min-grid",
        type=int,
        default=4,
        help="Minimum expected grid dimension (width or height) for peak detection. (Default: 4)",
    )

    parser.add_argument(
        "-o",
        "--output",
        metavar="FILENAME",
        dest="output_file",
        type=str,
        help="Save the output image (downsampled by default, or debug overlay if -d is used) to FILENAME.",
    )
    parser.add_argument(
        "-i",
        "--show",
        action="store_true",
        help="Display the output image (downsampled by default, or debug overlay if -d is used) using the default system viewer.",
    )
    parser.add_argument(
        "-d",
        "--debug",
        action="store_true",
        help="Enable debug mode: output/show a grid overlay instead of the downsampled image. Defaults to showing if -o or -i are not specified.",
    )

    parser.add_argument(
        "-q",
        "--quantize",
        type=int,
        default=8,
        help="Per-channel bit depth for quantization. If not specified, the image will not be quantized.",
    )

    parser.add_argument(
        "-b",
        "--remove-background",
        nargs="?",
        const="default",
        choices=["before", "after", "default"],
        help='Remove background (optionally specify "before" or "after")',
    )

    parser.add_argument(
        "-c",
        "--crop",
        action="store_true",
        help="Automatically crop the image to the first and last rows and columns where all pixels aren't transparent.",
    )

    parser.add_argument(
        "-a",
        "--ascii",
        nargs="?",
        choices=[1, 2],
        const=1,
        type=int,
    )

    parser.add_argument(
        "-s",
        "--symmetric",
        action="store_true",
        help="Enforce horizontal symmetry using confidence-based consensus. "
             "Useful for symmetric sprites with minor AI upscaling artifacts.",
    )

    args = parser.parse_args()

    # Ensure the crop argument is passed correctly
    if args.crop and args.remove_background not in [None, "default"]:
        parser.error(
            "The --crop option cannot be used with --remove-background set to 'before' or 'after'."
        )

    return args


def cli() -> None:
    """
    The main entry point for the command line interface.
    """
    args = parse_args()
    main(
        image_source=args.image_source,
        min_grid=args.min_grid,
        output_file=args.output_file,
        show=args.show,
        debug=args.debug,
        quantize=args.quantize,
        remove_background=args.remove_background,
        crop=args.crop,
        ascii_space_width=args.ascii,
        symmetric=args.symmetric,
    )


def parse_crop_scale_args() -> argparse.Namespace:
    """
    Parse arguments for the crop-and-scale command.
    """
    parser = argparse.ArgumentParser(
        prog="spritegrid-crop",
        description=(
            "Crop AI-generated images to content and scale to pixel art sizes. "
            "This tool auto-detects the subject in high-resolution images, crops away "
            "excess background, and scales to standard pixel art sizes (16x16, 32x32, 64x64, etc.) "
            "using NEAREST resampling to preserve hard pixel edges."
        ),
    )
    parser.add_argument(
        "image_source",
        type=str,
        help="Path to the local image file or URL of the image.",
    )
    parser.add_argument(
        "-s",
        "--size",
        type=parse_size,
        default=(32, 32),
        help=(
            "Target size for the output. Can be a single number (e.g., '32' for 32x32) "
            "or 'WxH' format (e.g., '32x48'). Default: 32"
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        metavar="FILENAME",
        dest="output_file",
        type=str,
        help="Save the output image to FILENAME.",
    )
    parser.add_argument(
        "-i",
        "--show",
        action="store_true",
        help="Display the output image using the default system viewer.",
    )
    parser.add_argument(
        "-p",
        "--padding",
        type=int,
        default=0,
        help="Padding (in pixels) to add around detected content before scaling. Default: 0",
    )
    parser.add_argument(
        "--alpha-threshold",
        type=int,
        default=0,
        help=(
            "Pixels with alpha <= this value are considered transparent. "
            "Increase to catch semi-transparent backgrounds. Default: 0"
        ),
    )
    parser.add_argument(
        "--color-tolerance",
        type=int,
        default=0,
        help=(
            "Color distance tolerance for background detection (used for RGB images "
            "or when --background-color is specified). Default: 0"
        ),
    )
    parser.add_argument(
        "--background-color",
        type=str,
        default=None,
        help=(
            "Specify background color as 'R,G,B' or 'R,G,B,A' (e.g., '255,255,255' for white). "
            "If not specified, uses alpha channel for RGBA images or corner pixel for RGB."
        ),
    )
    parser.add_argument(
        "--no-aspect",
        action="store_true",
        help="Force output to exact target size without maintaining aspect ratio.",
    )
    parser.add_argument(
        "--center",
        action="store_true",
        help=(
            "Center the scaled content on a canvas of the exact target size. "
            "Useful for ensuring consistent output dimensions."
        ),
    )

    return parser.parse_args()


def crop_scale_cli() -> None:
    """
    Entry point for the crop-and-scale command line interface.
    """
    from PIL import Image
    from .main import load_image, handle_output
    from .crop_and_scale import crop_and_scale, crop_and_scale_centered

    args = parse_crop_scale_args()

    # Parse background color if provided
    background_color = None
    if args.background_color:
        try:
            parts = [int(x.strip()) for x in args.background_color.split(",")]
            if len(parts) not in (3, 4):
                print(
                    f"Error: Background color must be 'R,G,B' or 'R,G,B,A', got: {args.background_color}",
                    file=sys.stderr,
                )
                sys.exit(1)
            background_color = tuple(parts)
        except ValueError:
            print(
                f"Error: Invalid background color format: {args.background_color}",
                file=sys.stderr,
            )
            sys.exit(1)

    # Load image
    print(f"Loading image from: {args.image_source}")
    image = load_image(args.image_source)

    if image is None:
        sys.exit(1)

    print(f"Image loaded successfully ({image.width}x{image.height}, Mode: {image.mode}).")

    # Determine target size
    target_size = args.size
    if target_size[0] == target_size[1]:
        target_size = target_size[0]  # Use single int for square targets

    # Process image
    try:
        if args.center:
            output_image = crop_and_scale_centered(
                image,
                target_size=target_size,
                padding=args.padding,
                background_color=background_color,
                alpha_threshold=args.alpha_threshold,
                color_tolerance=args.color_tolerance,
            )
        else:
            output_image = crop_and_scale(
                image,
                target_size=target_size,
                padding=args.padding,
                background_color=background_color,
                alpha_threshold=args.alpha_threshold,
                color_tolerance=args.color_tolerance,
                maintain_aspect=not args.no_aspect,
            )

        print(f"Processed image: {output_image.width}x{output_image.height}")

    except Exception as e:
        print(f"Error processing image: {e}", file=sys.stderr)
        sys.exit(1)

    # Handle output
    handle_output(
        output_image,
        args.output_file,
        args.show,
        is_debug=False,
        default_title=f"Crop & Scale: {args.image_source}",
    )


if __name__ == "__main__":
    cli()
