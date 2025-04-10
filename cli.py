import argparse
from src.main import main


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
        help="Path to the local image file or URL of the image.",
    )
    parser.add_argument(
        "--min-grid",
        type=int,
        default=4,
        help="Minimum expected grid dimension (width or height) for peak detection. (Default: 4)",
    )
    # --- Modified/New Arguments ---
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
        '-b', '--remove-background',
        nargs='?',
        const='default',
        choices=['before', 'after', 'default'],
        help='Remove background (optionally specify "before" or "after")'
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
