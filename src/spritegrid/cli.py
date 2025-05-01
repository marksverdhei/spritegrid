import argparse
from .main import main


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
    )


if __name__ == "__main__":
    cli()
