import numpy as np
from scipy.spatial.distance import cdist, euclidean
from PIL import Image


def convert_image_to_ascii(
    image: Image.Image,
    ascii_space_width: int = 1,
) -> str:
    assert ascii_space_width is not None, "ascii_space_width must be specified"
    assert ascii_space_width > 0, "ascii_space_width must be greater than 0"
    width, height = image.size
    image = image.load()
    strings = []
    for y in range(height):
        for x in range(width):
            r, g, b, *a = image[x, y]
            a = a[0] if a else 255
            if a == 0:
                # Transparent pixel -> uncolored space
                strings.append(" " * ascii_space_width)
            else:
                # ANSI escape: background color with truecolor
                strings.append(f"\x1b[48;2;{r};{g};{b}m" + (" " * ascii_space_width) + "\x1b[0m")
        strings.append("\n")
    return "".join(strings)


def naive_median(X: np.ndarray) -> np.ndarray:
    """
    Returns the naive median of points in X.

    By orip, released under zlib license.
    Lightly modified for readability.
    https://stackoverflow.com/questions/30299267/geometric-median-of-multidimensional-points

    """
    return np.median(X, axis=0)


def geometric_median(X: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """
    Returns the geometric median of points in X.

    By orip, released under zlib license.
    Lightly modified for readability.
    https://stackoverflow.com/questions/30299267/geometric-median-of-multidimensional-points

    """
    y = np.mean(X, 0)

    while True:
        D = cdist(X, [y])
        nonzeros = (D != 0)[:, 0]

        Dinv = 1 / D[nonzeros]
        Dinvs = np.sum(Dinv)
        W = Dinv / Dinvs
        T = np.sum(W * X[nonzeros], 0)

        num_zeros = len(X) - np.sum(nonzeros)
        if num_zeros == 0:
            y1 = T
        elif num_zeros == len(X):
            return y
        else:
            R = (T - y) * Dinvs
            r = np.linalg.norm(R)
            rinv = 0 if r == 0 else num_zeros / r
            y1 = max(0, 1 - rinv) * T + min(1, rinv) * y

        if euclidean(y, y1) < eps:
            return y1

        y = y1


def crop_to_content(image: Image.Image) -> Image.Image:
    """
    Automatically crop an image with an alpha channel to the first and last rows and columns
    where all pixels aren't transparent.

    Args:
        image: A PIL Image object with an alpha channel (RGBA mode)

    Returns:
        A cropped PIL Image object
    """
    # Ensure the image has an alpha channel
    if image.mode != "RGBA":
        return image

    # Get alpha channel
    alpha = np.array(image.split()[3])

    # Find the bounding box of non-transparent pixels
    # Get the non-zero alpha locations
    non_transparent = np.where(alpha > 0)

    if len(non_transparent[0]) == 0:  # No non-transparent pixels found
        return image

    # Find the bounding box
    min_y, max_y = non_transparent[0].min(), non_transparent[0].max()
    min_x, max_x = non_transparent[1].min(), non_transparent[1].max()

    # Add 1 to max values because PIL's crop is inclusive of the start coordinates
    # but exclusive of the end coordinates
    # Crop the image to the bounding box
    cropped_image = image.crop((min_x, min_y, max_x + 1, max_y + 1))

    return cropped_image


def enforce_symmetry(image: Image.Image) -> Image.Image:
    """
    Enforce horizontal symmetry on an image using confidence-based consensus.

    For each pair of horizontally mirrored pixels, if they differ, the pixel
    with higher "confidence" (further from the decision boundary between colors)
    is used for both positions.

    This is useful for pixel art that should be symmetric but has minor
    asymmetries from AI upscaling artifacts.

    Args:
        image: A PIL Image object (RGB or RGBA mode)

    Returns:
        A new PIL Image with enforced horizontal symmetry
    """
    if image.mode not in ["RGB", "RGBA"]:
        image = image.convert("RGBA")

    arr = np.array(image)
    h, w = arr.shape[:2]
    has_alpha = arr.shape[2] == 4

    # Calculate confidence for each pixel
    # Confidence = how "decisive" the color is (distance from gray/ambiguous)
    def pixel_confidence(pixel):
        if has_alpha and pixel[3] < 128:
            return 0  # Transparent pixels have no confidence

        r, g, b = pixel[0], pixel[1], pixel[2]

        # Confidence based on saturation and distance from middle gray
        max_c, min_c = max(r, g, b), min(r, g, b)
        saturation = (max_c - min_c) if max_c > 0 else 0

        # Distance from middle gray (128, 128, 128)
        brightness = (int(r) + int(g) + int(b)) / 3
        gray_distance = abs(brightness - 128)

        # Combined confidence: high saturation OR very dark/bright
        return saturation + gray_distance

    # Process each mirror pair
    result = arr.copy()

    for y in range(h):
        for x in range(w // 2):
            mx = w - 1 - x  # Mirror x position

            p1 = arr[y, x]
            p2 = arr[y, mx]

            # Check if pixels are different
            if not np.array_equal(p1, p2):
                c1 = pixel_confidence(p1)
                c2 = pixel_confidence(p2)

                # Use the pixel with higher confidence for both positions
                if c1 >= c2:
                    result[y, mx] = p1
                else:
                    result[y, x] = p2

    return Image.fromarray(result, image.mode)
