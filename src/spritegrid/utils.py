import numpy as np
from scipy.spatial.distance import cdist, euclidean
from PIL import Image


def convert_image_to_ascii(
    image: Image.Image,
    ascii_space_width: int = 1,
    reset_after_each_pixel: bool = False,
    character: str = " "
) -> str:
    assert ascii_space_width is not None, "ascii_space_width must be specified"
    assert ascii_space_width > 0, "ascii_space_width must be greater than 0"
    width, height = image.size
    image = image.load()
    strings = []
    yr, xr = range(height), range(width)
    if character == " ":
        templ = "\x1b[48;2;{};{};{}m"
    else:
        templ = "\x1b[38;2;{};{};{}m"

    reset_needed = False
    for y in yr:
        for x in xr:
            r, g, b, *a = image[x, y]
            a = a[0] if a else 255
            if a == 0:
                s = " " * ascii_space_width
                if reset_needed:
                    s = "\x1b[0m" + s

                # Transparent pixel -> uncolored space
                strings.append(s)
                reset_needed = False
            else:
                s = templ.format(r, g, b) + (character * ascii_space_width)

                # s += "\x1b[0m"
                if reset_after_each_pixel:
                    s += "\x1b[0m"
                else:
                    reset_needed = True

                strings.append(s)

        if reset_needed:
            strings.append("\x1b[0m")
            reset_needed = False
        strings.append("\n")

    return "".join(strings)


def convert_image_to_ascii_half_block(
    image: Image.Image,
) -> str:
    width, height = image.size
    image = image.load()
    strings = []
    for y in range(0, height, 2):
        for x in range(width):
            r1, g1, b1, *a1 = image[x, y]
            a1 = a1[0] if a1 else 255

            if y + 1 < height:
                r2, g2, b2, *a2 = image[x, y+1]
                a2 = a2[0] if a2 else 255
            else:
                r2, g2, b2, a2 = 0, 0, 0, 0

            if a1 == 0 and a2 == 0:
                strings.append(" ")
            elif a1 == 0:
                # Upper pixel is transparent, lower is not
                strings.append(f"\x1b[38;2;{r2};{g2};{b2}m\u2584")
            elif a2 == 0:
                # Lower pixel is transparent, upper is not
                strings.append(f"\x1b[38;2;{r1};{g1};{b1}m\u2580")
            else:
                # Both pixels are colored
                strings.append(f"\x1b[48;2;{r1};{g1};{b1}m\x1b[38;2;{r2};{g2};{b2}m\u2584")
        strings.append("\x1b[0m\n")
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
