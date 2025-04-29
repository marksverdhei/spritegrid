import numpy as np
from scipy.spatial.distance import cdist, euclidean
from PIL import Image


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
