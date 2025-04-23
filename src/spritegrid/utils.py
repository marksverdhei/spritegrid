import numpy as np
from scipy.spatial.distance import cdist, euclidean


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
