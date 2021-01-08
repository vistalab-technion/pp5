import numba
import numpy as np
from numpy.random import permutation
from scipy.spatial.distance import pdist, squareform


def tw_test(X: np.ndarray, Y: np.ndarray, k: int = 1000):
    """
    Calculates the Tw^2 Welch statistic based on distances.
    :param X: (N, nx) array containing sample X
    :param Y: (N, ny) array containing sample Y
    :param k: number of permutations for significance evaluation
    :return: Tw^2 statistic, p-value (significance).
    """
    # sample sizes
    nx = X.shape[1]
    ny = Y.shape[1]

    # pooled vectors
    Z = np.hstack((X, Y))

    # squared distances
    D = 0.5 * squareform(pdist(Z.T)) ** 2
    D = D.astype(np.float32)

    t2, p = _tw_test_inner(D, nx, ny, k)
    return t2, p


@numba.jit(nopython=True, parallel=False)  # parallel doesn't seem to help
def _tw_test_inner(D: np.ndarray, nx: int, ny: int, k: int):
    """
    Calculates p-value based on Tw^2 test.
    :param D: Matrix of squared distanced of two pooled samples (X and Y) of
        shape (nx+ny, nx+ny).
    :param nx: Number of observations from X.
    :param ny: Number of observations from Y.
    :param k: Number of permutations for significance evaluation.
    :return: Tw^2 statistic of un-permuted distances, and p-value.
    """
    t2 = tw2_statistic(D, nx, ny)
    ss = np.zeros(k)
    for i in range(k):
        idx = permutation(nx + ny)
        t2_perm = tw2_statistic(D[idx, :][:, idx], nx, ny)
        if t2 <= t2_perm:
            ss[i] = 1

    p = np.mean(ss)
    return t2, p


@numba.jit(nopython=True, parallel=False)
def tw2_statistic(D: np.ndarray, nx: int, ny: int):
    """
    Calculates T statistic of a distance matrix
    :param D: Matrix of squared distances of two pooled samples (X and Y) of
        shape (nx+ny, nx+ny).
    :param nx: Number of observations from X.
    :param ny: Number of observations from Y.
    :return: The T2 statistic.
    """
    factor = (nx + ny) / nx / ny
    sum_X = np.sum(D[0:nx, 0:nx])
    sum_Y = np.sum(D[nx:, nx:])
    sum_Z = np.sum(D)
    enumerator = sum_Z / (nx + ny) - sum_X / nx - sum_Y / ny
    denumerator = (sum_X / (nx ** 2) / (nx - 1)) + (sum_Y / (ny ** 2) / (ny - 1))
    if denumerator < 1e-12:  # prevent division by zero
        return 0.0
    return factor * enumerator / denumerator
