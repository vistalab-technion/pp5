"""Welch T-squared two-sample test."""

from typing import Callable, Optional, Tuple

import numpy as np
from numpy import ndarray
from scipy.spatial.distance import sqeuclidean

from pp5.stats.two_sample.common import two_sample_kernel_permutation_test


# @numba.jit(nopython=True, parallel=_NUMBA_PARALLEL)
def _tw2_statistic(D: ndarray, nx: int, ny: int, nx_idx=None, ny_idx=None) -> float:
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
    denumerator = (sum_X / (nx**2) / (nx - 1)) + (sum_Y / (ny**2) / (ny - 1))
    if denumerator < 1e-12:  # prevent division by zero
        return 0.0
    return float(factor * enumerator / denumerator)


def tw_test(
    X: ndarray,
    Y: ndarray,
    k: int,
    similarity_fn: Callable[[ndarray, ndarray], float] = sqeuclidean,
    kernel_fn: Callable[[ndarray], ndarray] = lambda x: x,
    k_min: Optional[int] = None,
    k_th: Optional[float] = float("inf"),
) -> Tuple[float, float, int]:
    """
    Applies a two-sample permutation test to determine whether the null hypothesis
    that two distributions are identical can be rejected, using the Tw^2 Welch
    statistic based on pairwise squared-euclidean distances.

    For parameters, see documentation of :obj:`two_sample_kernel_permutation_test`.

    :return: Tw^2 statistic value, p-value (significance).
    """
    return two_sample_kernel_permutation_test(
        X,
        Y,
        k,
        similarity_fn=similarity_fn,
        kernel_fn=kernel_fn,
        statistic_fn=_tw2_statistic,
        k_min=k_min,
        k_th=k_th,
    )
