"""MMD (maximum mean discrepancy) two-sample test."""

from typing import Callable, Optional, Tuple

import numpy as np
from numpy import ndarray
from scipy.spatial.distance import euclidean, pdist, squareform

from pp5.distributions.kde import gaussian_kernel
from pp5.stats.two_sample.common import two_sample_kernel_permutation_test


# @numba.jit(nopython=True, parallel=_NUMBA_PARALLEL)
def mmd_statistic(K: np.ndarray, nx: int, ny: int, nx_idx=None, ny_idx=None) -> float:
    """
    Calculates MMD statistic of a kernel matrix

    :param K: Matrix of inner products of two pooled samples (X and Y) of
        shape (nx+ny, nx+ny).
    :param nx: Number of observations from X.
    :param ny: Number of observations from Y.
    :return: The MMD statistic.
    """
    sum_X = np.sum(K[0:nx, 0:nx]) / nx / nx
    sum_Y = np.sum(K[nx:, nx:]) / ny / ny
    sum_XY = np.sum(K[nx:, 0:nx]) / nx / ny
    return float(sum_X + sum_Y - 2.0 * sum_XY)


# @numba.jit(nopython=True, parallel=_NUMBA_PARALLEL)
def mmd_statistic_unbiased(
    K: np.ndarray, nx: int, ny: int, nx_idx=None, ny_idx=None
) -> float:
    """
    Calculates the unbiased MMD statistic of a kernel matrix.

    Unlike :obj:`mmd_statistic`, the within-sample diagonal entries k(x,x) are
    excluded and the within-sample sums are normalized by n*(n-1). This yields a
    U-statistic whose expectation equals MMD^2 (and is zero under H0: P_X = P_Y).

    :param K: Matrix of inner products of two pooled samples (X and Y) of
        shape (nx+ny, nx+ny).
    :param nx: Number of observations from X.
    :param ny: Number of observations from Y.
    :return: The unbiased MMD statistic.
    """
    K_X = K[0:nx, 0:nx]
    K_Y = K[nx:, nx:]
    sum_X = (np.sum(K_X) - np.trace(K_X)) / nx / (nx - 1)
    sum_Y = (np.sum(K_Y) - np.trace(K_Y)) / ny / (ny - 1)
    sum_XY = np.sum(K[nx:, 0:nx]) / nx / ny
    return float(sum_X + sum_Y - 2.0 * sum_XY)


def mmd_test(
    X: ndarray,
    Y: ndarray,
    k: int,
    similarity_fn: Callable[[ndarray, ndarray], float] = euclidean,
    kernel_fn: Callable[[ndarray], ndarray] = gaussian_kernel,
    unbiased: bool = True,
    k_min: Optional[int] = None,
    k_th: Optional[float] = float("inf"),
) -> Tuple[float, float, int]:
    """
    Applies a two-sample permutation test to determine whether the null hypothesis
    that two distributions are identical can be rejected, using the MMD approach.

    For parameters, see documentation of :obj:`two_sample_kernel_permutation_test`.

    :param unbiased: Whether to use the unbiased MMD U-statistic (excludes the
        within-sample diagonal). If False, the biased V-statistic is used.
    :return: MMD statistic value, p-value (significance).
    """
    return two_sample_kernel_permutation_test(
        X,
        Y,
        k,
        similarity_fn=similarity_fn,
        kernel_fn=kernel_fn,
        statistic_fn=mmd_statistic_unbiased if unbiased else mmd_statistic,
        k_min=k_min,
        k_th=k_th,
    )


def mmd_test_fast(
    X: ndarray,
    Y: ndarray,
    k: int,
    similarity_fn: Callable[[ndarray, ndarray], float] = euclidean,
    kernel_fn: Callable[[ndarray], ndarray] = gaussian_kernel,
    unbiased: bool = True,
    k_min: Optional[int] = None,
    k_th: Optional[float] = float("inf"),
    rng=None,
) -> Tuple[float, float, int]:
    """
    Faster variant of :obj:`mmd_test`, using a Gram row-sum identity so each
    permutation only requires summing an (nx, nx) sub-block instead of
    re-materializing and re-summing the full (nx+ny, nx+ny) permuted kernel matrix.

    The row sums R and grand total T of the full pooled Gram matrix are precomputed
    once. For a permutation assigning indices S (|S|=nx) to the X role, the X-X
    block sum is ``K[S,:][:,S].sum()``, the cross-block sum follows from
    ``R[S].sum() - Sxx``, and the Y-Y block sum follows from ``T`` by
    inclusion-exclusion. This avoids ever permuting or re-summing the full pooled
    matrix on each iteration.

    Requires a self-normalized kernel (k(z,z)=1, e.g. a Gaussian kernel) when
    unbiased=True, so that the trace of each within-sample block is exactly nx
    (resp. ny) and can be subtracted as a scalar instead of an explicit trace.

    For parameters, see documentation of :obj:`two_sample_kernel_permutation_test`.

    :param unbiased: Whether to use the unbiased MMD U-statistic. If False, the
        biased V-statistic is used.
    :param rng: Object to draw permutations from via ``rng.permutation(n)``, e.g. a
        :obj:`numpy.random.Generator` (from :obj:`numpy.random.default_rng`) or the
        ``numpy.random`` module itself. Defaults to the ``numpy.random`` module
        (the global numpy random state), matching the rest of this module.
    :return: MMD statistic value, p-value (significance), number of permutations.
    """
    nx = X.shape[0]
    ny = Y.shape[0]
    if nx < 2 or ny < 2:
        raise ValueError(
            "Permutation test requires at least two observations in each sample"
        )

    assert k > 0
    assert k_th > 0

    Z = np.vstack((X, Y))  # (nx+ny, m)
    D = squareform(pdist(Z, metric=similarity_fn))
    K = kernel_fn(D)

    return mmd_permutation_test_from_kernel(
        K, nx, ny, k, unbiased=unbiased, k_min=k_min, k_th=k_th, rng=rng
    )


def mmd_permutation_test_from_kernel(
    K: ndarray,
    nx: int,
    ny: int,
    k: int,
    unbiased: bool = True,
    k_min: Optional[int] = None,
    k_th: Optional[float] = float("inf"),
    rng=None,
) -> Tuple[float, float, int]:
    """
    Core row-sum-identity permutation loop of :obj:`mmd_test_fast`, operating on
    an already-computed pooled Gram matrix ``K`` rather than computing it from
    raw observations. Useful when the same ``K`` is reused across multiple calls
    (e.g. running both the unbiased and biased MMD^2 statistic on the same pair
    without recomputing the kernel matrix twice).

    See :obj:`mmd_test_fast` for the row-sum identity, the self-normalized-kernel
    requirement when ``unbiased=True``, and the rest of the parameter semantics.

    :param K: Pooled Gram matrix of shape (nx+ny, nx+ny).
    :return: MMD statistic value, p-value (significance), number of permutations.
    """
    assert k > 0
    assert k_th > 0
    k_min = k_min or k
    rng = rng if rng is not None else np.random

    if unbiased:
        diag = np.diagonal(K)
        assert np.allclose(diag, 1.0), (
            "mmd_test_fast(unbiased=True) requires a self-normalized kernel "
            f"(k(z,z)==1); got diagonal values in [{diag.min()}, {diag.max()}]"
        )

    def _mmd_from_block_sums(_sxx: float, _syy: float, _sxy: float) -> float:
        # MMD^2 statistic from Gram-matrix block sums rather than the matrix
        # itself: Sxx/Syy/Sxy are the sums of the X-X, Y-Y (each including the
        # diagonal) and X-Y blocks respectively.
        if unbiased:
            return (
                (_sxx - nx) / (nx * (nx - 1))
                + (_syy - ny) / (ny * (ny - 1))
                - 2.0 * _sxy / (nx * ny)
            )
        return _sxx / nx**2 + _syy / ny**2 - 2.0 * _sxy / (nx * ny)

    # Row sums and grand total of the full pooled Gram matrix, computed once.
    R = K.sum(axis=1)  # (nx+ny,)
    T = R.sum()

    sxx0 = K[:nx, :nx].sum()
    sxy0 = K[:nx, nx:].sum()
    syy0 = K[nx:, nx:].sum()
    stat_val = _mmd_from_block_sums(sxx0, syy0, sxy0)

    pval, count, curr_permutation = 0, 0, 0
    for curr_permutation in range(1, k + 1):
        # Indices (in the pooled 0..nx+ny-1 order) currently assigned the X role.
        S = rng.permutation(nx + ny)[:nx]
        sxx = K[np.ix_(S, S)].sum()
        rs = R[S].sum()
        sxy = rs - sxx
        syy = T - 2.0 * rs + sxx
        stat_val_perm = _mmd_from_block_sums(sxx, syy, sxy)

        if stat_val <= stat_val_perm:
            count += 1

        # The smallest pval this test can detect is 1/(k+1).
        pval = (count + 1) / (curr_permutation + 1)

        # Check early termination criterion: high pval after k_min permutations.
        if (curr_permutation >= k_min) and (pval >= k_th * 1 / (curr_permutation + 1)):
            break

    return stat_val, pval, curr_permutation
