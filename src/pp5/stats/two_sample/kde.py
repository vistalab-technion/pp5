"""KDE-L1 two-sample test on the flat torus."""

from functools import partial
from typing import Callable, Optional, Tuple

import numpy as np
from numpy import ndarray

from pp5.distributions.kde import kde_2d, torus_gaussian_kernel_2d
from pp5.stats.two_sample.common import (
    two_sample_kernel_permutation_test,
    two_sample_kernel_permutation_test_inner,
)


# @numba.jit(nopython=True, parallel=_NUMBA_PARALLEL)
def _kde_statistic(
    K: np.ndarray,
    nx: int,
    ny: int,
    nx_idx: Optional[np.ndarray] = None,
    ny_idx: Optional[np.ndarray] = None,
) -> float:
    """
    Calculates KDE-based statistic of a kernel matrix

    :param K: Matrix of shape (nx+ny, M, M), containing the (M, M) contributions of
        N=nx+ny observations to the KDE estimate.
    :param nx: Number of observations from X.
    :param ny: Number of observations from Y.
    :return: The KDE statistic: L1 distance between the KDEs of X and Y.
    """

    # Apply a reduction to compute X and Y's KDEs from the contribution of each
    # of their observations
    if nx_idx is not None and ny_idx is not None:
        kde_X = K[nx_idx]
        kde_Y = K[ny_idx]
    else:
        kde_X = K[:nx]
        kde_Y = K[nx:]

    kde_X = np.sum(kde_X, axis=0)  # (nx, M, M) -> (M, M)
    kde_X /= np.sum(kde_X)

    kde_Y = np.sum(kde_Y, axis=0)  # (ny, M, M) -> (M, M)
    kde_Y /= np.sum(kde_Y)

    # w2_dist = w2_dist_sinkhorn(kde_X, kde_Y, sigma=1e-5, niter=250)[0]
    l1_dist = np.sum(np.abs(kde_X - kde_Y)).item()
    return l1_dist


def kde2d_test(
    X: ndarray,
    Y: ndarray,
    k: int,
    n_bins: int,
    grid_low: float,
    grid_high: float,
    dtype: np.dtype,
    kernel_fn: Callable,
    k_min: Optional[int] = None,
    k_th: Optional[float] = float("inf"),
) -> Tuple[float, float, int]:
    """
    Applies a two-sample permutation test to determine whether the null hypothesis
    that two distributions are identical can be rejected, using the KDE approach.

    For parameters, see documentation of :obj:`two_sample_kernel_permutation_test`.

    :param n_bins: Number of bins for KDE estimation.
    :param grid_low: Smallest value on the evaluation grid, inclusive.
    :param grid_high: Largest value on the evaluation grid, exclusive.
    :param kernel_fn: Kernel for the 2D KDE (not for the permutation test itself).
    :return: KDE statistic value, p-value (significance).
    """

    def _kde_2d_kernel_fn(Z: np.ndarray):
        # Z has shape (N, 2)

        K = kde_2d(
            x1=Z[:, 0],
            x2=Z[:, 1],
            kernel_fn=kernel_fn,
            n_bins=n_bins,
            grid_low=grid_low,
            grid_high=grid_high,
            dtype=dtype,
            # Disabling reduction is necessary to avoid re-calculating the entire KDE
            # on each permutation.
            reduce=False,
        )  # K is (n_bins,n_bins)

        # Transpose from (M, M, N) to (N, M, M) where N=nx+ny, so that we can permute
        # over the first dimension.
        return K.transpose(2, 0, 1)

    return two_sample_kernel_permutation_test(
        X,
        Y,
        k,
        similarity_fn=None,
        kernel_fn=_kde_2d_kernel_fn,
        statistic_fn=_kde_statistic,
        k_min=k_min,
        k_th=k_th,
    )


# -----------------------------------------------------------------------------
# Per-group KDE-L1 permutation test (the "double-slab trick").
#
# When comparing two codons with different CV-chosen bandwidths (sigma_1, sigma_2),
# we cannot precompute a single KDE slab per observation because the same observation
# may be assigned to group X or group Y across permutations. The fix: precompute TWO
# slabs per observation — one with sigma_1 and one with sigma_2 — and have the
# permutation loop pick the slab matching the role the observation currently plays.
#
# Bandwidth attaches to the group *role*, not to the observation. (sigma_1, sigma_2)
# are fixed nuisance constants precomputed once from the original labels; every
# permutation applies the same function T(Z, L; sigma_1, sigma_2). Under H0 the
# labels are exchangeable, so the permutation null is correctly sampled.
#
# See docs/kernel_bandwidth_cv.md for the full derivation.
# -----------------------------------------------------------------------------


def _kde_statistic_pergroup(
    K: Tuple[np.ndarray, np.ndarray],
    nx: int,
    ny: int,
    nx_idx: Optional[np.ndarray] = None,
    ny_idx: Optional[np.ndarray] = None,
) -> float:
    """KDE-L1 statistic from two slab stacks (one per group bandwidth).

    :param K: Tuple (K_x, K_y). Each has shape (nx+ny, M, M). Observation i
        contributes its K_x slab when assigned to group X, and its K_y slab when
        assigned to group Y.
    :param nx: Number of observations from X.
    :param ny: Number of observations from Y.
    :param nx_idx: Indices of observations currently assigned to group X (for a
        given permutation). If None, the unpermuted split [:nx] is used.
    :param ny_idx: Indices of observations currently assigned to group Y. If None,
        the unpermuted split [nx:] is used.
    :return: L1 distance between the two pooled, normalized KDEs.
    """
    K_x, K_y = K

    x_slabs = K_x[nx_idx] if nx_idx is not None else K_x[:nx]
    y_slabs = K_y[ny_idx] if ny_idx is not None else K_y[nx:]

    kde_X = np.sum(x_slabs, axis=0)
    kde_X /= np.sum(kde_X)

    kde_Y = np.sum(y_slabs, axis=0)
    kde_Y /= np.sum(kde_Y)

    return float(np.sum(np.abs(kde_X - kde_Y)).item())


def kde_2d_slab_stacks(
    Z: ndarray,
    n_bins: int,
    grid_low: float,
    grid_high: float,
    dtype: np.dtype,
    sigma_x_rad: float,
    sigma_y_rad: float,
) -> Tuple[ndarray, ndarray]:
    """
    Precomputes the per-observation KDE "slab" stacks used by the per-group
    KDE-L1 permutation test (naive and fast variants alike), for both group
    bandwidths.

    When ``sigma_x_rad == sigma_y_rad`` the second slab computation is skipped
    and the same array is returned for both, since the two bandwidths would
    otherwise produce an identical stack.

    :param Z: (nx+ny, 2) pooled phi/psi observations in radians.
    :param n_bins: M, grid size per axis.
    :param grid_low: Grid lower bound (inclusive), radians.
    :param grid_high: Grid upper bound (exclusive), radians.
    :param dtype: Slab dtype.
    :param sigma_x_rad: Bandwidth (radians) for the X group role.
    :param sigma_y_rad: Bandwidth (radians) for the Y group role.
    :return: Tuple (K_x, K_y), each of shape (nx+ny, n_bins, n_bins).
    """

    def _compute_slabs(sigma_rad: float) -> ndarray:
        slabs = kde_2d(
            x1=Z[:, 0],
            x2=Z[:, 1],
            kernel_fn=partial(torus_gaussian_kernel_2d, sigma=sigma_rad),
            n_bins=n_bins,
            grid_low=grid_low,
            grid_high=grid_high,
            dtype=dtype,
            reduce=False,
        )
        # kde_2d returns (M, M, N); permutation indexing needs (N, M, M).
        return slabs.transpose(2, 0, 1)

    K_x = _compute_slabs(sigma_x_rad)
    K_y = K_x if sigma_x_rad == sigma_y_rad else _compute_slabs(sigma_y_rad)
    return K_x, K_y


def kde2d_test_pergroup(
    X: ndarray,
    Y: ndarray,
    k: int,
    n_bins: int,
    grid_low: float,
    grid_high: float,
    dtype: np.dtype,
    sigma_x_rad: float,
    sigma_y_rad: float,
    k_min: Optional[int] = None,
    k_th: Optional[float] = float("inf"),
) -> Tuple[float, float, int]:
    """KDE-L1 permutation test on the torus with per-group bandwidths.

    Precomputes two stacks of per-sample kernel slabs on the M x M torus grid — one
    with ``sigma_x_rad`` (for observations assigned to group X) and one with
    ``sigma_y_rad`` (for group Y). See the block comment above for the statistical
    justification of the double-slab approach.

    When ``sigma_x_rad == sigma_y_rad`` this reduces to the fixed-bandwidth KDE-L1
    test and skips the second slab computation.

    :param X: (nx, 2) phi/psi observations in radians.
    :param Y: (ny, 2) phi/psi observations in radians.
    :param k: Number of permutations.
    :param n_bins: M — grid size per axis.
    :param grid_low: Grid lower bound (inclusive), radians.
    :param grid_high: Grid upper bound (exclusive), radians.
    :param dtype: Slab dtype.
    :param sigma_x_rad: Bandwidth (radians) for the X group role.
    :param sigma_y_rad: Bandwidth (radians) for the Y group role.
    :param k_min: Early termination minimum permutations.
    :param k_th: Early termination threshold.
    :return: (ddist, pval, n_permutations).
    """
    nx, ny = X.shape[0], Y.shape[0]
    if nx < 2 or ny < 2:
        raise ValueError(
            "Permutation test requires at least two observations in each sample"
        )

    # Pool the observations. Any permutation is expressed as a split of indices
    # 0..nx+ny-1 into (nx_idx, ny_idx).
    Z = np.vstack((X, Y))
    K_x, K_y = kde_2d_slab_stacks(
        Z, n_bins, grid_low, grid_high, dtype, sigma_x_rad, sigma_y_rad
    )

    k_min = k_min if k_min else k
    assert k > 0
    assert k_th is None or k_th > 0

    return two_sample_kernel_permutation_test_inner(
        (K_x, K_y),
        nx,
        ny,
        k,
        _kde_statistic_pergroup,
        permute_pairs=False,
        k_min=k_min,
        k_th=k_th if k_th is not None else float("inf"),
    )


def kde2d_test_pergroup_fast(
    X: ndarray,
    Y: ndarray,
    k: int,
    n_bins: int,
    grid_low: float,
    grid_high: float,
    dtype: np.dtype,
    sigma_x_rad: float,
    sigma_y_rad: float,
    k_min: Optional[int] = None,
    k_th: Optional[float] = float("inf"),
    rng=None,
    batch_size: int = 1000,
) -> Tuple[float, float, int]:
    """
    Faster variant of :obj:`kde2d_test_pergroup`, batching permutations through a
    BLAS matrix product (selection matrix x slab stack) instead of recomputing the
    KDE sums one permutation at a time.

    Uses the same double-slab precomputation as :obj:`kde2d_test_pergroup` and the
    same `X = idx[:nx]` permutation convention as the rest of this module (e.g.
    :obj:`mmd_test_fast`), so it reproduces :obj:`kde2d_test_pergroup` exactly
    under an identical seed, regardless of whether nx or ny is larger.

    Early termination (`k_min`/`k_th`) is only checked once per batch, not
    once per permutation as in the naive test, so with early termination enabled
    the exact number of permutations run may differ slightly from
    :obj:`kde2d_test_pergroup`; the returned p-value remains valid either way.

    :param X: (nx, 2) phi/psi observations in radians.
    :param Y: (ny, 2) phi/psi observations in radians.
    :param k: Number of permutations.
    :param n_bins: M, grid size per axis.
    :param grid_low: Grid lower bound (inclusive), radians.
    :param grid_high: Grid upper bound (exclusive), radians.
    :param dtype: Slab dtype.
    :param sigma_x_rad: Bandwidth (radians) for the X group role.
    :param sigma_y_rad: Bandwidth (radians) for the Y group role.
    :param k_min: Early termination minimum permutations.
    :param k_th: Early termination threshold.
    :param rng: Object to draw permutations from via ``rng.permutation(n)``, e.g. a
        :obj:`numpy.random.Generator` (from :obj:`numpy.random.default_rng`) or the
        ``numpy.random`` module itself. Defaults to the ``numpy.random`` module
        (the global numpy random state), matching the rest of this module.
    :param batch_size: Number of permutations drawn and evaluated per BLAS call.
    :return: (ddist, pval, n_permutations).
    """
    nx, ny = X.shape[0], Y.shape[0]
    if nx < 2 or ny < 2:
        raise ValueError(
            "Permutation test requires at least two observations in each sample"
        )

    Z = np.vstack((X, Y))  # (nx+ny, 2)
    K_x, K_y = kde_2d_slab_stacks(
        Z, n_bins, grid_low, grid_high, dtype, sigma_x_rad, sigma_y_rad
    )
    N = nx + ny
    # Flatten the (M, M) grid to P=M*M so a permutation's KDE sum is a single
    # (N,) . (N, P) reduction, batchable as a (batch, N) @ (N, P) matmul.
    K_x_flat = np.ascontiguousarray(K_x.reshape(N, -1), dtype=dtype)
    K_y_flat = (
        K_x_flat
        if K_y is K_x
        else np.ascontiguousarray(K_y.reshape(N, -1), dtype=dtype)
    )

    return kde_l1_permutation_test_from_slabs(
        K_x_flat,
        K_y_flat,
        nx,
        ny,
        k,
        k_min=k_min,
        k_th=k_th,
        rng=rng,
        batch_size=batch_size,
    )


def kde_l1_permutation_test_from_slabs(
    K_x_flat: ndarray,
    K_y_flat: ndarray,
    nx: int,
    ny: int,
    k: int,
    k_min: Optional[int] = None,
    k_th: Optional[float] = float("inf"),
    rng=None,
    batch_size: int = 1000,
) -> Tuple[float, float, int]:
    """
    Core BLAS-batched permutation loop of :obj:`kde2d_test_pergroup_fast`,
    operating on already-flattened ``(nx+ny, P)`` slab stacks rather than
    computing them from raw observations. Useful when the same slabs are reused
    across multiple calls (e.g. re-evaluating the p-value on a shrinking subset
    of observations during an adversarial breakdown-k analysis).

    :param K_x_flat: (nx+ny, P) contiguous slab stack for the X group role.
    :param K_y_flat: (nx+ny, P) contiguous slab stack for the Y group role
        (pass the same array as ``K_x_flat`` when both bandwidths match).
    :param nx: Number of observations from X.
    :param ny: Number of observations from Y.
    :param k: Number of permutations.
    :param k_min: Early termination minimum permutations.
    :param k_th: Early termination threshold.
    :param rng: Object to draw permutations from via ``rng.permutation(n)``.
        Defaults to the ``numpy.random`` module (the global numpy random
        state), matching the rest of this module.
    :param batch_size: Number of permutations drawn and evaluated per BLAS call.
    :return: (ddist, pval, n_permutations).
    """
    assert k > 0
    assert k_th is None or k_th > 0
    k_min = k_min if k_min else k
    k_th = k_th if k_th is not None else float("inf")
    rng = rng if rng is not None else np.random
    N = nx + ny

    def _l1(x_sum: ndarray, y_sum: ndarray) -> ndarray:
        # x_sum, y_sum: (..., P) unnormalized KDE sums for the X/Y role.
        x_norm = x_sum / x_sum.sum(axis=-1, keepdims=True)
        y_norm = y_sum / y_sum.sum(axis=-1, keepdims=True)
        return np.abs(x_norm - y_norm).sum(axis=-1)

    ddist = float(_l1(K_x_flat[:nx].sum(0), K_y_flat[nx:].sum(0)))

    pval, count, curr_permutation = 0, 0, 0
    while curr_permutation < k:
        b = min(batch_size, k - curr_permutation)
        # B[r, i] = 1 iff observation i plays the X role in the r-th permutation
        # of this batch; drawn one at a time (same rng.permutation(N) call per
        # permutation as the naive test) so the draw sequence is identical.
        B = np.zeros((b, N), dtype=K_x_flat.dtype)
        for r in range(b):
            B[r, rng.permutation(N)[:nx]] = 1.0
        x_sums = B @ K_x_flat  # (b, P)
        y_sums = (1.0 - B) @ K_y_flat  # (b, P)
        ddist_perm = _l1(x_sums, y_sums)  # (b,)

        count += int((ddist <= ddist_perm).sum())
        curr_permutation += b
        pval = (count + 1) / (curr_permutation + 1)

        if (curr_permutation >= k_min) and (pval >= k_th * 1 / (curr_permutation + 1)):
            break

    return ddist, pval, curr_permutation
