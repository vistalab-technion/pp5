from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from numpy import ndarray
from scipy.spatial.distance import pdist, squareform

from pp5.stats.two_sample.kde import (
    kde_2d_slab_stacks,
    kde_l1_permutation_test_from_slabs,
)
from pp5.stats.two_sample.mmd import mmd_permutation_test_from_kernel


def greedy_breakdown_k(
    n1: int,
    n2: int,
    stat_and_influence_fn: Callable[
        [List[int], List[int]], Tuple[float, ndarray, ndarray]
    ],
    pval_fn: Callable[[List[int], List[int]], Tuple[float, float]],
    thresh: float,
    k_grid: Sequence[int],
) -> Tuple[List[Dict], Optional[int], List[Tuple[str, int, float]]]:
    """
    Adversarial breakdown-k: greedily remove the single most influential
    remaining observation (re-ranked after every removal), recomputing the real
    p-value only at the checkpoints in `k_grid`, until the pair is no longer
    significant at `thresh`.

    breakdown-k is a greedy upper bound on the minimal kill-set: a more
    sophisticated "attack" could require fewer points. The removal ranking uses
    `stat_and_influence_fn`'s influence values as a proxy; the significance
    decision at each checkpoint always uses the real `pval_fn` recomputation.

    :param n1: Number of observations in group 1 (original, pre-removal).
    :param n2: Number of observations in group 2 (original, pre-removal).
    :param stat_and_influence_fn: Given the current (x_keep, y_keep) index
        lists (into the original group-1/group-2 observations), returns
        (base_stat, infl_1, infl_2), where infl_1[m] and infl_2[m] are the drop
        in the statistic if x_keep[m] (resp. y_keep[m]) were removed. Larger
        means more influential (removed first).
    :param pval_fn: Given the current (x_keep, y_keep), returns (ddist, pval)
        from a real recomputed significance test on the reduced samples.
    :param thresh: Significance threshold (e.g. a BH cutoff); the pair is
        significant while pval <= thresh.
    :param k_grid: Checkpoints (number of observations removed so far) at
        which to actually call pval_fn and check significance.
    :return: Tuple of:
        - rows: one dict per checkpoint in k_grid actually reached, with keys
          k, ddist, p, n1, n2, significant.
        - breakdown_k: the smallest k_grid checkpoint at which the pair is no
          longer significant, or None if it never breaks within k_grid.
        - removed_log: one (side, original_index, drop) tuple per removal
          actually performed, in removal order. side is "X" or "Y";
          original_index indexes into the original group-1/group-2 arrays.
    """
    x_keep = list(range(n1))
    y_keep = list(range(n2))
    rows: List[Dict] = []
    removed_log: List[Tuple[str, int, float]] = []
    breakdown_k: Optional[int] = None
    k_max = max(k_grid)

    for k in range(0, k_max + 1):
        if k in k_grid:
            ddist, pval = pval_fn(x_keep, y_keep)
            significant = pval <= thresh
            rows.append(
                dict(
                    k=k,
                    ddist=ddist,
                    p=pval,
                    n1=len(x_keep),
                    n2=len(y_keep),
                    significant=significant,
                )
            )
            if not significant and breakdown_k is None:
                breakdown_k = k

        if k == k_max or breakdown_k is not None:
            break

        # base_stat is not needed by the loop itself (stat_and_influence_fn
        # already folds it into infl_x/infl_y); exposed for callers that want
        # to log the ranking-time statistic.
        _base_stat, infl_x, infl_y = stat_and_influence_fn(x_keep, y_keep)

        # Find the single most influential remaining point. Strict ">" means
        # the first-encountered candidate wins exact ties: X before Y, and
        # within each side, the lower local index first.
        best_local_idx, best_drop, best_side = None, -np.inf, None
        for local_i, drop in enumerate(infl_x):
            if drop > best_drop:
                best_drop, best_local_idx, best_side = drop, local_i, "X"
        for local_j, drop in enumerate(infl_y):
            if drop > best_drop:
                best_drop, best_local_idx, best_side = drop, local_j, "Y"

        if best_side == "X":
            removed_original_idx = x_keep.pop(best_local_idx)
        else:
            removed_original_idx = y_keep.pop(best_local_idx)
        removed_log.append((best_side, removed_original_idx, float(best_drop)))

    return rows, breakdown_k, removed_log


def breakdown_k_kde(
    X: ndarray,
    Y: ndarray,
    n_bins: int,
    grid_low: float,
    grid_high: float,
    dtype: np.dtype,
    sigma_x_rad: float,
    sigma_y_rad: float,
    thresh: float,
    k_grid: Sequence[int],
    k_perm: int,
    k_min: Optional[int] = None,
    k_th: float = float("inf"),
    seed: Optional[int] = None,
    batch_size: int = 1000,
) -> Tuple[List[Dict], Optional[int], List[Tuple[str, int, float]]]:
    """
    Adversarial breakdown-k (see :obj:`greedy_breakdown_k`) for the KDE-L1
    statistic on the flat torus, with per-group bandwidths.

    Precomputes each observation's per-role KDE slab once (via
    :obj:`pp5.stats.two_sample.kde.kde_2d_slab_stacks`) and reuses it for both
    the O(1) leave-one-out influence ranking and the real permutation p-value
    recomputation at each `k_grid` checkpoint, instead of recomputing the KDE
    from raw angles at every checkpoint. Reduces to the fixed-bandwidth case
    when ``sigma_x_rad == sigma_y_rad`` (shared slabs).

    :param X: (n1, 2) phi/psi observations in radians (group 1, "X" role).
    :param Y: (n2, 2) phi/psi observations in radians (group 2, "Y" role).
    :param n_bins: M, grid size per axis.
    :param grid_low: Grid lower bound (inclusive), radians.
    :param grid_high: Grid upper bound (exclusive), radians.
    :param dtype: Slab dtype.
    :param sigma_x_rad: Bandwidth (radians) for the X group role.
    :param sigma_y_rad: Bandwidth (radians) for the Y group role.
    :param thresh: Significance threshold (e.g. a BH cutoff).
    :param k_grid: Checkpoints; see :obj:`greedy_breakdown_k`.
    :param k_perm: Number of permutations for the p-value recomputation at
        each checkpoint.
    :param k_min: Early termination minimum permutations for the permutation
        test (see :obj:`pp5.stats.two_sample.kde.kde_l1_permutation_test_from_slabs`).
    :param k_th: Early termination threshold for the permutation test.
    :param seed: If given, every checkpoint's permutation p-value is computed
        with a freshly-seeded ``numpy.random.default_rng(seed)`` (same seed
        every time, not one RNG advanced across checkpoints), so that
        differences between checkpoints come only from the removed
        observations, not from RNG draw luck. If None, the global
        ``numpy.random`` state is used (and does advance across checkpoints).
    :param batch_size: Number of permutations drawn and evaluated per BLAS call.
    :return: (rows, breakdown_k, removed_log); see :obj:`greedy_breakdown_k`.
    """
    n1, n2 = X.shape[0], Y.shape[0]
    Z = np.vstack([X, Y])
    K_x, K_y = kde_2d_slab_stacks(
        Z, n_bins, grid_low, grid_high, dtype, sigma_x_rad, sigma_y_rad
    )
    N = n1 + n2
    K_x_flat = np.ascontiguousarray(K_x.reshape(N, -1), dtype=dtype)
    K_y_flat = (
        K_x_flat
        if K_y is K_x
        else np.ascontiguousarray(K_y.reshape(N, -1), dtype=dtype)
    )

    def _l1(x_sum: ndarray, y_sum: ndarray) -> float:
        x_norm = x_sum / x_sum.sum()
        y_norm = y_sum / y_sum.sum()
        return float(np.abs(x_norm - y_norm).sum())

    def stat_and_influence_fn(x_keep, y_keep):
        x_keep = np.asarray(x_keep, dtype=int)
        y_pooled = n1 + np.asarray(y_keep, dtype=int)
        sx = K_x_flat[x_keep].sum(0)
        sy = K_y_flat[y_pooled].sum(0)
        base = _l1(sx, sy)
        infl_x = np.array([base - _l1(sx - K_x_flat[i], sy) for i in x_keep])
        infl_y = np.array([base - _l1(sx, sy - K_y_flat[j]) for j in y_pooled])
        return base, infl_x, infl_y

    def pval_fn(x_keep, y_keep):
        x_keep = np.asarray(x_keep, dtype=int)
        y_pooled = n1 + np.asarray(y_keep, dtype=int)
        pooled_idx = np.concatenate([x_keep, y_pooled])
        rng = np.random.default_rng(seed) if seed is not None else np.random
        ddist, pval, _ = kde_l1_permutation_test_from_slabs(
            K_x_flat[pooled_idx],
            K_y_flat[pooled_idx],
            len(x_keep),
            len(y_pooled),
            k_perm,
            k_min=k_min,
            k_th=k_th,
            rng=rng,
            batch_size=batch_size,
        )
        return ddist, pval

    return greedy_breakdown_k(n1, n2, stat_and_influence_fn, pval_fn, thresh, k_grid)


def breakdown_k_mmd(
    X: ndarray,
    Y: ndarray,
    similarity_fn: Callable[[ndarray, ndarray], float],
    kernel_fn: Callable[[ndarray], ndarray],
    unbiased: bool,
    thresh: float,
    k_grid: Sequence[int],
    k_perm: int,
    k_min: Optional[int] = None,
    k_th: float = float("inf"),
    seed: Optional[int] = None,
) -> Tuple[List[Dict], Optional[int], List[Tuple[str, int, float]]]:
    """
    Adversarial breakdown-k (see :obj:`greedy_breakdown_k`) for the (biased or
    unbiased) MMD^2 statistic.

    Unlike :obj:`breakdown_k_kde`, there is no shared representation that can
    be precomputed once and sliced at every checkpoint: the Gram matrix is
    rebuilt from the surviving observations at every influence-ranking step
    and at every `k_grid` checkpoint. The O(1) part is the leave-one-out
    update of the block sums (Sxx, Syy, Sxy) given that Gram matrix, not the
    Gram matrix construction itself.

    :param X: (n1, m) observations for group 1 ("X" role).
    :param Y: (n2, m) observations for group 2 ("Y" role).
    :param similarity_fn: h(x, y), see
        :obj:`pp5.stats.two_sample.common.two_sample_kernel_permutation_test`.
    :param kernel_fn: k(z), see
        :obj:`pp5.stats.two_sample.common.two_sample_kernel_permutation_test`.
    :param unbiased: Whether to use the unbiased MMD U-statistic (excludes the
        within-sample diagonal). If False, the biased V-statistic is used.
    :param thresh: Significance threshold (e.g. a BH cutoff).
    :param k_grid: Checkpoints; see :obj:`greedy_breakdown_k`.
    :param k_perm: Number of permutations for the p-value recomputation at
        each checkpoint.
    :param k_min: Early termination minimum permutations for the permutation
        test (see :obj:`pp5.stats.two_sample.mmd.mmd_permutation_test_from_kernel`).
    :param k_th: Early termination threshold for the permutation test.
    :param seed: If given, every checkpoint's permutation p-value is computed
        with a freshly-seeded ``numpy.random.default_rng(seed)`` (same seed
        every time, not one RNG advanced across checkpoints), so that
        differences between checkpoints come only from the removed
        observations, not from RNG draw luck. If None, the global
        ``numpy.random`` state is used (and does advance across checkpoints).
    :return: (rows, breakdown_k, removed_log); see :obj:`greedy_breakdown_k`.
    """
    n1, n2 = X.shape[0], Y.shape[0]

    def _kernel_matrix(Z: ndarray) -> ndarray:
        D = squareform(pdist(Z, metric=similarity_fn))
        return kernel_fn(D)

    def _mmd2(Sxx: float, Syy: float, Sxy: float, nx: int, ny: int) -> float:
        if unbiased:
            return (
                (Sxx - nx) / (nx * (nx - 1))
                + (Syy - ny) / (ny * (ny - 1))
                - 2.0 * Sxy / (nx * ny)
            )
        return Sxx / nx**2 + Syy / ny**2 - 2.0 * Sxy / (nx * ny)

    def stat_and_influence_fn(x_keep, y_keep):
        Z = np.vstack([X[x_keep], Y[y_keep]])
        K = _kernel_matrix(Z)
        nx, ny = len(x_keep), len(y_keep)
        Sxx = K[:nx, :nx].sum()
        Syy = K[nx:, nx:].sum()
        Sxy = K[:nx, nx:].sum()
        base = _mmd2(Sxx, Syy, Sxy, nx, ny)
        Rx_in = K[:nx, :nx].sum(1)
        Rx_cr = K[:nx, nx:].sum(1)
        Ry_in = K[nx:, nx:].sum(1)
        Ry_cr = K[nx:, :nx].sum(1)
        infl_x = np.array(
            [
                base
                - _mmd2(Sxx - 2 * Rx_in[i] + K[i, i], Syy, Sxy - Rx_cr[i], nx - 1, ny)
                for i in range(nx)
            ]
        )
        infl_y = np.array(
            [
                base
                - _mmd2(
                    Sxx,
                    Syy - 2 * Ry_in[j] + K[nx + j, nx + j],
                    Sxy - Ry_cr[j],
                    nx,
                    ny - 1,
                )
                for j in range(ny)
            ]
        )
        return base, infl_x, infl_y

    def pval_fn(x_keep, y_keep):
        K = _kernel_matrix(np.vstack([X[x_keep], Y[y_keep]]))
        rng = np.random.default_rng(seed) if seed is not None else np.random
        stat, pval, _ = mmd_permutation_test_from_kernel(
            K,
            len(x_keep),
            len(y_keep),
            k_perm,
            unbiased=unbiased,
            k_min=k_min,
            k_th=k_th,
            rng=rng,
        )
        return stat, pval

    return greedy_breakdown_k(n1, n2, stat_and_influence_fn, pval_fn, thresh, k_grid)
