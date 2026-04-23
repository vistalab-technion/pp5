"""
Cross-validation of KDE bandwidth (sigma) on the flat torus, plus pipeline-facing
helpers for per-(codon, SS) bandwidth selection.

Motivation and statistical justification live in docs/kernel_bandwidth_cv.md. The key
result used here is the "double-slab trick": when the KDE-L1 permutation test runs with
per-group bandwidths (sigma_1, sigma_2), both are treated as **fixed nuisance
constants** precomputed once from the original labels. The bandwidth follows the group
role — observation i assigned to group 1 is smoothed with sigma_1, assigned to group 2
it is smoothed with sigma_2. This keeps the test a valid permutation test.

This module provides:

- Primitives for LOO / K-fold CV log-likelihood on the torus KDE grid:
  `compute_kernel_slabs`, `compute_observation_bin_indices`,
  `cross_validate_log_likelihood_loo`, `cross_validate_log_likelihood_kfold`.
- A per-group CV driver `cross_validate_bandwidths_for_group`.
- Pipeline-facing helpers `run_bandwidth_cv`, `uniform_bandwidth_table`,
  `bandwidth_table_to_lookup` that build the (codon, SS) -> sigma_rad lookup consumed
  by the analysis pipeline.
"""
import functools
import logging
import multiprocessing as mp
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from pp5.distributions.kde import kde_2d, torus_gaussian_kernel_2d
from pp5.parallel import yield_async_results

LOGGER = logging.getLogger(__name__)

# Default CV configuration — chosen to match the notebook's committed (post-pilot)
# configuration in notebooks/kernel_bandwidth_cv.ipynb so the pipeline and the
# standalone notebook share one source of truth.
DEFAULT_CV_GRID_MIN_DEG: float = 1.0
DEFAULT_CV_GRID_MAX_DEG: float = 32.0
DEFAULT_CV_GRID_N: int = 100
DEFAULT_CV_N_MIN: int = 30
DEFAULT_CV_SCORE_TOLERANCE: float = 0.01
DEFAULT_CV_SEED: int = 42

# Column names of the bandwidth table (same columns as the notebook output).
BW_COL_CODON = "codon"
BW_COL_AA = "aa"
BW_COL_SS = "ss"
BW_COL_N = "n"
BW_COL_SIGMA_RAD = "sigma_cv_rad"
BW_COL_SIGMA_DEG = "sigma_cv_deg"
BW_COL_SIGMA_ARGMAX_DEG = "sigma_argmax_deg"
BW_COL_SCORE = "score_at_cv"
BW_COL_IS_FALLBACK = "is_fallback"
BW_COLUMNS = (
    BW_COL_CODON, BW_COL_AA, BW_COL_SS, BW_COL_N,
    BW_COL_SIGMA_RAD, BW_COL_SIGMA_DEG, BW_COL_SIGMA_ARGMAX_DEG,
    BW_COL_SCORE, BW_COL_IS_FALLBACK,
)


def default_sigma_grid_rad(
    grid_min_deg: float = DEFAULT_CV_GRID_MIN_DEG,
    grid_max_deg: float = DEFAULT_CV_GRID_MAX_DEG,
    grid_n: int = DEFAULT_CV_GRID_N,
) -> np.ndarray:
    """Return a log-spaced grid of bandwidths (radians), ascending."""
    return np.deg2rad(np.geomspace(grid_min_deg, grid_max_deg, grid_n))


def compute_kernel_slabs(
    phi_rad: np.ndarray,
    psi_rad: np.ndarray,
    sigma_rad: float,
    n_bins: int,
    grid_low: float,
    grid_high: float,
    dtype: np.dtype = np.float32,
) -> np.ndarray:
    """Return a (M, M, N) array of kernel slabs, each normalized to sum to 1.

    Each slab K[:, :, i] is the (normalized) Gaussian kernel of observation i evaluated
    on the M x M torus grid. Summing slabs and renormalizing yields the pooled KDE.

    The slab representation is what makes the permutation test efficient: we precompute
    slabs once per (codon-pair, sigma) and then each permutation is just a sum over a
    subset of slab indices — no kernel re-evaluation.
    """
    kernel_fn = functools.partial(torus_gaussian_kernel_2d, sigma=sigma_rad)
    return kde_2d(
        x1=phi_rad,
        x2=psi_rad,
        kernel_fn=kernel_fn,
        n_bins=n_bins,
        grid_low=grid_low,
        grid_high=grid_high,
        dtype=dtype,
        reduce=False,
    )


def compute_observation_bin_indices(
    phi_rad: np.ndarray,
    psi_rad: np.ndarray,
    n_bins: int,
    grid_low: float,
    grid_high: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Map each observation to its (row, col) bin index on the M x M KDE grid.

    Uses the same linspace convention as :func:`kde_2d`:
        grid[j] = grid_low + j * bin_size, bin_size = (grid_high - grid_low) / n_bins.
    Returns (bin_rows, bin_cols), each of shape (N,), clipped to [0, n_bins - 1].
    """
    bin_size = (grid_high - grid_low) / n_bins
    bin_rows = np.clip(((phi_rad - grid_low) / bin_size).astype(int), 0, n_bins - 1)
    bin_cols = np.clip(((psi_rad - grid_low) / bin_size).astype(int), 0, n_bins - 1)
    return bin_rows, bin_cols


def cross_validate_log_likelihood_loo(
    slabs: np.ndarray,
    bin_rows: np.ndarray,
    bin_cols: np.ndarray,
    eps: float = 1e-300,
) -> float:
    """Vectorized leave-one-out CV log-likelihood for a single bandwidth.

    Each slab must be normalized to sum to 1 (as returned by compute_kernel_slabs).
    LOO density for observation i = (sum_of_all_slabs - slab_i) / (N - 1), evaluated
    at the bin of observation i.

    :param slabs: (M, M, N) per-sample-normalized kernel slabs.
    :param bin_rows: (N,) row bin indices for each observation.
    :param bin_cols: (N,) column bin indices for each observation.
    :return: Sum of log leave-one-out densities across all N observations.
    """
    n = slabs.shape[2]

    # Sum all N slabs into a single (M, M) grid. Since each slab sums to 1, this grid
    # sums to N.
    sum_of_all_slabs = slabs.sum(axis=2)  # (M, M)

    # For each observation i we need the LOO density evaluated at x_i's bin:
    #
    #   p_{-i}(x_i) = [sum of all slabs at bin(x_i), minus slab i's value there]
    #                 / (N - 1)
    #
    # Instead of re-summing N-1 slabs for every i, we compute the full sum once and
    # subtract the single value that slab i contributes at bin(x_i). All N observations
    # are handled simultaneously via fancy indexing.
    observation_indices = np.arange(n)
    sum_of_remaining_at_xi = (
        sum_of_all_slabs[bin_rows, bin_cols]
        - slabs[bin_rows, bin_cols, observation_indices]
    )

    # Divide by (N-1): since each slab sums to 1, N-1 slabs sum to N-1.
    # Cast to float64 to avoid log(0) underflow in float32.
    loo_density_at_xi = sum_of_remaining_at_xi.astype(np.float64) / (n - 1)

    return float(np.sum(np.log(np.maximum(loo_density_at_xi, eps))))


def cross_validate_log_likelihood_kfold(
    slabs: np.ndarray,
    bin_rows: np.ndarray,
    bin_cols: np.ndarray,
    n_folds: int,
    seed: int,
    eps: float = 1e-300,
) -> float:
    """K-fold CV log-likelihood for a single bandwidth.

    Each slab must be normalized to sum to 1. Train density for a fold = sum of
    training slabs / n_train, evaluated at held-out observation bins.
    """
    n = slabs.shape[2]
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    log_lik = 0.0
    for train_indices, test_indices in kf.split(np.arange(n)):
        n_train = len(train_indices)
        train_sum = slabs[:, :, train_indices].sum(axis=2)  # (M, M)
        density_at_test = (
            train_sum[bin_rows[test_indices], bin_cols[test_indices]].astype(np.float64)
            / n_train
        )
        log_lik += float(np.sum(np.log(np.maximum(density_at_test, eps))))
    return log_lik


def cross_validate_bandwidths_for_group(
    group_key: tuple,
    phi_rad: np.ndarray,
    psi_rad: np.ndarray,
    sigma_grid_rad: np.ndarray,
    n_bins: int,
    grid_low: float,
    grid_high: float,
    cv_mode: str = "loo",
    n_folds: int = 5,
    seed: int = DEFAULT_CV_SEED,
    score_tolerance: float = DEFAULT_CV_SCORE_TOLERANCE,
) -> dict:
    """Run CV over all sigma values for one (codon, SS) group.

    :param group_key: (codon, ss) tuple identifying the group.
    :param phi_rad: Phi angles in radians, shape (N,).
    :param psi_rad: Psi angles in radians, shape (N,).
    :param sigma_grid_rad: Candidate bandwidths in radians, in ascending order.
    :param n_bins: Number of bins in each dimension of the KDE grid.
    :param grid_low: Lower bound of the KDE grid (inclusive).
    :param grid_high: Upper bound of the KDE grid (exclusive).
    :param cv_mode: "loo" or "kfold".
    :param n_folds: Number of folds (only used when cv_mode=="kfold").
    :param seed: Random seed (only used when cv_mode=="kfold").
    :param score_tolerance: Plateau-detection tolerance as a fraction of the finite LL
        range. sigma_cv is the smallest sigma whose LL is within
        score_tolerance * (max_LL - min_LL) of the maximum. Defaults to 0.01 (1%).
        Use 0.0 to recover literal argmax behaviour.
    :return: Dict with keys: codon, ss, n, sigma_grid_rad, scores, sigma_cv_rad
        (plateau-aware selection), sigma_argmax_rad (literal argmax), score_at_cv.
    """
    codon, ss = group_key
    n = len(phi_rad)

    # Bin indices are sigma-independent — compute once per group
    bin_rows, bin_cols = compute_observation_bin_indices(
        phi_rad, psi_rad, n_bins, grid_low, grid_high
    )

    scores = np.full(len(sigma_grid_rad), np.nan)
    for sigma_idx, sigma_rad in enumerate(sigma_grid_rad):
        slabs = compute_kernel_slabs(
            phi_rad, psi_rad, sigma_rad, n_bins, grid_low, grid_high
        )
        if cv_mode == "loo":
            scores[sigma_idx] = cross_validate_log_likelihood_loo(
                slabs, bin_rows, bin_cols
            )
        elif cv_mode == "kfold":
            scores[sigma_idx] = cross_validate_log_likelihood_kfold(
                slabs, bin_rows, bin_cols, n_folds, seed
            )
        else:
            raise ValueError(f"Invalid cv_mode: {cv_mode}")

    # Plateau-aware selection: sigma^ = min { sigma : LL(sigma) >= max_LL - eps }
    # where eps = score_tolerance * (max_LL - min_LL). Since sigma_grid_rad is
    # ascending, the first qualifying index is the smallest sigma on the plateau —
    # the most sensitive bandwidth that LL cannot distinguish from the optimum.
    finite_mask = np.isfinite(scores)
    max_score = float(np.nanmax(scores))
    score_range = max_score - float(np.nanmin(scores[finite_mask]))
    threshold = max_score - score_tolerance * score_range
    qualifying_indices = np.where(finite_mask & (scores >= threshold))[0]
    best_idx = int(qualifying_indices[0])
    argmax_idx = int(np.nanargmax(scores))

    return {
        "codon": codon,
        "ss": ss,
        "n": n,
        "sigma_grid_rad": sigma_grid_rad,
        "scores": scores,
        "sigma_cv_rad": float(sigma_grid_rad[best_idx]),
        "sigma_argmax_rad": float(sigma_grid_rad[argmax_idx]),
        "score_at_cv": float(scores[best_idx]),
    }


def run_bandwidth_cv(
    group_angles: Dict[Tuple[str, str], Tuple[np.ndarray, np.ndarray]],
    sigma_grid_rad: np.ndarray,
    n_bins: int,
    grid_low: float,
    grid_high: float,
    n_min: int = DEFAULT_CV_N_MIN,
    pool: Optional[mp.pool.Pool] = None,
    seed: int = DEFAULT_CV_SEED,
    score_tolerance: float = DEFAULT_CV_SCORE_TOLERANCE,
    cv_mode: str = "loo",
    n_folds: int = 5,
) -> pd.DataFrame:
    """Run per-(codon, SS) bandwidth cross-validation and build the bandwidth table.

    For every group with n >= n_min, runs LOO (or K-fold) CV across ``sigma_grid_rad``
    and records the selected bandwidth. Groups with n < n_min receive a **fallback
    sigma** equal to the median CV-chosen sigma of non-fallback groups sharing the same
    (amino acid, SS). This matches the policy in notebooks/kernel_bandwidth_cv.ipynb.

    Note on statistical validity (double-slab trick): the table produced here becomes a
    fixed nuisance-parameter lookup for the KDE-L1 permutation test. Each codon c in SS
    s gets one sigma_{c,s} from the *original* (un-permuted) labels. The test statistic
    T(Z, L; sigma_1, sigma_2) treats (sigma_1, sigma_2) as constants; permuting labels
    leaves them unchanged, so the permutation test remains valid. See
    docs/kernel_bandwidth_cv.md.

    :param group_angles: Dict mapping (ss, codon) -> (phi_rad, psi_rad) arrays.
    :param sigma_grid_rad: Ascending grid of candidate bandwidths (radians).
    :param n_bins: KDE grid size per axis.
    :param grid_low: KDE grid lower bound (inclusive).
    :param grid_high: KDE grid upper bound (exclusive).
    :param n_min: Minimum observations required to run CV. Groups below this get a
        fallback sigma (AA+SS median of non-fallback).
    :param pool: Optional multiprocessing pool. If provided, CV per group runs in
        parallel. If None, runs sequentially.
    :param seed: Random seed for K-fold (unused for LOO).
    :param score_tolerance: Plateau-detection tolerance, see
        :func:`cross_validate_bandwidths_for_group`.
    :param cv_mode: "loo" or "kfold".
    :param n_folds: Number of folds (only used for kfold).
    :return: DataFrame with columns BW_COLUMNS, one row per (codon, SS) group.
    """
    # Split groups into CV-eligible vs fallback.
    cv_groups, fallback_groups = [], []
    for (ss, codon), (phi_rad, psi_rad) in group_angles.items():
        n = len(phi_rad)
        if n >= n_min:
            cv_groups.append((ss, codon, phi_rad, psi_rad))
        else:
            fallback_groups.append((ss, codon, n))

    LOGGER.info(
        f"Bandwidth CV: {len(cv_groups)} groups with n>={n_min}; "
        f"{len(fallback_groups)} groups will get fallback sigma."
    )

    # Shared kwargs for each CV call.
    cv_kwargs = dict(
        sigma_grid_rad=sigma_grid_rad,
        n_bins=n_bins,
        grid_low=grid_low,
        grid_high=grid_high,
        cv_mode=cv_mode,
        n_folds=n_folds,
        seed=seed,
        score_tolerance=score_tolerance,
    )

    # Run CV in parallel if a pool is provided, else sequentially.
    if pool is not None:
        async_results = {}
        for ss, codon, phi_rad, psi_rad in cv_groups:
            async_results[(ss, codon)] = pool.apply_async(
                cross_validate_bandwidths_for_group,
                kwds=dict(
                    group_key=(codon, ss),
                    phi_rad=phi_rad,
                    psi_rad=psi_rad,
                    **cv_kwargs,
                ),
            )
        cv_results_list = [res for _, res in yield_async_results(async_results)]
    else:
        cv_results_list = [
            cross_validate_bandwidths_for_group(
                group_key=(codon, ss),
                phi_rad=phi_rad,
                psi_rad=psi_rad,
                **cv_kwargs,
            )
            for ss, codon, phi_rad, psi_rad in cv_groups
        ]

    # Build rows for the summary DataFrame.
    rows = [
        {
            BW_COL_CODON: r["codon"],
            BW_COL_AA: r["codon"].split("-")[0],
            BW_COL_SS: r["ss"],
            BW_COL_N: r["n"],
            BW_COL_SIGMA_RAD: r["sigma_cv_rad"],
            BW_COL_SIGMA_DEG: float(np.rad2deg(r["sigma_cv_rad"])),
            BW_COL_SIGMA_ARGMAX_DEG: float(np.rad2deg(r["sigma_argmax_rad"])),
            BW_COL_SCORE: r["score_at_cv"],
            BW_COL_IS_FALLBACK: False,
        }
        for r in cv_results_list
    ]

    # Append fallback rows: for each small group, use the median sigma among
    # non-fallback groups that share the same (AA, SS).
    if fallback_groups:
        df_cv = pd.DataFrame(rows)
        df_median_sigma = (
            df_cv.groupby([BW_COL_SS, BW_COL_AA])[BW_COL_SIGMA_DEG]
            .median()
            .reset_index()
        )
        median_lookup = {
            (row[BW_COL_SS], row[BW_COL_AA]): row[BW_COL_SIGMA_DEG]
            for _, row in df_median_sigma.iterrows()
        }
        n_missing_median = 0
        for ss, codon, n in fallback_groups:
            aa = codon.split("-")[0]
            fallback_sigma_deg = median_lookup.get((ss, aa))
            if fallback_sigma_deg is None or not np.isfinite(fallback_sigma_deg):
                # No non-fallback groups in the same (AA, SS) — fall back to the
                # global median across SS. Should be rare, but must not crash.
                n_missing_median += 1
                fallback_sigma_deg = float(df_cv[BW_COL_SIGMA_DEG].median())
            rows.append({
                BW_COL_CODON: codon,
                BW_COL_AA: aa,
                BW_COL_SS: ss,
                BW_COL_N: n,
                BW_COL_SIGMA_RAD: float(np.deg2rad(fallback_sigma_deg)),
                BW_COL_SIGMA_DEG: float(fallback_sigma_deg),
                BW_COL_SIGMA_ARGMAX_DEG: float("nan"),
                BW_COL_SCORE: float("nan"),
                BW_COL_IS_FALLBACK: True,
            })
        if n_missing_median > 0:
            LOGGER.warning(
                f"{n_missing_median} fallback group(s) had no (AA, SS) median; "
                "used global median instead."
            )

    df_bw = pd.DataFrame(rows, columns=list(BW_COLUMNS))
    df_bw = df_bw.sort_values([BW_COL_SS, BW_COL_CODON]).reset_index(drop=True)

    # Warn on grid-edge saturation (helps catch a grid that is too narrow).
    df_non_fb = df_bw[~df_bw[BW_COL_IS_FALLBACK]]
    if len(df_non_fb) > 0:
        grid_deg = np.rad2deg(sigma_grid_rad)
        at_lower = (df_non_fb[BW_COL_SIGMA_DEG] <= grid_deg[0] * 1.05).sum()
        at_upper = (df_non_fb[BW_COL_SIGMA_DEG] >= grid_deg[-1] * 0.95).sum()
        frac_edge = (at_lower + at_upper) / len(df_non_fb)
        if frac_edge > 0.05:
            LOGGER.warning(
                f"{frac_edge:.1%} of non-fallback groups ({at_lower + at_upper} / "
                f"{len(df_non_fb)}) have sigma_cv at the grid edges "
                f"({grid_deg[0]:.2f}°–{grid_deg[-1]:.2f}°). Consider widening the grid."
            )

    return df_bw


def uniform_bandwidth_table(
    group_sizes: Dict[Tuple[str, str], int],
    sigma_rad: float,
) -> pd.DataFrame:
    """Build a bandwidth table that assigns the same sigma to every (codon, SS) group.

    Used to route the fixed-bandwidth code path (``ddist_kernel_size > 0``) through the
    same per-group machinery as the CV path — no duplicate implementations.

    :param group_sizes: Dict mapping (ss, codon) -> n.
    :param sigma_rad: Bandwidth in radians to assign to every group.
    :return: DataFrame with columns BW_COLUMNS.
    """
    sigma_deg = float(np.rad2deg(sigma_rad))
    rows = [
        {
            BW_COL_CODON: codon,
            BW_COL_AA: codon.split("-")[0],
            BW_COL_SS: ss,
            BW_COL_N: n,
            BW_COL_SIGMA_RAD: float(sigma_rad),
            BW_COL_SIGMA_DEG: sigma_deg,
            BW_COL_SIGMA_ARGMAX_DEG: float("nan"),
            BW_COL_SCORE: float("nan"),
            BW_COL_IS_FALLBACK: False,
        }
        for (ss, codon), n in group_sizes.items()
    ]
    df = pd.DataFrame(rows, columns=list(BW_COLUMNS))
    return df.sort_values([BW_COL_SS, BW_COL_CODON]).reset_index(drop=True)


def bandwidth_table_to_lookup(df_bw: pd.DataFrame) -> Dict[Tuple[str, str], float]:
    """Convert a bandwidth DataFrame to a {(ss, codon): sigma_rad} dict for fast
    lookup inside the analysis pipeline's dispatch loop."""
    return {
        (row[BW_COL_SS], row[BW_COL_CODON]): float(row[BW_COL_SIGMA_RAD])
        for _, row in df_bw.iterrows()
    }
