"""Null-control replicate generation for adversarial breakdown-k calibration.

Two calibrations are supported, both preserving every observation's (phi, psi) --
and therefore every outlier -- while scrambling only which codon each observation
is labelled with:

- AA+SS randomization: shuffle codon labels within each (amino acid, secondary
  structure) group of the full dataset.
- Within-pair pooled shuffle: pool only the two codons under test and re-split at
  the same (n1, n2) sizes -- the tightest possible content match.
"""

from typing import Callable, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from numpy import ndarray


def randomized_codon_column(
    codon: ndarray, group_keys: Sequence[Tuple], seed: int
) -> ndarray:
    """Shuffle codon labels within each group, preserving per-group counts.

    :param codon: (n,) array of codon labels (e.g. `"L-CTC"`).
    :param group_keys: (n,) sequence of hashable group keys (e.g. (AA, secondary
        structure) tuples) to shuffle within; every position keeps its own
        (phi, psi), only its codon label is reassigned within its own group.
    :param seed: Seed for the within-group permutations.
    :return: (n,) array of shuffled codon labels.
    """
    assert len(codon) == len(group_keys)
    rng = np.random.default_rng(seed)
    shuffled = np.asarray(codon).copy()

    groups: Dict[Tuple, List[int]] = {}
    for i, key in enumerate(group_keys):
        groups.setdefault(key, []).append(i)

    for member_idx in groups.values():
        member_idx = np.asarray(member_idx)
        # Gather this group's own values in permuted order, write back in place:
        # within-group shuffle only, never mixing values across groups.
        shuffled[member_idx] = shuffled[member_idx[rng.permutation(len(member_idx))]]
    return shuffled


def gen_aa_ss_control_replicates(
    df: pd.DataFrame,
    ss: str,
    codon1: str,
    codon2: str,
    n_replicates: int,
    base_seed: int,
) -> Iterator[Tuple[ndarray, ndarray]]:
    """Yield AA+SS-randomized control replicates for one codon pair.

    Each replicate shuffles codon labels within (AA, condition_group) groups of the
    entire dataset, then re-extracts the two codons' (phi, psi) points under their
    new labels -- preserving every residue's (phi, psi) and the exact per-(AA, SS)
    codon counts.

    :param df: Dataset with columns `AA`, `condition_group`, `codon`, `phi`, `psi`
        (degrees).
    :param ss: Secondary-structure class to extract, e.g. `"HELIX"`.
    :param codon1: First codon label, e.g. `"L-CTC"`.
    :param codon2: Second codon label, e.g. `"L-TTG"`.
    :param n_replicates: Number of control replicates to generate.
    :param base_seed: Base seed; replicate `r` uses seed `base_seed + r`.
    :return: Iterator of `(n1, 2)` / `(n2, 2)` radian phi/psi arrays, one pair per
        replicate.
    """
    # Precompute group keys, original codon labels, and phi/psi for efficient
    # per-replicate shuffling and extraction.
    group_keys = list(zip(df["AA"], df["condition_group"]))
    codon_values = df["codon"].to_numpy()
    ss_mask = df["condition_group"].to_numpy() == ss
    phi_psi_deg = df[["phi", "psi"]].to_numpy()

    for r in range(n_replicates):
        shuffled_codon = randomized_codon_column(codon_values, group_keys, base_seed + r)
        # Extract the two codons' points under their new shuffled labels within
        # the target SS class.
        X = np.deg2rad(phi_psi_deg[ss_mask & (shuffled_codon == codon1)])
        Y = np.deg2rad(phi_psi_deg[ss_mask & (shuffled_codon == codon2)])
        yield X, Y


def gen_pooled_shuffle_replicates(
    X: ndarray, Y: ndarray, n_replicates: int, base_seed: int
) -> Iterator[Tuple[ndarray, ndarray]]:
    """Yield within-pair pooled-shuffle control replicates.

    Pools only `X` and `Y`'s own points and re-splits at the same (n1, n2) sizes --
    identical point cloud and identical outliers, labels randomized.

    :param X: `(n1, 2)` radian phi/psi observations for codon 1.
    :param Y: `(n2, 2)` radian phi/psi observations for codon 2.
    :param n_replicates: Number of control replicates to generate.
    :param base_seed: Base seed; replicate `r` uses seed `base_seed + r`.
    :return: Iterator of `(n1, 2)` / `(n2, 2)` radian phi/psi arrays, one pair per
        replicate.
    """
    Z = np.vstack([X, Y])
    n1, n2 = len(X), len(Y)
    for r in range(n_replicates):
        rng = np.random.default_rng(base_seed + r)
        perm = rng.permutation(n1 + n2)
        yield Z[perm[:n1]], Z[perm[n1:]]


def null_control_summary(
    replicate_pairs: Iterator[Tuple[ndarray, ndarray]],
    pval_fn: Callable[[ndarray, ndarray], Tuple[float, float]],
    breakdown_fn: Callable[
        [ndarray, ndarray, float, Sequence[int]], Tuple[List, Optional[int], List]
    ],
    thresh: float,
    k_grid_fn: Callable[[int], Sequence[int]],
) -> Dict[str, float]:
    """Baseline significance and adversarial breakdown-k over null-control replicates.

    For each replicate: skip if either group has fewer than 2 points; otherwise
    compute the baseline p-value, and only if it clears `thresh`, the adversarial
    breakdown-k (this mirrors the real-pair computation, and avoids the cost of an
    adversarial removal loop on replicates that are already non-significant at the
    full sample). Non-significant replicates get breakdown-k = 0 by definition.

    :param replicate_pairs: Iterator of (X, Y) control replicate observation arrays.
    :param pval_fn: Given (X, Y), returns (statistic, pval) for the baseline test.
    :param breakdown_fn: Given (X, Y, thresh, k_grid), returns
        (rows, breakdown_k, removed_log) as in
        :obj:`pp5.stats.breakdown.greedy_breakdown_k`.
    :param thresh: Significance threshold (e.g. a BH cutoff).
    :param k_grid_fn: Given `min(n1, n2)`, returns the checkpoint grid to use.
    :return: Dict with `n_replicates`, `frac_sig`, `bk_median`, `bk_max`, `min_p0`.
    """
    p0s: List[float] = []
    breakdown_ks: List[float] = []
    for X, Y in replicate_pairs:
        if len(X) < 2 or len(Y) < 2:
            continue
        _, p0 = pval_fn(X, Y)
        p0s.append(p0)
        if p0 <= thresh:
            k_grid = k_grid_fn(min(len(X), len(Y)))
            _, breakdown_k, _ = breakdown_fn(X, Y, thresh, k_grid)
            breakdown_ks.append(breakdown_k if breakdown_k is not None else k_grid[-1])
        else:
            breakdown_ks.append(0)

    assert p0s, "No control replicate had at least 2 points in both groups"
    p0s_arr = np.array(p0s)
    breakdown_ks_arr = np.array(breakdown_ks)
    return dict(
        n_replicates=len(p0s_arr),
        frac_sig=float((p0s_arr <= thresh).mean()),
        bk_median=float(np.median(breakdown_ks_arr)),
        bk_max=float(np.max(breakdown_ks_arr)),
        min_p0=float(p0s_arr.min()),
    )
