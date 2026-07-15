#!/usr/bin/env python
"""
MMD breakdown arm: unbiased MMD^2 (U-statistic) and biased RBF-MMD (V-statistic),
Gaussian kernel on the flat torus (sigma=10 deg). For the 6 candidate pairs:
baseline MMD^2 + permutation p, and breakdown-k for both estimators.

Significance is assessed at the per-SS BH thresholds established by the published
KDE tests (HELIX 0.00115, TURN 0.00057) so the 5-statistic side-by-side is on a
common footing; a statistic-specific MMD-BH (all 87 pairs) is a later refinement.

Fast permutation via the row-sum identity (sum only the smaller group's block);
clean O(1)-update leave-one-out for the influence ranking.
"""

import os
import sys
from functools import partial

import numpy as np
import pandas as pd

from pp5.dihedral import flat_torus_distance
from pp5.distributions.kde import gaussian_kernel
from pp5.stats.breakdown import breakdown_k_mmd
from pp5.stats.two_sample import mmd_permutation_test_from_kernel

sys.path.insert(0, "scripts/pnas2026")
from _common import PAIRS, SEED

# Positional arg overrides the default; default is the published/reproduced
# aggregated dataset (see out/pnas-2026/docs; no dependency on Alex's repro zip).
DS = sys.argv[1] if len(sys.argv) > 1 else "out/pnas-2026/dataset-processed/dataset.csv"
SIG = np.deg2rad(10.0)
KP = 5000
# KDE-L1's BH thresholds ("for a common footing"), reused here per Report 2 §2
# -- MMD's own statistic-specific BH threshold is a later refinement.
THR = {"HELIX": 0.0011494, "TURN": 0.0005747}


def kmat(Z):
    d = np.abs(Z[:, None, :] - Z[None, :, :])
    d = np.minimum(d, 2 * np.pi - d)
    return np.exp(-(d**2).sum(2) / (2 * SIG**2))


def mmd2(Sxx, Syy, Sxy, nx, ny, est):
    """Biased/unbiased MMD^2 from precomputed Gram-matrix block sums.

    No longer called by this module's own `perm_p`/`breakdown` (both delegate to
    the library's `mmd_permutation_test_from_kernel`/`breakdown_k_mmd`, which
    compute the statistic internally) -- kept as a public formula because
    `directional_speed_mmd.py` and `expression_confound_mmd.py` import it directly.
    """
    if est == "b":
        return Sxx / nx**2 + Syy / ny**2 - 2 * Sxy / (nx * ny)
    return (
        (Sxx - nx) / (nx * (nx - 1))
        + (Syy - ny) / (ny * (ny - 1))
        - 2 * Sxy / (nx * ny)
    )


def perm_p(K, nx, ny, est, k=KP, seed=SEED):
    obs, pval, _ = mmd_permutation_test_from_kernel(
        K,
        nx,
        ny,
        k,
        unbiased=(est == "u"),
        k_min=k,
        k_th=float("inf"),
        rng=np.random.default_rng(seed),
    )
    return obs, pval


def breakdown(X, Y, thresh, k_grid, est):
    """Adversarial breakdown-k via the library's generalized MMD wrapper.

    similarity_fn=flat_torus_distance + kernel_fn=gaussian_kernel(sigma=SIG)
    reproduces kmat(Z) exactly (validated: max abs diff ~1e-16, see
    tests/test_statistical_tests.py::TestMMD::test_composed_kernel_matches_torus_gaussian_kernel_2d).
    """
    rows, bk, _removed_log = breakdown_k_mmd(
        X,
        Y,
        similarity_fn=flat_torus_distance,
        kernel_fn=partial(gaussian_kernel, sigma=SIG),
        unbiased=(est == "u"),
        thresh=thresh,
        k_grid=k_grid,
        k_perm=KP,
        k_min=KP,
        k_th=float("inf"),
        seed=SEED,
    )
    return rows, bk


def main():
    df = pd.read_csv(DS)
    print(
        f"{'pair':<14}{'SS':<6}{'MMD2u p':>9}{'sig':>5}{'bk_u':>6}"
        f"{'RBF-MMD p':>11}{'sig':>5}{'bk_b':>6}",
        flush=True,
    )
    out = []
    for ss, c1, c2 in PAIRS:
        A = np.deg2rad(
            df[(df.condition_group == ss) & (df.codon == c1)][["phi", "psi"]].values
        )
        B = np.deg2rad(
            df[(df.condition_group == ss) & (df.codon == c2)][["phi", "psi"]].values
        )
        K = kmat(np.vstack([A, B]))
        _, pu = perm_p(K, len(A), len(B), "u")
        _, pb = perm_p(K, len(A), len(B), "b")
        nmin = min(len(A), len(B))
        kg = [k for k in [0, 1, 2, 3, 5, 8, 12, 20] if k <= nmin - 2]
        bku = breakdown(A, B, THR[ss], kg, "u")[1] if pu <= THR[ss] else None
        bkb = breakdown(A, B, THR[ss], kg, "b")[1] if pb <= THR[ss] else None
        su, sb = pu <= THR[ss], pb <= THR[ss]
        print(
            f"{c1+':'+c2.split('-')[1]:<14}{ss:<6}{pu:>9.4f}{('yes' if su else 'no'):>5}"
            f"{str(bku):>6}{pb:>11.4f}{('yes' if sb else 'no'):>5}{str(bkb):>6}",
            flush=True,
        )
        out.append(
            dict(
                SS=ss,
                pair=f"{c1}:{c2}",
                mmd2u_p=round(pu, 5),
                mmd2u_sig=su,
                mmd2u_bk=bku,
                rbfmmd_p=round(pb, 5),
                rbfmmd_sig=sb,
                rbfmmd_bk=bkb,
            )
        )
    os.makedirs("out/pnas-2026-repro", exist_ok=True)
    pd.DataFrame(out).to_csv("out/pnas-2026-repro/mmd_breakdown.csv", index=False)
    print("\nsaved: out/pnas-2026-repro/mmd_breakdown.csv")


if __name__ == "__main__":
    main()
