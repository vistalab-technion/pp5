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

import numpy as np
import pandas as pd

from pp5.stats.breakdown import greedy_breakdown_k
from pp5.stats.two_sample import _mmd_permutation_test_from_kernel

sys.path.insert(0, "scripts/pnas2026")
from _common import PAIRS, SEED

# Positional arg overrides the default; default is the published/reproduced
# aggregated dataset (see out/pnas-2026/docs; no dependency on Alex's repro zip).
DS = (
    sys.argv[1]
    if len(sys.argv) > 1
    else (
        "out/prec-collected/20211001_124553-aida-ex_EC-src_EC/results/"
        "pointwise_cdist-natcom/_intermediate_/dataset.csv"
    )
)
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
    if est == "b":
        return Sxx / nx**2 + Syy / ny**2 - 2 * Sxy / (nx * ny)
    return (
        (Sxx - nx) / (nx * (nx - 1))
        + (Syy - ny) / (ny * (ny - 1))
        - 2 * Sxy / (nx * ny)
    )


def perm_p(K, nx, ny, est, k=KP, seed=SEED):
    obs, pval, _ = _mmd_permutation_test_from_kernel(
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


def _breakdown_closures(X, Y, est):
    """Builds the (stat_and_influence_fn, pval_fn) closures greedy_breakdown_k
    needs: the O(1)-per-point leave-one-out MMD^2 formula (row-sum trick) for
    ranking, and the real permutation p-value (via perm_p) at checkpoints."""

    def stat_and_influence_fn(x_keep, y_keep):
        Z = np.vstack([X[x_keep], Y[y_keep]])
        K = kmat(Z)
        nx, ny = len(x_keep), len(y_keep)
        Sxx = K[:nx, :nx].sum()
        Syy = K[nx:, nx:].sum()
        Sxy = K[:nx, nx:].sum()
        base = mmd2(Sxx, Syy, Sxy, nx, ny, est)
        RxIn = K[:nx, :nx].sum(1)
        RxCr = K[:nx, nx:].sum(1)
        RyIn = K[nx:, nx:].sum(1)
        RyCr = K[nx:, :nx].sum(1)
        infl_x = np.array(
            [
                base
                - mmd2(Sxx - 2 * RxIn[i] + K[i, i], Syy, Sxy - RxCr[i], nx - 1, ny, est)
                for i in range(nx)
            ]
        )
        infl_y = np.array(
            [
                base
                - mmd2(
                    Sxx,
                    Syy - 2 * RyIn[j] + K[nx + j, nx + j],
                    Sxy - RyCr[j],
                    nx,
                    ny - 1,
                    est,
                )
                for j in range(ny)
            ]
        )
        return base, infl_x, infl_y

    def pval_fn(x_keep, y_keep):
        K = kmat(np.vstack([X[x_keep], Y[y_keep]]))
        return perm_p(K, len(x_keep), len(y_keep), est)

    return stat_and_influence_fn, pval_fn


def breakdown(X, Y, thresh, k_grid, est):
    stat_and_influence_fn, pval_fn = _breakdown_closures(X, Y, est)
    rows, bk, _removed_log = greedy_breakdown_k(
        n1=len(X),
        n2=len(Y),
        stat_and_influence_fn=stat_and_influence_fn,
        pval_fn=pval_fn,
        thresh=thresh,
        k_grid=k_grid,
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
