#!/usr/bin/env python
"""
CV-bandwidth arm of the breakdown analysis (colleague comment #1).

Uses the per-(codon, SS) cross-validated KDE bandwidths from the published
revision (kernel_bandwidths.csv in the zip) instead of a fixed 10 deg. The KDE-L1
statistic is the two-bandwidth "double-slab" permutation test (kde2d_test_pergroup):
each pooled point carries a slab at BOTH codons' sigmas; under a permutation it
uses the slab matching the group it currently plays. Vectorized (selection-matrix
x slab-stack) so a pair runs in seconds.

Reports baseline p (validate vs published) and, for FDR-significant pairs, the
adversarial breakdown-k.
"""

import os
import sys

import numpy as np
import pandas as pd

from pp5.stats.breakdown import breakdown_k_kde
from pp5.stats.two_sample import kde2d_test_pergroup_fast

sys.path.insert(0, "scripts/pnas2026")
from _common import PAIRS, SEED

# CV-selected per-(codon, SS) kernel bandwidths, produced by the published pipeline
# (see out/pnas-2026/docs); override via PP5_CVTAB_PATH if using a different table.
CVTAB = os.environ.get(
    "PP5_CVTAB_PATH", "out/pnas-2026/bandwidth_cv/kernel_bandwidths.csv"
)
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
NBINS, GLOW, GHIGH, DT = 128, -np.pi, np.pi, np.float64
K = 5000
BH = {"HELIX": 0.0005747126436781609, "TURN": 0.0005747126436781609}
PUB = {
    ("HELIX", "L-CTC", "L-TTG"): 0.00039992,
    ("HELIX", "L-CTC", "L-CTG"): 0.38022813,
    ("HELIX", "L-CTC", "L-CTT"): 0.04310345,
    ("HELIX", "R-AGG", "R-CGA"): 0.0041991603,
    ("TURN", "A-GCG", "A-GCT"): 0.00039992,
    ("TURN", "P-CCC", "P-CCG"): 0.061576355,
}


def perm_pval_cv(Z, nx, ny, sx_rad, sy_rad, k=K, seed=SEED, batch=1000):
    """double-slab vectorized permutation p; Z=[X;Y] radians (X=codon A at sx_rad)."""
    ddist, pval, _ = kde2d_test_pergroup_fast(
        Z[:nx],
        Z[nx:],
        k,
        n_bins=NBINS,
        grid_low=GLOW,
        grid_high=GHIGH,
        dtype=DT,
        sigma_x_rad=sx_rad,
        sigma_y_rad=sy_rad,
        k_min=k,
        k_th=float("inf"),
        rng=np.random.default_rng(seed),
        batch_size=batch,
    )
    return ddist, pval


def greedy_cv(A, B, sxr, syr, thresh, k_grid):
    """Adversarial breakdown-k via the library's generalized KDE-L1 wrapper,
    per-codon CV bandwidths (sxr for A/"X" role, syr for B/"Y" role)."""
    rows, bk, _log = breakdown_k_kde(
        A,
        B,
        n_bins=NBINS,
        grid_low=GLOW,
        grid_high=GHIGH,
        dtype=DT,
        sigma_x_rad=sxr,
        sigma_y_rad=syr,
        thresh=thresh,
        k_grid=k_grid,
        k_perm=K,
        k_min=K,
        k_th=float("inf"),
        seed=SEED,
    )
    return rows, bk


def main():
    cv = pd.read_csv(CVTAB).set_index(["codon", "ss"])["sigma_cv_deg"]
    df = pd.read_csv(DS)
    print(
        f"{'pair':<16}{'SS':<6}{'sigA':>6}{'sigB':>6}{'CV ddist':>9}{'CV p':>9}"
        f"{'pub p':>9}{'sig':>5}{'breakdown-k':>13}"
    )
    out = []
    for ss, c1, c2 in PAIRS:
        sA = float(cv.get((c1, ss)))
        sB = float(cv.get((c2, ss)))
        A = np.deg2rad(
            df[(df.condition_group == ss) & (df.codon == c1)][["phi", "psi"]].values
        )
        B = np.deg2rad(
            df[(df.condition_group == ss) & (df.codon == c2)][["phi", "psi"]].values
        )
        dd, p = perm_pval_cv(
            np.vstack([A, B]), len(A), len(B), np.deg2rad(sA), np.deg2rad(sB)
        )
        sig = p <= BH[ss]
        bk_txt = ""
        if sig:
            nmin = min(len(A), len(B))
            kg = [
                k
                for k in [
                    0,
                    1,
                    2,
                    3,
                    5,
                    8,
                    12,
                    20,
                    min(40, max(5, int(np.ceil(0.05 * nmin)))),
                ]
                if k <= nmin - 2
            ]
            rows, bk = greedy_cv(A, B, np.deg2rad(sA), np.deg2rad(sB), BH[ss], kg)
            bk_txt = f"{bk}" if bk else f">{max(kg)}"
        print(
            f"{c1+':'+c2.split('-')[1]:<16}{ss:<6}{sA:>6.1f}{sB:>6.1f}{dd:>9.3f}{p:>9.4f}"
            f"{PUB[(ss,c1,c2)]:>9.4f}{('yes' if sig else 'no'):>5}{bk_txt:>13}"
        )
        out.append(
            dict(
                SS=ss,
                pair=f"{c1}:{c2}",
                sigA=sA,
                sigB=sB,
                cv_ddist=round(dd, 4),
                cv_p=round(p, 5),
                pub_p=PUB[(ss, c1, c2)],
                significant=sig,
                breakdown_k=bk_txt,
            )
        )
    os.makedirs("out/pnas-2026-repro", exist_ok=True)
    pd.DataFrame(out).to_csv(
        "out/pnas-2026-repro/robustness_cv_summary.csv", index=False
    )
    print("\nsaved: out/pnas-2026-repro/robustness_cv_summary.csv")


if __name__ == "__main__":
    main()
