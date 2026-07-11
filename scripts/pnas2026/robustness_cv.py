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
from functools import partial
import numpy as np
import pandas as pd

from pp5.distributions.kde import kde_2d, torus_gaussian_kernel_2d

# CV-selected per-(codon, SS) kernel bandwidths, produced by the published pipeline
# (see out/pnas-2026/docs); override via PP5_CVTAB_PATH if using a different table.
CVTAB = os.environ.get("PP5_CVTAB_PATH", "out/pnas-2026/bandwidth_cv/kernel_bandwidths.csv")
# Positional arg overrides the default; default is the published/reproduced
# aggregated dataset (see out/pnas-2026/docs; no dependency on Alex's repro zip).
DS = sys.argv[1] if len(sys.argv) > 1 else (
    "out/prec-collected/20211001_124553-aida-ex_EC-src_EC/results/"
    "pointwise_cdist-natcom/_intermediate_/dataset.csv"
)
NBINS, GLOW, GHIGH, DT = 128, -np.pi, np.pi, np.float64
K, SEED = 5000, 12345
BH = {"HELIX": 0.0005747126436781609, "TURN": 0.0005747126436781609}
PAIRS = [("HELIX","L-CTC","L-TTG"),("HELIX","L-CTC","L-CTG"),("HELIX","L-CTC","L-CTT"),
         ("HELIX","R-AGG","R-CGA"),("TURN","A-GCG","A-GCT"),("TURN","P-CCC","P-CCG")]
PUB = {("HELIX","L-CTC","L-TTG"):0.00039992,("HELIX","L-CTC","L-CTG"):0.38022813,
       ("HELIX","L-CTC","L-CTT"):0.04310345,("HELIX","R-AGG","R-CGA"):0.0041991603,
       ("TURN","A-GCG","A-GCT"):0.00039992,("TURN","P-CCC","P-CCG"):0.061576355}


def slabs(A, sig):
    s = kde_2d(A[:,0], A[:,1], kernel_fn=partial(torus_gaussian_kernel_2d, sigma=sig),
               n_bins=NBINS, grid_low=GLOW, grid_high=GHIGH, dtype=DT, reduce=False)
    return s.transpose(2,0,1).reshape(A.shape[0], -1)


def _l1(sx, sy):
    a = sx/sx.sum(); b = sy/sy.sum(); return float(np.abs(a-b).sum())


def perm_pval_cv(Z, nx, ny, sx_rad, sy_rad, k=K, seed=SEED, batch=1000):
    """double-slab vectorized permutation p; Z=[X;Y] radians (X=codon A at sx_rad)."""
    N = nx+ny
    Sx = np.ascontiguousarray(slabs(Z, sx_rad))
    Sy = Sx if sx_rad == sy_rad else np.ascontiguousarray(slabs(Z, sy_rad))
    ddist = _l1(Sx[:nx].sum(0), Sy[nx:].sum(0))
    rng = np.random.default_rng(seed); count = done = 0
    while done < k:
        b = min(batch, k-done)
        Bx = np.zeros((b, N), DT)
        for r in range(b):
            Bx[r, rng.permutation(N)[:nx]] = 1.0
        SX = Bx @ Sx; SY = (1.0 - Bx) @ Sy
        L = np.abs(SX/SX.sum(1, keepdims=True) - SY/SY.sum(1, keepdims=True)).sum(1)
        count += int((ddist <= L).sum()); done += b
    return ddist, (count+1)/(k+1)


def greedy_cv(A, B, sxr, syr, thresh, k_grid):
    xs, ys = slabs(A, sxr), slabs(B, syr)
    xk, yk = list(range(len(A))), list(range(len(B)))
    rows, log, bk = [], [], None
    for k in range(0, max(k_grid)+1):
        sx, sy = xs[xk].sum(0), ys[yk].sum(0)
        if k in k_grid:
            _, p = perm_pval_cv(np.vstack([A[xk], B[yk]]), len(xk), len(yk), sxr, syr)
            rows.append((k, p, len(xk), len(yk), p <= thresh))
            if p > thresh and bk is None:
                bk = k
        if k == max(k_grid) or bk is not None:
            break
        base = _l1(sx, sy); best, bd, side = None, -1e9, None
        for i in xk:
            d = base - _l1(sx-xs[i], sy)
            if d > bd: bd, best, side = d, i, "X"
        for j in yk:
            d = base - _l1(sx, sy-ys[j])
            if d > bd: bd, best, side = d, j, "Y"
        (xk if side == "X" else yk).remove(best); log.append((side, best, bd))
    return rows, bk


def main():
    cv = pd.read_csv(CVTAB).set_index(["codon", "ss"])["sigma_cv_deg"]
    df = pd.read_csv(DS)
    print(f"{'pair':<16}{'SS':<6}{'sigA':>6}{'sigB':>6}{'CV ddist':>9}{'CV p':>9}"
          f"{'pub p':>9}{'sig':>5}{'breakdown-k':>13}")
    out = []
    for ss, c1, c2 in PAIRS:
        sA = float(cv.get((c1, ss))); sB = float(cv.get((c2, ss)))
        A = np.deg2rad(df[(df.condition_group==ss)&(df.codon==c1)][["phi","psi"]].values)
        B = np.deg2rad(df[(df.condition_group==ss)&(df.codon==c2)][["phi","psi"]].values)
        dd, p = perm_pval_cv(np.vstack([A, B]), len(A), len(B), np.deg2rad(sA), np.deg2rad(sB))
        sig = p <= BH[ss]
        bk_txt = ""
        if sig:
            nmin = min(len(A), len(B))
            kg = [k for k in [0,1,2,3,5,8,12,20, min(40,max(5,int(np.ceil(0.05*nmin))))] if k <= nmin-2]
            rows, bk = greedy_cv(A, B, np.deg2rad(sA), np.deg2rad(sB), BH[ss], kg)
            bk_txt = f"{bk}" if bk else f">{max(kg)}"
        print(f"{c1+':'+c2.split('-')[1]:<16}{ss:<6}{sA:>6.1f}{sB:>6.1f}{dd:>9.3f}{p:>9.4f}"
              f"{PUB[(ss,c1,c2)]:>9.4f}{('yes' if sig else 'no'):>5}{bk_txt:>13}")
        out.append(dict(SS=ss, pair=f"{c1}:{c2}", sigA=sA, sigB=sB, cv_ddist=round(dd,4),
                        cv_p=round(p,5), pub_p=PUB[(ss,c1,c2)], significant=sig, breakdown_k=bk_txt))
    os.makedirs("out/pnas-2026-repro", exist_ok=True)
    pd.DataFrame(out).to_csv("out/pnas-2026-repro/robustness_cv_summary.csv", index=False)
    print("\nsaved: out/pnas-2026-repro/robustness_cv_summary.csv")


if __name__ == "__main__":
    main()
