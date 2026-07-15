#!/usr/bin/env python
"""#2 expression confounder redone on unbiased MMD^2.
Within PaxDb abundance tertiles, recompute each pair's unbiased MMD^2 + permutation p.
MMD^2 is size-unbiased so the cross-stratum magnitudes are directly comparable."""

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "scripts/pnas2026")
from _common import PAIRS
from robustness_mmd import THR, kmat, mmd2, perm_p

# Positional arg overrides the default; default is the published/reproduced aggregated
# dataset -- the post-preprocessing/aggregation output of `analyze_pointwise.py`'s
# pipeline (see docs/pnas_2026.md), not the raw data-precs.csv.
DS = (
    sys.argv[1]
    if len(sys.argv) > 1
    else (
        "out/prec-collected/20211001_124553-aida-ex_EC-src_EC/results/"
        "pointwise_cdist-natcom/_intermediate_/dataset.csv"
    )
)
PAX = "out/pnas-2026-repro/paxdb_abundance.csv"
ROBUST = {"L-CTC:L-TTG", "A-GCG:A-GCT"}
MINN = 15


def main():
    df = pd.read_csv(DS)
    ab = pd.read_csv(PAX).set_index("unp_id")["abundance"]
    d = df.assign(abund=df.unp_id.map(ab)).dropna(subset=["abund"]).copy()
    d["la"] = np.log10(d.abund)
    q = d.la.quantile([1 / 3, 2 / 3]).values
    d["bin"] = np.where(d.la <= q[0], "LOW", np.where(d.la <= q[1], "MID", "HIGH"))
    print(
        f"# positions w/ abundance {len(d)}/{len(df)} ({100 * len(d) / len(df):.0f}%)\n"
    )
    print(
        f"{'pair':<14}{'SS':<6}{'bin':<6}{'n1':>5}{'n2':>5}{'MMD2u p':>9}{'sig':>5}{'MMD2u':>9}"
    )
    rows = []
    for ss, c1, c2 in PAIRS:
        tag = "*" if f"{c1}:{c2}" in ROBUST else " "
        for b in ["ALL", "LOW", "MID", "HIGH"]:
            sub = d[d.condition_group == ss]
            if b != "ALL":
                sub = sub[sub.bin == b]
            A = np.deg2rad(sub[sub.codon == c1][["phi", "psi"]].values)
            B = np.deg2rad(sub[sub.codon == c2][["phi", "psi"]].values)
            if len(A) < MINN or len(B) < MINN:
                print(
                    f"{c1 + ':' + c2.split('-')[1] + tag:<14}{ss:<6}{b:<6}{len(A):>5}{len(B):>5}{'(small)':>9}"
                )
                continue
            K = kmat(np.vstack([A, B]))
            nx, ny = len(A), len(B)
            m = mmd2(
                K[:nx, :nx].sum(), K[nx:, nx:].sum(), K[:nx, nx:].sum(), nx, ny, "u"
            )
            _, p = perm_p(K, nx, ny, "u")
            sig = p <= THR[ss]
            print(
                f"{c1 + ':' + c2.split('-')[1] + tag:<14}{ss:<6}{b:<6}{nx:>5}{ny:>5}{p:>9.4f}"
                f"{('yes' if sig else 'no'):>5}{m:>9.4f}"
            )
            rows.append(
                dict(
                    pair=f"{c1}:{c2}",
                    SS=ss,
                    robust=f"{c1}:{c2}" in ROBUST,
                    bin=b,
                    n1=nx,
                    n2=ny,
                    mmd2u_p=round(p, 5),
                    sig=sig,
                    mmd2u=round(m, 4),
                )
            )
        print()
    pd.DataFrame(rows).to_csv(
        "out/pnas-2026-repro/expression_confound_mmd.csv", index=False
    )
    print(
        "saved: out/pnas-2026-repro/expression_confound_mmd.csv   (* = robust-across-stats pair)"
    )


if __name__ == "__main__":
    main()
