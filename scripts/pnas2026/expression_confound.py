#!/usr/bin/env python
"""
#2 Expression confounder (Cope-Gilchrist), done properly.

Non-circular expression axis = PaxDb integrated protein abundance (E. coli K-12),
joined to each position's gene (unp_id). Positions split into LOW/MID/HIGH
abundance tertiles. For each robust pair, within each tertile we report:
  - KDE-L1 permutation p at the natural sample size (size-calibrated significance);
  - a SIZE-MATCHED distance (subsample both codons to a common n0, KDE-L1, averaged)
    so effect magnitudes are comparable across strata of different n.
Persistence of significance across abundance strata argues the codon-structure
signal is not an expression artifact.
"""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts/pnas2026")
from robustness_outliers import slab_rows_for, perm_pval

DS = "out/pnas-2026-repro/pointwise_cdist-SMOKE-kde_g_10-cr_none/_intermediate_/dataset.csv"
PAX = "out/pnas-2026-repro/paxdb_abundance.csv"
PAIRS = [("HELIX","L-CTC","L-TTG"),("HELIX","L-CTC","L-CTG"),("HELIX","L-CTC","L-CTT"),
         ("HELIX","R-AGG","R-CGA"),("TURN","A-GCG","A-GCT"),("TURN","P-CCC","P-CCG")]
BH = {"HELIX":0.0011494,"TURN":0.0011494}   # KDE-L1 bw=10 BH thresholds
N0, R, KPERM, SEED, MINN = 40, 25, 5000, 12345, 15


def ddist(sa, sb):
    a = sa.sum(0); a /= a.sum(); b = sb.sum(0); b /= b.sum()
    return float(np.abs(a - b).sum())


def main():
    df = pd.read_csv(DS)
    ab = pd.read_csv(PAX).set_index("unp_id")["abundance"]
    df["abund"] = df.unp_id.map(ab)
    cov = df.abund.notna().mean()
    d = df.dropna(subset=["abund"]).copy()
    d["la"] = np.log10(d.abund)
    q = d.la.quantile([1/3, 2/3]).values
    d["bin"] = np.where(d.la <= q[0], "LOW", np.where(d.la <= q[1], "MID", "HIGH"))
    print(f"# positions with PaxDb abundance: {len(d)}/{len(df)} ({100*cov:.0f}%); "
          f"tertile cuts (log10 ppm) {q[0]:.2f},{q[1]:.2f}")
    rng = np.random.default_rng(SEED)
    print(f"\n{'pair':<14}{'SS':<6}{'bin':<6}{'n1':>5}{'n2':>5}{'KDE-L1 p':>10}"
          f"{'sig?':>6}{'sized d':>9}")
    rows = []
    for ss, c1, c2 in PAIRS:
        for b in ["ALL", "LOW", "MID", "HIGH"]:
            sub = d[d.condition_group == ss]
            if b != "ALL":
                sub = sub[sub.bin == b]
            A = np.deg2rad(sub[sub.codon == c1][["phi", "psi"]].values)
            B = np.deg2rad(sub[sub.codon == c2][["phi", "psi"]].values)
            if len(A) < MINN or len(B) < MINN:
                print(f"{c1+':'+c2.split('-')[1]:<14}{ss:<6}{b:<6}{len(A):>5}{len(B):>5}"
                      f"{'(small)':>10}")
                continue
            slab = np.vstack([slab_rows_for(A), slab_rows_for(B)])
            dd0, p = perm_pval(slab, len(A), len(B), KPERM)
            n0 = min(len(A), len(B), N0)
            sized = np.mean([ddist(slab_rows_for(A[rng.choice(len(A), n0, replace=False)]),
                                   slab_rows_for(B[rng.choice(len(B), n0, replace=False)]))
                             for _ in range(R)])
            sig = p <= BH[ss]
            print(f"{c1+':'+c2.split('-')[1]:<14}{ss:<6}{b:<6}{len(A):>5}{len(B):>5}"
                  f"{p:>10.4f}{('yes' if sig else 'no'):>6}{sized:>9.3f}")
            rows.append(dict(pair=f"{c1}:{c2}", SS=ss, abund_bin=b, n1=len(A), n2=len(B),
                             kde_l1_p=round(p, 5), significant=sig, sized_dist=round(sized, 3)))
        print()
    pd.DataFrame(rows).to_csv("out/pnas-2026-repro/expression_confound.csv", index=False)
    print("saved: out/pnas-2026-repro/expression_confound.csv")


if __name__ == "__main__":
    main()
