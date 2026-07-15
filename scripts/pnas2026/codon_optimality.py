#!/usr/bin/env python
"""
Biological-significance probes on the codon-conditioned backbone signal.

#1 Directionality vs codon optimality
   - Optimality proxy = empirical within-amino-acid codon usage in the dataset
     (matches the known E. coli K-12 ranks: GCG/CTG/CCG/CGC dominant).
   - (a) Across ALL synonymous codon pairs (within AA, within SS, n>=MINN each):
         does a larger usage gap go with a larger backbone-distribution shift?
         -> Spearman( |Δusage| , torus-distance-between-codon-means ).
   - (b) For the robust pairs: tabulate usage, the optimal codon, the oriented
         (optimal − rare) angular shift, and each codon's distance from the
         (AA,SS) consensus.

#2 Expression stratification (Cope–Gilchrist confound)
   - Expression proxy = per-gene mean codon usage (CAI-like) over the gene's
     residues. Residues are split into LOW/MID/HIGH tertiles.
   - For each robust pair, the codon-pair (φ,ψ) difference (torus distance between
     circular means + label-permutation p) is recomputed WITHIN each tertile.
     Persistence within strata argues the signal is not a pure expression artifact.
"""

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "scripts/pnas2026")
from _common import PAIRS, SEED

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
MINN = 20
K = 2000


def cmean(deg):
    r = np.deg2rad(np.asarray(deg, float))
    return np.rad2deg(np.arctan2(np.sin(r).mean(), np.cos(r).mean()))


def tdist(p1, p2):  # flat-torus distance between two (phi,psi) points, deg
    d = np.abs(((np.array(p1) - np.array(p2) + 180) % 360) - 180)
    return float(np.hypot(*d))


def codon_mean(df, ss, codon):
    s = df[(df.condition_group == ss) & (df.codon == codon)]
    return (cmean(s.phi.values), cmean(s.psi.values)), len(s)


def perm_p_meandist(A, B, k=K, seed=SEED):
    """label-permutation p for torus distance between circular means of A,B."""
    rng = np.random.default_rng(seed)
    Z = np.vstack([A, B])
    nA = len(A)
    obs = tdist((cmean(A[:, 0]), cmean(A[:, 1])), (cmean(B[:, 0]), cmean(B[:, 1])))
    c = 0
    for _ in range(k):
        idx = rng.permutation(len(Z))
        a, b = Z[idx[:nA]], Z[idx[nA:]]
        d = tdist((cmean(a[:, 0]), cmean(a[:, 1])), (cmean(b[:, 0]), cmean(b[:, 1])))
        if d >= obs:
            c += 1
    return obs, (c + 1) / (k + 1)


def main():
    df = pd.read_csv(DS)
    df["AA"] = df.codon.str.split("-").str[0]
    df["cod"] = df.codon.str.split("-").str[1]
    usage = {
        aa: g.cod.value_counts(normalize=True).to_dict() for aa, g in df.groupby("AA")
    }

    # ---- #1a: usage gap vs structural distance, across all codon pairs --------
    rows = []
    for (aa, ss), g in df.groupby(["AA", "condition_group"]):
        counts = g.codon.value_counts()
        cods = [c for c in counts.index if counts[c] >= MINN]
        means = {
            c: (cmean(g[g.codon == c].phi), cmean(g[g.codon == c].psi)) for c in cods
        }
        for i in range(len(cods)):
            for j in range(i + 1, len(cods)):
                ca, cb = cods[i], cods[j]
                fa = usage[aa][ca.split("-")[1]]
                fb = usage[aa][cb.split("-")[1]]
                rows.append((aa, ss, ca, cb, abs(fa - fb), tdist(means[ca], means[cb])))
    P = pd.DataFrame(rows, columns=["AA", "SS", "cA", "cB", "usage_gap", "ang_dist"])
    from scipy.stats import spearmanr

    rho, pv = spearmanr(P.usage_gap, P.ang_dist)
    print(f"#1a  codon pairs (within AA,SS; n>={MINN} each): {len(P)}")
    print(
        f"     Spearman(|Δusage|, torus-dist-between-means) rho={rho:.3f}  p={pv:.2e}"
    )
    print(f"     usage_gap quartiles -> mean ang_dist:")
    P["qbin"] = pd.qcut(P.usage_gap, 4, labels=["Q1", "Q2", "Q3", "Q4"])
    print(
        P.groupby("qbin", observed=True)
        .ang_dist.agg(["mean", "median", "count"])
        .round(2)
        .to_string()
    )

    # ---- #1b: directionality for the robust pairs -----------------------------
    print("\n#1b  robust pairs — usage, optimal codon, oriented shift (optimal−rare):")
    print(
        f"{'pair':<16}{'SS':<6}{'f(c1)':>6}{'f(c2)':>6}  {'optimal':<8}"
        f"{'Δφ':>6}{'Δψ':>6}{'dist':>6}"
    )
    db = []
    for ss, c1, c2 in PAIRS:
        f1 = usage[c1.split("-")[0]][c1.split("-")[1]]
        f2 = usage[c2.split("-")[0]][c2.split("-")[1]]
        m1, _ = codon_mean(df, ss, c1)
        m2, _ = codon_mean(df, ss, c2)
        opt, rare = (c1, c2) if f1 >= f2 else (c2, c1)
        mo = m1 if opt == c1 else m2
        mr = m1 if rare == c1 else m2
        dphi = ((mo[0] - mr[0] + 180) % 360) - 180
        dpsi = ((mo[1] - mr[1] + 180) % 360) - 180
        print(
            f"{c1 + ':' + c2.split('-')[1]:<16}{ss:<6}{f1:>6.2f}{f2:>6.2f}  {opt:<8}"
            f"{dphi:>6.1f}{dpsi:>6.1f}{tdist(mo, mr):>6.1f}"
        )
        db.append(
            dict(
                pair=f"{c1}:{c2}",
                SS=ss,
                f_c1=round(f1, 3),
                f_c2=round(f2, 3),
                optimal=opt,
                dphi=round(dphi, 1),
                dpsi=round(dpsi, 1),
                dist=round(tdist(mo, mr), 1),
            )
        )
    pd.DataFrame(db).to_csv(
        "out/pnas-2026-repro/codon_optimality_pairs.csv", index=False
    )

    # ---- #2: expression stratification ---------------------------------------
    gene_opt = (
        df.assign(f=[usage[a][c] for a, c in zip(df.AA, df.cod)])
        .groupby("unp_id")
        .f.mean()
    )
    df["gene_opt"] = df.unp_id.map(gene_opt)
    qs = df.gene_opt.quantile([1 / 3, 2 / 3]).values
    df["expr_bin"] = np.where(
        df.gene_opt <= qs[0], "LOW", np.where(df.gene_opt <= qs[1], "MID", "HIGH")
    )
    print(
        "\n#2  codon-pair (φ,ψ) difference within gene-expression tertiles "
        "(proxy = per-gene mean codon usage):"
    )
    print(f"{'pair':<16}{'SS':<6}{'bin':<6}{'n1':>5}{'n2':>5}{'dist':>7}{'perm_p':>9}")
    s2 = []
    for ss, c1, c2 in PAIRS:
        for b in ["LOW", "MID", "HIGH"]:
            sub = df[(df.condition_group == ss) & (df.expr_bin == b)]
            A = sub[sub.codon == c1][["phi", "psi"]].values
            B = sub[sub.codon == c2][["phi", "psi"]].values
            if len(A) >= 5 and len(B) >= 5:
                dist, p = perm_p_meandist(A, B)
            else:
                dist, p = float("nan"), float("nan")
            print(
                f"{c1 + ':' + c2.split('-')[1]:<16}{ss:<6}{b:<6}{len(A):>5}{len(B):>5}"
                f"{dist:>7.1f}{p:>9.4f}"
            )
            s2.append(
                dict(
                    pair=f"{c1}:{c2}",
                    SS=ss,
                    expr_bin=b,
                    n1=len(A),
                    n2=len(B),
                    dist=round(dist, 1) if dist == dist else None,
                    perm_p=round(p, 4) if p == p else None,
                )
            )
    pd.DataFrame(s2).to_csv(
        "out/pnas-2026-repro/codon_optimality_strat.csv", index=False
    )
    print("\nsaved: codon_optimality_pairs.csv, codon_optimality_strat.csv")


if __name__ == "__main__":
    main()
