#!/usr/bin/env python
"""
Directional/speed analysis (#1), size-corrected.

Question: does codon translation-speed difference predict backbone-distribution
divergence — and does any relationship survive sample-size correction?

- Speed axis: |log(speed ratio)| from the digitized Chevance Fig.3 values.
- Structure axis, RAW: KDE-L1 distance between the two codons' aggregated (φ,ψ)
  at their natural (unequal) sample sizes  [carries the finite-sample bias].
- Structure axis, SIZE-MATCHED: subsample BOTH codons to a common n0, compute
  KDE-L1, average over R repeats  [same bias for every pair -> comparable].

Across all synonymous codon pairs (within AA, within SS, both codons >= n0) we
Spearman-correlate |log speed-ratio| with each structure axis. If the raw
correlation is positive but the size-matched one vanishes, the speed↔structure
trend was a sample-size artifact.
"""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "scripts/pnas2026")
from robustness_outliers import slab_rows_for  # (n,P) KDE slab rows
from _common import SEED

# Positional arg overrides the default; default is the published/reproduced aggregated
# dataset -- the post-preprocessing/aggregation output of `analyze_pointwise.py`'s
# pipeline (see docs/pnas_2026.md), not the raw data-precs.csv. No dependency on Alex's
# repro zip.
DS = (
    sys.argv[1]
    if len(sys.argv) > 1
    else (
        "out/prec-collected/20211001_124553-aida-ex_EC-src_EC/results/"
        "pointwise_cdist-natcom/_intermediate_/dataset.csv"
    )
)
N0, R, MINN = 40, 25, 40
ROBUST = {
    ("HELIX", "L-CTC", "L-TTG"),
    ("HELIX", "L-CTC", "L-CTG"),
    ("HELIX", "L-CTC", "L-CTT"),
    ("HELIX", "R-AGG", "R-CGA"),
    ("TURN", "A-GCG", "A-GCT"),
    ("TURN", "P-CCC", "P-CCG"),
}

# Chevance Fig.3 values (high = slow); RNA codons. Stops excluded.
SPD = {
    "UUU": 2.2,
    "UUC": 2.0,
    "UUA": 1.6,
    "UUG": 1.5,
    "UCU": 1.9,
    "UCC": 2.1,
    "UCA": 1.4,
    "UCG": 1.6,
    "UAU": 2.8,
    "UAC": 2.3,
    "UGU": 4.4,
    "UGC": 2.0,
    "UGG": 2.4,
    "CUU": 2.3,
    "CUC": 2.1,
    "CUA": 1.6,
    "CUG": 1.0,
    "CCU": 2.5,
    "CCC": 3.3,
    "CCA": 1.7,
    "CCG": 1.5,
    "CAU": 1.7,
    "CAC": 1.0,
    "CAA": 1.5,
    "CAG": 1.0,
    "CGU": 7.9,
    "CGC": 1.7,
    "CGA": 7.3,
    "CGG": 4.1,
    "AUU": 1.8,
    "AUC": 1.6,
    "AUA": 2.9,
    "AUG": 1.0,
    "ACU": 1.1,
    "ACC": 1.2,
    "ACA": 0.9,
    "ACG": 0.8,
    "AAU": 1.9,
    "AAC": 1.4,
    "AAA": 1.3,
    "AAG": 1.2,
    "AGU": 6.7,
    "AGC": 1.4,
    "AGA": 5.0,
    "AGG": 9.2,
    "GUU": 1.8,
    "GUC": 1.8,
    "GUA": 1.1,
    "GUG": 1.3,
    "GCU": 1.1,
    "GCC": 1.0,
    "GCA": 0.7,
    "GCG": 0.7,
    "GAU": 2.3,
    "GAC": 1.5,
    "GAA": 1.7,
    "GAG": 2.0,
    "GGU": 5.2,
    "GGC": 1.7,
    "GGA": 2.1,
    "GGG": 2.0,
}


def spd(codon):  # codon like 'L-CTC' -> RNA, lookup
    rna = codon.split("-")[1].replace("T", "U")
    return SPD.get(rna)


def ddist(slab_a, slab_b):
    a = slab_a.sum(0)
    a /= a.sum()
    b = slab_b.sum(0)
    b /= b.sum()
    return float(np.abs(a - b).sum())


def main():
    df = pd.read_csv(DS)
    df["AA"] = df.codon.str.split("-").str[0]
    rng = np.random.default_rng(SEED)
    rows = []
    for (aa, ss), g in df.groupby(["AA", "condition_group"]):
        cods = [c for c, n in g.codon.value_counts().items() if n >= MINN and spd(c)]
        ang = {c: np.deg2rad(g[g.codon == c][["phi", "psi"]].values) for c in cods}
        for i in range(len(cods)):
            for j in range(i + 1, len(cods)):
                ca, cb = cods[i], cods[j]
                A, Bv = ang[ca], ang[cb]
                raw = ddist(slab_rows_for(A), slab_rows_for(Bv))
                sub = np.mean(
                    [
                        ddist(
                            slab_rows_for(A[rng.choice(len(A), N0, replace=False)]),
                            slab_rows_for(Bv[rng.choice(len(Bv), N0, replace=False)]),
                        )
                        for _ in range(R)
                    ]
                )
                lsr = abs(np.log(spd(ca) / spd(cb)))
                rows.append(
                    dict(
                        AA=aa,
                        SS=ss,
                        cA=ca,
                        cB=cb,
                        nA=len(A),
                        nB=len(Bv),
                        lsr=lsr,
                        raw=raw,
                        sized=sub,
                        robust=(ss, ca, cb) in ROBUST or (ss, cb, ca) in ROBUST,
                    )
                )
    P = pd.DataFrame(rows)
    P.to_csv("out/pnas-2026-repro/directional_speed.csv", index=False)
    from scipy.stats import spearmanr

    print(
        f"synonymous codon pairs (within AA,SS; n>={MINN} each): {len(P)}   "
        f"subsample n0={N0}, R={R}"
    )
    for col, lab in [
        ("raw", "RAW distance (unequal n)"),
        ("sized", f"SIZE-MATCHED distance (both -> n0={N0})"),
    ]:
        rho, pv = spearmanr(P.lsr, P[col])
        print(f"  Spearman(|log speed-ratio|, {lab:<34}) rho={rho:+.3f}  p={pv:.2g}")
    print("\n  also: does sample-size predict raw distance? (the bias)")
    P["minn"] = P[["nA", "nB"]].min(axis=1)
    rho, pv = spearmanr(P.minn, P.raw)
    print(
        f"  Spearman(min sample size, RAW distance) rho={rho:+.3f}  p={pv:.2g}  "
        f"(negative => small n inflates distance)"
    )
    print("\n  robust pairs:")
    print(
        P[P.robust][["SS", "cA", "cB", "nA", "nB", "lsr", "raw", "sized"]]
        .round(3)
        .to_string(index=False)
    )
    print("\nsaved: out/pnas-2026-repro/directional_speed.csv")


if __name__ == "__main__":
    main()
