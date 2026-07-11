#!/usr/bin/env python
"""#1 directional/speed redone on unbiased MMD^2 (size-unbiased -> no subsampling).
Correlate |log Chevance speed-ratio| with unbiased MMD^2 across synonymous pairs."""
import sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, "scripts/pnas2026")
from mmd_breakdown import kmat, mmd2
from per_aa_directional import SPD

DS = "out/pnas-2026-repro/pointwise_cdist-SMOKE-kde_g_10-cr_none/_intermediate_/dataset.csv"
MINN, CAP, SEED = 40, 800, 12345
ROBUST = {("TURN","A-GCG","A-GCT"), ("HELIX","L-CTC","L-TTG")}


def mmd2u(A, B):
    K = kmat(np.vstack([A, B])); nx, ny = len(A), len(B)
    Sxx = K[:nx, :nx].sum(); Syy = K[nx:, nx:].sum(); Sxy = K[:nx, nx:].sum()
    return mmd2(Sxx, Syy, Sxy, nx, ny, "u")


def spd(c): return SPD.get(c.split("-")[1].replace("T", "U"))


def main():
    df = pd.read_csv(DS); df["AA"] = df.codon.str.split("-").str[0]
    rng = np.random.default_rng(SEED); rows = []
    for (aa, ss), g in df.groupby(["AA", "condition_group"]):
        cods = [c for c, n in g.codon.value_counts().items() if n >= MINN and spd(c)]
        ang = {}
        for c in cods:
            v = np.deg2rad(g[g.codon == c][["phi", "psi"]].values)
            if len(v) > CAP: v = v[rng.choice(len(v), CAP, replace=False)]
            ang[c] = v
        for i in range(len(cods)):
            for j in range(i+1, len(cods)):
                ca, cb = cods[i], cods[j]
                m = mmd2u(ang[ca], ang[cb])
                lsr = abs(np.log(spd(ca)/spd(cb)))
                rob = (ss, ca, cb) in ROBUST or (ss, cb, ca) in ROBUST
                rows.append((aa, ss, ca, cb, min(len(ang[ca]), len(ang[cb])), lsr, m, rob))
    P = pd.DataFrame(rows, columns=["AA","SS","cA","cB","minn","lsr","mmd2u","robust"])
    P.to_csv("out/pnas-2026-repro/directional_speed_mmd.csv", index=False)
    print(f"pairs: {len(P)}  (codons capped at {CAP})")
    print("Spearman(min n, MMD2u)            rho=%+.3f p=%.2g  (confirms size-unbiased)"
          % spearmanr(P.minn, P.mmd2u))
    print("Spearman(|log speed-ratio|, MMD2u) rho=%+.3f p=%.2g"
          % spearmanr(P.lsr, P.mmd2u))
    print("\nrobust pairs:")
    print(P[P.robust][["SS","cA","cB","minn","lsr","mmd2u"]].round(4).to_string(index=False))
    print("\nsaved: out/pnas-2026-repro/directional_speed_mmd.csv")


if __name__ == "__main__":
    main()
