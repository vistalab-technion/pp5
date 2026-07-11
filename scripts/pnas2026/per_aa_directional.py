#!/usr/bin/env python
"""
Per-amino-acid directional test: within each amino acid, does codon translation
speed (Chevance, high = slow) correlate with the backbone angle?

Uses codon CIRCULAR MEANS (φ̄, ψ̄) — far less sample-size-biased than the density
distances. Secondary structure is controlled by centering each codon mean on its
(AA, SS) group mean (Δφ, Δψ). For each AA we Spearman-correlate codon speed with
Δφ and Δψ across its (codon, SS) entries, with a within-AA speed-label permutation
p-value. Proline is the a-priori candidate (EF-P stalling; PPII vs α along ψ).
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

DS = "out/pnas-2026-repro/pointwise_cdist-SMOKE-kde_g_10-cr_none/_intermediate_/dataset.csv"
MINN, KPERM, SEED = 40, 5000, 12345
SPD = {
 "UUU":2.2,"UUC":2.0,"UUA":1.6,"UUG":1.5,"UCU":1.9,"UCC":2.1,"UCA":1.4,"UCG":1.6,
 "UAU":2.8,"UAC":2.3,"UGU":4.4,"UGC":2.0,"UGG":2.4,
 "CUU":2.3,"CUC":2.1,"CUA":1.6,"CUG":1.0,"CCU":2.5,"CCC":3.3,"CCA":1.7,"CCG":1.5,
 "CAU":1.7,"CAC":1.0,"CAA":1.5,"CAG":1.0,"CGU":7.9,"CGC":1.7,"CGA":7.3,"CGG":4.1,
 "AUU":1.8,"AUC":1.6,"AUA":2.9,"AUG":1.0,"ACU":1.1,"ACC":1.2,"ACA":0.9,"ACG":0.8,
 "AAU":1.9,"AAC":1.4,"AAA":1.3,"AAG":1.2,"AGU":6.7,"AGC":1.4,"AGA":5.0,"AGG":9.2,
 "GUU":1.8,"GUC":1.8,"GUA":1.1,"GUG":1.3,"GCU":1.1,"GCC":1.0,"GCA":0.7,"GCG":0.7,
 "GAU":2.3,"GAC":1.5,"GAA":1.7,"GAG":2.0,"GGU":5.2,"GGC":1.7,"GGA":2.1,"GGG":2.0,
}
AA3 = {"A":"Ala","R":"Arg","N":"Asn","D":"Asp","C":"Cys","Q":"Gln","E":"Glu","G":"Gly",
       "H":"His","I":"Ile","L":"Leu","K":"Lys","M":"Met","F":"Phe","P":"Pro","S":"Ser",
       "T":"Thr","W":"Trp","Y":"Tyr","V":"Val"}


def cmean(d):
    r = np.deg2rad(d); return float(np.rad2deg(np.arctan2(np.sin(r).mean(), np.cos(r).mean())))


def cdiff(a, b):  # circular a-b in (-180,180]
    return ((a - b + 180) % 360) - 180


def main():
    df = pd.read_csv(DS)
    df["AA"] = df.codon.str.split("-").str[0]
    df["rna"] = df.codon.str.split("-").str[1].str.replace("T", "U")
    # codon means per (AA, SS, codon)
    g = df.groupby(["AA", "condition_group", "codon", "rna"])
    rec = g.agg(n=("phi", "size"), phibar=("phi", cmean), psibar=("psi", cmean)).reset_index()
    rec = rec[rec.n >= MINN].copy()
    rec["speed"] = rec.rna.map(SPD)
    rec = rec.dropna(subset=["speed"])
    # SS-center within (AA, SS)
    for ax, col in [("phibar", "dphi"), ("psibar", "dpsi")]:
        ssmean = rec.groupby(["AA", "condition_group"])[ax].transform(cmean)
        rec[col] = cdiff(rec[ax].values, ssmean.values)

    rng = np.random.default_rng(SEED)
    out = []
    for aa, sub in rec.groupby("AA"):
        codons = sub.rna.unique()
        if len(codons) < 3 or len(sub) < 4:
            continue
        sp = sub.speed.values
        res = {}
        for col in ["dphi", "dpsi"]:
            rho, _ = spearmanr(sp, sub[col].values)
            # permute speed across the AA's distinct codons
            cod2sp = dict(zip(codons, [SPD[c] for c in codons]))
            cnt = 0
            for _ in range(KPERM):
                perm = dict(zip(codons, rng.permutation([cod2sp[c] for c in codons])))
                spp = sub.rna.map(perm).values
                rp, _ = spearmanr(spp, sub[col].values)
                if abs(rp) >= abs(rho):
                    cnt += 1
            res[col] = (rho, (cnt + 1) / (KPERM + 1))
        out.append(dict(AA=AA3.get(aa, aa), n_codons=len(codons), n_entries=len(sub),
                        rho_phi=round(res["dphi"][0], 2), p_phi=round(res["dphi"][1], 4),
                        rho_psi=round(res["dpsi"][0], 2), p_psi=round(res["dpsi"][1], 4)))
    O = pd.DataFrame(out).sort_values("p_psi")
    print("=== per-AA: codon speed vs SS-centered backbone angle (circular means) ===")
    print(O.to_string(index=False))
    O.to_csv("out/pnas-2026-repro/per_aa_directional.csv", index=False)

    # focused proline view
    print("\n=== Proline: codon speed vs mean ψ (PPII axis) per SS ===")
    p = rec[rec.AA == "P"].sort_values(["condition_group", "speed"])
    print(p[["condition_group", "rna", "n", "speed", "phibar", "psibar"]]
          .round(1).to_string(index=False))
    print("\nsaved: out/pnas-2026-repro/per_aa_directional.csv")


if __name__ == "__main__":
    main()
