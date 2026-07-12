#!/usr/bin/env python
"""
#5 B-factor / flexibility analysis for the robust codon pairs (POSITION level).

B is z-normalised within each structure, then averaged per aggregation position
(unp_id, unp_idx, SS) — the same unit as the main analysis (no pseudoreplication).
Per robust pair:
  (a) do the two codons differ in flexibility (position B_z)?  [Mann-Whitney]
  (b) are these positions rigid or flexible vs the proteome (B_z ~ 0)?
  (c) does the codon (φ,ψ) difference live in RIGID or FLEXIBLE positions?
      split positions at the pair's median B_z; torus distance between codon
      circular means + label-permutation p in each stratum.
"""
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

DP = "out/prec-collected/20211001_124553-aida-ex_EC-src_EC/data-precs.csv"
CONS = {
    "E": "SHEET",
    "H": "HELIX",
    "G": "OTHER",
    "I": "OTHER",
    "T": "TURN",
    "S": "OTHER",
    "B": "OTHER",
    "-": None,
    "": None,
}
PAIRS = [
    ("HELIX", "L-CTC", "L-TTG"),
    ("HELIX", "L-CTC", "L-CTG"),
    ("HELIX", "L-CTC", "L-CTT"),
    ("HELIX", "R-AGG", "R-CGA"),
    ("TURN", "A-GCG", "A-GCT"),
    ("TURN", "P-CCC", "P-CCG"),
]
K, SEED = 2000, 12345


def cmean(d):
    r = np.deg2rad(np.asarray(d, float))
    return np.rad2deg(np.arctan2(np.sin(r).mean(), np.cos(r).mean()))


def tdist(a, b):
    d = np.abs(((np.array(a) - np.array(b) + 180) % 360) - 180)
    return float(np.hypot(*d))


def perm_p(A, B, k=K, seed=SEED):
    rng = np.random.default_rng(seed)
    Z = np.vstack([A, B])
    nA = len(A)
    obs = tdist((cmean(A[:, 0]), cmean(A[:, 1])), (cmean(B[:, 0]), cmean(B[:, 1])))
    c = sum(
        tdist(
            (cmean(Z[i[:nA], 0]), cmean(Z[i[:nA], 1])),
            (cmean(Z[i[nA:], 0]), cmean(Z[i[nA:], 1])),
        )
        >= obs
        for i in (rng.permutation(len(Z)) for _ in range(k))
    )
    return obs, (c + 1) / (k + 1)


def main():
    dp = pd.read_csv(
        DP,
        usecols=[
            "pdb_id",
            "unp_id",
            "unp_idx",
            "codon",
            "secondary",
            "phi",
            "psi",
            "bfactor",
        ],
        low_memory=False,
    )
    dp = dp.dropna(subset=["unp_idx", "phi", "psi", "bfactor"])
    dp["unp_idx"] = dp["unp_idx"].astype(int)
    dp["cg"] = dp["secondary"].map(lambda s: CONS.get(str(s), None))
    dp = dp.dropna(subset=["cg"])
    g = dp.groupby("pdb_id")["bfactor"]
    dp["bz"] = (dp["bfactor"] - g.transform("mean")) / g.transform("std")
    dp = dp.dropna(subset=["bz"])

    # aggregate to positions: mean B_z + circular-mean (phi,psi)
    pos = (
        dp.groupby(["unp_id", "unp_idx", "cg", "codon"])
        .agg(
            bz=("bz", "mean"),
            phi=("phi", lambda s: cmean(s)),
            psi=("psi", lambda s: cmean(s)),
            n=("bz", "size"),
        )
        .reset_index()
    )

    print(
        "=== #5 B-factor / flexibility (position level; B_z within-structure z-score) ==="
    )
    print(
        f"{'pair':<14}{'SS':<6}{'n1':>5}{'n2':>5}{'Bz med1':>9}{'Bz med2':>9}{'MWU p':>8}"
        f"{'flex?':>7}"
    )
    rows, strat = [], []
    for ss, c1, c2 in PAIRS:
        sub = pos[pos.cg == ss]
        a = sub[sub.codon == c1.split("-")[1]]
        b = sub[sub.codon == c2.split("-")[1]]
        if len(a) < 5 or len(b) < 5:
            continue
        _, p = mannwhitneyu(a.bz, b.bz, alternative="two-sided")
        med = pd.concat([a.bz, b.bz]).median()
        tag = "rigid" if med < -0.1 else "flex" if med > 0.1 else "~avg"
        print(
            f"{c1+':'+c2.split('-')[1]:<14}{ss:<6}{len(a):>5}{len(b):>5}"
            f"{a.bz.median():>9.2f}{b.bz.median():>9.2f}{p:>8.3f}{tag:>7}"
        )
        rows.append(
            dict(
                pair=f"{c1}:{c2}",
                SS=ss,
                n1=len(a),
                n2=len(b),
                bz_med1=round(a.bz.median(), 2),
                bz_med2=round(b.bz.median(), 2),
                mwu_p=round(p, 4),
                pair_bz_median=round(med, 2),
                flex=tag,
            )
        )
        both = pd.concat([a, b])
        thr = both.bz.median()
        for lab, mask in [("rigid", both.bz <= thr), ("flexible", both.bz > thr)]:
            s = both[mask]
            aa = s[s.codon == c1.split("-")[1]][["phi", "psi"]].values
            bb = s[s.codon == c2.split("-")[1]][["phi", "psi"]].values
            d, pp = (
                perm_p(aa, bb) if len(aa) >= 5 and len(bb) >= 5 else (np.nan, np.nan)
            )
            strat.append(
                dict(
                    pair=f"{c1}:{c2}",
                    SS=ss,
                    stratum=lab,
                    n1=len(aa),
                    n2=len(bb),
                    dist=round(d, 1) if d == d else None,
                    perm_p=round(pp, 4) if pp == pp else None,
                )
            )

    print("\n=== (c) codon (φ,ψ) difference in RIGID vs FLEXIBLE positions ===")
    print(
        f"{'pair':<14}{'SS':<6}{'stratum':<10}{'n1':>5}{'n2':>5}{'dist':>7}{'perm_p':>9}"
    )
    for r in strat:
        d = r["dist"] if r["dist"] is not None else float("nan")
        pp = r["perm_p"] if r["perm_p"] is not None else float("nan")
        print(
            f"{r['pair']:<14}{r['SS']:<6}{r['stratum']:<10}{r['n1']:>5}{r['n2']:>5}{d:>7}{pp:>9}"
        )
    pd.DataFrame(rows).to_csv("out/pnas-2026-repro/bfactor_pairs.csv", index=False)
    pd.DataFrame(strat).to_csv("out/pnas-2026-repro/bfactor_strat.csv", index=False)
    print("\nsaved: bfactor_pairs.csv, bfactor_strat.csv")


if __name__ == "__main__":
    main()
