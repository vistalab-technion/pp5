#!/usr/bin/env python
"""
Spurious-but-significant reference for breakdown-k (colleague comment #2).

control breakdown-k = 0 is tautological (no control pair was significant). Instead
we *manufacture* spurious rejections matched to each real pair and measure their
breakdown-k. Construction: pool one amino acid's residues in one SS, pick a random
circular-aware direction in (φ,ψ), and split a size-matched subset into two
pseudo-codon groups by that direction + tunable noise. The split is independent of
true codon identity, so the rejection is spurious; we binary-search the noise so the
pseudo-pair's p matches the real pair's. Statistic = unbiased MMD^2.

If real pairs need MORE adversarial deletions than these matched fakes, breakdown-k
discriminates real from spurious at equal significance.
"""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts/pnas2026")
from mmd_breakdown import kmat, perm_p, breakdown, THR

DS = "out/pnas-2026-repro/pointwise_cdist-SMOKE-kde_g_10-cr_none/_intermediate_/dataset.csv"
SEED, NREP = 12345, 10
# real pairs to calibrate: (SS, AA, n1, n2, target p, real breakdown-k under MMD2u)
REAL = [("TURN", "A", 350, 203, 0.0002, 8), ("HELIX", "L", 528, 599, 0.0008, 5)]


def embed(ang):  # (n,2) radians -> circular 4D
    return np.column_stack([np.cos(ang[:, 0]), np.sin(ang[:, 0]),
                            np.cos(ang[:, 1]), np.sin(ang[:, 1])])


def split_by_dir(ang, n1, n2, noise, rng):
    sub = ang[rng.choice(len(ang), n1+n2, replace=False)]
    u = rng.normal(size=4); u /= np.linalg.norm(u)
    score = embed(sub) @ u + noise * rng.normal(size=len(sub))
    order = np.argsort(score)
    return sub[order[:n1]], sub[order[n1:]]   # group A (low score), group B (high)


def tune_noise(ang, n1, n2, target_p, ss, rng):
    """binary-search noise so MMD2u p ~ target_p (low noise -> very significant)."""
    lo, hi = 0.02, 3.0
    for _ in range(9):
        mid = np.sqrt(lo*hi)
        A, B = split_by_dir(ang, n1, n2, mid, np.random.default_rng(rng.integers(1e9)))
        K = kmat(np.vstack([A, B])); _, p = perm_p(K, len(A), len(B), "u", k=2000)
        if p <= target_p:      # too significant -> add noise
            lo = mid
        else:
            hi = mid
    return mid


def main():
    df = pd.read_csv(DS); df["AA"] = df.codon.str.split("-").str[0]
    rng = np.random.default_rng(SEED)
    print("Spurious-significant reference (unbiased MMD^2), matched n & p:\n")
    for ss, aa, n1, n2, tp, real_bk in REAL:
        pool = np.deg2rad(df[(df.condition_group == ss) & (df.AA == aa)][["phi", "psi"]].values)
        print(f"=== match {aa}-pair in {ss}: n=({n1},{n2}), target p~{tp}, "
              f"real breakdown-k={real_bk}  (pool n={len(pool)}) ===")
        bks, ps = [], []
        for r in range(NREP):
            noise = tune_noise(pool, n1, n2, tp, ss, rng)
            A, B = split_by_dir(pool, n1, n2, noise, np.random.default_rng(SEED+r))
            K = kmat(np.vstack([A, B])); _, p = perm_p(K, len(A), len(B), "u")
            if p > THR[ss]:
                bks.append(0); ps.append(p); continue
            nmin = min(n1, n2); kg = [k for k in [0,1,2,3,5,8,12,20] if k <= nmin-2]
            _, bk = breakdown(A, B, THR[ss], kg, "u")
            bks.append(bk if bk is not None else max(kg)+1); ps.append(p)
        bks = np.array([b if b is not None else 0 for b in bks])
        print(f"  spurious p: median={np.median(ps):.4f}; "
              f"spurious breakdown-k: {sorted(bks)}  median={int(np.median(bks))} max={int(bks.max())}")
        print(f"  --> REAL breakdown-k = {real_bk}  vs spurious median {int(np.median(bks))} "
              f"(real {'>' if real_bk>np.median(bks) else '<='} spurious)\n")


if __name__ == "__main__":
    main()
