#!/usr/bin/env python
"""
Outlier-robustness analysis for the PNAS-2026 brief-report significant pairs.

For each codon pair FDR-rejected in the published runs we ask, without deleting
data arbitrarily:

  (1) Adversarial breakdown-k (fragility): greedily remove the observations most
      INFLUENTIAL on the KDE-L1 statistic (re-ranked each step), recomputing the
      permutation p-value, and report the smallest number of removals that pushes
      the pair above the published BH threshold for its SS class. This is a greedy
      UPPER BOUND on the minimal breakdown-set: a smarter attacker could need fewer.
      Each removed point's (phi,psi) + source is logged so we can see whether the
      breakdown deletes genuine Ramachandran outliers or ordinary bulk points.

  (2) Bounded-influence statistic (no deletion): re-test the full samples with a
      standard Gaussian-RBF MMD on the flat torus (saturating kernel -> each point
      contributes O(1/n)). Survival under MMD = the rejection is not a tail artifact.

  (3) Control calibration (AA+SS randomization, the paper's null): for each real
      pair, R replicates that shuffle codon labels within (AA, SS) -- preserving
      every residue's (phi,psi) and exact codon counts, hence the SAME outliers --
      and run the IDENTICAL adversarial breakdown on the matched pair. Reports the
      null fraction significant at baseline and the null breakdown-k distribution,
      so K_real becomes interpretable relative to the null.

Faithful to the published kde_g fixed-bandwidth=10deg run:
  samples = deg2rad(phi,psi); n_bins=128; grid=[-pi,pi); float64; sigma=10deg.
KDE-L1 ddist validated to ~1e-9 and permutation p to MC error vs the repo's
kde2d_test_pergroup; the permutation here is vectorized (selection-matrix x slabs)
for speed but uses the identical statistic and (count+1)/(k+1) p-value convention.
"""

import os
import sys
from functools import partial

import numpy as np
import pandas as pd

OUTDIR = os.environ.get("PP5_ROBUST_OUTDIR", "out/pnas-2026-repro")

from pp5.stats.breakdown import breakdown_k_kde
from pp5.stats.two_sample import (
    kde_l1_permutation_test_from_slabs,
    mmd_statistic,
    two_sample_kernel_permutation_test_inner,
)
from pp5.distributions.kde import kde_2d, torus_gaussian_kernel_2d

sys.path.insert(0, "scripts/pnas2026")
from _common import PAIRS, SEED

# ---- settings matching the published kde_g bw=10 run -------------------------
BW_DEG = 10.0
SIGMA_RAD = np.deg2rad(BW_DEG)
NBINS, GLOW, GHIGH, DT = 128, -np.pi, np.pi, np.float64
K_REAL = 5000  # permutations for real-pair p-values
K_CTRL = 2000  # permutations for control baselines (resolves the BH thresh)
N_REPLICATES = 30  # AA+SS control replicates per pair

# Published BH p-value thresholds (kde-l1 bw=10, Real data) per SS class.
BH_THRESH = {"HELIX": 0.0011494252873563218, "TURN": 0.0011494252873563218}

DS_PATH = (
    sys.argv[1] if len(sys.argv) > 1 else "out/pnas-2026/dataset-processed/dataset.csv"
)
# Raw per-structure file with PDB provenance (pdb_id includes chain; res_id is the
# PDB residue number). Aggregated dataset positions map (unp_id, unp_idx) -> 1+ rows.
DP_PATH = "out/pnas-2026/dataset-raw/data-precs.csv"


def load_provenance(path):
    """(unp_id, unp_idx) -> list of 'pdb_id,res_id' strings (pdb_id = 'PDBID:CHAIN')."""
    dp = pd.read_csv(path, usecols=["pdb_id", "unp_id", "res_id", "unp_idx"])
    dp = dp.dropna(subset=["unp_idx"])
    prov = {}
    for pid, uid, rid, uidx in zip(dp.pdb_id, dp.unp_id, dp.res_id, dp.unp_idx):
        prov.setdefault((uid, int(uidx)), []).append(f"{pid},{rid}")
    return prov


# ---- core KDE-L1 machinery ---------------------------------------------------
def angles(df, ss, codon):
    sub = df[(df["condition_group"] == ss) & (df["codon"] == codon)]
    return np.deg2rad(sub[["phi", "psi"]].values), sub[["unp_id", "unp_idx"]].values


def slab_rows_for(A):
    """(n, P=NBINS*NBINS) per-point KDE slab rows (fixed bandwidth)."""
    s = kde_2d(
        x1=A[:, 0],
        x2=A[:, 1],
        kernel_fn=partial(torus_gaussian_kernel_2d, sigma=SIGMA_RAD),
        n_bins=NBINS,
        grid_low=GLOW,
        grid_high=GHIGH,
        dtype=DT,
        reduce=False,
    )
    return s.transpose(2, 0, 1).reshape(s.shape[2], -1)  # (n, P)


def _l1(xsum, T):
    a = xsum / xsum.sum(-1, keepdims=True)
    ysum = T - xsum
    b = ysum / ysum.sum(-1, keepdims=True)
    return np.abs(a - b).sum(-1)


def perm_pval(slab_rows, nx, ny, k, seed=SEED, batch=1000):
    """Vectorized KDE-L1 permutation p-value (selection-matrix x slab stack).

    Identical statistic and p-value convention as kde2d_test_pergroup_fast;
    delegates to the library's shared implementation, standardized on the
    X=idx[:nx] permutation convention used throughout pp5.stats.two_sample
    (not a "sum the smaller group" shortcut, which would silently disagree
    with that convention whenever nx > ny).
    """
    slab_rows = np.ascontiguousarray(slab_rows, dtype=DT)
    ddist, pval, _ = kde_l1_permutation_test_from_slabs(
        slab_rows,
        slab_rows,
        nx,
        ny,
        k,
        k_min=k,
        k_th=float("inf"),
        rng=np.random.default_rng(seed),
        batch_size=batch,
    )
    return ddist, pval


def greedy_breakdown(X, Y, thresh, k_grid, k_perm):
    """Adversarially remove most-influential points (re-ranked), recompute p at
    each k in k_grid, via the library's generalized KDE-L1 breakdown wrapper
    (equal-bandwidth case: sigma_x_rad == sigma_y_rad == SIGMA_RAD reduces to
    this arm's fixed-bandwidth setup, sharing slabs internally). Returns
    (rows, breakdown_k, removed_log)."""
    return breakdown_k_kde(
        X,
        Y,
        n_bins=NBINS,
        grid_low=GLOW,
        grid_high=GHIGH,
        dtype=DT,
        sigma_x_rad=SIGMA_RAD,
        sigma_y_rad=SIGMA_RAD,
        thresh=thresh,
        k_grid=k_grid,
        k_perm=k_perm,
        k_min=k_perm,
        k_th=float("inf"),
        seed=SEED,
    )


# ---- bounded-influence MMD cross-check --------------------------------------
def mmd_pval(X, Y, k, seed=SEED):
    np.random.seed(seed)
    Z = np.vstack((X, Y))
    diff = np.abs(Z[:, None, :] - Z[None, :, :])
    diff = np.minimum(diff, 2 * np.pi - diff)
    D2 = np.sum(diff**2, axis=2)
    Kgram = np.exp(-D2 / (2.0 * SIGMA_RAD**2))  # standard RBF, bounded [0,1]
    stat, pval, _ = two_sample_kernel_permutation_test_inner(
        Kgram,
        len(X),
        len(Y),
        k,
        mmd_statistic,
        permute_pairs=True,
        k_min=k,
        k_th=float("inf"),
    )
    return stat, pval


# ---- AA+SS randomized control (paper's null) --------------------------------
def randomized_codon_column(df, seed):
    """Shuffle codon labels within each (AA, condition_group) group -> a control
    replicate. Preserves every (phi,psi) and exact per-(AA,SS) codon counts."""
    rng = np.random.default_rng(seed)
    out = df["codon"].to_numpy().copy()
    for _, idx in df.groupby(["AA", "condition_group"]).indices.items():
        out[idx] = out[idx][rng.permutation(len(idx))]
    return out


def gen_aass(df, ss, c1, c2, R, base_seed):
    """AA+SS control: shuffle codons within (AA, SS), yield the matched pair."""
    for r in range(R):
        dr = df.copy()
        dr["codon"] = randomized_codon_column(df, base_seed + r)
        Xr, _ = angles(dr, ss, c1)
        Yr, _ = angles(dr, ss, c2)
        yield Xr, Yr


def gen_pooled(X, Y, R, base_seed):
    """Within-pair pooled shuffle: pool ONLY these two codons' points and re-split
    at the same (nA, nB) sizes. Exact same point cloud + outliers, labels randomized."""
    Z = np.vstack([X, Y])
    nA, nB = len(X), len(Y)
    for r in range(R):
        rng = np.random.default_rng(base_seed + r)
        perm = rng.permutation(nA + nB)
        yield Z[perm[:nA]], Z[perm[nA:]]


def run_control(replicate_pairs, thresh, k_perm):
    """Baseline p + adversarial breakdown over control replicates. Non-significant
    replicates have breakdown-k = 0 (already dead)."""
    p0s, bks = [], []
    for Xr, Yr in replicate_pairs:
        if len(Xr) < 2 or len(Yr) < 2:
            continue
        xr, yr = slab_rows_for(Xr), slab_rows_for(Yr)
        _, pr = perm_pval(np.vstack([xr, yr]), len(Xr), len(Yr), k_perm)
        p0s.append(pr)
        if pr <= thresh:
            g = k_grid_for(min(len(Xr), len(Yr)))
            _, bkr, _ = greedy_breakdown(Xr, Yr, thresh, g, k_perm)
            bks.append(bkr if bkr is not None else g[-1])
        else:
            bks.append(0)
    p0s = np.array(p0s)
    return dict(
        R=len(p0s),
        frac_sig=float((p0s <= thresh).mean()),
        bk_median=float(np.median(bks)),
        bk_max=float(np.max(bks)),
        min_p0=float(p0s.min()),
    )


def k_grid_for(nmin):
    kcap = min(40, max(5, int(np.ceil(0.05 * nmin))))
    g = sorted(set([0, 1, 2, 3, 5, 8, 12, 20, kcap]))
    return [k for k in g if k <= nmin - 2]


def main():
    print(f"# dataset: {DS_PATH}", flush=True)
    df = pd.read_csv(DS_PATH)
    if "AA" not in df.columns:
        df["AA"] = df["codon"].str.split("-").str[0]
    print(
        f"# rows={len(df)} bw={BW_DEG} K_real={K_REAL} K_ctrl={K_CTRL} "
        f"R={N_REPLICATES} seed={SEED}",
        flush=True,
    )
    prov = load_provenance(DP_PATH)
    print(
        f"# provenance: {len(prov)} (unp_id,unp_idx) positions from {DP_PATH}\n",
        flush=True,
    )

    summary, worst_rows = [], []
    for pi, (ss, c1, c2) in enumerate(PAIRS):
        X, idx1 = angles(df, ss, c1)
        Y, idx2 = angles(df, ss, c2)
        n1, n2 = len(X), len(Y)
        thresh = BH_THRESH[ss]
        nmin = min(n1, n2)
        kg = k_grid_for(nmin)

        print(f"=== {ss}  {c1}:{c2}   n1={n1} n2={n2} ===", flush=True)
        xs, ys = slab_rows_for(X), slab_rows_for(Y)

        d0, p0 = perm_pval(np.vstack([xs, ys]), n1, n2, K_REAL)
        print(
            f"  KDE-L1 baseline: ddist={d0:.4f} p={p0:.5f} "
            f"(BH thresh={thresh:.5f}, sig={p0 <= thresh})",
            flush=True,
        )

        rows, bk, rmlog = greedy_breakdown(X, Y, thresh, kg, K_REAL)
        bk_txt = (
            f"{bk}  [{bk/nmin*100:.1f}% of smaller group]"
            if bk
            else f">{max(kg)} (never broke)"
        )
        print(f"  adversarial breakdown-k (real) = {bk_txt}", flush=True)
        print(
            "   p(k): "
            + "  ".join(
                f"{r['k']}={r['p']:.4f}{'*' if r['significant'] else ''}" for r in rows
            ),
            flush=True,
        )
        # The ordered breakdown-set: the adversarially-worst samples (the minimal set
        # whose removal breaks significance, or top-5 if it never broke).
        n_show = bk if bk else min(5, len(rmlog))
        print(
            f"   adversarially-worst samples (codon, unp_id:unp_idx -> pdb_id:chain,res_id):",
            flush=True,
        )
        for rank, (side, i, drop) in enumerate(rmlog[:n_show], 1):
            codon = c1 if side == "X" else c2
            uid, uidx = (idx1 if side == "X" else idx2)[i]
            a = np.rad2deg((X if side == "X" else Y)[i])
            pdbs = prov.get((uid, int(uidx)), ["<no-pdb>"])
            print(
                f"     {rank}. {codon}  {uid}:{uidx}  phi={a[0]:.0f} psi={a[1]:.0f} "
                f"drop={drop:+.4f}  [{'; '.join(pdbs)}]",
                flush=True,
            )
            worst_rows.append(
                dict(
                    SS=ss,
                    pair=f"{c1}:{c2}",
                    rank=rank,
                    codon=codon,
                    unp_id=uid,
                    unp_idx=int(uidx),
                    phi=round(float(a[0]), 1),
                    psi=round(float(a[1]), 1),
                    influence_drop=round(float(drop), 5),
                    n_pdb=len(pdbs),
                    pdb_provenance="; ".join(pdbs),
                )
            )

        ms, mp = mmd_pval(X, Y, K_REAL)
        print(
            f"  MMD baseline (RBF, bounded): stat={ms:.3e} p={mp:.5f} "
            f"(sig={mp <= thresh})",
            flush=True,
        )

        # ---- controls: AA+SS (primary) and within-pair pooled shuffle (secondary)
        sd = SEED + 7919 * pi
        ca = run_control(gen_aass(df, ss, c1, c2, N_REPLICATES, sd), thresh, K_CTRL)
        cp = run_control(gen_pooled(X, Y, N_REPLICATES, sd + 13), thresh, K_CTRL)
        print(
            f"  CONTROL AA+SS    (R={ca['R']}): frac_sig={ca['frac_sig']:.2f} "
            f"null bk med={ca['bk_median']:.0f} max={ca['bk_max']:.0f} "
            f"min_p0={ca['min_p0']:.4f}",
            flush=True,
        )
        print(
            f"  CONTROL pooled   (R={cp['R']}): frac_sig={cp['frac_sig']:.2f} "
            f"null bk med={cp['bk_median']:.0f} max={cp['bk_max']:.0f} "
            f"min_p0={cp['min_p0']:.4f}",
            flush=True,
        )
        print()

        summary.append(
            dict(
                SS=ss,
                pair=f"{c1}:{c2}",
                n1=n1,
                n2=n2,
                kde_l1_p=round(p0, 6),
                mmd_p=round(mp, 6),
                breakdown_k_real=bk,
                breakdown_pct=(bk / nmin * 100) if bk else None,
                aass_frac_sig=ca["frac_sig"],
                aass_bk_max=ca["bk_max"],
                aass_min_p0=round(ca["min_p0"], 6),
                pooled_frac_sig=cp["frac_sig"],
                pooled_bk_max=cp["bk_max"],
                pooled_min_p0=round(cp["min_p0"], 6),
            )
        )

    os.makedirs(OUTDIR, exist_ok=True)
    sdf = pd.DataFrame(summary)
    out = f"{OUTDIR}/robustness_outliers_summary.csv"
    sdf.to_csv(out, index=False)
    wdf = pd.DataFrame(worst_rows)
    wout = f"{OUTDIR}/robustness_worst_samples.csv"
    wdf.to_csv(wout, index=False)
    print("=== SUMMARY ===")
    print(sdf.to_string(index=False))
    print("\n=== ADVERSARIALLY-WORST SAMPLES (breakdown-sets) ===")
    print(wdf.to_string(index=False))
    print(f"\nsaved: {out}\nsaved: {wout}")


if __name__ == "__main__":
    main()
