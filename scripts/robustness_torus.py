#!/usr/bin/env python
"""
Torus (projected-Wasserstein, 4-fixed-projection) arm of the outlier-robustness
analysis. Mirrors scripts/robustness_outliers.py but uses the `torus_p` statistic
that produced the w2torus rejections.

- Authoritative significance: the analytical `torus_p` test (no permutations) via
  the vendored R `torustest`, 4 fixed geodesics [1,0],[0,1],[1,1],[2,3], with the
  data-independent null simulated once (2000 sims, n=30) and reused.
- Adversarial breakdown: rank observations by their influence on a fast numpy
  circular-Wasserstein-1 proxy of the projected statistic (one-pass leave-one-out
  ordering), remove the most influential cumulatively, and recompute the
  authoritative torus_p p-value at each k. breakdown-k = smallest removal count
  that pushes the pair above the published BH threshold for its SS class.
  (The proxy is used only for ordering; every significance decision is the real
  torus_p p-value. One-pass ordering is a conservative upper bound on the minimal
  kill-set, as for the KDE-L1 arm.)
- Controls: AA+SS randomization (primary) and within-pair pooled shuffle
  (secondary), identical to the KDE-L1 arm.
"""
import os
import sys
from functools import partial

import numpy as np
import pandas as pd

OUTDIR = os.environ.get("PP5_ROBUST_OUTDIR", "out/pnas-2026-repro")

import rpy2.robjects as robjects
from pp5.stats.two_sample import (
    torus_projection_test_null_samples,
    R_TORUSTEST_GEODESIC,
    PY2R_CONVERTER,
)

# shared helpers from the KDE-L1 arm
from robustness_outliers import (
    angles, load_provenance, randomized_codon_column, k_grid_for, DS_PATH, DP_PATH,
)

GEODESICS = np.array([[1, 0], [0, 1], [1, 1], [2, 3]], dtype=float)  # 4 fixed
GLOW, GHIGH = -np.pi, np.pi
N_NULL_SIMS, N_NULL_SAMPLE = 2000, 30
N_REPLICATES = 30
SEED = 12345
GRIDN = 256

# Published BH thresholds for w2torus(4-fixed), Real data, per SS class.
BH_THRESH = {"HELIX": 0.0022988505747126436, "TURN": 0.0005747126436781609}

PAIRS = [
    ("HELIX", "L-CTC", "L-TTG"),
    ("HELIX", "L-CTC", "L-CTG"),
    ("HELIX", "L-CTC", "L-CTT"),
    ("HELIX", "R-AGG", "R-CGA"),
    ("TURN",  "A-GCG", "A-GCT"),
    ("TURN",  "P-CCC", "P-CCG"),
]

# ---- authoritative torus_p via R (cached null) ------------------------------
_SIM_NULL = None
_TEST_FN = None


def _torus_setup():
    global _SIM_NULL, _TEST_FN
    if _SIM_NULL is None:
        _SIM_NULL = torus_projection_test_null_samples(N_NULL_SIMS, N_NULL_SAMPLE, n_cores=4)
        _TEST_FN = robjects.globalenv[R_TORUSTEST_GEODESIC]


def _scale01(Z):
    return (Z - GLOW) / (GHIGH - GLOW)


def torus_pval(X, Y):
    """(w2_stat, pval) from the analytical torus_p test, 4 fixed projections."""
    _torus_setup()
    Xs, Ys = _scale01(X), _scale01(Y)
    with robjects.conversion.localconverter(PY2R_CONVERTER):
        res = _TEST_FN(
            sample_1=Xs, sample_2=Ys, n_geodesics=4, NC_geodesic=1,
            geodesic_list=GEODESICS, sim_null=_SIM_NULL, return_stat=True,
        )
        return res["stat"].item(), res["pval"].item()


# ---- fast circular-W1 proxy (for influence ordering only) -------------------
_GRID = (np.arange(GRIDN) + 0.5) / GRIDN


def _proj(Z01, g):
    return np.mod(g[0] * Z01[:, 0] + g[1] * Z01[:, 1], 1.0)


def proxy_stat(X01, Y01):
    tot = 0.0
    for g in GEODESICS:
        ua, ub = _proj(X01, g), _proj(Y01, g)
        ca = np.searchsorted(np.sort(ua), _GRID, side="right") / len(ua)
        cb = np.searchsorted(np.sort(ub), _GRID, side="right") / len(ub)
        d = ca - cb
        tot += np.mean(np.abs(d - np.median(d)))
    return tot / len(GEODESICS)


def influence_onepass(X01, Y01):
    """Per-point influence on the proxy stat (stat_full - stat_without_point).
    Returns (infl_x, infl_y), vectorised over points via count updates."""
    base = proxy_stat(X01, Y01)
    nA, nB = len(X01), len(Y01)
    infl_x = np.zeros(nA)
    infl_y = np.zeros(nB)
    for g in GEODESICS:
        ua, ub = _proj(X01, g), _proj(Y01, g)
        ca = np.searchsorted(np.sort(ua), _GRID, side="right").astype(float)
        cb = np.searchsorted(np.sort(ub), _GRID, side="right").astype(float)
        cb_n = cb / nB
        ca_n = ca / nA
        # remove one X point with value v: counts_a -> ca - (v<=grid); n -> nA-1
        leqx = (ua[:, None] <= _GRID[None, :]).astype(float)        # (nA, GRIDN)
        dX = (ca[None, :] - leqx) / (nA - 1) - cb_n[None, :]
        infl_x += base / len(GEODESICS) - np.mean(np.abs(dX - np.median(dX, axis=1, keepdims=True)), axis=1) / 1
        leqy = (ub[:, None] <= _GRID[None, :]).astype(float)        # (nB, GRIDN)
        dY = ca_n[None, :] - (cb[None, :] - leqy) / (nB - 1)
        infl_y += base / len(GEODESICS) - np.mean(np.abs(dY - np.median(dY, axis=1, keepdims=True)), axis=1) / 1
    # infl accumulated per-geodesic mean; base/len*4 == base, consistent with proxy_stat mean
    return infl_x, infl_y


def breakdown_torus(X, Y, thresh, k_grid):
    """One-pass influence ordering + authoritative torus_p p-value at each k."""
    X01, Y01 = _scale01(X), _scale01(Y)
    infl_x, infl_y = influence_onepass(X01, Y01)
    # global descending order over both groups
    order = sorted(
        [("X", i, infl_x[i]) for i in range(len(X))] +
        [("Y", j, infl_y[j]) for j in range(len(Y))],
        key=lambda t: -t[2],
    )
    rows, removed_log, breakdown_k = [], [], None
    xmask = np.ones(len(X), bool)
    ymask = np.ones(len(Y), bool)
    for k in range(0, max(k_grid) + 1):
        if k in k_grid:
            stat, pval = torus_pval(X[xmask], Y[ymask])
            sig = pval <= thresh
            rows.append(dict(k=k, p=pval, stat=stat,
                             n1=int(xmask.sum()), n2=int(ymask.sum()), significant=sig))
            if not sig and breakdown_k is None:
                breakdown_k = k
                break
        if k < len(order):
            side, idx, drop = order[k]
            (xmask if side == "X" else ymask)[idx] = False
            removed_log.append((side, idx, drop))
    return rows, breakdown_k, removed_log


def run_control(replicate_pairs, thresh, k_grid_fn):
    p0s, bks = [], []
    for Xr, Yr in replicate_pairs:
        if len(Xr) < 2 or len(Yr) < 2:
            continue
        _, pr = torus_pval(Xr, Yr)
        p0s.append(pr)
        if pr <= thresh:
            _, bkr, _ = breakdown_torus(Xr, Yr, thresh, k_grid_fn(min(len(Xr), len(Yr))))
            bks.append(bkr if bkr is not None else 0)
        else:
            bks.append(0)
    p0s = np.array(p0s)
    return dict(R=len(p0s), frac_sig=float((p0s <= thresh).mean()),
                bk_max=float(np.max(bks)), min_p0=float(p0s.min()))


def gen_aass(df, ss, c1, c2, R, base_seed):
    for r in range(R):
        dr = df.copy()
        dr["codon"] = randomized_codon_column(df, base_seed + r)
        Xr, _ = angles(dr, ss, c1)
        Yr, _ = angles(dr, ss, c2)
        yield Xr, Yr


def gen_pooled(X, Y, R, base_seed):
    Z = np.vstack([X, Y])
    nA, nB = len(X), len(Y)
    for r in range(R):
        rng = np.random.default_rng(base_seed + r)
        perm = rng.permutation(nA + nB)
        yield Z[perm[:nA]], Z[perm[nA:]]


def main():
    ds = sys.argv[1] if len(sys.argv) > 1 else DS_PATH
    df = pd.read_csv(ds)
    if "AA" not in df.columns:
        df["AA"] = df["codon"].str.split("-").str[0]
    print(f"# torus_p 4-fixed | dataset rows={len(df)} R={N_REPLICATES} seed={SEED}", flush=True)
    prov = load_provenance(DP_PATH)
    print(f"# provenance positions: {len(prov)}\n", flush=True)

    summary, worst_rows = [], []
    for pi, (ss, c1, c2) in enumerate(PAIRS):
        X, idx1 = angles(df, ss, c1)
        Y, idx2 = angles(df, ss, c2)
        n1, n2 = len(X), len(Y)
        thresh = BH_THRESH[ss]
        nmin = min(n1, n2)
        kg = k_grid_for(nmin)

        print(f"=== {ss}  {c1}:{c2}   n1={n1} n2={n2} ===", flush=True)
        stat0, p0 = torus_pval(X, Y)
        sig0 = p0 <= thresh
        print(f"  torus_p baseline: stat={stat0:.4f} p={p0:.5f} "
              f"(BH thresh={thresh:.5f}, sig={sig0})", flush=True)

        bk = None
        if sig0:
            rows, bk, rmlog = breakdown_torus(X, Y, thresh, kg)
            bk_txt = f"{bk}  [{bk/nmin*100:.1f}%]" if bk else f">{max(kg)} (never broke)"
            print(f"  adversarial breakdown-k (real) = {bk_txt}", flush=True)
            print("   p(k): " + "  ".join(
                f"{r['k']}={r['p']:.4f}{'*' if r['significant'] else ''}" for r in rows), flush=True)
            n_show = bk if bk else min(5, len(rmlog))
            print("   adversarially-worst samples (codon, unp_id:idx -> pdb):", flush=True)
            for rank, (side, i, drop) in enumerate(rmlog[:n_show], 1):
                codon = c1 if side == "X" else c2
                uid, uidx = (idx1 if side == "X" else idx2)[i]
                a = np.rad2deg((X if side == "X" else Y)[i])
                pdbs = prov.get((uid, int(uidx)), ["<no-pdb>"])
                print(f"     {rank}. {codon} {uid}:{uidx} phi={a[0]:.0f} psi={a[1]:.0f} "
                      f"[{'; '.join(pdbs[:6])}{' …' if len(pdbs) > 6 else ''}]", flush=True)
                worst_rows.append(dict(
                    SS=ss, pair=f"{c1}:{c2}", rank=rank, codon=codon, unp_id=uid,
                    unp_idx=int(uidx), phi=round(float(a[0]), 1), psi=round(float(a[1]), 1),
                    influence=round(float(drop), 6), n_pdb=len(pdbs),
                    pdb_provenance="; ".join(pdbs)))
        else:
            print("  (not significant under torus_p; no breakdown)", flush=True)

        sd = SEED + 7919 * pi
        ca = run_control(gen_aass(df, ss, c1, c2, N_REPLICATES, sd), thresh, k_grid_for)
        cp = run_control(gen_pooled(X, Y, N_REPLICATES, sd + 13), thresh, k_grid_for)
        print(f"  CONTROL AA+SS  (R={ca['R']}): frac_sig={ca['frac_sig']:.2f} "
              f"null bk max={ca['bk_max']:.0f} min_p0={ca['min_p0']:.4f}", flush=True)
        print(f"  CONTROL pooled (R={cp['R']}): frac_sig={cp['frac_sig']:.2f} "
              f"null bk max={cp['bk_max']:.0f} min_p0={cp['min_p0']:.4f}\n", flush=True)

        summary.append(dict(
            SS=ss, pair=f"{c1}:{c2}", n1=n1, n2=n2, torus_p=round(p0, 6),
            torus_sig=sig0, breakdown_k=bk, breakdown_pct=(bk / nmin * 100) if bk else None,
            aass_frac_sig=ca["frac_sig"], pooled_frac_sig=cp["frac_sig"],
            aass_min_p0=round(ca["min_p0"], 6), pooled_min_p0=round(cp["min_p0"], 6)))

    os.makedirs(OUTDIR, exist_ok=True)
    sdf = pd.DataFrame(summary)
    out = f"{OUTDIR}/robustness_torus_summary.csv"
    sdf.to_csv(out, index=False)
    if worst_rows:
        pd.DataFrame(worst_rows).to_csv(f"{OUTDIR}/robustness_torus_worst_samples.csv", index=False)
    print("=== TORUS SUMMARY ===")
    print(sdf.to_string(index=False))
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()
