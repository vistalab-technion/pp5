#!/usr/bin/env python
"""
Expand each adversarial breakdown-set sample into its per-PDB-structure angles, with
the aggregated (circular-mean) angle and the circular spread (circular std).

For each (unp_id, unp_idx) in the worst-sample tables we pull every contributing
structure from the raw data-precs.csv, list its individual (phi, psi), and report:
  - aggregated angle  : circular mean of phi, psi over the structures
                        (matches the pipeline 'cent' centroid used in the test)
  - spread            : circular std of phi, psi (Mardia: sqrt(-2 ln R)), degrees

Outputs:
  robustness_worst_angles_detail.csv   (one row per structure)
  robustness_worst_angles_summary.csv  (one row per breakdown-set entry)
"""

import os

import numpy as np
import pandas as pd

DP = "out/pnas-2026/dataset-raw/data-precs.csv"
WIDE_DEG = 15.0  # flag entries whose circular std exceeds this on either axis
OUTDIR = os.environ.get("WSA_OUTDIR", "out/pnas-2026-repro")
LABEL = os.environ.get("WSA_LABEL", "A")  # appendix letter
DESC = os.environ.get("WSA_DESC", "")  # extra title text
REMOVED = os.environ.get("WSA_REMOVED", "")  # removed_structures.csv to exclude
ARMS = {
    "KDE-L1": f"{OUTDIR}/robustness_worst_samples.csv",
    "torus_p": f"{OUTDIR}/robustness_torus_worst_samples.csv",
}


def circ_mean_deg(deg):
    r = np.deg2rad(np.asarray(deg, float))
    return float(np.rad2deg(np.arctan2(np.sin(r).mean(), np.cos(r).mean())))


def circ_std_deg(deg):
    r = np.deg2rad(np.asarray(deg, float))
    R = np.hypot(np.cos(r).mean(), np.sin(r).mean())
    R = min(max(R, 1e-12), 1.0)
    return float(np.rad2deg(np.sqrt(max(0.0, -2.0 * np.log(R))))) + 0.0


def main():
    dp = pd.read_csv(
        DP,
        usecols=["pdb_id", "unp_id", "res_id", "unp_idx", "phi", "psi"],
        low_memory=False,
    )
    dp = dp.dropna(subset=["unp_idx"])
    dp["unp_idx"] = dp["unp_idx"].astype(int)

    # optionally drop outlier structures so spreads/means reflect the cleaned aggregation
    removed_set = set()
    if REMOVED:
        rm = pd.read_csv(REMOVED)
        removed_set = set(
            zip(
                rm["unp_id"],
                rm["unp_idx"].astype(int),
                rm["pdb"].astype(str) + ":" + rm["chain"].astype(str),
            )
        )
        if removed_set:
            mask = [
                (u, i, p) in removed_set
                for u, i, p in zip(
                    dp["unp_id"], dp["unp_idx"], dp["pdb_id"].astype(str)
                )
            ]
            dp = dp[~np.array(mask)].reset_index(drop=True)
    g = dp.groupby(["unp_id", "unp_idx"])

    detail, summary = [], []
    for arm, path in ARMS.items():
        w = pd.read_csv(path)
        for _, row in w.iterrows():
            key = (row["unp_id"], int(row["unp_idx"]))
            try:
                sub = g.get_group(key).sort_values("pdb_id")
            except KeyError:
                continue
            phis, psis = sub["phi"].values, sub["psi"].values
            phbar, psbar = circ_mean_deg(phis), circ_mean_deg(psis)
            phsd, pssd = circ_std_deg(phis), circ_std_deg(psis)
            wide = max(phsd, pssd) > WIDE_DEG
            for _, s in sub.iterrows():
                pid = s["pdb_id"]
                chain = pid.split(":")[1] if ":" in str(pid) else ""
                detail.append(
                    dict(
                        arm=arm,
                        SS=row["SS"],
                        pair=row["pair"],
                        rank=int(row["rank"]),
                        codon=row["codon"],
                        unp_id=row["unp_id"],
                        unp_idx=int(row["unp_idx"]),
                        pdb_id=str(pid).split(":")[0],
                        chain=chain,
                        res_id=s["res_id"],
                        phi=round(float(s["phi"]), 1),
                        psi=round(float(s["psi"]), 1),
                        # aggregate stats repeated per row (full angles + the summary together)
                        n_pdb=len(sub),
                        phi_mean=round(phbar, 1),
                        phi_cstd=round(phsd, 1),
                        psi_mean=round(psbar, 1),
                        psi_cstd=round(pssd, 1),
                        wide_spread=wide,
                    )
                )
            summary.append(
                dict(
                    arm=arm,
                    SS=row["SS"],
                    pair=row["pair"],
                    rank=int(row["rank"]),
                    codon=row["codon"],
                    unp_id=row["unp_id"],
                    unp_idx=int(row["unp_idx"]),
                    n_pdb=len(sub),
                    phi_mean=round(phbar, 1),
                    phi_cstd=round(phsd, 1),
                    psi_mean=round(psbar, 1),
                    psi_cstd=round(pssd, 1),
                    wide_spread=wide,
                )
            )

    dd = pd.DataFrame(detail)
    ds = pd.DataFrame(summary)
    dd.to_csv(f"{OUTDIR}/robustness_worst_angles_detail.csv", index=False)
    ds.to_csv(f"{OUTDIR}/robustness_worst_angles_summary.csv", index=False)

    # markdown appendix: PDB structures + mean angle ± circular std (no per-structure
    # angles — those live in the detail CSV).
    # one markdown table per (SS, pair); Unicode symbols in the body font (no math).
    H = "| # | codon | position | φ ± σ (°) | ψ ± σ (°) | n | structures |"
    SEP = (
        "|--:|:"
        + "-" * 8
        + "|:"
        + "-" * 12
        + "|:"
        + "-" * 16
        + "|:"
        + "-" * 16
        + "|--:|:"
        + "-" * 32
        + "|"
    )
    desc = f" ({DESC})" if DESC else ""
    src = "outlier structures removed" if REMOVED else "matches the pipeline centroid"
    md = [
        f"## Appendix {LABEL} — breakdown-set samples: contributing structures and "
        f"aggregated angle{desc}\n",
        f"Aggregated angle = circular mean (φ ± σφ, ψ ± σψ) over the contributing "
        f"structures ({src}); σ is the circular standard deviation (Mardia), in "
        f"degrees. The **position** is marked WIDE when σ > {WIDE_DEG:.0f}° on either "
        f"axis. Per-structure angles are in `robustness_worst_angles_detail.csv`.\n",
    ]
    for arm in ARMS:
        md.append(f"\n### {arm} arm\n")
        sa = ds[ds.arm == arm]
        for (ss, pair), grp in sa.groupby(["SS", "pair"], sort=False):
            md.append(f"\n#### {ss}  {pair}\n")
            md.append(H)
            md.append(SEP)
            for _, r in grp.iterrows():
                pos = f"{r['unp_id']}:{r['unp_idx']}" + (
                    " **WIDE**" if r["wide_spread"] else ""
                )
                if (
                    r["n_pdb"] > 1
                ):  # only report spread when there is more than one structure
                    phi = f"{r['phi_mean']:+.1f} ± {r['phi_cstd']:.1f}"
                    psi = f"{r['psi_mean']:+.1f} ± {r['psi_cstd']:.1f}"
                else:
                    phi = f"{r['phi_mean']:+.1f}"
                    psi = f"{r['psi_mean']:+.1f}"
                di = dd[
                    (dd.arm == arm)
                    & (dd.pair == pair)
                    & (dd.unp_id == r["unp_id"])
                    & (dd.unp_idx == r["unp_idx"])
                ]
                pdbs = "; ".join(
                    f"{x.pdb_id}:{x.chain},{x.res_id}" for x in di.itertuples()
                )
                md.append(
                    f"| {r['rank']} | {r['codon']} | {pos} | {phi} | {psi} | "
                    f"{r['n_pdb']} | {pdbs} |"
                )
    with open(f"{OUTDIR}/robustness_worst_angles_appendix.md", "w") as f:
        f.write("\n".join(md) + "\n")

    # spread check
    wide = ds[ds.wide_spread]
    print(f"=== SPREAD CHECK ({OUTDIR}, circular std > {WIDE_DEG:.0f}° on phi/psi) ===")
    print(
        f"{len(wide)} of {len(ds)} breakdown-set entries flagged WIDE "
        f"({(ds.n_pdb > 1).sum()} are multi-structure):"
    )
    if len(wide):
        print(
            wide[
                [
                    "arm",
                    "SS",
                    "pair",
                    "codon",
                    "unp_id",
                    "unp_idx",
                    "n_pdb",
                    "phi_mean",
                    "phi_cstd",
                    "psi_mean",
                    "psi_cstd",
                ]
            ].to_string(index=False)
        )
    m = ds[ds.n_pdb > 1]
    print(
        f"\nmulti-structure entries: φ σ median={m.phi_cstd.median():.1f}° "
        f"max={m.phi_cstd.max():.1f}°; ψ σ median={m.psi_cstd.median():.1f}° "
        f"max={m.psi_cstd.max():.1f}°"
    )
    print(
        "\nsaved: robustness_worst_angles_detail.csv (full per-structure angles + mean±σ)"
    )
    print("saved: robustness_worst_angles_summary.csv")
    print("saved: robustness_worst_angles_appendix.md")


if __name__ == "__main__":
    main()
