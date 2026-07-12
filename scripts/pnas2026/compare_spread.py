#!/usr/bin/env python
"""
Compare the conformational spread (per-position circular std of phi, psi across
the contributing PDB structures) of the adversarial kill-set ("sensitive")
residues against the rest of the sample.

Background sets (multi-structure positions, n>=2, where spread is defined):
  - codon-matched : all positions whose codon is one of the kill-set codons
  - global        : all positions in the collected dataset
Spread metric per position: s_max = max(circ_std_phi, circ_std_psi), degrees.
"""
import numpy as np
import pandas as pd

DP = "out/prec-collected/20211001_124553-aida-ex_EC-src_EC/data-precs.csv"
SUMM = "out/pnas-2026-repro/robustness_worst_angles_summary.csv"
WIDE = 15.0


def per_position_cstd(dp):
    d = dp.copy()
    for a in ("phi", "psi"):
        d[a + "_c"] = np.cos(np.deg2rad(d[a]))
        d[a + "_s"] = np.sin(np.deg2rad(d[a]))
    g = d.groupby(["unp_id", "unp_idx"])
    out = pd.DataFrame({"n": g.size(), "codon": g["codon"].first()})
    for a in ("phi", "psi"):
        R = np.hypot(g[a + "_c"].mean(), g[a + "_s"].mean()).clip(1e-12, 1.0)
        out[a + "_cstd"] = np.rad2deg(np.sqrt(np.maximum(0.0, -2.0 * np.log(R))))
    out["s_max"] = out[["phi_cstd", "psi_cstd"]].max(axis=1)
    return out.reset_index()


def describe(label, s):
    s = np.asarray(s, float)
    q = np.percentile(s, [25, 50, 75, 90]) if len(s) else [np.nan] * 4
    print(
        f"{label:<24} n={len(s):>6}  median={q[1]:5.1f}  IQR=[{q[0]:.1f},{q[2]:.1f}]  "
        f"p90={q[3]:5.1f}  %>{WIDE:.0f}deg={100*np.mean(s > WIDE):4.1f}  "
        f"max={s.max() if len(s) else float('nan'):.1f}"
    )
    return q


def main():
    dp = pd.read_csv(
        DP, usecols=["unp_id", "unp_idx", "codon", "phi", "psi"], low_memory=False
    ).dropna(subset=["unp_idx", "phi", "psi"])
    dp["unp_idx"] = dp["unp_idx"].astype(int)
    pos = per_position_cstd(dp)
    pos["codon_aac"] = pos["codon"].astype(str)  # raw codon, e.g. 'CTC'

    summ = pd.read_csv(SUMM)
    ks = set(zip(summ["unp_id"], summ["unp_idx"].astype(int)))
    ks_codons = set(c.split("-")[1] for c in summ["codon"].unique())  # e.g. 'CTC'

    pos["is_ks"] = [(u, i) in ks for u, i in zip(pos["unp_id"], pos["unp_idx"])]
    multi = pos[pos["n"] >= 2]

    sens = multi[multi["is_ks"]]
    codon_rest = multi[(~multi["is_ks"]) & (multi["codon_aac"].isin(ks_codons))]
    global_rest = multi[~multi["is_ks"]]

    print(f"# kill-set codons: {sorted(ks_codons)}")
    print(
        f"# multi-structure positions total={len(multi)}, "
        f"kill-set(multi)={len(sens)}\n"
    )
    print("Spread metric = max(circular-std phi, circular-std psi), degrees\n")
    describe("SENSITIVE (kill-set)", sens["s_max"])
    describe("rest, codon-matched", codon_rest["s_max"])
    describe("rest, global", global_rest["s_max"])

    # Mann-Whitney: is sensitive spread different from codon-matched rest?
    try:
        from scipy.stats import mannwhitneyu

        u, p = mannwhitneyu(sens["s_max"], codon_rest["s_max"], alternative="two-sided")
        print(
            f"\nMann-Whitney sensitive vs codon-matched rest: p={p:.3g} "
            f"(median {sens['s_max'].median():.1f} vs {codon_rest['s_max'].median():.1f})"
        )
    except Exception as e:
        print(f"(MWU skipped: {e})")

    # per-axis too
    print("\nper-axis medians (deg):")
    for lab, grp in [
        ("SENSITIVE", sens),
        ("codon-rest", codon_rest),
        ("global-rest", global_rest),
    ]:
        print(
            f"  {lab:<12} phi_cstd={grp['phi_cstd'].median():.1f}  psi_cstd={grp['psi_cstd'].median():.1f}"
        )

    out = "out/pnas-2026-repro/spread_comparison.csv"
    rows = []
    for lab, grp in [
        ("sensitive_killset", sens),
        ("rest_codon_matched", codon_rest),
        ("rest_global", global_rest),
    ]:
        s = grp["s_max"]
        rows.append(
            dict(
                group=lab,
                n=len(s),
                median=round(s.median(), 1),
                q25=round(s.quantile(0.25), 1),
                q75=round(s.quantile(0.75), 1),
                p90=round(s.quantile(0.90), 1),
                frac_wide=round((s > WIDE).mean(), 3),
                max=round(s.max(), 1),
            )
        )
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()
