#!/usr/bin/env python
"""
Outlier-robust re-aggregation of the pointwise dataset.

Within each aggregation group (unp_id, unp_idx, consolidated-SS) that has >=3
contributing PDB structures, a structure is an OUTLIER if its (phi, psi) lies more
than T degrees (flat-torus distance) from the group's robust centre (circular mean
after one trimming pass). Outliers are removed and the centroid recomputed from the
remaining structures. Applied uniformly to every position in the dataset.

Outputs:
  dataset_clean.csv         : original dataset.csv with phi/psi (and group_size/std)
                              recomputed on the outlier-free structures
  removed_structures.csv    : every removed PDB structure, with its angle and the
                              group consensus it disagreed with
"""
import sys
import numpy as np
import pandas as pd

T = 60.0  # flat-torus distance (deg) from group centre to call a structure an outlier
CONS = {"E": "SHEET", "H": "HELIX", "G": "OTHER", "I": "OTHER",
        "T": "TURN", "S": "OTHER", "B": "OTHER", "-": None, "": None}

DS = sys.argv[1] if len(sys.argv) > 1 else (
    "out/prec-collected/20211001_124553-aida-ex_EC-src_EC/results/"
    "pointwise_cdist-natcom/_intermediate_/dataset.csv")
DP = "out/prec-collected/20211001_124553-aida-ex_EC-src_EC/data-precs.csv"
OUT_DS = "out/pnas-2026-repro-clean/dataset_clean.csv"
OUT_RM = "out/pnas-2026-repro-clean/removed_structures.csv"


def cd(a, b):  # circular |difference| in degrees
    return np.abs(((a - b + 180) % 360) - 180)


def cmean_cols(df, by):
    r_phi, r_psi = np.deg2rad(df["phi"]), np.deg2rad(df["psi"])
    gg = df.assign(cphi=np.cos(r_phi), sphi=np.sin(r_phi),
                   cpsi=np.cos(r_psi), spsi=np.sin(r_psi)).groupby(by)
    m = gg[["cphi", "sphi", "cpsi", "spsi"]].transform("mean")
    return (np.rad2deg(np.arctan2(m["sphi"], m["cphi"])),
            np.rad2deg(np.arctan2(m["spsi"], m["cpsi"])))


def main():
    import os
    os.makedirs("out/pnas-2026-repro-clean", exist_ok=True)
    ds = pd.read_csv(DS)
    dp = pd.read_csv(DP, usecols=["pdb_id", "unp_id", "unp_idx", "secondary", "phi", "psi"],
                     low_memory=False).dropna(subset=["unp_idx", "phi", "psi"])
    dp["unp_idx"] = dp["unp_idx"].astype(int)
    dp["cg"] = dp["secondary"].map(lambda s: CONS.get(str(s), None))
    dp = dp.dropna(subset=["cg"]).reset_index(drop=True)
    BY = ["unp_id", "unp_idx", "cg"]
    dp["n"] = dp.groupby(BY)["phi"].transform("size")

    # pass 1: centre = circular mean of all
    c1phi, c1psi = cmean_cols(dp, BY)
    d1 = np.hypot(cd(dp["phi"], c1phi), cd(dp["psi"], c1psi))
    flag1 = (d1 > T) & (dp["n"] >= 3)

    # pass 2: recompute centre on pass-1 inliers, re-flag (robust to masking)
    inl = dp[~flag1]
    c2phi_i, c2psi_i = cmean_cols(inl, BY)
    c2 = pd.DataFrame({"cphi": c2phi_i, "cpsi": c2psi_i}, index=inl.index)
    # map the inlier-centre back to all rows of each group
    centre = inl.assign(cphi=c2["cphi"], cpsi=c2["cpsi"]).groupby(BY)[["cphi", "cpsi"]].first()
    cm = dp.merge(centre, on=BY, how="left")
    d2 = np.hypot(cd(dp["phi"], cm["cphi"].values), cd(dp["psi"], cm["cpsi"].values))
    # only groups that still have >=2 inliers after removal may drop outliers
    outlier = (d2 > T) & (dp["n"] >= 3)
    n_out = pd.Series(outlier.values, index=dp.index).groupby([dp[b] for b in BY]).transform("sum")
    keepable = (dp["n"] - n_out) >= 2
    outlier = outlier & keepable.values

    dp["centre_phi"] = cm["cphi"].values
    dp["centre_psi"] = cm["cpsi"].values
    dp["dist"] = d2
    dp["outlier"] = outlier.values

    removed = dp[dp["outlier"]].copy()
    removed["chain"] = removed["pdb_id"].str.split(":").str[1]
    removed["pdb"] = removed["pdb_id"].str.split(":").str[0]
    removed = removed.rename(columns={"cg": "SS"})[
        ["unp_id", "unp_idx", "SS", "pdb", "chain", "phi", "psi",
         "centre_phi", "centre_psi", "dist", "n"]].round(1)
    removed.to_csv(OUT_RM, index=False)

    # cleaned aggregation over non-outlier structures
    clean = dp[~dp["outlier"]]
    cphi, cpsi = cmean_cols(clean, BY)
    agg = clean.assign(phi_c=cphi, psi_c=cpsi).groupby(BY).agg(
        phi_clean=("phi_c", "first"), psi_clean=("psi_c", "first"),
        n_clean=("phi", "size")).reset_index()
    # circular std (max axis) on cleaned, for record
    rr = clean.assign(cp=np.cos(np.deg2rad(clean.phi)), spp=np.sin(np.deg2rad(clean.phi)),
                      cs=np.cos(np.deg2rad(clean.psi)), ss=np.sin(np.deg2rad(clean.psi))).groupby(BY)
    Rphi = np.hypot(rr.cp.mean(), rr.spp.mean()).clip(1e-12, 1)
    Rpsi = np.hypot(rr.cs.mean(), rr.ss.mean()).clip(1e-12, 1)
    std = pd.DataFrame({"std_clean": np.maximum(np.rad2deg(np.sqrt(-2 * np.log(Rphi))),
                                                np.rad2deg(np.sqrt(-2 * np.log(Rpsi))))}).reset_index()
    agg = agg.merge(std, on=BY)

    # write cleaned dataset = original rows with phi/psi/group_size/group_std replaced
    out = ds.merge(agg, left_on=["unp_id", "unp_idx", "condition_group"], right_on=BY, how="left")
    hit = out["phi_clean"].notna()
    out.loc[hit, "phi"] = out.loc[hit, "phi_clean"]
    out.loc[hit, "psi"] = out.loc[hit, "psi_clean"]
    out.loc[hit, "group_size"] = out.loc[hit, "n_clean"]
    out.loc[hit, "group_std"] = out.loc[hit, "std_clean"]
    out = out[ds.columns]
    out.to_csv(OUT_DS, index=False)

    npos_changed = int((agg["n_clean"] < dp.groupby(BY)["phi"].size().reset_index()["phi"].reindex(
        agg.set_index(BY).index).values).sum()) if False else len(removed.groupby(["unp_id", "unp_idx", "SS"]))
    print(f"T = {T} deg")
    print(f"structures removed: {len(removed)} "
          f"({100*len(removed)/len(dp):.3f}% of all per-structure rows)")
    print(f"positions affected: {removed.groupby(['unp_id','unp_idx','SS']).ngroups} "
          f"of {dp.groupby(BY).ngroups} aggregation groups")
    print(f"saved cleaned dataset: {OUT_DS}  (rows={len(out)})")
    print(f"saved removed list:    {OUT_RM}")
    # confirm the known case
    k = removed[(removed.unp_id == "P33590") & (removed.unp_idx == 421)]
    print("\nKnown case P33590:421 removed structures:")
    print(k.to_string(index=False) if len(k) else "  (none — check)")


if __name__ == "__main__":
    main()
