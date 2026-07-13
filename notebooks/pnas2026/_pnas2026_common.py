"""Shared p-value loading and BH-rejection helpers for the PNAS-2026 paper notebooks.

Both `w2torus-analysis.ipynb` and `robustness-figures.ipynb` import this module so they
can never disagree about which `cc-pvals.csv` files feed the analysis or what counts as
significant under a given statistic.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from pp5.stats.mht import mht_bh

BASE_PATH = Path(__file__).resolve().parent.parent.parent / "out" / "pnas-2026"
RESULTS_DIR = BASE_PATH / "results"
PVALS_FILENAME = "pvals/cc-pvals.csv"

# Full registry of runs: every statistic that has a sweep, x {real, control}.
LABELLED_TAGS: dict[str, str] = {
    # torustest, w2torus metric (no permutation test)
    "w2torus(2-random); none": "pointwise_cdist-t=1-bs=1-k=0-nmax=0-torus_p_nproj=2_randproj=True-cr=none-noself",
    "w2torus(2-random); aa_ss": "pointwise_cdist-t=1-bs=1-k=0-nmax=0-torus_p_nproj=2_randproj=True-cr=aa_ss-noself",
    "w2torus(4-fixed); none": "pointwise_cdist-t=1-bs=1-k=0-nmax=0-torus_p_nproj=4_randproj=False-cr=none-noself",
    "w2torus(4-fixed); aa_ss": "pointwise_cdist-t=1-bs=1-k=0-nmax=0-torus_p_nproj=4_randproj=False-cr=aa_ss-noself",
    # Permutation test with w2torus metric
    "w2torus-perm(2-random); none": "pointwise_cdist-t=1-bs=1-k=5000-nmax=0-torus_perm_nproj=2_randproj=True-cr=none-noself",
    "w2torus-perm(2-random); aa_ss": "pointwise_cdist-t=1-bs=1-k=5000-nmax=0-torus_perm_nproj=2_randproj=True-cr=aa_ss-noself",
    "w2torus-perm(4-fixed); none": "pointwise_cdist-t=1-bs=1-k=5000-nmax=0-torus_perm_nproj=4_randproj=False-cr=none-noself",
    "w2torus-perm(4-fixed); aa_ss": "pointwise_cdist-t=1-bs=1-k=5000-nmax=0-torus_perm_nproj=4_randproj=False-cr=aa_ss-noself",
    # Permutation test with KDE-L1 metric
    "kde-l1(bw=10.0); none": "pointwise_cdist-manual-t_1-bs_1-k_5000-nmax_0-kde_g_10.0-cr_none-noself",
    "kde-l1(bw=10.0); aa_ss": "pointwise_cdist-manual-t_1-bs_1-k_5000-nmax_0-kde_g_10.0-cr_aa_ss-noself",
    "kde-l1(bw=CV); none": "pointwise_cdist-t=1-bs=1-k=5000-nmax=0-kde_g_bw=cv-cr=none-noself",
    "kde-l1(bw=CV); aa_ss": "pointwise_cdist-t=1-bs=1-k=5000-nmax=0-kde_g_bw=cv-cr=aa_ss-noself",
    # Permutation test, MMD^2 (unbiased U-statistic and biased V-statistic)
    "mmd(unbiased); none": "pointwise_cdist-t=1-bs=1-k=5000-nmax=0-mmd_bw=10-cr=none-noself",
    "mmd(unbiased); aa_ss": "pointwise_cdist-t=1-bs=1-k=5000-nmax=0-mmd_bw=10-cr=aa_ss-noself",
    "mmd(biased); none": "pointwise_cdist-t=1-bs=1-k=5000-nmax=0-mmd_biased_bw=10-cr=none-noself",
    "mmd(biased); aa_ss": "pointwise_cdist-t=1-bs=1-k=5000-nmax=0-mmd_biased_bw=10-cr=aa_ss-noself",
}


def labelled_csv_paths(
    labelled_tags: dict[str, str] = LABELLED_TAGS,
) -> dict[str, Path]:
    """Resolve each display label to its `cc-pvals.csv` path under `RESULTS_DIR`.

    :param labelled_tags: Mapping of display label (e.g. `"mmd(unbiased); none"`) to
        result-folder tag.
    :return: Mapping of display label to the resolved `cc-pvals.csv` path.
    """
    return {
        label: RESULTS_DIR / tag / PVALS_FILENAME for label, tag in labelled_tags.items()
    }


def load_all_pvals(labelled_tags: dict[str, str] = LABELLED_TAGS) -> pd.DataFrame:
    """Load and concatenate every labelled `cc-pvals.csv` into one long-form table.

    Each label is expected to be `"<stat_test>; <codon_randomization>"` (e.g.
    `"mmd(unbiased); none"`); the parsed halves are stamped onto every row of that
    file's rows as the `stat_test` and `codon_randomization` columns.

    :param labelled_tags: Mapping of display label to result-folder tag.
    :return: Concatenated DataFrame with `cc-pvals.csv`'s own columns
        (`condition_group, subgroup1, subgroup2, n1, n2, pval, ddist, significant, ...`)
        plus `stat_test` and `codon_randomization`.
    """
    csv_paths = labelled_csv_paths(labelled_tags)
    for label, path in csv_paths.items():
        assert path.is_file(), f"Missing pvals CSV for {label!r}: {path}"

    frames = []
    for label, path in csv_paths.items():
        stat_test_name, randomization_type = (part.strip() for part in label.split(";"))
        df = pd.read_csv(path)
        df["stat_test"] = stat_test_name
        df["codon_randomization"] = randomization_type
        frames.append(df)
    return pd.concat(frames, axis=0, ignore_index=True)


def compute_bh_rejections(df_all_pvals: pd.DataFrame, fdr: float = 0.05) -> pd.DataFrame:
    """Compute each statistic's own self-consistent Benjamini-Hochberg threshold.

    For every (stat_test, codon_randomization, condition_group) group, pools that
    group's own p-values and derives its own BH(fdr) threshold via `mht_bh` -- never
    a threshold borrowed from a different statistic or a different secondary-structure
    class.

    :param df_all_pvals: Long-form pvals table with columns `pval`, `subgroup1`,
        `subgroup2`, `stat_test`, `codon_randomization`, `condition_group`.
    :param fdr: Desired false discovery rate.
    :return: DataFrame with one row per (stat_test, codon_randomization, SS): columns
        `stat_test, codon_randomization, SS, n_hypotheses, fdr, bh_pvalue_threshold,
        n_rejected, rejected_pairs` (`rejected_pairs` is a `list[str]` of `"c1:c2"`).
    """
    group_cols = ["stat_test", "codon_randomization", "condition_group"]
    rows = []
    for (stat_test, randomization, ss), df_group in df_all_pvals.groupby(
        group_cols, sort=False
    ):
        pvals = df_group["pval"].to_numpy()
        n_hypotheses = len(pvals)
        assert n_hypotheses >= 1, f"Empty p-value group: {(stat_test, randomization, ss)}"

        # BH(fdr) threshold computed from this group's own p-values only.
        p_thresh = mht_bh(q=fdr, pvals=pvals)
        rejected_mask = pvals <= p_thresh
        rejected = df_group[rejected_mask]
        rejected_pairs = [
            f"{c1}:{c2}" for c1, c2 in zip(rejected["subgroup1"], rejected["subgroup2"])
        ]
        rows.append(
            dict(
                stat_test=stat_test,
                codon_randomization=randomization,
                SS=ss,
                n_hypotheses=n_hypotheses,
                fdr=fdr,
                bh_pvalue_threshold=p_thresh,
                n_rejected=int(rejected_mask.sum()),
                rejected_pairs=rejected_pairs,
            )
        )
    return pd.DataFrame(rows)
