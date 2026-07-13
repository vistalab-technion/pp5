import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "notebooks" / "pnas2026"))
from _pnas2026_common import compute_bh_rejections, load_all_pvals  # noqa: E402


class TestComputeBhRejections:
    def _make_pvals(self, pvals: list[float], ss: str, stat_test: str, randomization: str):
        n = len(pvals)
        return pd.DataFrame(
            dict(
                condition_group=[ss] * n,
                subgroup1=[f"A-AAA{i}" for i in range(n)],
                subgroup2=[f"A-AAB{i}" for i in range(n)],
                pval=pvals,
                stat_test=[stat_test] * n,
                codon_randomization=[randomization] * n,
            )
        )

    def test_rejects_only_pvals_at_or_below_own_threshold(self):
        # 10 hypotheses, FDR=0.05: BH threshold rejects the 2 smallest pvals here.
        pvals = [0.001, 0.002, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
        df = self._make_pvals(pvals, "HELIX", "stat_a", "none")
        df_rej = compute_bh_rejections(df, fdr=0.05)
        assert len(df_rej) == 1
        row = df_rej.iloc[0]
        assert row["n_hypotheses"] == 10
        assert row["n_rejected"] == 2
        assert set(row["rejected_pairs"]) == {"A-AAA0:A-AAB0", "A-AAA1:A-AAB1"}

    def test_thresholds_are_self_consistent_per_statistic(self):
        # Two statistics with different pval distributions in the same SS must each
        # get their own threshold -- one should reject, the other should not, even
        # though they share a condition_group.
        strict = self._make_pvals([0.001] + [0.9] * 9, "HELIX", "strict_stat", "none")
        lenient = self._make_pvals([0.04] + [0.9] * 9, "HELIX", "lenient_stat", "none")
        df = pd.concat([strict, lenient], ignore_index=True)
        df_rej = compute_bh_rejections(df, fdr=0.05).set_index("stat_test")
        assert df_rej.loc["strict_stat", "n_rejected"] == 1
        assert df_rej.loc["lenient_stat", "n_rejected"] == 0

    def test_groups_by_ss_and_randomization_independently(self):
        helix = self._make_pvals([0.001] + [0.9] * 9, "HELIX", "stat_a", "none")
        turn = self._make_pvals([0.9] * 10, "TURN", "stat_a", "none")
        control = self._make_pvals([0.9] * 10, "HELIX", "stat_a", "aa_ss")
        df = pd.concat([helix, turn, control], ignore_index=True)
        df_rej = compute_bh_rejections(df, fdr=0.05)
        assert len(df_rej) == 3  # (HELIX, none), (TURN, none), (HELIX, aa_ss)


class TestLoadAllPvals:
    def test_raises_on_missing_csv(self, tmp_path):
        with pytest.raises(AssertionError):
            load_all_pvals({"missing; none": "no-such-tag"})
