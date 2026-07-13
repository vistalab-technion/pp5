import multiprocessing as mp

import numpy as np
import pandas as pd
import pytest

from pp5.analysis.pointwise import (
    AGGREGATION_TYPE_CENTROID,
    PHI_COL,
    PSI_COL,
    PointwiseCodonDistanceAnalyzer,
    _subgroup_outlier_mask,
)


def _subgroup_df(phi_deg, psi_deg) -> pd.DataFrame:
    return pd.DataFrame({PHI_COL: phi_deg, PSI_COL: psi_deg})


class TestSubgroupOutlierMask:
    def test_below_min_group_size_keeps_all(self):
        # n=2 is below the 3-structure minimum for outlier removal, even though
        # the two points are far apart on the torus.
        df = _subgroup_df([0.0, 170.0], [0.0, 170.0])
        mask = _subgroup_outlier_mask(df)
        assert mask.tolist() == [True, True]

    def test_removes_clear_outlier(self):
        # 3 identical points plus one far outlier (170 deg away): the outlier
        # should be excluded, matching Alex's clean_aggregation.py T=60deg rule.
        df = _subgroup_df([0.0, 0.0, 0.0, 170.0], [0.0, 0.0, 0.0, 170.0])
        mask = _subgroup_outlier_mask(df)
        assert mask.tolist() == [True, True, True, False]

    def test_safety_net_cancels_removal_below_min_keep(self):
        # 3 points, symmetric-ish: the naive pass-1 mean sits near the single
        # point at 0deg, flagging the other two (100, -100) as outliers. But
        # removing both would leave only 1 structure (< min_keep=2), so the
        # safety net cancels removal entirely for this group.
        df = _subgroup_df([0.0, 100.0, -100.0], [0.0, 0.0, 0.0])
        mask = _subgroup_outlier_mask(df)
        assert mask.tolist() == [True, True, True]

    def test_two_pass_requalifies_point_flagged_in_pass_one(self):
        # Verified numerically: pass-1 mean (over all 5 points) is pulled far
        # enough toward the two 150deg outliers that it flags the two 0deg
        # points too (dist ~66.6deg > 60). Pass-2 recomputes the centre from
        # the single pass-1 inlier (the 55deg point) alone, which lands close
        # enough to the 0deg points (dist=58deg < 60) that they "re-qualify"
        # and are correctly kept in the final result. The 150deg points remain
        # outliers in both passes.
        df = _subgroup_df([0.0, 0.0, 55.0, 150.0, 150.0], [0.0, 0.0, 0.0, 0.0, 0.0])
        mask = _subgroup_outlier_mask(df)
        assert mask.tolist() == [True, True, True, False, False]


class TestStrictAggregationWiring:
    """
    Exercises strict_aggregation end-to-end through
    PointwiseCodonDistanceAnalyzer._preprocess_dataset, using a small synthetic
    dataset (no dependency on any real, gitignored data files) so this test is
    portable and runs anywhere.
    """

    @pytest.fixture
    def dataset_dir(self, tmp_path):
        # Group P00001:10 (codon AAA=Lys, secondary H=HELIX): 4 structures,
        # 3 identical inliers + 1 clear outlier (170,170), >=3 structures so
        # outlier removal is eligible under strict aggregation.
        # Group P00002:20: only 2 structures (below the size-3 threshold), so
        # it should be identical in both modes regardless of the flag.
        rows = [
            ("1AAA:A", "P00001", 10, "AAA", 1.00, "H", -60.0, -45.0),
            ("1BBB:A", "P00001", 10, "AAA", 1.00, "H", -60.0, -45.0),
            ("1CCC:A", "P00001", 10, "AAA", 1.00, "H", -60.0, -45.0),
            ("1DDD:A", "P00001", 10, "AAA", 1.00, "H", 170.0, 170.0),
            ("1EEE:A", "P00002", 20, "AAA", 1.00, "H", -60.0, -45.0),
            ("1FFF:A", "P00002", 20, "AAA", 1.00, "H", 170.0, 170.0),
        ]
        df = pd.DataFrame(
            rows,
            columns=[
                "pdb_id",
                "unp_id",
                "unp_idx",
                "codon",
                "codon_score",
                "secondary",
                "phi",
                "psi",
            ],
        )
        df.to_csv(tmp_path / "data-precs.csv", index=False)
        return tmp_path

    def _preprocessed(self, dataset_dir, strict_aggregation: bool) -> pd.DataFrame:
        analyzer = PointwiseCodonDistanceAnalyzer(
            dataset_dir=dataset_dir,
            out_dir=dataset_dir / f"out-{strict_aggregation}",
            aggregation_type=AGGREGATION_TYPE_CENTROID,
            aggregation_min_group_size=1,
            ignore_omega=True,
            strict_aggregation=strict_aggregation,
        )
        with mp.Pool(processes=1) as pool:
            analyzer._preprocess_dataset(pool)
        return analyzer._load_intermediate("dataset")

    def test_flag_off_keeps_all_structures(self, dataset_dir):
        df = self._preprocessed(dataset_dir, strict_aggregation=False)
        row = df[(df.unp_id == "P00001") & (df.unp_idx == 10)].iloc[0]
        assert row.group_size == 4

    def test_flag_on_excludes_outlier_structure(self, dataset_dir):
        df = self._preprocessed(dataset_dir, strict_aggregation=True)
        row = df[(df.unp_id == "P00001") & (df.unp_idx == 10)].iloc[0]
        assert row.group_size == 3
        # Dihedral.circular_centroid computes in float32 internally, so allow
        # for that precision (not related to the strict-aggregation logic).
        assert row.phi == pytest.approx(-60.0, abs=1e-4)
        assert row.psi == pytest.approx(-45.0, abs=1e-4)
        assert row.group_std == pytest.approx(0.0, abs=1e-4)

    def test_small_group_unaffected_by_flag(self, dataset_dir):
        # Below the 3-structure minimum: identical output regardless of flag.
        df_off = self._preprocessed(dataset_dir, strict_aggregation=False)
        df_on = self._preprocessed(dataset_dir, strict_aggregation=True)
        row_off = df_off[(df_off.unp_id == "P00002") & (df_off.unp_idx == 20)].iloc[0]
        row_on = df_on[(df_on.unp_id == "P00002") & (df_on.unp_idx == 20)].iloc[0]
        assert row_off.group_size == row_on.group_size == 2
        assert row_off.phi == pytest.approx(row_on.phi)
        assert row_off.psi == pytest.approx(row_on.psi)

    def test_strict_aggregation_requires_centroid_aggregation(self, dataset_dir):
        with pytest.raises(ValueError):
            PointwiseCodonDistanceAnalyzer(
                dataset_dir=dataset_dir,
                aggregation_type="none",
                strict_aggregation=True,
            )
