import pytest

from pp5.pgroup import ProteinGroup


class TestFromPDBRef(object):
    @pytest.fixture(autouse=True)
    def setup(self):
        pass

    @pytest.mark.parametrize("match_len", [2, 1])
    def test_default(self, match_len):
        pgroup = ProteinGroup.from_pdb_ref(
            "1NKD:A",
            parallel=False,
            sa_min_aligned_residues=40,
            match_len=match_len,
            compare_contacts=True,
        )
        assert isinstance(pgroup, ProteinGroup)
        assert pgroup.num_query_structs >= 4
        assert len(pgroup.to_csv())

    def test_specific_example(self):
        pgroup = ProteinGroup.from_pdb_ref(
            "4NE4:A",
            parallel=False,
            match_len=2,
            compare_contacts=True,
            resolution_cutoff=1.8,
            b_max=50,
            sa_outlier_cutoff=2.5,
            angle_aggregation="max_res",
            query_predicate=(lambda q: "5TEU" in q),
        )

        assert pgroup.num_matches >= 276
