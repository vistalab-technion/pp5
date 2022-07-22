import pytest

from pp5.pgroup import ProteinGroup


class TestFromPDBRef(object):
    @pytest.fixture(autouse=True)
    def setup(self):
        pass

    @pytest.mark.parametrize("match_len", [1, 2])
    def test_default(self, match_len):
        # Just an initial rudimentary test to check the code is alive
        pgroup = ProteinGroup.from_pdb_ref(
            "1NKD:A", parallel=False, sa_min_aligned_residues=40, match_len=match_len,
        )
        assert isinstance(pgroup, ProteinGroup)
        assert pgroup.num_query_structs >= 4
        assert len(pgroup.to_csv())
