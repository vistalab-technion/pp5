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
