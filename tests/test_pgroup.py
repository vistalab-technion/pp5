import pytest

from pp5.pgroup import ProteinGroup


class TestFromPDBRef(object):
    @pytest.fixture(autouse=True)
    def setup(self):
        pass

    def test_default(self):
        # Just an initial rudimentary test to check the code is alive
        pgroup = ProteinGroup.from_pdb_ref("1NKD:A", parallel=False)
        assert isinstance(pgroup, ProteinGroup)
        assert pgroup.num_query_structs >= 4
        assert len(pgroup.to_csv())
