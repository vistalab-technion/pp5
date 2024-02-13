import pytest

from pp5.pgroup import ProteinGroup
from pp5.external_dbs.pdb import PDB_AFLD, PDB_RCSB, PDB_REDO


class TestFromPDBRef(object):
    @pytest.mark.parametrize("pdb_source", [PDB_RCSB, PDB_REDO])
    @pytest.mark.parametrize("match_len", [2, 1])
    def test_default(self, match_len, pdb_source):
        pgroup = ProteinGroup.from_pdb_ref(
            "1NKD:A",
            pdb_source=pdb_source,
            parallel=False,
            sa_min_aligned_residues=40,
            match_len=match_len,
            compare_contacts=True,
        )
        assert isinstance(pgroup, ProteinGroup)
        assert pgroup.num_query_structs >= 4
        assert len(pgroup.to_csv())
