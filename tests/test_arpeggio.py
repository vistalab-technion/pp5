import os

import pytest

from tests import get_tmp_path
from pp5.arpeggio import Arpeggio
from pp5.external_dbs.pdb import PDB_DOWNLOAD_SOURCES

CONDA_ENV_NAME = "arpeggio"


@pytest.mark.skipif(
    not Arpeggio.can_execute(use_conda_env=CONDA_ENV_NAME), reason="no arpeggio"
)
class TestArpeggio:
    @pytest.fixture(autouse=False, scope="class", params=[*PDB_DOWNLOAD_SOURCES.keys()])
    def pdb_source(self, request):
        return request.param

    @pytest.fixture(autouse=False, scope="class", params=["2WUR:A"])
    def pdb_id(self, request):
        return request.param

    @pytest.fixture(autouse=False, scope="class")
    def arpeggio(self, pdb_source, pdb_id) -> Arpeggio:
        return Arpeggio(
            out_dir=get_tmp_path("arpeggio", clear=True),
            interaction_cutoff=0.1,
            use_conda_env=CONDA_ENV_NAME,
            cache=True,
            pdb_source=pdb_source,
        )

    @pytest.fixture(autouse=False, scope="class")
    def pre_compute(self, arpeggio, pdb_id):
        # pre-compute at class level
        _ = arpeggio.contacts_df(pdb_id)

    def test_no_chain(self, arpeggio):
        with pytest.raises(ValueError, match="chain"):
            arpeggio.contacts_df("2WUR")

    def test_contacts_df(self, arpeggio, pdb_id, pre_compute):
        df_single = arpeggio.contacts_df(pdb_id, single_sided=True)
        df_double = arpeggio.contacts_df(pdb_id, single_sided=False)
        assert len(df_single) > 0
        assert len(df_single) * 2 == len(df_double)

    def test_residue_contacts(self, arpeggio, pdb_id, pre_compute):
        df = arpeggio.residue_contacts_df(pdb_id)
        assert len(df) > 0
        assert df["contact_count"].sum() > 0
