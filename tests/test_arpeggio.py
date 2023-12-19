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
    @pytest.fixture(autouse=True, params=[*PDB_DOWNLOAD_SOURCES.keys()])
    def setup(self, request):
        pdb_source = request.param

        self.arpeggio = Arpeggio(
            out_dir=get_tmp_path("arpeggio", clear=True),
            interaction_cutoff=0.1,
            use_conda_env=CONDA_ENV_NAME,
            cache=True,
            pdb_source=pdb_source,
        )

    def test_no_chain(self):
        with pytest.raises(ValueError, match="chain"):
            self.arpeggio.contact_df("2WUR")

    def test_from_pdb(self):
        df_single = self.arpeggio.contact_df("2WUR:A", single_sided=True)
        df_double = self.arpeggio.contact_df("2WUR:A", single_sided=False)
        assert len(df_single) > 0
        assert len(df_single) * 2 == len(df_double)
