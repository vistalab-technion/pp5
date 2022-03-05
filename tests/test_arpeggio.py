import os

import pytest

from tests import get_tmp_path
from pp5.align import Arpeggio

CONDA_ENV_NAME = "arpeggio"


@pytest.mark.skipif(
    not Arpeggio.can_execute(use_conda_env=CONDA_ENV_NAME), reason="no arpeggio"
)
class TestArpeggio:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.arpeggio = Arpeggio(
            out_dir=get_tmp_path("arpeggio", clear=True),
            interaction_cutoff=0.1,
            use_conda_env=CONDA_ENV_NAME,
            cache=True,
        )

    def test_no_chain(self):
        with pytest.raises(ValueError, match="chain"):
            self.arpeggio.contact_df("2WUR")

    def test_from_pdb(self):
        df = self.arpeggio.contact_df("2WUR:A")
        assert len(df) > 0
