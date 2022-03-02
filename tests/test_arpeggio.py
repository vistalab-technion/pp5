import os

import pytest

from pp5.align import Arpeggio

CONDA_ENV_NAME = "arpeggio"


@pytest.mark.skipif(
    not Arpeggio.can_execute(use_conda_env=CONDA_ENV_NAME), reason="no arpeggio"
)
class TestArpeggio:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.arpeggio = Arpeggio(interaction_cutoff=0.1, use_conda_env=CONDA_ENV_NAME,)

    def test_no_chain(self):
        with pytest.raises(ValueError, match="chain"):
            self.arpeggio.pdb("2WUR")

    def test_from_pdb(self):
        df = self.arpeggio.pdb("2WUR:A")
        assert len(df) > 0
