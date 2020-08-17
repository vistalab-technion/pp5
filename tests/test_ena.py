import pytest

import pp5.external_dbs.ena as ena

import tests
import tests.utils

NO_INTERNET = not tests.utils.has_internet()


@pytest.mark.skipif(NO_INTERNET, reason="Needs internet")
class TestENA:
    def test_ena_seq(self):
        test_enaid = "CAA28212.1"
        seq = ena.ena_seq(test_enaid)
        assert len(seq) == 495

    def test_ena_non_existing(self):
        with pytest.raises(IOError):
            test_enaid = "ABC12345.1"
            seq = ena.ena_seq(test_enaid)
