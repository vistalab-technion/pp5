import pytest

import pp5.external_dbs.ena as ena

import tests
import tests.utils

NO_INTERNET = not tests.utils.has_internet()


@pytest.mark.skipif(NO_INTERNET, reason='Needs internet')
class TestENA:
    @classmethod
    def setup_class(cls):
        cls.test_enaid = 'CAA28212.1'
        cls.TEMP_PATH = tests.utils.get_tmp_path('ena')

    def test_unp_record(self):
        seq = ena.ena_seq(self.test_enaid, self.TEMP_PATH)
        assert len(seq) == 495
