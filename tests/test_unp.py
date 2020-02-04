import pytest

import pp5.external_dbs.unp as unp
import tests
import tests.utils

TEMP_PATH = tests.TEST_RESOURCES_PATH.joinpath('unp/tmp')
NO_INTERNET = not tests.utils.has_internet()


@pytest.mark.skipif(NO_INTERNET, reason='Needs internet')
class TestUNP:
    @classmethod
    def setup_class(cls):
        cls.test_id = 'P00720'
        cls.TEMP_PATH = tests.utils.get_tmp_path('unp')

    def test_unp_record(self):
        unp_rec = unp.unp_record(self.test_id, self.TEMP_PATH)
        assert unp_rec.sequence_length == 164
