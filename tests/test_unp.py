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
        cls.TEMP_PATH = tests.utils.get_tmp_path('unp')

    def test_unp_record(self):
        test_id = 'P00720'
        unp_rec = unp.unp_record(test_id, self.TEMP_PATH)
        assert unp_rec.sequence_length == 164

    def test_unp_download(self):
        test_id = 'P00720'
        path = unp.unp_download(test_id, self.TEMP_PATH)
        assert path == self.TEMP_PATH.joinpath(f'{test_id}.txt')

    def test_unp_download_with_redirect(self):
        # This UNP id causes a redirect to
        test_id = 'P31217'
        replacement_id = unp.replacement_ids(test_id)[0]
        assert replacement_id != test_id

        path = unp.unp_download(test_id, self.TEMP_PATH)

        assert path == self.TEMP_PATH.joinpath(f'{replacement_id}.txt')

    def test_unp_download_with_invalid_id(self):
        with pytest.raises(IOError, match='404'):
            path = unp.unp_download('P000000', self.TEMP_PATH)
