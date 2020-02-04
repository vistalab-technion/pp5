import os

import pytest

import pp5.external_dbs.pdb as pdb
import tests
import tests.utils

NO_INTERNET = not tests.utils.has_internet()


@pytest.mark.skipif(NO_INTERNET, reason='Needs internet')
class TestPDB:
    @classmethod
    def setup_class(cls):
        cls.test_id = '102L'
        cls.TEMP_PATH = tests.utils.get_tmp_path('pdb')

    @classmethod
    def teardown_class(cls):
        pass

    def setup(self):
        pass

    def test_pdb_download(self):
        path = pdb.pdb_download(self.test_id, self.TEMP_PATH)
        expected_path = self.TEMP_PATH.joinpath(f'{self.test_id.lower()}.cif')
        assert path == expected_path
        assert os.path.isfile(expected_path)

    def test_pdb_struct(self):
        struct = pdb.pdb_struct(self.test_id, self.TEMP_PATH)
        chains = list(struct.get_chains())
        assert len(chains) == 1

    def test_pdbid_to_unpids(self):
        ids = pdb.pdbid_to_unpids(self.test_id, self.TEMP_PATH)
        assert len(ids) == 1
        assert ids[0] == 'P00720'

    def test_pdbid_to_unpids2(self):
        test_id = '4HHB'
        ids = pdb.pdbid_to_unpids(test_id, self.TEMP_PATH)
        assert len(ids) == 2
        assert ids == ['P69905', 'P68871']
