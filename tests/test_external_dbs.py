import os

import pytest

import pp5.external_dbs as external_dbs
import tests.utils

TEMP_PATH = tests.TEST_RESOURCES_PATH.joinpath('external_dbs/tmp')
NO_INTERNET = not tests.utils.has_internet()


def clear_tmp_path():
    os.makedirs(TEMP_PATH, exist_ok=True)
    for f in TEMP_PATH.glob('*'):
        os.remove(f)


@pytest.mark.skipif(NO_INTERNET, reason='Needs internet')
class TestPDB:
    @classmethod
    def setup_class(cls):
        cls.test_pdbid = '102L'
        clear_tmp_path()

    @classmethod
    def teardown_class(cls):
        pass

    def setup(self):
        pass

    def test_pdb_download(self):
        path = external_dbs.pdb_download(self.test_pdbid, TEMP_PATH)
        expected_path = TEMP_PATH.joinpath(f'{self.test_pdbid.lower()}.cif')
        assert path == expected_path
        assert os.path.isfile(expected_path)

    def test_pdb_struct(self):
        struct = external_dbs.pdb_struct(self.test_pdbid, TEMP_PATH)
        chains = list(struct.get_chains())
        assert len(chains) == 1

    def test_pdbid_to_unpids(self):
        ids = external_dbs.pdbid_to_unpids(self.test_pdbid, TEMP_PATH)
        assert len(ids) == 1
        assert ids[0] == 'P00720'

    def test_pdbid_to_unpids2(self):
        test_pdbid = '4HHB'
        ids = external_dbs.pdbid_to_unpids(test_pdbid, TEMP_PATH)
        assert len(ids) == 2
        assert ids == ['P69905', 'P68871']


@pytest.mark.skipif(NO_INTERNET, reason='Needs internet')
class TestUNP:
    @classmethod
    def setup_class(cls):
        cls.test_unpid = 'P00720'
        clear_tmp_path()

    def test_unp_record(self):
        unp_rec = external_dbs.unp_record(self.test_unpid, TEMP_PATH)
        assert unp_rec.sequence_length == 164


@pytest.mark.skipif(NO_INTERNET, reason='Needs internet')
class TestENA:
    @classmethod
    def setup_class(cls):
        cls.test_enaid = 'CAA28212.1'
        clear_tmp_path()

    def test_unp_record(self):
        seq = external_dbs.ena_seq(self.test_enaid, TEMP_PATH)
        assert len(seq) == 495
