import os
import random
import string

import pytest
from Bio.PDB.PDBExceptions import PDBConstructionException

import pp5.external_dbs.pdb as pdb
import tests
import tests.utils

NO_INTERNET = not tests.utils.has_internet()


class TestSplitID:
    @staticmethod
    def _random_pdb_id(with_chain=False):
        base_id = str.join('',
                           [str(random.randrange(10))] +
                           random.choices(string.ascii_letters, k=3))
        if not with_chain:
            return base_id

        return base_id + f':{random.choice(string.ascii_letters)}'

    def test_split_no_chain(self):
        for _ in range(100):
            expected_id = self._random_pdb_id()
            id, chain = pdb.split_id(expected_id)
            assert id == expected_id, expected_id
            assert chain is None, expected_id

    def test_split_with_chain(self):
        for _ in range(100):
            full_id = self._random_pdb_id(with_chain=True)
            expected_id, expected_chain = full_id.split(":")
            id, chain = pdb.split_id(full_id)
            assert id == expected_id, full_id
            assert chain == expected_chain, full_id

    def test_doesnt_start_with_digit(self):
        for _ in range(100):
            full_id = self._random_pdb_id(with_chain=bool(random.randrange(2)))
            invalid_id = f'{random.choice(string.ascii_letters)}{full_id[1:]}'

            with pytest.raises(ValueError) as exc_info:
                pdb.split_id(invalid_id)

    def test_too_long(self):
        for _ in range(100):
            full_id = self._random_pdb_id(with_chain=bool(random.randrange(2)))
            invalid_id = f'{str(random.randrange(10))}{full_id}'

            with pytest.raises(ValueError) as exc_info:
                pdb.split_id(invalid_id)


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
        assert set(ids) == {'P69905', 'P68871'}


@pytest.mark.skipif(NO_INTERNET, reason='Needs internet')
class TestPDBQueries:
    @classmethod
    def setup_class(cls):
        cls.TEMP_PATH = tests.utils.get_tmp_path('pdb')

    def test_resolution_query(self):
        min_res = 0.4
        max_res = 0.5
        query = pdb.PDBResolutionQuery(min_res, max_res)
        pdb_ids = query.execute()

        assert len(pdb_ids) >= 2

        for id in pdb_ids:
            try:
                pdb_s = pdb.pdb_struct(id)
                assert min_res <= pdb_s.header['resolution'] <= max_res
            except PDBConstructionException as e:
                pass

    def test_expression_system_query(self):
        expr_sys = 'Escherichia coli BL21(DE3)'
        comp_type = 'equals'

        query = pdb.PDBExpressionSystemQuery(expr_sys, comp_type)
        pdb_ids = query.execute()

        assert len(pdb_ids) > 17100

    def test_composite_query(self):
        query = pdb.PDBCompositeQuery(
            pdb.PDBExpressionSystemQuery('Escherichia coli BL21(DE3)'),
            pdb.PDBResolutionQuery(0.5, 0.8)
        )

        pdb_ids = query.execute()

        assert len(pdb_ids) >= 3
