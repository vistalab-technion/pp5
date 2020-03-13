import os
import random
import string
from urllib.request import urlopen

import pandas as pd
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
            assert id == expected_id.upper(), expected_id
            assert chain is None, expected_id

    def test_split_with_chain(self):
        for _ in range(100):
            full_id = self._random_pdb_id(with_chain=True)
            expected_id, expected_chain = full_id.split(":")
            id, chain = pdb.split_id(full_id)
            assert id == expected_id.upper(), full_id
            assert chain == expected_chain.upper(), full_id

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
        cls.TEMP_PATH = tests.get_tmp_path('pdb')

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

    def test_pdb_metadata(self):
        pdb_ids = ['1MWC', '1b0y']

        # See all fields at: https://www.rcsb.org/pdb/results/reportField.do
        url = f'http://www.rcsb.org/pdb/rest/customReport.xml?' \
              f'pdbids={",".join(pdb_ids)}' \
              f'&customReportColumns=taxonomyId,expressionHost,source,' \
              f'resolution,structureTitle,entityId' \
              f'&service=wsfile&format=csv'

        with urlopen(url) as f:
            df = pd.read_csv(f)
            df_groups = df.groupby('structureId')

        for pdb_id in pdb_ids:
            meta = pdb.pdb_metadata(pdb_id)
            expected = df_groups.get_group(pdb_id.upper()).iloc[0]
            chain = expected['chainId']

            assert meta.pdb_id == pdb_id.upper()
            assert meta.title == expected['structureTitle']
            assert meta.src_org == expected['source']
            assert meta.src_org_id == expected['taxonomyId']
            assert meta.host_org == expected['expressionHost']
            assert meta.resolution == expected['resolution']
            assert meta.chain_entities[chain] == expected['entityId']


@pytest.mark.skipif(NO_INTERNET, reason='Needs internet')
class TestPDBToUNP:
    @classmethod
    def setup_class(cls):
        cls.TEMP_PATH = tests.get_tmp_path('pdb')

    def _check(self, pdb_id, expected_unp_id):
        actual_unp_id = pdb.pdbid_to_unpid(pdb_id, self.TEMP_PATH)
        assert actual_unp_id == expected_unp_id

    def test_no_chain_single_unp(self):
        self._check('102L', 'P00720')

    def test_with_chain_single_unp(self):
        self._check('102L:A', 'P00720')

    def test_no_chain_multi_unp(self):
        test_id = '4HHB'
        expected_unp_ids = {'P69905', 'P68871'}

        id = pdb.pdbid_to_unpid(test_id, self.TEMP_PATH)
        assert id in expected_unp_ids

    def test_with_chain_multi_unp(self):
        for test_id in {'4HHB:A', '4HHB:C'}:
            self._check(test_id, 'P69905')

        for test_id in {'4HHB:B', '4HHB:D'}:
            self._check(test_id, 'P68871')

    def test_with_invalid_chain(self):
        with pytest.raises(ValueError, match='chain Z'):
            pdb.pdbid_to_unpid('4HHB:Z')


@pytest.mark.skipif(NO_INTERNET, reason='Needs internet')
class TestPDBQueries:
    @classmethod
    def setup_class(cls):
        cls.TEMP_PATH = tests.get_tmp_path('pdb')

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
