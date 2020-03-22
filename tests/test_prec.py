import pickle
import pytest

from pp5.protein import ProteinRecord, ProteinInitError
from pp5.external_dbs import unp
from tests import get_tmp_path


class TestCreation:
    def test_from_unp_default_selector(self):
        unp_id = 'P00720'
        prec = ProteinRecord.from_unp(unp_id)
        assert prec.unp_id == unp_id

        # Default selection should be by resolution
        xrs = sorted(unp.find_pdb_xrefs(unp_id), key=lambda x: x.resolution)
        assert prec.pdb_base_id == xrs[0].pdb_id
        assert prec.pdb_chain_id == xrs[0].chain_id

    def test_from_unp_with_name_selector(self):
        unp_id = 'P00720'
        prec = ProteinRecord.from_unp(unp_id, xref_selector=lambda x: x.pdb_id)
        assert prec.unp_id == unp_id
        assert prec.pdb_id == '102L:A'

    def test_from_pdb_with_chain(self):
        pdb_id = '102L:A'
        prec = ProteinRecord.from_pdb(pdb_id)
        assert prec.unp_id == 'P00720'
        assert prec.pdb_id == pdb_id

    def test_from_pdb_without_chain(self):
        pdb_id = '102L'
        prec = ProteinRecord.from_pdb(pdb_id)
        assert prec.unp_id == 'P00720'
        assert prec.pdb_id == f'{pdb_id}:A'

    def test_from_pdb_invalid_chain(self):
        with pytest.raises(ProteinInitError):
            ProteinRecord.from_pdb('102L:Z')

    def test_from_pdb_entity(self):
        pdb_id = '4HHB'
        prec = ProteinRecord.from_pdb_entity(pdb_id, 2)
        assert prec.pdb_chain_id == 'B'

    def test_from_pdb_entity_with_invalid_entity(self):
        with pytest.raises(ProteinInitError):
            ProteinRecord.from_pdb_entity('4HHB', 3)

    def test_from_pdb_invalid_pdbid(self):
        with pytest.raises(ProteinInitError):
            ProteinRecord.from_pdb('0AAA')

    def test_init_no_chain(self):
        unp_id = 'P00720'
        pdb_id = '5JDT'
        prec = ProteinRecord(unp_id, pdb_id)
        assert prec.unp_id == 'P00720'
        assert prec.pdb_id == f'{pdb_id}:A'

    def test_init_with_chain(self):
        unp_id = 'P00720'
        pdb_id = '5JDT:A'
        prec = ProteinRecord(unp_id, pdb_id)
        assert prec.unp_id == 'P00720'
        assert prec.pdb_id == pdb_id

    def test_init_with_mismatching_pdb_id(self):
        with pytest.raises(ProteinInitError):
            ProteinRecord('P00720', '2WUR:A')

        with pytest.raises(ProteinInitError):
            ProteinRecord('P00720', '4GY3')


class TestSave:
    def test_save(self):
        pdb_id = '102L:A'
        prec = ProteinRecord.from_pdb(pdb_id)
        filepath = prec.save(out_dir=get_tmp_path('prec'))

        with open(str(filepath), 'rb') as f:
            prec2 = pickle.load(f)

        assert prec == prec2
