import pickle

import pytest

import pp5
from tests import get_tmp_path
from pp5.protein import ProteinRecord, ProteinInitError
from pp5.external_dbs import unp


class TestFromUnp:
    def test_from_unp_default_selector(self):
        unp_id = "P00720"
        prec = ProteinRecord.from_unp(unp_id)
        assert prec.unp_id == unp_id

        # Default selection should be by resolution
        xrs = sorted(unp.find_pdb_xrefs(unp_id), key=lambda x: x.resolution)
        assert prec.pdb_base_id == xrs[0].pdb_id
        assert prec.pdb_chain_id == xrs[0].chain_id

    def test_from_unp_with_name_selector(self):
        unp_id = "P00720"
        prec = ProteinRecord.from_unp(unp_id, xref_selector=lambda x: x.pdb_id)
        assert prec.unp_id == unp_id
        assert prec.pdb_id == "102L:A"


class TestFromPDB:
    def test_with_chain(self):
        pdb_id = "102L:A"
        prec = ProteinRecord.from_pdb(pdb_id)
        assert prec.unp_id == "P00720"
        assert prec.pdb_id == pdb_id

    def test_without_chain(self):
        pdb_id = "102L"
        prec = ProteinRecord.from_pdb(pdb_id)
        assert prec.unp_id == "P00720"
        assert prec.pdb_id == f"{pdb_id}:A"

    def test_invalid_chain(self):
        with pytest.raises(ProteinInitError):
            ProteinRecord.from_pdb("102L:Z")

    def test_entity(self):
        pdb_id = "4HHB:2"
        prec = ProteinRecord.from_pdb(pdb_id)
        assert prec.pdb_chain_id == "B"

    def test_numerical_chain(self):
        pdb_id = "4N6V:9"
        prec = ProteinRecord.from_pdb(pdb_id)
        assert prec.pdb_base_id == "4N6V"
        assert prec.pdb_chain_id == "9"

    def test_ambiguous_numerical_entity_and_chain(self):
        # In this rare case it's impossible to know if entity or chain!
        # In this PDB structure the chains are numeric and there's also an
        # entity with id=1.
        # We expect the ':1' to be treated as an entity, and the first
        # associated chain should be returned.
        pdb_id = "4N6V:1"
        prec = ProteinRecord.from_pdb(pdb_id)
        assert prec.pdb_base_id == "4N6V"
        assert prec.pdb_chain_id == "0"

    def test_entity_with_invalid_entity(self):
        with pytest.raises(ProteinInitError):
            ProteinRecord.from_pdb("4HHB:3")

    def test_invalid_pdbid(self):
        with pytest.raises(ProteinInitError):
            ProteinRecord.from_pdb("0AAA")

    def test_multiple_unp_ids_for_same_pdb_chain_no_strict_pdb_xref(self):
        prec = ProteinRecord.from_pdb("3SG4:A", strict_pdb_xref=False,)
        assert prec.unp_id == "P0DP29"
        assert prec.pdb_id == "3SG4:A"

        prec = ProteinRecord.from_pdb("3SG4", strict_pdb_xref=False,)
        assert prec.unp_id == "P0DP29"
        assert prec.pdb_id == "3SG4:A"

    def test_multiple_unp_ids_for_same_pdb_chain(self):
        with pytest.raises(ProteinInitError):
            ProteinRecord.from_pdb("3SG4:A")

        with pytest.raises(ProteinInitError):
            ProteinRecord.from_pdb("3SG4")


class TestInit:
    def test_init_no_chain(self):
        unp_id = "P00720"
        pdb_id = "5JDT"
        prec = ProteinRecord(unp_id, pdb_id)
        assert prec.unp_id == "P00720"
        assert prec.pdb_id == f"{pdb_id}:A"

    def test_init_with_chain(self):
        unp_id = "P00720"
        pdb_id = "5JDT:A"
        prec = ProteinRecord(unp_id, pdb_id)
        assert prec.unp_id == "P00720"
        assert prec.pdb_id == pdb_id

    def test_init_with_mismatching_pdb_id(self):
        with pytest.raises(ProteinInitError):
            ProteinRecord("P00720", "2WUR:A")

        with pytest.raises(ProteinInitError):
            ProteinRecord("P00720", "4GY3")

    def test_no_strict_xref_with_no_xref_in_pdb(self):
        prec = ProteinRecord("Q6LDG3", "3SG4:A", strict_unp_xref=False)
        assert prec.unp_id == "Q6LDG3"
        assert prec.pdb_id == "3SG4:A"

    def test_no_strict_xref_with_no_xref_in_pdb_and_no_chain(self):
        with pytest.raises(ProteinInitError, match="and no chain provided"):
            ProteinRecord("Q6LDG3", "3SG4", strict_unp_xref=False)

    def test_strict_xref_with_no_matching_xref_in_pdb(self):
        with pytest.raises(ProteinInitError):
            ProteinRecord("P42212", "2QLE:A")

    def test_no_strict_xref_with_no_matching_xref_in_pdb(self):
        prec = ProteinRecord("P42212", "2QLE:A", strict_unp_xref=False)
        assert prec.unp_id == "P42212"
        assert prec.pdb_id == "2QLE:A"


class TestSave:
    @classmethod
    def setup_class(cls):
        cls.TEMP_PATH = get_tmp_path("prec")

    @pytest.mark.parametrize("pdb_id", ["2WUR:A", "102L", "5NL4:A"])
    def test_save_roundtrip(self, pdb_id):
        prec = ProteinRecord.from_pdb(pdb_id)
        filepath = prec.save(out_dir=self.TEMP_PATH)

        with open(str(filepath), "rb") as f:
            prec2 = pickle.load(f)

        assert prec == prec2


class TestCache:
    @classmethod
    def setup_class(cls):
        cls.CACHE_DIR = get_tmp_path("prec_cache")

    @pytest.mark.parametrize("pdb_id", ["1MWC:A", "4N6V:7"])
    def test_from_pdb_with_cache(self, pdb_id):
        prec = ProteinRecord.from_pdb(pdb_id, cache=True, cache_dir=self.CACHE_DIR)

        filename = f"{pdb_id.replace(':', '_')}.prec"
        expected_filepath = self.CACHE_DIR.joinpath(filename)
        assert expected_filepath.is_file()

        with open(str(expected_filepath), "rb") as f:
            loaded_prec = pickle.load(f)
        assert prec == loaded_prec

        loaded_prec = ProteinRecord.from_cache(pdb_id, cache_dir=self.CACHE_DIR)
        assert prec == loaded_prec

    @pytest.mark.parametrize("pdb_id", ["1B0Y:A",])
    def test_from_cache_non_existent_id(self, pdb_id):
        prec = ProteinRecord.from_cache(pdb_id, cache_dir=self.CACHE_DIR)

        filename = f"{pdb_id.replace(':', '_')}.prec"
        expected_filepath = self.CACHE_DIR.joinpath(filename)
        assert not expected_filepath.is_file()
        assert prec is None
