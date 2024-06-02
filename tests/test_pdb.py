import os
import math
import random
import string
from pprint import pprint
from urllib.request import urlopen

import pandas as pd
import pytest
from pytest import approx

import tests
import tests.utils
import pp5.external_dbs.pdb as pdb

NO_INTERNET = not tests.utils.has_internet()


def _random_pdb_id(id_type="plain", min_chain_len=1, max_chain_len=3) -> str:
    """
    Creates a random PDB id either as a base id only, with a chain or with an entity.
    :param id_type: 'plain' (only base), 'chain' (with chain id), 'entity' (with
        entity id).
    :param min_chain_len: Minimal allowed length of a chain/entity id added.
    :param max_chain_len: Maximal allowed length of a chain/entity id added.
    :return: The generated id.
    """

    base_id = str.join(
        "", [str(random.randrange(10))] + random.choices(string.ascii_letters, k=3)
    )

    if id_type == "plain":
        return base_id
    elif id_type == "chain":
        extra_chars = string.ascii_letters
    elif id_type == "entity":
        extra_chars = string.digits
    else:
        raise ValueError(f"Unexpected type: {id_type}")

    extra_letters = [
        random.choice(extra_chars)
        # Note: randint max is inclusive
        for _ in range(random.randint(min_chain_len, max_chain_len))
    ]

    return base_id + f":{str.join('', extra_letters)}"


class TestSplitID:
    @pytest.fixture(autouse=True)
    def setup_fixture(self):
        self.n = 100

    def test_split_no_chain(self):
        for _ in range(self.n):
            expected_id = _random_pdb_id()
            id, chain = pdb.split_id(expected_id)
            assert id == expected_id.upper(), expected_id
            assert chain is None, expected_id

    def test_split_with_chain(self):
        for _ in range(self.n):
            full_id = _random_pdb_id(id_type="chain")
            expected_id, expected_chain = full_id.split(":")
            id, chain = pdb.split_id(full_id)
            assert id == expected_id.upper(), full_id
            assert chain == expected_chain.upper(), full_id

    def test_split_id_when_entity_given(self):
        for _ in range(self.n):
            expected_id = _random_pdb_id(id_type="entity")
            id, chain = pdb.split_id(expected_id)
            expected_id = expected_id.split(":")[0]  # remove entity
            assert id == expected_id.upper(), expected_id
            assert chain is None, expected_id


class TestSplitIDWithEntity:
    @pytest.fixture(autouse=True)
    def setup_fixture(self):
        self.n = 100

    def test_split_base_only(self):
        for _ in range(self.n):
            expected_id = _random_pdb_id()
            id, chain, entity = pdb.split_id_with_entity(expected_id)
            assert id == expected_id.upper(), expected_id
            assert chain is None, expected_id
            assert entity is None, expected_id

    def test_split_with_chain(self):
        for _ in range(self.n):
            full_id = _random_pdb_id(id_type="chain")
            expected_id, expected_chain = full_id.split(":")
            id, chain, entity = pdb.split_id_with_entity(full_id)
            assert id == expected_id.upper(), full_id
            assert chain == expected_chain.upper(), full_id
            assert entity is None, full_id

    def test_split_with_entity(self):
        for _ in range(self.n):
            full_id = _random_pdb_id(id_type="entity")
            expected_id, expected_entity = full_id.split(":")
            id, chain, entity = pdb.split_id_with_entity(full_id)
            assert id == expected_id.upper(), full_id
            assert chain is None, full_id
            assert entity == expected_entity, full_id

    def test_doesnt_start_with_digit(self):
        for _ in range(self.n):
            full_id = _random_pdb_id(
                id_type=random.choice(["plain", "chain", "entity"])
            )
            invalid_id = f"{random.choice(string.ascii_letters)}{full_id[1:]}"
            with pytest.raises(ValueError) as exc_info:
                pdb.split_id(invalid_id)

    def test_too_long(self):
        for _ in range(self.n):
            full_id = _random_pdb_id(
                id_type=random.choice(["plain", "chain", "entity"])
            )
            invalid_id = f"{str(random.randrange(10))}{full_id}"
            with pytest.raises(ValueError) as exc_info:
                pdb.split_id(invalid_id)

    def test_chain_too_long(self):
        for _ in range(self.n):
            invalid_id = _random_pdb_id(
                id_type="chain", min_chain_len=4, max_chain_len=9
            )
            with pytest.raises(ValueError) as exc_info:
                pdb.split_id(invalid_id)

    def test_entity_too_long(self):
        for _ in range(self.n):
            invalid_id = _random_pdb_id(
                id_type="entity", min_chain_len=4, max_chain_len=9
            )
            with pytest.raises(ValueError) as exc_info:
                pdb.split_id(invalid_id)


@pytest.fixture(scope="class", params=["1MWC:A", "2WUR:A", "4N6V:1", "1DWI:A"])
def pdb_id(request):
    return request.param


@pytest.fixture(params=pdb.PDB_DOWNLOAD_SOURCES.keys())
def pdb_source(request):
    return request.param


@pytest.mark.skipif(NO_INTERNET, reason="Needs internet")
class TestPDBDownload:
    @pytest.fixture(scope="class")
    def temp_path(self):
        return tests.get_tmp_path("data/pdb", clear=True)

    def test_pdb_download(self, pdb_id, pdb_source, temp_path):
        path = pdb.pdb_download(pdb_id, pdb_dir=temp_path, pdb_source=pdb_source)

        pdb_base_id, pdb_chain = pdb.split_id(pdb_id)
        assert os.path.isfile(path)
        assert str(path).startswith(str(temp_path))
        assert pdb_base_id.lower() in path.stem
        assert pdb_source in path.stem
        assert ".cif" in path.suffix

    def test_pdb_struct(self, pdb_id, pdb_source, temp_path):
        struct = pdb.pdb_struct(pdb_id, pdb_dir=temp_path, pdb_source=pdb_source)

        pdb_base_id, pdb_chain = pdb.split_id(pdb_id)
        assert pdb_base_id == struct.get_id()

    def test_exception_chimeric_chain(self):
        with pytest.raises(ValueError, match="Can't determine unique uniprot id"):
            pdb.pdb_download("3SG4:A", pdb_source=pdb.PDB_AFLD)


@pytest.mark.skipif(NO_INTERNET, reason="Needs internet")
class TestPDBMetadata:
    @pytest.fixture(scope="class")
    def metadata(self, pdb_id):
        return pdb.PDBMetadata(pdb_id)

    def test_metadata_properties(self, metadata, pdb_id):
        pdb_base_id, pdb_chain = pdb.split_id(pdb_id)
        assert metadata.pdb_id == pdb_base_id

    def test_as_dict(self, metadata):
        d = metadata.as_dict()  # evaluates all metadata properties
        pprint(d)

    def test_cache(self, metadata):
        path = metadata.to_cache()
        cache_attrs = metadata.cache_attribs()
        assert path.exists()
        assert path.is_file()
        metadata_ = pdb.PDBMetadata.from_cache(cache_attribs=cache_attrs)

        assert metadata == metadata_

    @pytest.mark.parametrize("cache", [True, False], ids=["cache=True", "cache=False"])
    def test_from_pdb(self, pdb_id, cache):
        pdb_base_id, chain_id = pdb.split_id(pdb_id)
        metadata = pdb.PDBMetadata.from_pdb(pdb_id, cache=cache)
        assert metadata.pdb_id == pdb_base_id

    @pytest.mark.parametrize(
        "seq_to_str", [False, True], ids=["seq_to_str=False", "seq_to_str=True"]
    )
    def test_as_dict_chain(self, metadata, seq_to_str):
        for chain_id in metadata.chain_ids:
            d = metadata.as_dict(chain_id=chain_id, seq_to_str=seq_to_str)
            print(f" === {chain_id=} === ")
            pprint(d)

    @staticmethod
    def _check_unp(pdb_id, expected_unp_id):
        actual_unp_id = pdb.PDBMetadata.pdb_id_to_unp_id(pdb_id)
        assert actual_unp_id == expected_unp_id

    def test_no_chain_single_unp(self):
        self._check_unp("102L", "P00720")

    def test_with_chain_single_unp(self):
        self._check_unp("102L:A", "P00720")

    def test_no_chain_multi_unp_strict(self):
        test_id = "4HHB"
        with pytest.raises(ValueError, match="Multiple Uniprot IDs"):
            pdb.PDBMetadata.pdb_id_to_unp_id(test_id)

    def test_no_chain_multi_unp_not_strict(self):
        test_id = "4HHB"
        expected_unp_ids = {"P69905", "P68871"}
        actual_unp_id = pdb.PDBMetadata.pdb_id_to_unp_id(test_id, strict=False)
        assert actual_unp_id in expected_unp_ids

    @pytest.mark.parametrize("test_id", ["4HHB:A", "4HHB:C"])
    def test_with_chain_multi_unp_1(self, test_id):
        self._check_unp(test_id, "P69905")

    @pytest.mark.parametrize("test_id", ["4HHB:B", "4HHB:D"])
    def test_with_chain_multi_unp_2(self, test_id):
        self._check_unp(test_id, "P68871")

    def test_with_invalid_chain(self):
        with pytest.raises(ValueError, match="chain Z"):
            pdb.PDBMetadata.pdb_id_to_unp_id("4HHB:Z")

    @pytest.mark.parametrize("test_id", ["5LTR", "5LTR:A"])
    def test_with_no_xref_in_file(self, test_id):
        self._check_unp(test_id, "B1PNC0")

    @pytest.mark.parametrize("test_id", ["5EJU", "4DXP"])
    def test_with_no_xref_in_file_and_pdb(self, test_id):
        with pytest.raises(ValueError, match="No Uniprot entries"):
            pdb.PDBMetadata.pdb_id_to_unp_id(test_id)

    @pytest.mark.parametrize("test_id", ["3G53", "3G53:A"])
    def test_with_no_struct_ref_entry(self, test_id):
        self._check_unp(test_id, "P02213")

    @pytest.mark.parametrize(
        ("test_id", "unp_ids"),
        [("3SG4:A", {"P11799", "P42212", "P0DP29"}), ("4IK8:A", {"K4DIE3", "P42212"})],
    )
    def test_multi_unp_for_single_chain_no_strict(self, test_id, unp_ids):
        actual_unp_id = pdb.PDBMetadata.pdb_id_to_unp_id(test_id, strict=False)
        assert actual_unp_id in unp_ids

    @pytest.mark.parametrize("test_id", ["3SG4:A", "4IK8:A"])
    def test_multi_unp_for_single_chain_strict(self, test_id):
        with pytest.raises(ValueError, match="chimeric"):
            pdb.PDBMetadata.pdb_id_to_unp_id(test_id)
