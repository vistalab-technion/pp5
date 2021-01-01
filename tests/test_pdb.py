import os
import math
import random
import string
from urllib.request import urlopen

import pandas as pd
import pytest
from pytest import approx

import tests
import tests.utils
import pp5.external_dbs.pdb as pdb

NO_INTERNET = not tests.utils.has_internet()


def _random_pdb_id(id_type="plain", min_chain_len=1, max_chain_len=2) -> str:
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
    def setup(self):
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
    def setup(self):
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
                id_type="chain", min_chain_len=3, max_chain_len=9
            )
            with pytest.raises(ValueError) as exc_info:
                pdb.split_id(invalid_id)

    def test_entity_too_long(self):
        for _ in range(self.n):
            invalid_id = _random_pdb_id(
                id_type="entity", min_chain_len=3, max_chain_len=9
            )
            with pytest.raises(ValueError) as exc_info:
                pdb.split_id(invalid_id)


@pytest.mark.skipif(NO_INTERNET, reason="Needs internet")
class TestPDBDownload:
    @classmethod
    def setup_class(cls):
        cls.test_id = "102L"
        # Use temp path to force download (file won't exists there)
        cls.TEMP_PATH = tests.get_tmp_path("pdb")

    @classmethod
    def teardown_class(cls):
        pass

    def setup(self):
        pass

    def test_pdb_download(self):
        path = pdb.pdb_download(self.test_id, pdb_dir=self.TEMP_PATH)
        expected_path = self.TEMP_PATH.joinpath(f"{self.test_id.lower()}.cif")
        assert path == expected_path
        assert os.path.isfile(expected_path)

    def test_pdb_struct(self):
        struct = pdb.pdb_struct(self.test_id, pdb_dir=self.TEMP_PATH)
        chains = list(struct.get_chains())
        assert len(chains) == 1


@pytest.mark.skipif(NO_INTERNET, reason="Needs internet")
@pytest.mark.skip(reason="Legacy PDB API is down, should remove")
class TestPDBMetadata:
    def test_pdb_metadata(self):
        pdb_ids = ["1MWC", "1b0y"]

        # See all fields at: https://www.rcsb.org/pdb/results/reportField.do
        url = (
            f"http://www.rcsb.org/pdb/rest/customReport?"
            f'pdbids={",".join(pdb_ids)}'
            f"&customReportColumns=taxonomyId,expressionHost,source,"
            f"resolution,structureTitle,entityId,rFree,rWork,spaceGroup,"
            f"ligandId,sequence,crystallizationTempK,phValue"
            f"&service=wsfile&format=csv"
        )

        with urlopen(url) as f:
            df = pd.read_csv(f)
            df_groups = df.groupby("structureId")

        for pdb_id in pdb_ids:
            meta = pdb.PDBMetadata(pdb_id)
            pdb_id_group: pd.DataFrame = df_groups.get_group(pdb_id.upper())
            expected = pdb_id_group.iloc[0]
            chain = expected["chainId"]

            assert meta.pdb_id == pdb_id.upper()
            assert meta.title == expected["structureTitle"]
            assert meta.src_org == expected["source"]
            assert meta.src_org_id == expected["taxonomyId"]
            assert meta.host_org == expected["expressionHost"]
            assert meta.resolution == expected["resolution"]
            assert meta.chain_entities[chain] == expected["entityId"]
            if not meta.r_free:
                assert math.isnan(expected["rFree"])
            else:
                assert meta.r_free == approx(expected["rFree"], abs=1e-3)
            if not meta.r_work:
                assert math.isnan(expected["rWork"])
            else:
                assert meta.r_work == approx(expected["rWork"], abs=1e-3)
            if not meta.cg_ph:
                assert math.isnan(expected["phValue"])
            else:
                assert meta.cg_ph == approx(expected["phValue"], abs=1e-3)
            if not meta.cg_temp:
                assert math.isnan(expected["crystallizationTempK"])
            else:
                assert meta.cg_temp == approx(
                    expected["crystallizationTempK"], abs=1e-3
                )

            assert meta.space_group == expected["spaceGroup"]

            expected_ligands = pdb_id_group["ligandId"]
            assert meta.ligands.split(",") == list(set(expected_ligands))

            chain_groups = pdb_id_group.groupby("chainId")
            for chain, group in chain_groups:
                seq = meta.entity_sequence[meta.chain_entities[chain]]
                expected_seq = group.iloc[0]["sequence"]
                assert seq == expected_seq


@pytest.mark.skipif(NO_INTERNET, reason="Needs internet")
class TestPDB2UNP:
    @staticmethod
    def _check(pdb_id, expected_unp_id):
        actual_unp_id = pdb.PDB2UNP.pdb_id_to_unp_id(pdb_id)
        assert actual_unp_id == expected_unp_id

    def test_no_chain_single_unp(self):
        self._check("102L", "P00720")

    def test_with_chain_single_unp(self):
        self._check("102L:A", "P00720")

    def test_no_chain_multi_unp_strict(self):
        test_id = "4HHB"
        with pytest.raises(ValueError, match="Multiple Uniprot IDs"):
            pdb.PDB2UNP.pdb_id_to_unp_id(test_id)

    def test_no_chain_multi_unp_not_strict(self):
        test_id = "4HHB"
        expected_unp_ids = {"P69905", "P68871"}
        actual_unp_id = pdb.PDB2UNP.pdb_id_to_unp_id(test_id, strict=False)
        assert actual_unp_id in expected_unp_ids

    @pytest.mark.parametrize("test_id", ["4HHB:A", "4HHB:C"])
    def test_with_chain_multi_unp_1(self, test_id):
        self._check(test_id, "P69905")

    @pytest.mark.parametrize("test_id", ["4HHB:B", "4HHB:D"])
    def test_with_chain_multi_unp_2(self, test_id):
        self._check(test_id, "P68871")

    def test_with_invalid_chain(self):
        with pytest.raises(ValueError, match="chain Z"):
            pdb.PDB2UNP.pdb_id_to_unp_id("4HHB:Z")

    @pytest.mark.parametrize("test_id", ["5LTR", "5LTR:A"])
    def test_with_no_xref_in_file(self, test_id):
        self._check(test_id, "B1PNC0")

    @pytest.mark.parametrize("test_id", ["5EJU", "4DXP"])
    def test_with_no_xref_in_file_and_pdb(self, test_id):
        with pytest.raises(ValueError, match="No Uniprot entries"):
            pdb.PDB2UNP.pdb_id_to_unp_id(test_id)

    @pytest.mark.parametrize("test_id", ["3G53", "3G53:A"])
    def test_with_no_struct_ref_entry(self, test_id):
        self._check(test_id, "P02213")

    @pytest.mark.parametrize(
        ("test_id", "unp_id"), [("3SG4:A", "P42212"), ("4IK8:A", "P42212")],
    )
    def test_multi_unp_for_single_chain_no_strict(self, test_id, unp_id):
        actual_unp_id = pdb.PDB2UNP.pdb_id_to_unp_id(test_id, strict=False)
        assert actual_unp_id == unp_id

    @pytest.mark.parametrize("test_id", ["3SG4:A", "4IK8:A"])
    def test_multi_unp_for_single_chain_strict(self, test_id):
        with pytest.raises(ValueError, match="chimeric"):
            pdb.PDB2UNP.pdb_id_to_unp_id(test_id)
