import pickle
from math import isnan

import pytest

import pp5
from tests import get_tmp_path
from pp5.prec import ProteinRecord, ResidueRecord
from pp5.align import DEFAULT_ARPEGGIO_ARGS, Arpeggio
from pp5.utils import ProteinInitError
from pp5.codons import UNKNOWN_AA
from pp5.external_dbs import unp
from pp5.external_dbs.pdb import PDB_DOWNLOAD_SOURCES


@pytest.fixture(
    autouse=False,
    scope="class",
    params=[True, False],
    ids=["with_altlocs", "no_altlocs"],
)
def with_altlocs(request):
    return request.param


@pytest.fixture(
    autouse=False,
    scope="class",
    params=[False, True],
    ids=["no_backbone", "with_backbone"],
)
def with_backbone(request):
    return request.param


class TestMethods:
    @pytest.fixture(autouse=False, scope="class", params=["102L:A", "2WUR:A"])
    def prec(self, with_altlocs, with_backbone, request):
        pdb_id = request.param
        prec = ProteinRecord.from_pdb(
            pdb_id, with_altlocs=with_altlocs, with_backbone=with_backbone
        )
        return prec

    def test_backbone(self, prec, with_backbone):
        res: ResidueRecord
        for res in prec:
            if res.name == UNKNOWN_AA or isnan(res.angles.phi):
                continue
            if with_backbone:
                assert res.backbone_coords is not None
                assert res.backbone_coords.shape == (4, 3)
            else:
                assert res.backbone_coords is None

    @pytest.mark.parametrize("with_ids", [True, False], ids=["with_ids", "no_ids"])
    @pytest.mark.parametrize(
        "with_contacts",
        [False, True],
        ids=["no_contacts", "with_contacts"],
    )
    def test_to_dataframe(self, prec, with_ids, with_backbone, with_contacts):
        if isinstance(with_contacts, dict):
            if not Arpeggio.can_execute(**DEFAULT_ARPEGGIO_ARGS):
                pytest.skip()

        df = prec.to_dataframe(with_ids=with_ids, with_contacts=with_contacts)
        assert len(df) == len(prec)

        if with_ids:
            assert "unp_id" in df.columns
            assert "pdb_id" in df.columns

        if with_backbone:
            assert "backbone" in df.columns

        if with_contacts:
            assert "contact_count" in df.columns

    @pytest.mark.skipif(
        not Arpeggio.can_execute(**DEFAULT_ARPEGGIO_ARGS), reason="no arpeggio"
    )
    def test_contacts(self, prec):
        assert 0 < len(prec.contacts.items()) <= len(prec)

        for res_id, res_contacts in prec.contacts.items():
            assert res_id in prec
            assert len(res_contacts.contact_aas) == res_contacts.contact_count


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
    @pytest.mark.parametrize("pdb_id", ["2WUR:A"])
    @pytest.mark.parametrize("pdb_source", tuple(PDB_DOWNLOAD_SOURCES))
    def test_pdb_source(self, pdb_id, pdb_source):
        prec = ProteinRecord.from_pdb(pdb_id, pdb_source=pdb_source)
        assert prec.pdb_id == pdb_id
        assert prec.pdb_source == pdb_source

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
        prec = ProteinRecord.from_pdb(
            "3SG4:A",
            strict_pdb_xref=False,
        )
        assert prec.unp_id == "P42212"
        assert prec.pdb_id == "3SG4:A"

        prec = ProteinRecord.from_pdb(
            "3SG4",
            strict_pdb_xref=False,
        )
        assert prec.unp_id == "P42212"
        assert prec.pdb_id == "3SG4:A"

    def test_multiple_unp_ids_for_same_pdb_chain(self):
        with pytest.raises(ProteinInitError):
            ProteinRecord.from_pdb("3SG4:A")

        with pytest.raises(ProteinInitError):
            ProteinRecord.from_pdb("3SG4")


class TestInit:
    def test_init_no_chain(self):
        unp_id = "P00720"
        pdb_id = "102L"
        prec = ProteinRecord(unp_id, pdb_id)
        assert prec.unp_id == "P00720"
        assert prec.pdb_id == f"{pdb_id}:A"

    def test_init_with_chain(self):
        unp_id = "P00720"
        pdb_id = "102L:A"
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
        cls.TEMP_PATH = get_tmp_path("data/prec")

    @pytest.mark.parametrize("pdb_id", ["2WUR:A", "102L", "5NL4:A"])
    def test_save_roundtrip(self, pdb_id, with_altlocs):
        prec = ProteinRecord.from_pdb(pdb_id, with_altlocs=with_altlocs)
        filepath = prec.save(out_dir=self.TEMP_PATH)

        with open(str(filepath), "rb") as f:
            prec2 = pickle.load(f)

        assert prec == prec2


class TestCache:
    @pytest.fixture(scope="class")
    def cache_dir(self):
        return get_tmp_path("data/prec_cache")

    @pytest.mark.parametrize("pdb_id", ["1MWC:A", "4N6V:1"])
    @pytest.mark.parametrize("pdb_source", tuple(PDB_DOWNLOAD_SOURCES))
    def test_from_pdb_with_cache(self, pdb_id, pdb_source, cache_dir, with_altlocs):
        cache_dir = cache_dir / f"{pdb_source}"
        prec = ProteinRecord.from_pdb(
            pdb_id,
            pdb_source=pdb_source,
            cache=True,
            cache_dir=cache_dir,
            strict_unp_xref=False,
            with_altlocs=with_altlocs,
        )

        filename = f"{prec.pdb_id.replace(':', '_')}-{pdb_source}.prec"
        expected_filepath = cache_dir.joinpath(filename)
        assert expected_filepath.is_file()

        with open(str(expected_filepath), "rb") as f:
            loaded_prec = pickle.load(f)
        assert prec == loaded_prec

        loaded_prec = ProteinRecord.from_cache(
            prec.pdb_id, pdb_source=pdb_source, cache_dir=cache_dir
        )
        assert prec == loaded_prec

    @pytest.mark.parametrize(
        "pdb_id",
        ["1B0Y:A", "1GRL:A"],
    )
    @pytest.mark.parametrize("pdb_source", tuple(PDB_DOWNLOAD_SOURCES))
    def test_from_cache_non_existent_id(self, pdb_id, pdb_source, cache_dir):
        prec = ProteinRecord.from_cache(
            pdb_id, pdb_source=pdb_source, cache_dir=cache_dir
        )

        filename = f"{pdb_id.replace(':', '_')}-{pdb_source}.prec"
        expected_filepath = cache_dir.joinpath(filename)
        assert not expected_filepath.is_file()
        assert prec is None
