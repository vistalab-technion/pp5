import pytest

import tests
import tests.utils
import pp5.external_dbs.unp as unp

NO_INTERNET = not tests.utils.has_internet()


@pytest.mark.skipif(NO_INTERNET, reason="Needs internet")
class TestUNPDownload:
    @classmethod
    @pytest.fixture(scope="class")
    def unp_dir(cls):
        return tests.get_tmp_path("data/unp")

    def test_unp_record(self, unp_dir):
        test_id = "P00720"
        unp_rec = unp.unp_record(test_id, unp_dir=unp_dir)
        assert unp_rec.sequence_length == 164

    def test_unp_download(self, unp_dir):
        test_id = "P42212"
        path = unp.unp_download(test_id, unp_dir=unp_dir)
        assert path == unp_dir.joinpath(f"{test_id}.txt")

    def test_unp_download_with_redirect(self, unp_dir):
        # This UNP id causes a redirect to a few replacement ids.
        test_id = "P31217"
        replacement_id = unp.unp_replacement_ids(test_id)[0]
        assert replacement_id in ["P62707", "P62708", "P62709", "P62710"]

        path = unp.unp_download(test_id, unp_dir=unp_dir)

        assert path == unp_dir.joinpath(f"{replacement_id}.txt")

    def test_unp_download_with_invalid_id(self, unp_dir):
        with pytest.raises(IOError, match="400"):
            path = unp.unp_download("P000000", unp_dir=unp_dir)


class TestUNPRecord:
    @pytest.fixture(params=["P00720", "Q72J47"])
    def unp_id(self, request):
        return request.param

    def test_unp_record(self, unp_id):
        unp_rec = unp.unp_record(unp_id)
        assert unp_id in unp_rec.accessions

    def test_as_record(self, unp_id):
        rec = unp.as_record(unp_id)
        expected_rec = unp.unp_record(unp_id)
        assert rec.accessions == expected_rec.accessions

    def test_as_record_from_record(self, unp_id):
        orig_rec = unp.unp_record(unp_id)
        rec = unp.as_record(orig_rec)
        assert rec == orig_rec


@pytest.mark.skipif(NO_INTERNET, reason="Needs internet")
class TestENAXRefs:
    def test_ena_multi_type(self):
        molecule_types = ("mRNA", "Genomic_DNA")
        xrefs = unp.find_ena_xrefs("P42212", molecule_types)

        # Copied from Uniprot site
        expected_xrefs = {
            "AAA27722.1",
            "AAA27721.1",
            "AAA58246.1",
            "CAA65278.1",
            "AAB18957.1",
        }
        for expected_xref in expected_xrefs:
            assert expected_xref in xrefs

    def test_ena_single_type(self):
        molecule_types = "Genomic_DNA"
        xrefs = unp.find_ena_xrefs("P42212", molecule_types)

        expected_xrefs = {"AAB18957.1"}
        for expected_xref in expected_xrefs:
            assert expected_xref in xrefs


@pytest.mark.skipif(NO_INTERNET, reason="Needs internet")
class TestPDBXRefs:
    @pytest.fixture(autouse=True)
    def setup_fixture(self):
        self.xrefs = unp.find_pdb_xrefs("P00720", method="x-ray")

    def test_single_chain(self):
        xs = [x for x in self.xrefs if x.pdb_id == "102L"]
        assert len(xs) == 1
        assert xs[0].chain_id == "A"
        assert xs[0].seq_len == 163

    def test_multi_chain(self):
        xs = [x for x in self.xrefs if x.pdb_id == "150L"]
        assert len(xs) == 4
        assert set(x.chain_id for x in xs) == {"A", "B", "C", "D"}
        assert all(x.seq_len == 163 for x in xs)

    def test_multi_chain_with_other_proteins(self):
        xs = [x for x in self.xrefs if x.pdb_id == "6QAJ"]
        assert len(xs) == 2
        assert set(x.chain_id for x in xs) == {"A", "B"}
        assert all(x.seq_len == 159 for x in xs)
