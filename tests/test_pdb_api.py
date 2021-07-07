from contextlib import contextmanager

import pytest

import pp5.external_dbs.pdb_api as pdb_api
from pp5.external_dbs.pdb import split_id_with_entity


@contextmanager
def _does_not_raise(*args, **kwargs):
    yield


class _BasicQuery(pdb_api.PDBQuery):
    def __init__(self, **base_kwargs):
        super().__init__(**base_kwargs)

    def _raw_query_data(self) -> dict:
        return {"type": "terminal", "service": "text"}

    def description(self) -> str:
        return "Basic query"


class TestRawDataAPI:
    @pytest.mark.parametrize("pdb_base_id", ["4HHB", "2wur", "3sG4"])
    def test_entry(self, pdb_base_id):
        result = pdb_api.execute_raw_data_query(pdb_base_id)
        assert result["rcsb_id"] == pdb_base_id.upper()

    @pytest.mark.parametrize(
        ["pdb_base_id", "entity_id"], [["4HHB", "1"], ["4hhb", 2]],
    )
    def test_entity(self, pdb_base_id, entity_id):
        result = pdb_api.execute_raw_data_query(pdb_base_id, entity_id=entity_id)
        assert result["rcsb_id"] == f"{pdb_base_id.upper()}_{str(entity_id).upper()}"

    @pytest.mark.parametrize(
        ["pdb_base_id", "chain_id"], [["4HHB", "A"], ["4hhb", "b"]],
    )
    def test_chain(self, pdb_base_id, chain_id):
        result = pdb_api.execute_raw_data_query(pdb_base_id, chain_id=chain_id)
        assert result["rcsb_id"] == f"{pdb_base_id.upper()}.{str(chain_id).upper()}"

    @pytest.mark.parametrize(
        ["pdb_base_id", "chain_id", "entity_id", "ex_regex"],
        [
            [None, None, "1", "Must provide base"],
            ["", None, "1", "Must provide base"],
            ["4HHb:1", None, None, "must not include chain or entity"],
            ["4HHb_1", None, 1, "must not include chain or entity"],
            ["4HHb.A", "A", None, "must not include chain or entity"],
            ["4HHB", "A", "1", "not both"],
        ],
    )
    def test_invalid_inputs(self, pdb_base_id, chain_id, entity_id, ex_regex):
        with pytest.raises(ValueError, match=ex_regex):
            pdb_api.execute_raw_data_query(
                pdb_base_id, chain_id=chain_id, entity_id=entity_id
            )

    @pytest.mark.parametrize(
        ["pdb_base_id", "chain_id", "entity_id", "raise_on_error", "ex_regex"],
        [
            ["4HHB", "Z", None, True, "404"],
            ["4HHB", "", 12, True, "404"],
            #
            ["4HHB", "Z", None, False, ""],
            ["4HHB", "", 12, False, ""],
            #
            ["AAAA", None, None, True, "404"],
            ["BBBB", None, None, True, "404"],
            #
            ["333Z", None, None, False, ""],
            ["AAAAA", None, None, False, ""],
            #
        ],
    )
    def test_invalid_pdb_ids(
        self, pdb_base_id, chain_id, entity_id, raise_on_error, ex_regex
    ):

        if raise_on_error:
            expected_behaviour = pytest.raises(pdb_api.PDBAPIException, match=ex_regex)
            expected_result = None
        else:
            expected_behaviour = _does_not_raise()
            expected_result = {}

        actual_result = None
        with expected_behaviour:
            actual_result = pdb_api.execute_raw_data_query(
                pdb_base_id,
                chain_id=chain_id,
                entity_id=entity_id,
                raise_on_error=raise_on_error,
            )

        assert actual_result == expected_result


class TestPDBQuery(object):
    @pytest.fixture(autouse=True)
    def setup(self):
        pass

    def test_count(self):
        query = _BasicQuery(return_type=pdb_api.PDBQuery.ReturnType.ENTRY)
        count = query.count()
        results = query.execute()
        assert count > 170500
        assert len(results) == count

    @pytest.mark.parametrize(
        "return_type",
        [
            pdb_api.PDBQuery.ReturnType.ENTRY,
            pdb_api.PDBQuery.ReturnType.ENTITY,
            pdb_api.PDBQuery.ReturnType.CHAIN,
        ],
    )
    def test_return_type(self, return_type):
        query = _BasicQuery(return_type=return_type)
        results = query.execute()
        for i in range(100):
            base_id, chain_id, entity_id = split_id_with_entity(results[i])

            if return_type == pdb_api.PDBQuery.ReturnType.ENTRY:
                assert base_id
                assert not chain_id
                assert not entity_id

            if return_type == pdb_api.PDBQuery.ReturnType.CHAIN:
                assert base_id
                assert chain_id
                assert not entity_id

            if return_type == pdb_api.PDBQuery.ReturnType.ENTITY:
                assert base_id
                assert not chain_id
                assert entity_id


class TestPDBUnstructuredQuery(object):
    def test_search_structure_name(self):
        pdb_base_id = "4HHB"
        query = pdb_api.PDBUnstructuredSearchQuery(
            query_value=pdb_base_id, return_type=pdb_api.PDBQuery.ReturnType.CHAIN
        )
        results = query.execute()
        # This structure has 4 chains
        assert len(results) == 4
        for result in results:
            pdb_id, chain_id, entity_id = split_id_with_entity(result)
            assert pdb_id == pdb_base_id
            assert chain_id
            assert not entity_id

    @pytest.mark.parametrize(
        # Values obtained by manually searching using the website
        ["query_text", "min_results"],
        [["X-Ray Diffraction", 163085], ["Spodoptera", 6041]],
    )
    def test_search_full_text(self, query_text, min_results):
        query = pdb_api.PDBUnstructuredSearchQuery(
            query_value=query_text, return_type=pdb_api.PDBQuery.ReturnType.ENTRY
        )
        results = query.execute()
        assert len(results) >= min_results


class TestPDBAttributeSearchQuery:
    def test_numerical_field(self):
        query = pdb_api.PDBAttributeSearchQuery(
            attribute_name="rcsb_entry_info.diffrn_resolution_high.value",
            attribute_value=1.0,
            comparison_type="less",
            return_type=pdb_api.PDBQuery.ReturnType.ENTRY,
        )
        results = query.execute()
        assert len(results) >= 810

    def test_textual_field(self):
        query = pdb_api.PDBAttributeSearchQuery(
            attribute_name="rcsb_entity_source_organism.taxonomy_lineage.name",
            attribute_value="Spodoptera",
            comparison_type="contains_words",
            return_type=pdb_api.PDBQuery.ReturnType.ENTRY,
        )
        results = query.execute()
        assert len(results) >= 35


class TestPDBExpressionSystemQuery:
    def test_1(self):
        query = pdb_api.PDBExpressionSystemQuery(
            expr_sys="Spodoptera", comparison_type="exact_match"
        )
        results = query.execute()
        assert len(results) >= 7416


class TestPDBSourceTaxonomyIdQuery:
    def test_existing(self):
        query = pdb_api.PDBSourceTaxonomyIdQuery(taxonomy_id=1098)
        results = query.execute()
        assert len(results) >= 4

    def test_non_existing(self):
        query = pdb_api.PDBSourceTaxonomyIdQuery(taxonomy_id=10981234)
        assert len(query.execute()) == 0
        assert query.count() == 0


class TestPDBCompositeQuery:
    def test_and(self):
        query = pdb_api.PDBCompositeQuery(
            pdb_api.PDBExpressionSystemQuery("Homo Sapiens", "contains_phrase"),
            pdb_api.PDBSourceTaxonomyIdQuery(9606),
            logical_operator="and",
            return_type=pdb_api.PDBQuery.ReturnType.ENTRY,
        )

        results = query.execute()
        assert len(results) >= 2753

    def test_or(self):
        query = pdb_api.PDBCompositeQuery(
            pdb_api.PDBExpressionSystemQuery("Homo Sapiens", "contains_words"),
            pdb_api.PDBExpressionSystemQuery("Spodoptera", "exact_match"),
            logical_operator="or",
            return_type=pdb_api.PDBQuery.ReturnType.ENTRY,
        )

        results = query.execute()
        assert len(results) >= 9991

    def test_nested_composites(self):
        query = pdb_api.PDBCompositeQuery(
            pdb_api.PDBExpressionSystemQuery("Homo sapiens", "exact_match"),
            pdb_api.PDBXRayResolutionQuery(resolution=1.0, comparison_operator="less"),
            logical_operator="or",
            return_type=pdb_api.PDBQuery.ReturnType.ENTRY,
        )

        results = query.execute()
        assert len(results) >= 4991


class TestPDBXRayResolutionQuery:
    def test_1(self):
        query = pdb_api.PDBXRayResolutionQuery(
            resolution=1.1, return_type=pdb_api.PDBQuery.ReturnType.ENTITY
        )
        results = query.execute()
        assert len(results) >= 2798


class TestPDBRFreeQuery:
    def test_1(self):
        query = pdb_api.PDBRFreeQuery(
            rfree=0.1, return_type=pdb_api.PDBQuery.ReturnType.ENTITY
        )
        results = query.execute()
        assert len(results) >= 63
