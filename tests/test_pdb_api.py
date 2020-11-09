import pytest

import pp5.external_dbs.pdb_api as pdb_api
from pp5.external_dbs.pdb import split_id_with_entity


class _BasicQuery(pdb_api.PDBQuery):
    def __init__(self, **base_kwargs):
        super().__init__(**base_kwargs)

    def _raw_query_data(self) -> dict:
        return {"type": "terminal", "service": "text"}

    def description(self) -> str:
        return "Basic query"


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
