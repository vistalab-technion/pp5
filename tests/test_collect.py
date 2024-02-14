from random import randint
from pathlib import Path

import pytest

import pp5
from tests import get_tmp_path
from pp5.collect import (
    DATASET_DIRNAME,
    ALL_STRUCTS_FILENAME,
    COLLECTION_METADATA_FILENAME,
    ProteinRecordCollector,
)


class TestPrecCollector(object):
    @pytest.fixture(scope="class")
    def collection_nproc(self):
        return 4

    @pytest.fixture(scope="class")
    def collection_out_dir(self):
        return get_tmp_path("prec-collected-tests")

    @pytest.fixture(scope="class")
    def collection_out_tag(self):
        return f"tag-{randint(0, 1000)}"

    @pytest.fixture(scope="class")
    def collection_result(
        self, collection_nproc, collection_out_dir, collection_out_tag
    ):
        pp5.set_config("MAX_PROCESSES", collection_nproc)

        collector = ProteinRecordCollector(
            resolution=0.75,
            with_altlocs=True,
            with_contacts=True,
            with_backbone=True,
            entity_single_chain=False,
            seq_similarity_thresh=1.0,
            write_zip=True,
            out_dir=collection_out_dir,
            out_tag=collection_out_tag,
        )

        return collector.collect()

    def test_collection_result(
        self, collection_result, collection_out_dir, collection_out_tag
    ):
        assert collection_result["n_collected"] > 10
        assert collection_result["n_query_results"] > 10
        assert collection_result["n_entries"] > 2000
        assert collection_result["out_tag"] == collection_out_tag
        for step_result in collection_result["steps"]:
            assert "SUCCESS" in step_result

        out_dir = Path(collection_result["out_dir"])
        assert out_dir.is_dir()
        assert out_dir.is_relative_to(collection_out_dir)

        assert (out_dir / DATASET_DIRNAME).is_dir()
        assert (out_dir / COLLECTION_METADATA_FILENAME).is_file()
        assert (out_dir / f"{ALL_STRUCTS_FILENAME}.csv").is_file()

        collection_id = out_dir.name
        assert (out_dir / f"{collection_id}.zip").is_file()

        csv_files = tuple((out_dir / DATASET_DIRNAME).glob("*.csv"))
        assert collection_result["n_collected_filtered"] == len(csv_files)
