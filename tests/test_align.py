import pytest

from pp5.align import PYMOL_ALIGN_SYMBOL, StructuralAlignment, pymol
from pp5.external_dbs.pdb import PDB_AFLD, PDB_DOWNLOAD_SOURCES


class TestStructuralAlign(object):
    @pytest.fixture(autouse=True)
    def setup(self):
        pymol_pre_state = pymol.get_names("all")
        yield

        # Make sure that after each test pymol's state is unchanged
        pymol_post_state = pymol.get_names("all")
        assert pymol_pre_state == pymol_post_state

    @pytest.mark.parametrize("with_chain", [False, True])
    @pytest.mark.parametrize("backbone_only", [False, True])
    @pytest.mark.parametrize("pdb_source", PDB_DOWNLOAD_SOURCES.keys())
    def test_align_to_self(self, backbone_only, with_chain, pdb_source):
        pdb_id = "2WUR"
        if with_chain or pdb_source == PDB_AFLD:
            pdb_id += ":A"
        sa = StructuralAlignment(pdb_id, pdb_id, backbone_only=backbone_only)
        assert sa.rmse == 0

    def test_outlier_rejection_cutoff_example(self):
        pdb1, pdb2 = "4NE4:A", "5TEU:A"
        sa_20 = StructuralAlignment(
            pdb1, pdb2, backbone_only=True, outlier_rejection_cutoff=2.0
        )
        sa_25 = StructuralAlignment(
            pdb1, pdb2, backbone_only=True, outlier_rejection_cutoff=2.5
        )

        sub_seq = "KADD"
        sub_idx = sa_20.aligned_seq_1.index(sub_seq)
        assert sa_25.aligned_seq_1.index(sub_seq) == sub_idx

        sub_slice = slice(sub_idx, sub_idx + len(sub_seq))
        assert sa_20.aligned_stars[sub_slice] != PYMOL_ALIGN_SYMBOL * len(sub_seq)
        assert sa_25.aligned_stars[sub_slice] == PYMOL_ALIGN_SYMBOL * len(sub_seq)

    @pytest.mark.parametrize("backbone_only", [False, True])
    @pytest.mark.parametrize("outlier_rejection_cutoff", [2.0, 2.5])
    @pytest.mark.parametrize("pdb_source", PDB_DOWNLOAD_SOURCES.keys())
    def test_cache(self, backbone_only, outlier_rejection_cutoff, pdb_source):
        pdb1, pdb2 = "4NE4:A", "5TEU:A"

        # Should not exist in cache
        sa_cached = StructuralAlignment.from_cache(
            pdb1,
            pdb2,
            pdb_source=pdb_source,
            backbone_only=backbone_only,
            outlier_rejection_cutoff=outlier_rejection_cutoff,
        )
        assert sa_cached is None

        # Should be created and saved to cache
        sa = StructuralAlignment.from_pdb(
            pdb1,
            pdb2,
            cache=True,
            pdb_source=pdb_source,
            backbone_only=backbone_only,
            outlier_rejection_cutoff=outlier_rejection_cutoff,
        )

        # Should exist in cache
        sa_cached = StructuralAlignment.from_cache(
            pdb1,
            pdb2,
            pdb_source=pdb_source,
            backbone_only=backbone_only,
            outlier_rejection_cutoff=outlier_rejection_cutoff,
        )
        assert sa_cached is not None

        # Cached version should be the same
        assert sa == sa_cached
