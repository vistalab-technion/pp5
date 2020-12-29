import pytest

from pp5.align import StructuralAlignment, pymol


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
    def test_align_to_self(self, backbone_only, with_chain):
        pdb_id = "2WUR"
        if with_chain:
            pdb_id += ":A"
        sa = StructuralAlignment(pdb_id, pdb_id, backbone_only=backbone_only)
        assert sa.rmse == 0
