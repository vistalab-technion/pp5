import numpy as np
import pytest

from pp5.stats import mht_bh


class TestMHT_BH(object):
    @pytest.fixture(autouse=True)
    def setup(self):
        pass

    @pytest.mark.parametrize("q", [0.05, 0.1])
    @pytest.mark.parametrize("m", [10, 100])
    def test_linear_1(self, m, q):
        # pvals equal exactly to bh(q) line: the threshold should be the last
        pvals = (np.arange(m) + 1) * (q / m)
        t = mht_bh(q, pvals)
        assert t == pvals[-1]

    @pytest.mark.parametrize("q", [0.05, 0.1])
    @pytest.mark.parametrize("m", [10, 100])
    def test_linear_2(self, m, q):
        # pvals equal exactly to bh(q) line: the threshold should be the last
        bhq_thresh = (np.arange(m) + 1) * (q / m)

        pvals = np.copy(bhq_thresh)
        # Set a large pval at the end
        pvals[-1] = 0.9
        # Set a pval very slightly larger than threshold in the middle
        pvals[m // 2] *= 1.01

        t = mht_bh(q, pvals)

        # The chosen threshold should be one pval before last even though there was a
        # larger one in the middle
        assert t == pvals[-2]

    @pytest.mark.parametrize("q", [-0.1, 0, 1, 1.1])
    def test_invalid_q(self, q):
        with pytest.raises(ValueError, match="q must be"):
            mht_bh(q, np.array([0.1, 0.2, 0.3]))

    @pytest.mark.parametrize("m", [0, 1])
    def test_invalid_m(self, m):
        with pytest.raises(ValueError, match="Need at least two"):
            mht_bh(0.1, np.arange(m).astype(np.float))
