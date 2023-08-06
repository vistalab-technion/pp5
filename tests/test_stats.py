import numpy as np
import pytest
import matplotlib.pyplot as plt

from pp5.stats import mht_bh
from pp5.stats.two_sample import torus_w2_gof_test
from pp5.distributions.vonmises import BvMMixtureDiscreteDistribution


class TestMHTBH(object):
    Q = [0.05, 0.1]
    M = [10, 100]

    @pytest.fixture(autouse=True)
    def setup(self):
        pass

    @pytest.mark.parametrize("q", Q)
    @pytest.mark.parametrize("m", M)
    def test_pvals_equal_to_thresh(self, m, q):
        # pvals equal exactly to bh(q) line: the threshold should be the last
        pvals = (np.arange(m) + 1) * (q / m)
        t = mht_bh(q, pvals)
        assert t == pvals[-1]

    @pytest.mark.parametrize("q", Q)
    @pytest.mark.parametrize("m", M)
    def test_middle_pval_greater_than_thresh(self, m, q):
        bhq_thresh = (np.arange(m) + 1) * (q / m)

        pvals = np.copy(bhq_thresh)
        # Set a large pval at the end
        pvals[-1] = 0.9
        # Set a pval very slightly larger than threshold in the middle
        pvals[m // 2] *= 1.01

        t = mht_bh(q, pvals)

        # The chosen threshold should be one pval before last even though there was a
        # larger one in the middle
        assert t == bhq_thresh[-2]

    @pytest.mark.parametrize("q", Q)
    @pytest.mark.parametrize("m", M)
    def test_all_above(self, m, q):
        # all pvals are above the threshold
        bhq_thresh = (np.arange(m) + 1) * (q / m)
        pvals = bhq_thresh * 1.1
        t = mht_bh(q, pvals)
        assert t == 0.0

    @pytest.mark.parametrize("q", Q)
    @pytest.mark.parametrize("m", M)
    def test_all_below(self, m, q):
        # all pvals are below the threshold
        bhq_thresh = (np.arange(m) + 1) * (q / m)
        pvals = bhq_thresh * 0.9
        t = mht_bh(q, pvals)
        assert t == bhq_thresh[-1]

    @pytest.mark.parametrize("q", [-0.1, 0, 1, 1.1])
    def test_invalid_q(self, q):
        with pytest.raises(ValueError, match="q must be"):
            mht_bh(q, np.array([0.1, 0.2, 0.3]))

    @pytest.mark.parametrize("m", [0, 1])
    def test_invalid_m(self, m):
        with pytest.raises(ValueError, match="Need at least two"):
            mht_bh(0.1, np.arange(m).astype(float))


class TestTorusW2Test:
    @pytest.fixture
    def bvm_dist1(self, request):
        dist = BvMMixtureDiscreteDistribution(
            k1=0,
            k2=0,
            A=1,
            mu=[[0.5, 0.5]],
            # alpha=[1],
            gridsize=1024 * 1,
            two_pi=False,
        )
        return dist

    @pytest.fixture
    def bvm_dist2(self, request):
        dist = BvMMixtureDiscreteDistribution(
            k1=1,
            k2=1,
            A=2,
            mu=[[0.1, 0.1]],
            # alpha=[0.3, 0.7],
            gridsize=1024,
            two_pi=False,
        )
        return dist

    def test_1(self, bvm_dist1, bvm_dist2):
        X = bvm_dist1.sample(500)
        Y = bvm_dist1.sample(250)
        Z = bvm_dist2.sample(500)

        pval_xy = torus_w2_gof_test(X, Y)
        pval_xz = torus_w2_gof_test(X, Z)
        print(f"{pval_xy=},{pval_xz=}")

        # X, Y come from the same distribution
        assert pval_xy > 0.5

        # X, Z come from different distributions
        assert pval_xz < 0.1

    def test_uniformity(self, bvm_dist1):

        Ns = [10, 100, 200]
        M = 100
        pvals = np.empty((M,), dtype=float)
        for N in Ns:
            for i in range(M):
                X = bvm_dist1.sample(N)
                Y = bvm_dist1.sample(N)
                pvals[i] = torus_w2_gof_test(X, Y)

            plt.hist(pvals, bins=25, density=False, label=f"N={N}")

        plt.xlim([0, 1])
        plt.xlabel("p-value (UB)")
        plt.suptitle(rf"twosample.ubound.torus.test under $H_0$")
        plt.legend()
        plt.savefig("tests/out/pvals-ubound-synth_control.png", dpi=150)
        # plt.show()
