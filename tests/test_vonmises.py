import numpy as np
import pytest
from pytest import approx, raises

from pp5.distributions.vonmises import (
    BvMKernelDensityEstimator,
    BvMMixtureDiscreteDistribution,
)


class TestBvMMixtureDiscreteDistribution:
    @pytest.fixture(autouse=False, params=[True, False], ids=["2pi", "-pi"])
    def setup(self, request):
        self.dist = BvMMixtureDiscreteDistribution(
            k1=[1, 2],
            k2=[2, 3],
            A=[[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
            mu=[[1, 2], [3, 4]],
            alpha=[0.3, 0.7],
            gridsize=512,
            two_pi=request.param,
        )

    @pytest.mark.parametrize(
        ["gridsize", "two_pi"],
        [
            (128, True),
            (512, False),
        ],
    )
    @pytest.mark.parametrize(
        ["k1", "k2", "A", "mu", "alpha"],
        [
            (1, 2, 3, None, None),
            (1, [2], 3, None, None),
            (1, [2], [[4, 5], [6, 7]], None, None),
            (1, 2, 3, [[1, 2]], 1.0),
            ([1], [2], [3], [[1, 2]], 1.0),
            ([1, 2], [2, 3], [3, 4], [[1, 2], [3, 4]], [0.3, 0.7]),
            (
                [1, 2],
                [2, 3],
                [[[3, 4], [5, 6]], [[7, 8], [9, 10]]],
                [[1, 2], [3, 4]],
                [0.3, 0.7],
            ),
            (
                [1, 2],
                [2, 3],
                [np.eye(2), np.eye(2)],
                [[1, 2], [3, 4]],
                [0.3, 0.7],
            ),
            (
                np.array([1, 2]),
                np.array([2, 3]),
                np.array([np.eye(2), np.eye(2)]),
                np.array([[1, 2], [3, 4]]),
                np.array([0.3, 0.7]),
            ),
        ],
    )
    def test_init(self, k1, k2, A, mu, alpha, gridsize, two_pi):
        dist = BvMMixtureDiscreteDistribution(k1, k2, A, mu, alpha, gridsize, two_pi)
        assert len(dist.grid) == gridsize

        if two_pi:
            grid_expected_min, grid_expected_max = (0.0, 2 * np.pi)
        else:
            grid_expected_min, grid_expected_max = (-np.pi, np.pi)
        grid_min, grid_max = dist.grid[0], dist.grid[-1] + 2 * np.pi / gridsize
        assert (grid_min, grid_max) == approx((grid_expected_min, grid_expected_max))

        assert dist.pdf.shape == (gridsize, gridsize)
        assert np.sum(dist.pdf) == approx(1.0)

    @pytest.mark.parametrize(
        ["k1", "k2", "A", "mu", "alpha"],
        [
            (1, 2, 3, None, 2.0),
            (1, 2, 3, 4, 1.0),
            (
                [1, 2],
                [2, 3],
                [3, 4],
                [[1, 2], [3, 4]],
                [0.3, 0.3, 0.4],
            ),
            (
                [1, 2],
                [2, 3],
                [3, 4],
                [[1, 2], [3, 4]],
                [0.3, 0.8],
            ),
            (
                [1, 2],
                [2, 3],
                [3, 4],
                [[1, 2, 3], [3, 4, 5]],
                [0.3, 0.7],
            ),
            (
                [1, 2],
                [2, 3],
                [3, 4],
                [np.eye(3), np.eye(3)],
                [0.3, 0.7],
            ),
            (
                [1, 2],
                [2, 3],
                [3, 4],
                np.zeros((3, 4, 5, 6)),
                [0.3, 0.7],
            ),
            (
                [1, 2, 4],
                [2, 3],
                [3, 4],
                [[1, 2], [4, 5]],
                [0.3, 0.7],
            ),
        ],
    )
    def test_init_fail(self, k1, k2, A, mu, alpha):
        with raises(AssertionError):
            BvMMixtureDiscreteDistribution(k1, k2, A, mu, alpha)

    @pytest.mark.parametrize("n", [1, 10, 100, 1000])
    def test_sample(self, setup, n):
        samples = self.dist.sample(n)
        min_sample, max_sample = np.min(samples), np.max(samples)

        assert samples.shape == (n, 2)
        assert min_sample >= self.dist.grid[0]
        assert max_sample <= (self.dist.grid[-1] + 2 * np.pi / self.dist.gridsize)

    @pytest.mark.parametrize("n", [0, 10])
    def test_plot_number(self, setup, n):
        fig, ax = self.dist.plot(samples=n)


class TestBvMKernelDensityEstimator(object):
    @pytest.mark.parametrize(
        ["n_bins", "k1", "k2", "k3", "batchsize"],
        [
            (0, 1, 2, 3, 16),
            (1, 1, 2, 3, 16),
            (100, None, 2, 3, 16),
            (100, 1, None, 3, 16),
            (100, 1, 2, None, 16),
        ],
    )
    def test_init_fail(self, n_bins, k1, k2, k3, batchsize):
        with pytest.raises(AssertionError):
            BvMKernelDensityEstimator(n_bins, k1, k2, k3, batchsize)

    def test_estimate(self):
        kde = BvMKernelDensityEstimator(
            n_bins=256, k1=10, k2=20, k3=-2, dtype=np.float64, batchsize=64
        )
        phi_psi = np.random.uniform(-np.pi, np.pi, size=(500, 2))
        p = kde(phi_psi[:, 0], phi_psi[:, 1])

        assert p.shape == (kde.n_bins, kde.n_bins)
        assert p.dtype == kde.dtype

    @pytest.mark.parametrize("n_samples", [100, 1000, 5000])
    @pytest.mark.parametrize("grid_size", [128, 256])
    def test_estimate_real_mixture(self, grid_size, n_samples):
        bvm = BvMMixtureDiscreteDistribution(
            k1=[0.7, 1.0],
            k2=[1.3, 3.3],
            A=[2, -1],
            mu=[[1, 2], [-0.1, -0.2]],
            alpha=[0.3, 0.7],
            two_pi=False,
            gridsize=grid_size,
        )
        samples = bvm.sample(n_samples)
        kde = BvMKernelDensityEstimator(
            n_bins=grid_size,
            k1=30,
            k2=30,
            k3=2,
        )
        kde_pdf = kde(samples[:, 0], samples[:, 1])

        dist_norm = np.linalg.norm(bvm.pdf - kde_pdf)

        assert dist_norm < 1e-2
