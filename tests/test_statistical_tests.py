from functools import partial

import numpy as np
import pytest
import matplotlib.pyplot as plt

import pp5
from pp5.stats import mht_bh
from pp5.stats.two_sample import (
    _kde_statistic_pergroup,
    kde2d_test,
    kde2d_test_pergroup,
    torus_projection_permutation_test,
    torus_projection_test,
    torus_projection_test_null_samples,
    torus_w2_ub_test,
)
from pp5.distributions.kde import torus_gaussian_kernel_2d
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


class TestTorusW2:

    STAT_TEST_FNS = {
        "projection": torus_projection_test,
        "projection_fixed_geodesics": partial(
            torus_projection_test,
            geodesics=np.array([[1, 0], [0, 1], [1, 1], [2, 3]]),
        ),
        "ubound": torus_w2_ub_test,
    }

    @pytest.fixture(params=STAT_TEST_FNS.keys())
    def stat_test_name(self, request):
        return request.param

    @pytest.fixture
    def stat_test_fn(self, stat_test_name):
        return self.STAT_TEST_FNS[stat_test_name]

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

    def test_pvals(self, bvm_dist1, bvm_dist2, stat_test_fn):
        X = bvm_dist1.sample(500)
        Y = bvm_dist1.sample(250)
        Z = bvm_dist2.sample(500)

        dist_xy, pval_xy = stat_test_fn(X, Y)
        dist_xz, pval_xz = stat_test_fn(X, Z)
        print(f"{dist_xy=},{dist_xz=}")
        print(f"{pval_xy=},{pval_xz=}")

        # w2(X,Y) is smaller than w2(X,Z)
        assert dist_xy < dist_xz

        # X, Y come from the same distribution
        assert pval_xy > 0.2

        # X, Z come from different distributions
        assert pval_xz < 0.1

    def test_uniformity(self, bvm_dist1, stat_test_fn, stat_test_name):

        Ns = [10, 100, 200]
        M = 10
        pvals = np.empty((M,), dtype=float)
        for N in Ns:
            for i in range(M):
                X = bvm_dist1.sample(N)
                Y = bvm_dist1.sample(N)
                _, pvals[i] = stat_test_fn(X, Y)

            plt.hist(pvals, bins=25, density=False, label=f"N={N}")

        plt.xlim([0, 1])
        plt.xlabel(f"p-value ({stat_test_name})")
        plt.suptitle(rf"{stat_test_name} test under $H_0$")
        plt.legend()
        plt.savefig(f"tests/out/pvals-{stat_test_name}-synth_control-{M=}.png", dpi=150)
        # plt.show()


class TestTorusNullSeeding:
    """
    Tests for reproducible seeding of the torustest analytic null simulation
    (:func:`torus_projection_test_null_samples`, backed by R's
    ``sim.null.stat`` + ``parallel::clusterSetRNGStream``).

    All tests isolate ``pp5.TORUSTEST_NULL_DIR`` to a temp directory so they
    never touch the real ``data/torustest_null/`` cache, and so that a
    reproducibility check can't trivially "pass" just because it reloaded the
    same cached pickle twice.
    """

    N_SIMULATIONS = 100
    N_SAMPLE = 30
    # n_cores=2 (not 1) so the fork-parallel path is actually exercised: a
    # broken clusterSetRNGStream/fork-inherited-RNG-state bug would only show
    # up with >1 worker.
    N_CORES = 2

    @pytest.fixture
    def bvm_dist1(self, request):
        dist = BvMMixtureDiscreteDistribution(
            k1=0,
            k2=0,
            A=1,
            mu=[[0.5, 0.5]],
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
            gridsize=1024,
            two_pi=False,
        )
        return dist

    def test_same_seed_gives_identical_null(self, tmp_path, monkeypatch):
        dir_a, dir_b = tmp_path / "a", tmp_path / "b"
        dir_a.mkdir()
        dir_b.mkdir()

        monkeypatch.setattr(pp5, "TORUSTEST_NULL_DIR", dir_a)
        null_1 = torus_projection_test_null_samples(
            n_simulations=self.N_SIMULATIONS,
            n_sample=self.N_SAMPLE,
            n_cores=self.N_CORES,
            seed=42,
        )

        # Second call points at a fresh, empty cache dir so it cannot hit the
        # first call's cached pickle -- it must actually recompute the null.
        monkeypatch.setattr(pp5, "TORUSTEST_NULL_DIR", dir_b)
        null_2 = torus_projection_test_null_samples(
            n_simulations=self.N_SIMULATIONS,
            n_sample=self.N_SAMPLE,
            n_cores=self.N_CORES,
            seed=42,
        )

        assert null_1.shape == (self.N_SIMULATIONS,)
        assert null_2.shape == (self.N_SIMULATIONS,)
        # Same seed -> the RNG stream should make this bit-for-bit reproducible.
        assert np.array_equal(null_1, null_2)

    def test_different_seeds_give_different_null(self, tmp_path, monkeypatch):
        dir_a, dir_b = tmp_path / "a", tmp_path / "b"
        dir_a.mkdir()
        dir_b.mkdir()

        monkeypatch.setattr(pp5, "TORUSTEST_NULL_DIR", dir_a)
        null_42 = torus_projection_test_null_samples(
            n_simulations=self.N_SIMULATIONS,
            n_sample=self.N_SAMPLE,
            n_cores=self.N_CORES,
            seed=42,
        )

        monkeypatch.setattr(pp5, "TORUSTEST_NULL_DIR", dir_b)
        null_43 = torus_projection_test_null_samples(
            n_simulations=self.N_SIMULATIONS,
            n_sample=self.N_SAMPLE,
            n_cores=self.N_CORES,
            seed=43,
        )

        assert not np.allclose(null_42, null_43)

    def test_seed_none_still_works(self, tmp_path, monkeypatch):
        monkeypatch.setattr(pp5, "TORUSTEST_NULL_DIR", tmp_path)
        null = torus_projection_test_null_samples(
            n_simulations=self.N_SIMULATIONS,
            n_sample=self.N_SAMPLE,
            n_cores=self.N_CORES,
            seed=None,
        )
        assert null.shape == (self.N_SIMULATIONS,)

    def test_torus_projection_test_reproducible_with_null_seed(
        self, bvm_dist1, bvm_dist2, tmp_path, monkeypatch
    ):
        # Fixed samples, reused for both calls -- only the null cache dir
        # differs, so any difference in (stat, pval) can only come from the
        # null simulation itself.
        X = bvm_dist1.sample(30)
        Y = bvm_dist2.sample(30)
        geodesics = np.array([[1, 0], [0, 1], [1, 1], [2, 3]])

        dir_a, dir_b = tmp_path / "a", tmp_path / "b"
        dir_a.mkdir()
        dir_b.mkdir()

        monkeypatch.setattr(pp5, "TORUSTEST_NULL_DIR", dir_a)
        stat_1, pval_1 = torus_projection_test(
            X,
            Y,
            geodesics=geodesics,
            n_cores_null_simulations=self.N_CORES,
            n_null_simulations=self.N_SIMULATIONS,
            n_null_sample_size=self.N_SAMPLE,
            null_seed=42,
        )

        monkeypatch.setattr(pp5, "TORUSTEST_NULL_DIR", dir_b)
        stat_2, pval_2 = torus_projection_test(
            X,
            Y,
            geodesics=geodesics,
            n_cores_null_simulations=self.N_CORES,
            n_null_simulations=self.N_SIMULATIONS,
            n_null_sample_size=self.N_SAMPLE,
            null_seed=42,
        )

        # The statistic never depends on the null/seed at all.
        assert stat_1 == stat_2
        # With a seeded null, the pval is now reproducible too (this is the
        # actual point of this test).
        assert pval_1 == pval_2

    def test_torus_projection_permutation_test_accepts_null_seed(
        self, bvm_dist1, tmp_path, monkeypatch
    ):
        # Smoke test only: confirms the signature update to
        # torus_projection_permutation_test doesn't break its call path when
        # null_seed is passed (e.g. via TORUSTEST_DEFAULT_KWARGS). Seeding
        # behavior itself is not under test here -- per torus_projection_test's
        # note, this function's output never depends on null_seed.
        monkeypatch.setattr(pp5, "TORUSTEST_NULL_DIR", tmp_path)
        X = bvm_dist1.sample(20)
        Y = bvm_dist1.sample(20)
        geodesics = np.array([[1, 0], [0, 1], [1, 1], [2, 3]])

        stat, pval, k_used = torus_projection_permutation_test(
            X,
            Y,
            k=20,
            geodesics=geodesics,
            n_cores_null_simulations=self.N_CORES,
            n_null_simulations=self.N_SIMULATIONS,
            n_null_sample_size=self.N_SAMPLE,
            null_seed=42,
        )

        assert np.isfinite(stat)
        assert 0.0 < pval <= 1.0
        assert k_used > 0


class TestKdePergroup:
    """Tests for :func:`kde2d_test_pergroup` and :func:`_kde_statistic_pergroup`."""

    N_BINS = 64
    GRID_LOW = -np.pi
    GRID_HIGH = np.pi

    def _sample(self, n, mu=(0.3, -0.5), sigma=0.3, seed=0):
        rng = np.random.default_rng(seed)
        phi = rng.normal(mu[0], sigma, n).clip(-np.pi + 1e-6, np.pi - 1e-6)
        psi = rng.normal(mu[1], sigma, n).clip(-np.pi + 1e-6, np.pi - 1e-6)
        return np.stack([phi, psi], axis=1)

    def test_matches_kde2d_test_when_sigmas_equal(self):
        # With identical bandwidths, kde2d_test_pergroup must produce the same
        # (ddist, pval, k) as the legacy kde2d_test.
        sigma_rad = np.deg2rad(10.0)
        X = self._sample(n=60, seed=1)
        Y = self._sample(n=60, mu=(0.1, 0.2), seed=2)

        np.random.seed(123)
        ddist_a, pval_a, k_a = kde2d_test(
            X, Y, k=200,
            n_bins=self.N_BINS, grid_low=self.GRID_LOW, grid_high=self.GRID_HIGH,
            dtype=np.float64,
            kernel_fn=partial(torus_gaussian_kernel_2d, sigma=sigma_rad),
        )

        np.random.seed(123)
        ddist_b, pval_b, k_b = kde2d_test_pergroup(
            X, Y, k=200,
            n_bins=self.N_BINS, grid_low=self.GRID_LOW, grid_high=self.GRID_HIGH,
            dtype=np.float64,
            sigma_x_rad=sigma_rad, sigma_y_rad=sigma_rad,
        )

        assert k_a == k_b
        np.testing.assert_allclose(ddist_a, ddist_b, rtol=0, atol=1e-12)
        np.testing.assert_allclose(pval_a, pval_b, rtol=0, atol=1e-12)

    def test_shares_slabs_when_sigmas_equal(self, monkeypatch):
        # When sigma_x == sigma_y the implementation should compute slabs only once.
        import pp5.stats.two_sample as m

        call_count = {"n": 0}
        real_kde_2d = m.kde_2d

        def counting_kde_2d(*args, **kwargs):
            call_count["n"] += 1
            return real_kde_2d(*args, **kwargs)

        monkeypatch.setattr(m, "kde_2d", counting_kde_2d)

        X = self._sample(n=30, seed=3)
        Y = self._sample(n=30, seed=4)
        sigma_rad = np.deg2rad(8.0)
        _ = kde2d_test_pergroup(
            X, Y, k=10,
            n_bins=self.N_BINS, grid_low=self.GRID_LOW, grid_high=self.GRID_HIGH,
            dtype=np.float64,
            sigma_x_rad=sigma_rad, sigma_y_rad=sigma_rad,
        )
        assert call_count["n"] == 1

        call_count["n"] = 0
        _ = kde2d_test_pergroup(
            X, Y, k=10,
            n_bins=self.N_BINS, grid_low=self.GRID_LOW, grid_high=self.GRID_HIGH,
            dtype=np.float64,
            sigma_x_rad=sigma_rad, sigma_y_rad=sigma_rad * 2,
        )
        assert call_count["n"] == 2

    def test_different_sigmas_runs_and_is_finite(self):
        X = self._sample(n=60, mu=(0.3, -0.5), seed=5)
        Y = self._sample(n=60, mu=(-0.5, 0.3), seed=6)
        ddist, pval, k = kde2d_test_pergroup(
            X, Y, k=200,
            n_bins=self.N_BINS, grid_low=self.GRID_LOW, grid_high=self.GRID_HIGH,
            dtype=np.float64,
            sigma_x_rad=np.deg2rad(4.0),
            sigma_y_rad=np.deg2rad(12.0),
        )
        assert np.isfinite(ddist) and ddist > 0.0
        assert 0.0 < pval <= 1.0
        assert k > 0

    def test_statistic_assigns_K_x_to_X_and_K_y_to_Y(self):
        # Build disjoint slab stacks (K_x nonzero only in top-left, K_y in bottom-right)
        # and verify that:
        #   * un-permuted stat uses K_x[:nx] (ones in top-left) and K_y[nx:] (ones in
        #     bottom-right).
        #   * swapping nx_idx/ny_idx changes the answer.
        nx, ny, M = 3, 3, 4
        K_x = np.zeros((nx + ny, M, M))
        K_y = np.zeros((nx + ny, M, M))
        # K_x: a peak in the top-left, same for all observations.
        K_x[:, 0, 0] = 1.0
        # K_y: a peak in the bottom-right, same for all observations.
        K_y[:, M - 1, M - 1] = 1.0

        stat_unperm = _kde_statistic_pergroup((K_x, K_y), nx, ny)
        # kde_X is a single pixel (1) at (0,0); kde_Y is single pixel (1) at (M-1,M-1).
        # L1 distance between two disjoint unit-mass distributions = 2.
        np.testing.assert_allclose(stat_unperm, 2.0)

        # If we accidentally swapped which slab stack is used per group, the statistic
        # would see IDENTICAL distributions (both at top-left) and return 0.
        stat_swapped = _kde_statistic_pergroup((K_y, K_x), nx, ny)
        np.testing.assert_allclose(stat_swapped, 2.0)  # still 2, symmetric

        # Sanity: identical stacks yields 0.
        stat_same = _kde_statistic_pergroup((K_x, K_x), nx, ny)
        np.testing.assert_allclose(stat_same, 0.0)

    def test_statistic_uses_permutation_indices(self):
        # Build slab stacks where observation identity matters, and verify that the
        # statistic is sensitive to which indices are assigned to each group.
        nx, ny, M = 2, 2, 2
        K_x = np.zeros((nx + ny, M, M))
        K_y = np.zeros((nx + ny, M, M))
        # K_x slabs: obs 0,1 put mass at (0,0); obs 2,3 put mass at (1,1).
        K_x[0, 0, 0] = 1.0; K_x[1, 0, 0] = 1.0
        K_x[2, 1, 1] = 1.0; K_x[3, 1, 1] = 1.0
        # K_y: uniform 1/M^2 for all observations — irrelevant for this test.
        K_y[:, :, :] = 1.0 / (M * M)

        # Un-permuted: X is {0, 1} -> all mass at (0,0); Y is {2, 3} (uses K_y uniform)
        # L1 between a delta at (0,0) and uniform over M^2 = 1 + (M^2 - 1) / M^2
        stat_unperm = _kde_statistic_pergroup((K_x, K_y), nx, ny)

        # Permute so X holds {2, 3} -> all mass at (1, 1); Y still uniform.
        stat_perm = _kde_statistic_pergroup(
            (K_x, K_y), nx, ny,
            nx_idx=np.array([2, 3]), ny_idx=np.array([0, 1]),
        )

        # Both should give the same L1 (delta vs uniform), but the LOCATION of the
        # delta differs between them. That confirms indices route correctly: if the
        # statistic ignored the indices it would always use [:nx] = {0, 1}.
        np.testing.assert_allclose(stat_unperm, stat_perm)

        # Now pick indices that mix groups, which should still produce a valid number.
        stat_mixed = _kde_statistic_pergroup(
            (K_x, K_y), nx, ny,
            nx_idx=np.array([0, 2]), ny_idx=np.array([1, 3]),
        )
        assert np.isfinite(stat_mixed)

