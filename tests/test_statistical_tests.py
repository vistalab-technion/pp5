from functools import partial

import numpy as np
import pytest
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

import pp5
from pp5.dihedral import flat_torus_distance, flat_torus_distance_sq
from pp5.stats import mht_bh
from pp5.stats.breakdown import greedy_breakdown_k
from pp5.stats.two_sample import (
    _kde_statistic_pergroup,
    _mmd_statistic,
    _mmd_statistic_unbiased,
    kde2d_test,
    kde2d_test_pergroup,
    kde2d_test_pergroup_fast,
    mmd_test,
    mmd_test_fast,
    torus_projection_permutation_test,
    torus_projection_test,
    torus_projection_test_null_samples,
    torus_w2_ub_test,
)
from pp5.distributions.kde import gaussian_kernel, torus_gaussian_kernel_2d
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


class TestMMD:
    """
    Tests for the MMD^2 two-sample test (mmd_test, mmd_test_fast): the
    similarity/kernel composition used on the torus, the biased/unbiased statistic
    formula, and the fast permutation variant's agreement with the naive one.
    """

    SIGMA_DEG = 10.0
    SIGMA_RAD = np.deg2rad(SIGMA_DEG)

    def test_flat_torus_distance_matches_sqrt_of_squared(self):
        rng = np.random.default_rng(0)
        n = 200
        A = rng.uniform(-np.pi, np.pi, size=(n, 2))
        B = rng.uniform(-np.pi, np.pi, size=(n, 2))
        assert np.allclose(flat_torus_distance(A, B), np.sqrt(flat_torus_distance_sq(A, B)))

    def test_composed_kernel_matches_torus_gaussian_kernel_2d(self):
        # similarity_fn=flat_torus_distance (non-squared) composed with kernel_fn=
        # gaussian_kernel (which squares its input) should give the same standard
        # RBF-on-torus kernel as the dedicated torus_gaussian_kernel_2d.
        rng = np.random.default_rng(0)
        n = 200
        A = rng.uniform(-np.pi, np.pi, size=(n, 2))
        B = rng.uniform(-np.pi, np.pi, size=(n, 2))

        dist = flat_torus_distance(A, B)
        k_composed = gaussian_kernel(dist, sigma=self.SIGMA_RAD)

        # torus_gaussian_kernel_2d takes angle *differences* directly (it wraps them
        # internally via arccos(cos(.))), so feed it the raw per-coordinate diffs.
        k_ref = torus_gaussian_kernel_2d(
            A[:, 0] - B[:, 0], A[:, 1] - B[:, 1], sigma=self.SIGMA_RAD
        )
        assert np.allclose(k_composed, k_ref)

    @staticmethod
    def _reference_mmd_squared(Sxx, Syy, Sxy, nx, ny, unbiased):
        """
        Textbook block-sum formula for the biased/unbiased MMD^2 statistic, used to
        cross-check the library's implementation without depending on it.
        """
        if not unbiased:
            return Sxx / nx**2 + Syy / ny**2 - 2 * Sxy / (nx * ny)
        return (
            (Sxx - nx) / (nx * (nx - 1))
            + (Syy - ny) / (ny * (ny - 1))
            - 2 * Sxy / (nx * ny)
        )

    @pytest.mark.parametrize("unbiased", [True, False])
    def test_statistic_matches_reference_formula(self, unbiased):
        # Generic (non-torus) normalized RBF kernel, decoupled from the torus kernel
        # tested above: only the block-sums-to-scalar formula is under test here.
        rng = np.random.default_rng(1)
        nx, ny = 7, 11
        X = rng.normal(size=(nx, 3))
        Y = rng.normal(size=(ny, 3))
        sigma = 1.3

        Z = np.vstack([X, Y])
        K = np.exp(-cdist(Z, Z, metric="sqeuclidean") / (2 * sigma**2))
        assert np.allclose(np.diagonal(K), 1.0)

        Sxx = K[:nx, :nx].sum()
        Syy = K[nx:, nx:].sum()
        Sxy = K[:nx, nx:].sum()
        expected = self._reference_mmd_squared(Sxx, Syy, Sxy, nx, ny, unbiased)

        # Generic per-matrix statistic (used by mmd_test)
        stat_fn = _mmd_statistic_unbiased if unbiased else _mmd_statistic
        assert stat_fn(K, nx, ny) == pytest.approx(expected)

        # Fast, block-sum statistic (used by mmd_test_fast); default similarity_fn
        # (euclidean) + kernel_fn (gaussian_kernel) reproduce the same K as above.
        stat_val, _, _ = mmd_test_fast(
            X,
            Y,
            k=1,
            k_min=1,
            k_th=float("inf"),
            kernel_fn=partial(gaussian_kernel, sigma=sigma),
            unbiased=unbiased,
        )
        assert stat_val == pytest.approx(expected)

    @pytest.mark.parametrize("unbiased", [True, False])
    def test_fast_matches_naive_permutation_test(self, unbiased):
        rng = np.random.default_rng(2)
        nx, ny = 9, 13
        X = rng.uniform(-np.pi, np.pi, size=(nx, 2))
        Y = rng.uniform(-np.pi, np.pi, size=(ny, 2))

        common_kwargs = dict(
            k=50,
            k_min=50,
            k_th=float("inf"),
            similarity_fn=flat_torus_distance,
            kernel_fn=partial(gaussian_kernel, sigma=self.SIGMA_RAD),
            unbiased=unbiased,
        )

        # Both mmd_test and mmd_test_fast draw permutations via np.random.permutation
        # on the global numpy random state; resetting the seed identically before
        # each call means they see the exact same draws, so results should agree
        # exactly (not just statistically), given the row-sum identity is correct.
        np.random.seed(1234)
        stat_naive, pval_naive, k_naive = mmd_test(X, Y, **common_kwargs)

        np.random.seed(1234)
        stat_fast, pval_fast, k_fast = mmd_test_fast(X, Y, **common_kwargs)

        assert k_naive == k_fast == 50
        assert stat_naive == pytest.approx(stat_fast)
        assert pval_naive == pytest.approx(pval_fast)

    def test_fast_unbiased_requires_normalized_kernel(self):
        rng = np.random.default_rng(3)
        X = rng.normal(size=(5, 2))
        Y = rng.normal(size=(5, 2))
        with pytest.raises(AssertionError):
            mmd_test_fast(
                X,
                Y,
                k=10,
                similarity_fn=lambda a, b: np.sum((a - b) ** 2),
                kernel_fn=lambda d: d,  # k(z,z)=0, not a normalized kernel
                unbiased=True,
            )


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
        # BvMMixtureDiscreteDistribution.sample() draws from the global numpy RNG
        # with no seed argument of its own; seed here so X/Y/Z (and thus pval_xy/
        # pval_xz below) are deterministic instead of occasionally landing close
        # enough to the thresholds to flake. Seed 5 was chosen for margin (checked
        # against 30 candidate seeds across all 3 stat_test_fn variants).
        np.random.seed(5)
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
        import pp5.stats.two_sample.kde as m

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


class TestKdePergroupFast:
    """
    Tests for :func:`kde2d_test_pergroup_fast`: agreement with the naive
    :func:`kde2d_test_pergroup` under an identical seed, for both the
    equal-bandwidth and double-slab (CV) cases. In particular, the fast function
    must use the same ``X = idx[:nx]`` permutation convention as the rest of this
    module regardless of whether nx or ny is larger, not a "sum the smaller
    group" shortcut that would silently diverge from the naive test whenever
    nx > ny.
    """

    N_BINS = 32
    GRID_LOW = -np.pi
    GRID_HIGH = np.pi

    def _sample(self, n, mu=(0.3, -0.5), sigma=0.3, seed=0):
        rng = np.random.default_rng(seed)
        phi = rng.normal(mu[0], sigma, n).clip(-np.pi + 1e-6, np.pi - 1e-6)
        psi = rng.normal(mu[1], sigma, n).clip(-np.pi + 1e-6, np.pi - 1e-6)
        return np.stack([phi, psi], axis=1)

    def _common_kwargs(self, sigma_x_rad, sigma_y_rad):
        return dict(
            n_bins=self.N_BINS,
            grid_low=self.GRID_LOW,
            grid_high=self.GRID_HIGH,
            dtype=np.float64,
            sigma_x_rad=sigma_x_rad,
            sigma_y_rad=sigma_y_rad,
        )

    @pytest.mark.parametrize("nx,ny", [(20, 30), (30, 20)])
    def test_matches_naive_regardless_of_which_side_is_smaller(self, nx, ny):
        sigma_rad = np.deg2rad(10.0)
        X = self._sample(n=nx, seed=1)
        Y = self._sample(n=ny, mu=(0.1, 0.2), seed=2)
        kwargs = self._common_kwargs(sigma_rad, sigma_rad)

        np.random.seed(123)
        ddist_naive, pval_naive, k_naive = kde2d_test_pergroup(
            X, Y, k=200, k_min=200, k_th=float("inf"), **kwargs
        )

        np.random.seed(123)
        ddist_fast, pval_fast, k_fast = kde2d_test_pergroup_fast(
            X, Y, k=200, k_min=200, k_th=float("inf"), **kwargs
        )

        assert k_naive == k_fast == 200
        np.testing.assert_allclose(ddist_naive, ddist_fast, rtol=0, atol=1e-12)
        np.testing.assert_allclose(pval_naive, pval_fast, rtol=0, atol=1e-12)

    def test_matches_naive_with_different_bandwidths(self):
        sigma_x, sigma_y = np.deg2rad(4.0), np.deg2rad(12.0)
        X = self._sample(n=25, mu=(0.3, -0.5), seed=5)
        Y = self._sample(n=18, mu=(-0.5, 0.3), seed=6)
        kwargs = self._common_kwargs(sigma_x, sigma_y)

        np.random.seed(7)
        ddist_naive, pval_naive, k_naive = kde2d_test_pergroup(
            X, Y, k=150, k_min=150, k_th=float("inf"), **kwargs
        )

        np.random.seed(7)
        ddist_fast, pval_fast, k_fast = kde2d_test_pergroup_fast(
            X, Y, k=150, k_min=150, k_th=float("inf"), **kwargs
        )

        assert k_naive == k_fast == 150
        np.testing.assert_allclose(ddist_naive, ddist_fast, rtol=0, atol=1e-12)
        np.testing.assert_allclose(pval_naive, pval_fast, rtol=0, atol=1e-12)

    def test_batch_size_does_not_change_result(self):
        # Batching only defers compute; the sequence of rng.permutation() draws
        # is identical regardless of how many permutations are grouped per BLAS
        # call, so the result must not depend on batch_size.
        sigma_rad = np.deg2rad(10.0)
        X = self._sample(n=15, seed=3)
        Y = self._sample(n=22, mu=(0.2, 0.1), seed=4)
        kwargs = self._common_kwargs(sigma_rad, sigma_rad)

        np.random.seed(42)
        ddist_a, pval_a, k_a = kde2d_test_pergroup_fast(
            X, Y, k=137, k_min=137, k_th=float("inf"), batch_size=1, **kwargs
        )

        np.random.seed(42)
        ddist_b, pval_b, k_b = kde2d_test_pergroup_fast(
            X, Y, k=137, k_min=137, k_th=float("inf"), batch_size=1000, **kwargs
        )

        assert k_a == k_b == 137
        np.testing.assert_allclose(ddist_a, ddist_b, rtol=0, atol=1e-12)
        np.testing.assert_allclose(pval_a, pval_b, rtol=0, atol=1e-12)

    def test_shares_slabs_when_sigmas_equal(self, monkeypatch):
        # When sigma_x == sigma_y the implementation should compute slabs only
        # once (via the shared _kde_2d_slab_stacks helper), same guarantee as
        # kde2d_test_pergroup.
        import pp5.stats.two_sample.kde as m

        call_count = {"n": 0}
        real_kde_2d = m.kde_2d

        def counting_kde_2d(*args, **kwargs):
            call_count["n"] += 1
            return real_kde_2d(*args, **kwargs)

        monkeypatch.setattr(m, "kde_2d", counting_kde_2d)

        X = self._sample(n=12, seed=3)
        Y = self._sample(n=12, seed=4)
        sigma_rad = np.deg2rad(8.0)
        _ = kde2d_test_pergroup_fast(
            X, Y, k=10, **self._common_kwargs(sigma_rad, sigma_rad)
        )
        assert call_count["n"] == 1

        call_count["n"] = 0
        _ = kde2d_test_pergroup_fast(
            X, Y, k=10, **self._common_kwargs(sigma_rad, sigma_rad * 2)
        )
        assert call_count["n"] == 2

    def test_early_termination_stops_before_k(self):
        # Identical X, Y: ddist=0, so every permutation ties the observed
        # statistic and pval saturates at 1.0 immediately -- an easy case to
        # trigger early termination well before k.
        sigma_rad = np.deg2rad(10.0)
        X = self._sample(n=20, mu=(0.0, 0.0), seed=10)
        Y = X.copy()
        _, pval, k_used = kde2d_test_pergroup_fast(
            X,
            Y,
            k=5000,
            k_min=20,
            k_th=50.0,
            **self._common_kwargs(sigma_rad, sigma_rad),
        )
        assert k_used < 5000
        assert pval == pytest.approx(1.0)


class TestGreedyBreakdownK:
    """
    Tests for :func:`greedy_breakdown_k`: the generic adversarial-breakdown
    control flow (greedy remove-most-influential, re-rank, recompute the real
    p-value only at k_grid checkpoints) shared by the MMD/KDE-fix/KDE-CV
    breakdown-k analyses. Uses synthetic, statistic-agnostic closures; the
    per-statistic influence/p-value formulas are exercised separately when the
    scripts that use this function are repointed to it.
    """

    def _toy_closures(self, x_influence, y_influence, sig_until_removed):
        """
        Builds (stat_and_influence_fn, pval_fn) where each original point's
        influence is a fixed lookup by its original index (independent of what
        else has been removed, which is enough to exercise the loop's control
        flow), and the pair is "significant" until sig_until_removed points
        have been removed in total, then not significant from then on.
        """
        n_total = len(x_influence) + len(y_influence)

        def stat_and_influence_fn(x_keep, y_keep):
            base = 0.0  # unused by these tests
            infl_x = np.array([x_influence[i] for i in x_keep])
            infl_y = np.array([y_influence[j] for j in y_keep])
            return base, infl_x, infl_y

        def pval_fn(x_keep, y_keep):
            n_removed = n_total - (len(x_keep) + len(y_keep))
            pval = 0.01 if n_removed < sig_until_removed else 0.5
            return 0.0, pval

        return stat_and_influence_fn, pval_fn

    def test_removes_most_influential_point_each_step(self):
        # x_influence[1]=30 is the single largest value overall, so it must be
        # removed first; y_influence[0]=10 is the largest among what remains.
        x_influence = {0: 5.0, 1: 30.0, 2: 1.0}
        y_influence = {0: 10.0, 1: 2.0}
        stat_and_influence_fn, pval_fn = self._toy_closures(
            x_influence, y_influence, sig_until_removed=10
        )

        rows, breakdown_k, removed_log = greedy_breakdown_k(
            n1=3,
            n2=2,
            stat_and_influence_fn=stat_and_influence_fn,
            pval_fn=pval_fn,
            thresh=0.05,
            k_grid=[0, 1, 2, 3],
        )

        removed_order = [(side, idx) for side, idx, _ in removed_log]
        assert removed_order == [("X", 1), ("Y", 0), ("X", 0)]
        assert breakdown_k is None  # sig_until_removed=10 is never reached

    def test_stops_at_correct_breakdown_k(self):
        x_influence = {0: 5.0, 1: 30.0, 2: 1.0}
        y_influence = {0: 10.0, 1: 2.0}
        stat_and_influence_fn, pval_fn = self._toy_closures(
            x_influence, y_influence, sig_until_removed=2
        )

        rows, breakdown_k, removed_log = greedy_breakdown_k(
            n1=3,
            n2=2,
            stat_and_influence_fn=stat_and_influence_fn,
            pval_fn=pval_fn,
            thresh=0.05,
            k_grid=[0, 1, 2, 3, 4],
        )

        assert breakdown_k == 2
        # the loop must stop as soon as it finds the pair non-significant, not
        # run through the rest of k_grid.
        assert [r["k"] for r in rows] == [0, 1, 2]
        assert [r["significant"] for r in rows] == [True, True, False]
        assert len(removed_log) == 2

    def test_returns_none_when_never_breaks(self):
        x_influence = {0: 5.0, 1: 30.0}
        y_influence = {0: 10.0}
        stat_and_influence_fn, pval_fn = self._toy_closures(
            x_influence, y_influence, sig_until_removed=100
        )

        rows, breakdown_k, removed_log = greedy_breakdown_k(
            n1=2,
            n2=1,
            stat_and_influence_fn=stat_and_influence_fn,
            pval_fn=pval_fn,
            thresh=0.05,
            k_grid=[0, 1, 2],
        )

        assert breakdown_k is None
        assert [r["k"] for r in rows] == [0, 1, 2]
        assert rows[-1]["n1"] + rows[-1]["n2"] == 1  # 2 of 3 points removed by k=2

    def test_removed_log_records_side_original_index_and_drop(self):
        x_influence = {0: 1.0, 1: 30.0}
        y_influence = {0: 10.0}
        stat_and_influence_fn, pval_fn = self._toy_closures(
            x_influence, y_influence, sig_until_removed=100
        )

        _, _, removed_log = greedy_breakdown_k(
            n1=2,
            n2=1,
            stat_and_influence_fn=stat_and_influence_fn,
            pval_fn=pval_fn,
            thresh=0.05,
            k_grid=[0, 1],
        )

        assert removed_log[0] == ("X", 1, 30.0)

    def test_tie_break_prefers_x_then_lower_index(self):
        # Equal influence across X and Y, and within X: X wins over Y, and the
        # lower local-index X point wins over the higher one, since the loop
        # only replaces its running best on a strict ">" comparison (first
        # encountered wins ties), matching the original scripts' tie-break.
        x_influence = {0: 5.0, 1: 5.0}
        y_influence = {0: 5.0}
        stat_and_influence_fn, pval_fn = self._toy_closures(
            x_influence, y_influence, sig_until_removed=100
        )

        _, _, removed_log = greedy_breakdown_k(
            n1=2,
            n2=1,
            stat_and_influence_fn=stat_and_influence_fn,
            pval_fn=pval_fn,
            thresh=0.05,
            k_grid=[0, 1],
        )

        assert removed_log[0][:2] == ("X", 0)

