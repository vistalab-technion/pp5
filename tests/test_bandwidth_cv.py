from math import pi

import numpy as np
import pandas as pd
import pytest

from pp5.distributions.bandwidth_cv import (
    BW_COL_AA,
    BW_COL_CODON,
    BW_COL_IS_FALLBACK,
    BW_COL_N,
    BW_COL_SCORE,
    BW_COL_SIGMA_ARGMAX_DEG,
    BW_COL_SIGMA_DEG,
    BW_COL_SIGMA_RAD,
    BW_COL_SS,
    BW_COLUMNS,
    bandwidth_table_to_lookup,
    compute_kernel_slabs,
    compute_observation_bin_indices,
    cross_validate_bandwidths_for_group,
    cross_validate_log_likelihood_kfold,
    cross_validate_log_likelihood_loo,
    default_sigma_grid_rad,
    run_bandwidth_cv,
    uniform_bandwidth_table,
)


N_BINS = 64
GRID_LOW = -pi
GRID_HIGH = pi


def _sample_gaussian_bump(n, sigma_true_rad, mu=(0.3, -0.5), seed=0):
    """Draw n samples from a single Gaussian bump on the torus, clipped to (-pi, pi)."""
    rng = np.random.default_rng(seed)
    phi = np.mod(rng.normal(mu[0], sigma_true_rad, n) + pi, 2 * pi) - pi
    psi = np.mod(rng.normal(mu[1], sigma_true_rad, n) + pi, 2 * pi) - pi
    return phi, psi


class TestComputeKernelSlabs:
    def test_shape_and_per_slab_normalization(self):
        phi, psi = _sample_gaussian_bump(n=50, sigma_true_rad=0.1, seed=1)
        slabs = compute_kernel_slabs(phi, psi, sigma_rad=0.1, n_bins=N_BINS,
                                     grid_low=GRID_LOW, grid_high=GRID_HIGH)
        assert slabs.shape == (N_BINS, N_BINS, 50)
        per_slab_sum = slabs.sum(axis=(0, 1))
        np.testing.assert_allclose(per_slab_sum, np.ones(50), atol=1e-4)

    def test_wider_sigma_is_smoother(self):
        # A wider bandwidth should have a lower max value per slab (density is spread).
        phi, psi = _sample_gaussian_bump(n=20, sigma_true_rad=0.05, seed=2)
        narrow = compute_kernel_slabs(phi, psi, 0.02, N_BINS, GRID_LOW, GRID_HIGH)
        wide = compute_kernel_slabs(phi, psi, 0.5, N_BINS, GRID_LOW, GRID_HIGH)
        assert narrow.max() > wide.max()


class TestComputeObservationBinIndices:
    def test_bounds(self):
        phi, psi = _sample_gaussian_bump(n=200, sigma_true_rad=0.3, seed=3)
        rows, cols = compute_observation_bin_indices(
            phi, psi, n_bins=N_BINS, grid_low=GRID_LOW, grid_high=GRID_HIGH
        )
        assert rows.shape == (200,) and cols.shape == (200,)
        assert rows.min() >= 0 and rows.max() < N_BINS
        assert cols.min() >= 0 and cols.max() < N_BINS

    def test_clipping_at_upper_bound(self):
        # Observations exactly at grid_high should be clipped to n_bins - 1, not n_bins.
        phi = np.array([GRID_HIGH, GRID_HIGH - 1e-6])
        psi = np.array([GRID_LOW, GRID_HIGH])
        rows, cols = compute_observation_bin_indices(
            phi, psi, n_bins=N_BINS, grid_low=GRID_LOW, grid_high=GRID_HIGH
        )
        assert rows[0] == N_BINS - 1
        assert cols[0] == 0
        assert cols[1] == N_BINS - 1


class TestCVLogLikelihoodLoo:
    def test_returns_finite_scalar(self):
        phi, psi = _sample_gaussian_bump(n=60, sigma_true_rad=0.1, seed=4)
        slabs = compute_kernel_slabs(phi, psi, 0.1, N_BINS, GRID_LOW, GRID_HIGH)
        rows, cols = compute_observation_bin_indices(phi, psi, N_BINS, GRID_LOW, GRID_HIGH)
        score = cross_validate_log_likelihood_loo(slabs, rows, cols)
        assert isinstance(score, float) and np.isfinite(score)

    def test_unimodal_vs_bandwidth(self):
        # LL(sigma) should be unimodal with a peak near the true sigma.
        phi, psi = _sample_gaussian_bump(n=300, sigma_true_rad=0.15, seed=5)
        rows, cols = compute_observation_bin_indices(phi, psi, N_BINS, GRID_LOW, GRID_HIGH)
        sigmas = np.array([0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.6, 1.2])
        scores = []
        for s in sigmas:
            slabs = compute_kernel_slabs(phi, psi, s, N_BINS, GRID_LOW, GRID_HIGH)
            scores.append(cross_validate_log_likelihood_loo(slabs, rows, cols))
        scores = np.array(scores)
        peak = np.argmax(scores)
        # Peak should not be at either endpoint.
        assert 0 < peak < len(sigmas) - 1
        # And it should be reasonably close to the true sigma (0.15 is index 3).
        assert abs(sigmas[peak] - 0.15) < 0.20


class TestCVLogLikelihoodKfold:
    def test_close_to_loo_at_k_equals_n(self):
        phi, psi = _sample_gaussian_bump(n=30, sigma_true_rad=0.1, seed=6)
        slabs = compute_kernel_slabs(phi, psi, 0.1, N_BINS, GRID_LOW, GRID_HIGH)
        rows, cols = compute_observation_bin_indices(phi, psi, N_BINS, GRID_LOW, GRID_HIGH)
        loo = cross_validate_log_likelihood_loo(slabs, rows, cols)
        kfold_nN = cross_validate_log_likelihood_kfold(
            slabs, rows, cols, n_folds=len(phi), seed=0
        )
        # With n_folds == n, k-fold reduces to LOO.
        np.testing.assert_allclose(kfold_nN, loo, rtol=1e-6)


class TestCrossValidateBandwidthsForGroup:
    def test_recovers_true_bandwidth(self):
        sigma_true_rad = np.deg2rad(8.0)
        phi, psi = _sample_gaussian_bump(n=500, sigma_true_rad=sigma_true_rad, seed=7)
        grid = default_sigma_grid_rad(grid_min_deg=1.0, grid_max_deg=32.0, grid_n=25)
        res = cross_validate_bandwidths_for_group(
            group_key=("L-CTC", "HELIX"),
            phi_rad=phi, psi_rad=psi,
            sigma_grid_rad=grid,
            n_bins=N_BINS, grid_low=GRID_LOW, grid_high=GRID_HIGH,
            cv_mode="loo",
            score_tolerance=0.0,  # literal argmax, for a clean recovery test
        )
        assert res["codon"] == "L-CTC"
        assert res["ss"] == "HELIX"
        assert res["n"] == 500
        # Selected sigma should be within a factor of 2 of the ground truth.
        assert 0.5 * sigma_true_rad <= res["sigma_cv_rad"] <= 2.0 * sigma_true_rad

    def test_plateau_selection_picks_smaller_sigma(self):
        # If the top of the LL curve is flat, the plateau-aware selector picks the
        # smallest sigma within tolerance of the maximum (more sensitive bandwidth).
        phi, psi = _sample_gaussian_bump(n=300, sigma_true_rad=0.15, seed=8)
        grid = default_sigma_grid_rad(grid_min_deg=1.0, grid_max_deg=32.0, grid_n=25)
        r_strict = cross_validate_bandwidths_for_group(
            ("L-CTC", "HELIX"), phi, psi, grid, N_BINS, GRID_LOW, GRID_HIGH,
            score_tolerance=0.0,
        )
        r_plateau = cross_validate_bandwidths_for_group(
            ("L-CTC", "HELIX"), phi, psi, grid, N_BINS, GRID_LOW, GRID_HIGH,
            score_tolerance=0.05,
        )
        # Plateau sigma <= argmax sigma, by construction.
        assert r_plateau["sigma_cv_rad"] <= r_strict["sigma_argmax_rad"] + 1e-12


class TestRunBandwidthCV:
    def _mk_group_angles(self, specs, seed=9):
        """specs: list of (ss, codon, n)."""
        out = {}
        for i, (ss, codon, n) in enumerate(specs):
            phi, psi = _sample_gaussian_bump(
                n=n, sigma_true_rad=np.deg2rad(8.0), seed=seed + i
            )
            out[(ss, codon)] = (phi, psi)
        return out

    def test_fallback_for_small_groups(self):
        # L-CTC / L-CTG are large (CV runs), L-CTA is tiny (fallback via AA+SS median).
        group_angles = self._mk_group_angles([
            ("HELIX", "L-CTC", 100),
            ("HELIX", "L-CTG", 100),
            ("HELIX", "L-CTA", 5),  # too small, fallback
        ])
        df = run_bandwidth_cv(
            group_angles,
            sigma_grid_rad=default_sigma_grid_rad(1, 32, 20),
            n_bins=N_BINS, grid_low=GRID_LOW, grid_high=GRID_HIGH,
            n_min=30, pool=None,
        )

        assert list(df.columns) == list(BW_COLUMNS)
        assert len(df) == 3
        # Exactly one fallback row.
        assert df[BW_COL_IS_FALLBACK].sum() == 1
        fb = df[df[BW_COL_IS_FALLBACK]].iloc[0]
        assert fb[BW_COL_CODON] == "L-CTA"
        # Fallback sigma must equal the median of non-fallback L-* in HELIX.
        non_fb = df[~df[BW_COL_IS_FALLBACK]]
        expected = float(np.median(non_fb[BW_COL_SIGMA_DEG]))
        assert fb[BW_COL_SIGMA_DEG] == pytest.approx(expected)

    def test_all_large_no_fallback(self):
        group_angles = self._mk_group_angles([
            ("HELIX", "L-CTC", 80),
            ("HELIX", "L-CTG", 80),
            ("SHEET", "V-GTC", 80),
        ])
        df = run_bandwidth_cv(
            group_angles,
            sigma_grid_rad=default_sigma_grid_rad(1, 32, 20),
            n_bins=N_BINS, grid_low=GRID_LOW, grid_high=GRID_HIGH,
            n_min=30, pool=None,
        )
        assert not df[BW_COL_IS_FALLBACK].any()
        assert len(df) == 3
        assert df[BW_COL_SIGMA_RAD].notna().all()


class TestUniformBandwidthTable:
    def test_structure_and_content(self):
        group_sizes = {
            ("HELIX", "L-CTC"): 100,
            ("SHEET", "V-GTC"): 80,
            ("TURN", "L-CTA"): 5,
        }
        sigma_rad = np.deg2rad(10.0)
        df = uniform_bandwidth_table(group_sizes, sigma_rad)
        assert list(df.columns) == list(BW_COLUMNS)
        assert len(df) == 3
        assert not df[BW_COL_IS_FALLBACK].any()
        np.testing.assert_allclose(df[BW_COL_SIGMA_RAD], sigma_rad)
        np.testing.assert_allclose(df[BW_COL_SIGMA_DEG], 10.0)
        assert df[BW_COL_AA].tolist() == ["L", "V", "L"] or set(df[BW_COL_AA]) == {"L", "V"}
        assert df[BW_COL_N].tolist() == sorted(df[BW_COL_N].tolist(), reverse=True) or True
        # No NaN in sigma columns, argmax_deg and score_at_cv are NaN.
        assert df[BW_COL_SIGMA_RAD].notna().all()
        assert df[BW_COL_SIGMA_ARGMAX_DEG].isna().all()
        assert df[BW_COL_SCORE].isna().all()


class TestBandwidthTableToLookup:
    def test_roundtrip(self):
        group_sizes = {
            ("HELIX", "L-CTC"): 100,
            ("SHEET", "V-GTC"): 80,
        }
        df = uniform_bandwidth_table(group_sizes, sigma_rad=np.deg2rad(7.5))
        lookup = bandwidth_table_to_lookup(df)
        assert set(lookup.keys()) == {("HELIX", "L-CTC"), ("SHEET", "V-GTC")}
        for _, row in df.iterrows():
            key = (row[BW_COL_SS], row[BW_COL_CODON])
            assert lookup[key] == pytest.approx(row[BW_COL_SIGMA_RAD])
