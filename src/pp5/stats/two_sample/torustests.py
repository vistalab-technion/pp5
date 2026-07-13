"""Two-sample tests on the flat torus (Wasserstein-2 based), via the vendored
R ``torustest`` package."""
import logging
import pickle
from typing import Optional, Tuple

import numpy as np
import rpy2.robjects as robjects
import rpy2.robjects.numpy2ri
from filelock import FileLock
from numpy import ndarray

import pp5
from pp5.stats.two_sample.common import two_sample_kernel_permutation_test

_LOG = logging.getLogger(__name__)

# Names of R functions we use from the torustest R package.
R_TORUSTEST_GEODESIC = "twosample.geodesic.torus.test"
R_TORUSTEST_UBOUND = "twosample.ubound.torus.test"
R_TORUSTEST_SIM_NULL_STAT = "sim.null.stat"

# Setup conversion between python and R objects
PY2R_CONVERTER = robjects.default_converter + robjects.numpy2ri.converter
PY2R_CONVERTER.py2rpy.register(type(None), lambda _: robjects.NULL)


def torus_w2_ub_test(
    X: ndarray,
    Y: ndarray,
    grid_low: float = -np.pi,
    grid_high: float = np.pi,
) -> Tuple[float, float]:
    """

    Two-sample test for the torus using the Wasserstein-2 distance. The computed
    p-value is an upper bound of the pvalue for the null hypothesis that X and Y are
    samples from the same distribution.

    Uses code from: https://github.com/gonzalez-delgado/torustest

    González-Delgado J, González-Sanz A, Cortés J, Neuvial P: Two-sample
    goodness-of-fit tests on the flat torus based on Wasserstein distance and their
    relevance to structural biology. Electron. J. Statist., 17(1): 1547–1586, 2023.

    :param X: First sample observations, of shape (n, 2).
    :param Y: Second sample observations, of shape (n, 2).
    :param grid_low: Smallest value on the evaluation grid, inclusive.
    :param grid_high: Largest value on the evaluation grid, exclusive.
    :return: Tuple containing:
    - w2 distance
    - pvalue (upper bound)
    """

    # Scale X, Y from e.g. [-pi,pi) x [-pi,pi) to [0,1) x [0,1]
    def _scale(Z: ndarray) -> ndarray:
        return (Z - grid_low) / (grid_high - grid_low)

    X, Y = _scale(X), _scale(Y)

    # Get R-function to invoke for performing the test
    test_fn_r = robjects.globalenv[R_TORUSTEST_UBOUND]

    # Create a converter that converts np.ndarray to R array
    with robjects.conversion.localconverter(PY2R_CONVERTER) as cv:
        result = test_fn_r(X, Y, return_stat=True)
        return result["stat"].item(), result["pval"].item()


def torus_projection_test(
    X: ndarray,
    Y: ndarray,
    grid_low: float = -np.pi,
    grid_high: float = np.pi,
    n_cores: int = 2,
    n_geodesics: int = 2,
    geodesics: Optional[np.ndarray] = None,
    n_cores_null_simulations: int = 8,
    n_null_simulations: int = 2000,
    n_null_sample_size: int = 30,
    null_seed: Optional[int] = 42,
) -> Tuple[float, float]:
    """

    Two-sample test for the torus using the projections onto closed geodesics.
    The computed p-value is global pvalue obtained by n_gedesics(min(pvals))
    where pvals are the per-projection p-values.

    Uses code from: https://github.com/gonzalez-delgado/torustest

    González-Delgado J, González-Sanz A, Cortés J, Neuvial P: Two-sample
    goodness-of-fit tests on the flat torus based on Wasserstein distance and their
    relevance to structural biology. Electron. J. Statist., 17(1): 1547–1586, 2023.

    :param X: First sample observations, of shape (n, 2).
    :param Y: Second sample observations, of shape (n, 2).
    :param grid_low: Smallest value on the evaluation grid, inclusive.
    :param grid_high: Largest value on the evaluation grid, exclusive.
    :param n_cores: Number of cores to use for running on multiple processes.
    :param n_geodesics: Number of geodesics lines to sample.
    :param geodesics: An (n, 2) array. Each row is a vector defining a geodesic line on
        the torus.
    :param n_cores_null_simulations: Number of cores to use for null simulations.
    :param n_null_simulations: Number of simulations to run for the null distribution.
    :param n_null_sample_size: Number of samples to use for each null simulation.
    :param null_seed: Optional seed for reproducible null simulation. If None, the
        simulation is unseeded (previous behavior) and results will differ across cache
        misses. If provided, the simulated null (and thus any p-value derived from it)
        is reproducible across environments and re-runs, for the same seed and
        parameters. See :obj:`torus_projection_test_null_samples`.
    :return: Tuple containing:
    - w2 distance (mean over all projections)
    - pvalue (upper bound)
    """

    # Scale X, Y from e.g. [-pi,pi) x [-pi,pi) to [0,1) x [0,1]
    def _scale(Z: ndarray) -> ndarray:
        return (Z - grid_low) / (grid_high - grid_low)

    X, Y = _scale(X), _scale(Y)

    # Get R-function to invoke for performing the test
    test_fn_r = robjects.globalenv[R_TORUSTEST_GEODESIC]

    sim_null_dist = torus_projection_test_null_samples(
        n_simulations=n_null_simulations,
        n_sample=n_null_sample_size,
        n_cores=n_cores_null_simulations,
        seed=null_seed,
    )

    if geodesics is not None:
        n_geodesics, _2 = geodesics.shape
        assert _2 == 2

    # Create a converter that converts np.ndarray to R array
    with robjects.conversion.localconverter(PY2R_CONVERTER) as cv:
        result = test_fn_r(
            sample_1=X,
            sample_2=Y,
            n_geodesics=n_geodesics,
            NC_geodesic=n_cores,
            geodesic_list=geodesics,
            sim_null=sim_null_dist,
            return_stat=True,
        )
        return result["stat"].item(), result["pval"].item()


def torus_projection_test_null_samples(
    n_simulations: int,
    n_sample: int,
    n_cores: int = 8,
    seed: Optional[int] = 42,
) -> np.ndarray:
    """
    Sample from the null distribution of the wasserstein statistic on S^1.

    :param n_simulations: Number of simulations to perform.
    :param n_sample: Sample size in each simulation.
    :param n_cores: Number of cores to use for running on multiple processes.
    :param seed: Optional seed for reproducible null simulation. If None,
        the simulation is unseeded and results will differ across cache misses (new
        machine, new cache, different n_simulations/ n_sample). If provided, the
        simulated null (and thus any p-value derived from it) is reproducible across
        environments and re-runs, for the same seed and parameters.
    :return: An array of simulated wasserstein statistics, of shape (n_simulations,).
    """

    seed_suffix = f"_seed{seed}" if seed is not None else ""
    filename = f"torustest_null_{n_simulations:05d}_{n_sample:05d}{seed_suffix}.pkl"
    filepath = pp5.TORUSTEST_NULL_DIR / filename
    lock_filepath = str(filepath).replace(".pkl", ".lock")
    sim_null_fn_r = robjects.globalenv[R_TORUSTEST_SIM_NULL_STAT]

    with FileLock(lock_filepath):
        if filepath.exists():
            with open(filepath, "rb") as f:
                sim_null_dist = pickle.load(f)

        else:
            with robjects.conversion.localconverter(PY2R_CONVERTER) as cv:
                _LOG.info(
                    f"Calculating torustest null distribution {n_simulations=}, "
                    f"{n_sample=}, {seed=}..."
                )
                sim_null_dist = sim_null_fn_r(
                    NR=n_simulations, NC=n_cores, n=n_sample, seed=seed
                )
                with open(filepath, "wb") as f:
                    pickle.dump(sim_null_dist, f)
                _LOG.info(f"Saved torustest null distribution to {filepath}")

        return sim_null_dist


def torus_projection_permutation_test(
    X: ndarray,
    Y: ndarray,
    # params for permutations:
    k: int,
    k_min: Optional[int] = None,
    k_th: Optional[float] = float("inf"),
    # params for single torus projection test:
    grid_low: float = -np.pi,
    grid_high: float = np.pi,
    n_cores: int = 2,
    n_geodesics: int = 2,
    geodesics: Optional[np.ndarray] = None,
    n_cores_null_simulations: int = 8,
    n_null_simulations: int = 2000,
    n_null_sample_size: int = 30,
    null_seed: Optional[int] = None,
    #
) -> Tuple[float, float, int]:
    """
    Applies a two-sample permutation test to determine whether the null hypothesis
    that two distributions are identical can be rejected, using the test statistic from
    the projected torus test.

    For parameters, see documentation of :obj:`two_sample_kernel_permutation_test`
    and :obj:`torus_projection_test`.

    Note: null_seed has no effect on this function's returned statistic/p-value. The
    inner `torus_projection_test` call's p-value is discarded here (only its test
    statistic, which does not depend on the null distribution/seed at all, is used for
    permutations). It is accepted here purely so that this function accepts the same
    kwargs as :obj:`torus_projection_test` (needed since ``TORUSTEST_DEFAULT_KWARGS`` is
    splatted into both).
    """

    # Helper function to adapt between the input from the permutation test and the torus
    # projection test
    def _torus_projection_statistic(
        K: np.ndarray,
        nx: int,
        ny: int,
        nx_idx: Optional[np.ndarray] = None,
        ny_idx: Optional[np.ndarray] = None,
    ) -> float:
        # K will have shape (Nx+Ny, 2)
        if nx_idx is not None and ny_idx is not None:
            X = K[nx_idx, :]
            Y = K[ny_idx, :]
        else:
            X = K[:nx, :]
            Y = K[nx:, :]

        assert X.shape == (nx, 2)
        assert Y.shape == (ny, 2)

        stat, _pval = torus_projection_test(
            X,
            Y,
            grid_low=grid_low,
            grid_high=grid_high,
            n_cores=n_cores,
            n_geodesics=n_geodesics,
            geodesics=geodesics,
            n_cores_null_simulations=n_cores_null_simulations,
            n_null_simulations=n_null_simulations,
            n_null_sample_size=n_null_sample_size,
            null_seed=null_seed,
        )

        # Ignore pval, use test statistic for permutations
        return stat

    return two_sample_kernel_permutation_test(
        X,
        Y,
        k,
        statistic_fn=_torus_projection_statistic,
        # Disable similarity function and kernel, so statistic_fn will just see the
        # original samples
        similarity_fn=None,
        kernel_fn=None,
        k_min=k_min,
        k_th=k_th,
    )
