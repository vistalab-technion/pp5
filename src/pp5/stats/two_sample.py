from typing import Tuple, Callable, Optional

import numba
import numpy as np
import rpy2.robjects as robjects
import rpy2.robjects.numpy2ri
from numpy import ndarray
from scipy.spatial.distance import pdist, squareform, sqeuclidean

from pp5.distributions.kde import kde_2d, gaussian_kernel

_NUMBA_PARALLEL = False

# Names of R functions we use from the torustest R package.
R_TORUSTEST_GEODESIC = "twosample.geodesic.torus.test"
R_TORUSTEST_UBOUND = "twosample.ubound.torus.test"


# @numba.jit(nopython=True, parallel=_NUMBA_PARALLEL)
def _tw2_statistic(D: ndarray, nx: int, ny: int, nx_idx=None, ny_idx=None) -> float:
    """
    Calculates T statistic of a distance matrix
    :param D: Matrix of squared distances of two pooled samples (X and Y) of
        shape (nx+ny, nx+ny).
    :param nx: Number of observations from X.
    :param ny: Number of observations from Y.
    :return: The T2 statistic.
    """
    factor = (nx + ny) / nx / ny
    sum_X = np.sum(D[0:nx, 0:nx])
    sum_Y = np.sum(D[nx:, nx:])
    sum_Z = np.sum(D)
    enumerator = sum_Z / (nx + ny) - sum_X / nx - sum_Y / ny
    denumerator = (sum_X / (nx**2) / (nx - 1)) + (sum_Y / (ny**2) / (ny - 1))
    if denumerator < 1e-12:  # prevent division by zero
        return 0.0
    return float(factor * enumerator / denumerator)


# @numba.jit(nopython=True, parallel=_NUMBA_PARALLEL)
def _mmd_statistic(K: np.ndarray, nx: int, ny: int, nx_idx=None, ny_idx=None) -> float:
    """
    Calculates MMD statistic of a kernel matrix

    :param K: Matrix of inner products of two pooled samples (X and Y) of
        shape (nx+ny, nx+ny).
    :param nx: Number of observations from X.
    :param ny: Number of observations from Y.
    :return: The MMD statistic.
    """
    sum_X = np.sum(K[0:nx, 0:nx]) / nx / nx
    sum_Y = np.sum(K[nx:, nx:]) / ny / ny
    sum_XY = np.sum(K[nx:, 0:nx]) / nx / ny
    return float(sum_X + sum_Y - 2.0 * sum_XY)


# @numba.jit(nopython=True, parallel=_NUMBA_PARALLEL)
def _kde_statistic(
    K: np.ndarray,
    nx: int,
    ny: int,
    nx_idx: Optional[np.ndarray] = None,
    ny_idx: Optional[np.ndarray] = None,
) -> float:
    """
    Calculates KDE-based statistic of a kernel matrix

    :param K: Matrix of shape (nx+ny, M, M), containing the (M, M) contributions of
        N=nx+ny observations to the KDE estimate.
    :param nx: Number of observations from X.
    :param ny: Number of observations from Y.
    :return: The KDE statistic: L1 distance between the KDEs of X and Y.
    """

    # Apply a reduction to compute X and Y's KDEs from the contribution of each
    # of their observations
    if nx_idx is not None and ny_idx is not None:
        kde_X = K[nx_idx]
        kde_Y = K[ny_idx]
    else:
        kde_X = K[:nx]
        kde_Y = K[nx:]

    kde_X = np.sum(kde_X, axis=0)  # (nx, M, M) -> (M, M)
    kde_X /= np.sum(kde_X)

    kde_Y = np.sum(kde_Y, axis=0)  # (ny, M, M) -> (M, M)
    kde_Y /= np.sum(kde_Y)

    # w2_dist = w2_dist_sinkhorn(kde_X, kde_Y, sigma=1e-5, niter=250)[0]
    l1_dist = np.sum(np.abs(kde_X - kde_Y)).item()
    return l1_dist


def tw_test(
    X: ndarray,
    Y: ndarray,
    k: int,
    similarity_fn: Callable[[ndarray, ndarray], float] = sqeuclidean,
    kernel_fn: Callable[[ndarray], ndarray] = lambda x: x,
    k_min: Optional[int] = None,
    k_th: Optional[float] = float("inf"),
) -> Tuple[float, float, int]:
    """
    Applies a two-sample permutation test to determine whether the null hypothesis
    that two distributions are identical can be rejected, using the Tw^2 Welch
    statistic based on pairwise squared-euclidean distances.

    For parameters, see documentation of :obj:`two_sample_kernel_permutation_test`.

    :return: Tw^2 statistic value, p-value (significance).
    """
    return two_sample_kernel_permutation_test(
        X,
        Y,
        k,
        similarity_fn=similarity_fn,
        kernel_fn=kernel_fn,
        statistic_fn=_tw2_statistic,
        k_min=k_min,
        k_th=k_th,
    )


def mmd_test(
    X: ndarray,
    Y: ndarray,
    k: int,
    similarity_fn: Callable[[ndarray, ndarray], float] = sqeuclidean,
    kernel_fn: Callable[[ndarray], ndarray] = gaussian_kernel,
    k_min: Optional[int] = None,
    k_th: Optional[float] = float("inf"),
) -> Tuple[float, float, int]:
    """
    Applies a two-sample permutation test to determine whether the null hypothesis
    that two distributions are identical can be rejected, using the MMD approach.

    For parameters, see documentation of :obj:`two_sample_kernel_permutation_test`.

    :return: MMD statistic value, p-value (significance).
    """
    return two_sample_kernel_permutation_test(
        X,
        Y,
        k,
        similarity_fn=similarity_fn,
        kernel_fn=kernel_fn,
        statistic_fn=_mmd_statistic,
        k_min=k_min,
        k_th=k_th,
    )


def kde2d_test(
    X: ndarray,
    Y: ndarray,
    k: int,
    n_bins: int,
    grid_low: float,
    grid_high: float,
    dtype: np.dtype,
    kernel_fn: Callable,
    k_min: Optional[int] = None,
    k_th: Optional[float] = float("inf"),
) -> Tuple[float, float, int]:
    """
    Applies a two-sample permutation test to determine whether the null hypothesis
    that two distributions are identical can be rejected, using the KDE approach.

    For parameters, see documentation of :obj:`two_sample_kernel_permutation_test`.

    :param n_bins: Number of bins for KDE estimation.
    :param grid_low: Smallest value on the evaluation grid, inclusive.
    :param grid_high: Largest value on the evaluation grid, exclusive.
    :param kernel_fn: Kernel for the 2D KDE (not for the permutation test itself).
    :return: KDE statistic value, p-value (significance).
    """

    def _kde_2d_kernel_fn(Z: np.ndarray):
        # Z has shape (N, 2)

        K = kde_2d(
            x1=Z[:, 0],
            x2=Z[:, 1],
            kernel_fn=kernel_fn,
            n_bins=n_bins,
            grid_low=grid_low,
            grid_high=grid_high,
            dtype=dtype,
            # Disabling reduction is necessary to avoid re-calculating the entire KDE
            # on each permutation.
            reduce=False,
        )

        # Transpose from (M, M, N) to (N, M, M) where N=nx+ny, so that we can permute
        # over the first dimension.
        return K.transpose(2, 0, 1)

    return two_sample_kernel_permutation_test(
        X,
        Y,
        k,
        similarity_fn=None,
        kernel_fn=_kde_2d_kernel_fn,
        statistic_fn=_kde_statistic,
        k_min=k_min,
        k_th=k_th,
    )


def two_sample_kernel_permutation_test(
    X: ndarray,
    Y: ndarray,
    k: int,
    similarity_fn: Optional[Callable[[ndarray, ndarray], float]],
    kernel_fn: Callable[[ndarray], ndarray],
    statistic_fn: Callable[[ndarray, int, int], float],
    k_min: Optional[int] = None,
    k_th: Optional[float] = float("inf"),
) -> Tuple[float, float, int]:
    """
    Applies a two-sample permutation test to determine whether the null hypothesis
    that two distributions are identical can be rejected.

    If a similarity_fn is provided, the observations will be transformed using a Kernel
    function of the form K(X, Y) = k(h(x,y)), where h(x, y) is a scalar similarity
    function, and k(z) is a scalar univariate kernel to be applied on the similarity
    metric.
    For example,
        - RBF kernel K(x, y): Set h(x, y) = ||x-y||^2 and k(z) = exp(Ɣ z^2/σ^2).
        - Polynomial kernel K(x, y): Set h(x, y) = x^T y and k(z) = (Ɣ z + r)^d.
        - Linear kernel K(x, y): Set h(x, y) = x^T y and k(z) = z.
    The observation from X, Y will be pooled into Z = [X; Y], and a Gram matrix K
    will be computed, such that K[i,j] = k(h(z_i, z_j)).

    If similarity_fn is not provided, the the kernel function k(z) will be applied to
    all samples (from X and Y). The kernel doesn't have to be a scalar function in
    this case. The resulting matrix K will be of shape (nx+ny, M) where M is the output
    dimension of the kernel.

    A test-statistic s(Z; n_x, n_y) will be applied to K and to permutations of K which
    mix between the groups X and Y, where n_x and n_y are the number of observations
    in X and Y respectively.

    :param X: (n_x, m) array containing a sample X, where n_x is the number of
        observations in the sample and m is the dimension of each observation.
    :param Y: (n_y, m) array containing sample Y with n_y observations of dimension m.
    :param k: number of permutations for significance evaluation
    :param similarity_fn: h(x, y), a scalar bivariate similarity function.
    :param kernel_fn: k(z), a scalar univariate kernel function.
        The full bivariate kernel will be K(x,y)=k(h(x,y)).
    :param statistic_fn: A callable describing the statistic.
    :param k_min: Minimal number of permutations to run. Setting this to a
        truthy value enables early termination: when the number of permutations k
        exceeds this number and pvalue >= ddist_k_th * 1/(k+1), no more
        permutations will be performed.
    :param k_th: Early termination threshold for permutation test. Can be
        thought of as a factor of the smallest pvalue 1/(k+1). I.e. if k_th=50,
        then if after k_min permutations the pvalue is 50 times larger than it's
        smallest possible value - terminate.
    :return: Tuple containing:
        - statistic value for (X, Y)
        - p-value (significance) for the null-hypothesis that P_X = P_Y
        - number of permutations that were performed (could be less than k if early
          termination was used).
    """
    # sample sizes
    nx = X.shape[0]
    ny = Y.shape[0]
    if nx < 2 or ny < 2:
        raise ValueError(
            "Permutation test requires at least two observations in each sample"
        )

    assert k > 0
    assert k_th > 0
    k_min = k_min or k

    # pooled vectors
    Z = np.vstack((X, Y))  # (nx+ny, m)

    # pairwise distances
    if similarity_fn is not None:
        # D is (nx+ny, nx+ny)
        D = squareform(pdist(Z, metric=similarity_fn))  # type:ignore
        permute_pairs = True
    else:
        # D is (nx+ny, m)
        D = Z
        permute_pairs = False

    # inner products
    K = kernel_fn(D)  # in general can be (nx+ny, m')

    return _two_sample_kernel_permutation_test_inner(
        K, nx, ny, k, statistic_fn, permute_pairs, k_min, k_th
    )


# @numba.jit(nopython=True, parallel=_NUMBA_PARALLEL)
def _two_sample_kernel_permutation_test_inner(
    K: ndarray,
    nx: int,
    ny: int,
    k: int,
    statistic_fn: Callable[[ndarray, int, int], float],
    permute_pairs: bool,
    k_min: int,
    k_th: float,
) -> Tuple[float, float, int]:
    """
    Calculates p-value for an H0 of P_X=P_Y, based on permutation testing.

    :param K: Observations matrix of of two pooled samples (X and Y) of shape
        (N, M, *) where '*' means any number of additional dims, and N=nx+ny the
        total number of observations in the sample X and sample Y combined.
    :param nx: Number of observations from X.
    :param ny: Number of observations from Y.
    :param k: Number of permutations for significance evaluation.
    :param statistic_fn: Test statistic function.
        A callable taking an observations array K, nx, and ny, and returning a measure of
        similarity between the sample X and the sample Y.
    :param k_min: Early termination min permutations.
    :param k_th: Early termination threshold.
    :param permute_pairs: If False, K will be permuted on first axes only,
        i.e. K[p, ...]. If True, then K will be treated as a Gram matrix of pairwise
        distances and permuted as K[p,:][:,p], where p are permuted indices.
    :return: Statistic value of un-permuted distances, p-value for an H0 of P_X=P_Y,
        and the number of permutations that were evaluated.
    """

    # Value of statistic on the un-permuted data
    stat_val = statistic_fn(K, nx, ny)

    pval, count, curr_permutation = 0, 0, 0
    for curr_permutation in range(1, k + 1):
        idx = np.random.permutation(nx + ny)

        if permute_pairs:
            K_perm = K[idx, :][:, idx]
            stat_val_perm = statistic_fn(K_perm, nx, ny)
        else:
            nx_idx = idx[:nx]
            ny_idx = idx[nx:]
            stat_val_perm = statistic_fn(K, nx, ny, nx_idx, ny_idx)

        if stat_val <= stat_val_perm:
            count += 1

        # The smallest pval this test can detect is 1/(k+1).
        # When count is zero, pval should be 1/(i+1).
        pval = (count + 1) / (curr_permutation + 1)

        # Early termination criterion: minimal number of permutations reached,
        # and pval is larger than some factor (k_th) times the smallest possible pvalue.
        if (curr_permutation >= k_min) and (pval >= k_th * 1 / (curr_permutation + 1)):
            break

    # Calculate pval, and make sure it's not zero (it's possible that no iteration
    # produced stat_val <= stat_val_perm, but that doesn't mean the true pval is zero).
    return stat_val, pval, curr_permutation


def torus_w2_ub_test(X: ndarray, Y: ndarray) -> float:
    """

    Two-sample test for the torus using the Wasserstein-2 distance. Returns an upper
    bound for the pvalue.

    Uses code from: https://github.com/gonzalez-delgado/torustest

    González-Delgado J, González-Sanz A, Cortés J, Neuvial P: Two-sample
    goodness-of-fit tests on the flat torus based on Wasserstein distance and their
    relevance to structural biology. Electron. J. Statist., 17(1): 1547–1586, 2023.

    :param X: First sample observations, of shape (n, 2).
    :param Y: Second sample observations, of shape (n, 2).
    :return: Upper bound of p-value for the null hypothesis that X and Y are samples
        from the same distribution.
    """

    # Scale X, Y from [-pi,pi) x [-pi,pi) to [0,1) x [0,1]
    X = (X + np.pi) / (2 * np.pi)
    Y = (Y + np.pi) / (2 * np.pi)

    # Get R-function to invoke for performing the test
    test_fn_r = robjects.globalenv[R_TORUSTEST_UBOUND]

    # Create a converter that converts np.ndarray to R array
    np_conversion = robjects.default_converter + robjects.numpy2ri.converter
    with robjects.conversion.localconverter(np_conversion) as cv:
        pval = test_fn_r(X, Y)
        return pval.item()
