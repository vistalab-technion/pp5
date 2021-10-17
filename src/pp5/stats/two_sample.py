from typing import Any, Tuple, Callable, Optional

import numba
import numpy as np
from numpy import ndarray
from scipy.spatial.distance import pdist, squareform, sqeuclidean

from pp5.vonmises import kde_2d

_NUMBA_PARALLEL = False


@numba.jit(nopython=True, parallel=_NUMBA_PARALLEL)
def _tw2_statistic(D: ndarray, nx: int, ny: int) -> float:
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
    denumerator = (sum_X / (nx ** 2) / (nx - 1)) + (sum_Y / (ny ** 2) / (ny - 1))
    if denumerator < 1e-12:  # prevent division by zero
        return 0.0
    return float(factor * enumerator / denumerator)


@numba.jit(nopython=True, parallel=_NUMBA_PARALLEL)
def _mmd_statistic(K: np.ndarray, nx: int, ny: int) -> float:
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


@numba.jit(nopython=True, parallel=_NUMBA_PARALLEL)
def _kde_statistic(K: np.ndarray, nx: int, ny: int) -> float:
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
    kde_X = np.sum(K[:nx, ...], axis=0)  # (nx, M, M) -> (M, M)
    kde_X /= np.sum(kde_X)
    kde_Y = np.sum(K[nx:, ...], axis=0)  # (ny, M, M) -> (M, M)
    kde_Y /= np.sum(kde_Y)

    l1_dist = np.mean(np.abs(kde_X - kde_Y)).item()
    return l1_dist


def identity_kernel(x: ndarray, **kw):
    """
    Identity function, to use as a kernel.
    """
    return x


@numba.jit(nopython=True, parallel=_NUMBA_PARALLEL)
def gaussian_kernel(x: ndarray, sigma: float = 1):
    """
    Gaussian kernel function.
    """
    return np.exp(-0.5 * (x ** 2) / sigma ** 2)


def tw_test(
    X: ndarray,
    Y: ndarray,
    k: int,
    similarity_fn: Callable[[ndarray, ndarray], float] = sqeuclidean,
    kernel_fn: Callable[[ndarray], ndarray] = identity_kernel,
) -> Tuple[float, float]:
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
    )


def mmd_test(
    X: ndarray,
    Y: ndarray,
    k: int,
    similarity_fn: Callable[[ndarray, ndarray], float] = sqeuclidean,
    kernel_fn: Callable[[ndarray], ndarray] = gaussian_kernel,
) -> Tuple[float, float]:
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
    )


def kde2d_test(
    X: ndarray,
    Y: ndarray,
    k: int,
    n_bins: int,
    grid_low: float,
    grid_high: float,
    kernel_fn: Callable,
) -> Tuple[float, float]:
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
            dtype=np.float64,
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
    )


def two_sample_kernel_permutation_test(
    X: ndarray,
    Y: ndarray,
    k: int,
    similarity_fn: Optional[Callable[[ndarray, ndarray], float]],
    kernel_fn: Callable[[ndarray], ndarray],
    statistic_fn: Callable[[ndarray, int, int], float],
) -> Tuple[float, float]:
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
    :return: Tuple containing the statistic value for (X, Y) and the p-value (
        significance) for the null-hypothesis that P_X = P_Y.
    """
    # sample sizes
    nx = X.shape[0]
    ny = Y.shape[0]
    if nx < 2 or ny < 2:
        raise ValueError(
            "Permutation test requires at least two observations in each sample"
        )

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
        K, nx, ny, k, statistic_fn, permute_pairs
    )


@numba.jit(nopython=True, parallel=_NUMBA_PARALLEL)
def _two_sample_kernel_permutation_test_inner(
    K: ndarray,
    nx: int,
    ny: int,
    k: int,
    statistic_fn: Callable[[ndarray, int, int], float],
    permute_pairs: bool,
) -> Tuple[float, float]:
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
    :param permute_pairs: If False, K will be permuted on first axes only,
        i.e. K[p, ...]. If True, then K will be treated as a Gram matrix of pairwise
        distances and permuted as K[p,:][:,p], where p are permuted indices.
    :return: Statistic value of un-permuted distances, and p-value for an H0 of P_X=P_Y.
    """

    stat_val = statistic_fn(K, nx, ny)
    ss = np.zeros(k)
    for i in range(k):
        idx = np.random.permutation(nx + ny)

        if permute_pairs:
            K_perm = K[idx, :][:, idx]
        else:
            K_perm = K[idx]

        stat_val_perm = statistic_fn(K_perm, nx, ny)
        if stat_val <= stat_val_perm:
            ss[i] = 1

    # Calculate pval, and make sure it's not zero (it's possible that no iteration
    # produced stat_val <= stat_val_perm, but that doesn't mean the true pval is zero).
    # The smallest pval we can detect is 1/(k+1).
    p_val = np.mean(ss).item()
    p_val = max(p_val, 1 / (k + 1))
    return stat_val, p_val
