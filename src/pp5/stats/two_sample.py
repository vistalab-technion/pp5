from typing import Any, Tuple, Callable, Optional

import numba
import numpy as np
from numpy import ndarray
from numpy.random import permutation
from scipy.spatial.distance import pdist, squareform, sqeuclidean

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
def _identity_kernel(x: ndarray):
    """
    Identity function, to use as a kernel.
    """
    return x


@numba.jit(nopython=True, parallel=_NUMBA_PARALLEL)
def _gaussian_kernel(x: ndarray, sigma: float = 1):
    """
    Gaussian kernel function.
    """
    return np.exp(-0.5 * (x ** 2) / sigma ** 2)


def tw_test(
    X: ndarray,
    Y: ndarray,
    k: int = 1000,
    similarity_fn: Callable[[ndarray, ndarray], float] = sqeuclidean,
    kernel_fn: Callable[[ndarray], ndarray] = _identity_kernel,
    kernel_kwargs: Optional[dict] = None,
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
        kernel_kwargs=kernel_kwargs,
    )


def mmd_test(
    X: ndarray,
    Y: ndarray,
    k: int = 1000,
    similarity_fn: Callable[[ndarray, ndarray], float] = sqeuclidean,
    kernel_fn: Callable[[ndarray], ndarray] = _gaussian_kernel,
    kernel_kwargs: Optional[dict] = None,
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
        kernel_kwargs=kernel_kwargs,
    )


def two_sample_kernel_permutation_test(
    X: ndarray,
    Y: ndarray,
    k: int,
    similarity_fn: Callable[[ndarray, ndarray], float],
    kernel_fn: Callable[[ndarray, Optional[Any]], ndarray],
    statistic_fn: Callable[[ndarray, int, int], float],
    kernel_kwargs: Optional[dict] = None,
) -> Tuple[float, float]:
    """
    Applies a two-sample permutation test to determine whether the null hypothesis
    that two distributions are identical can be rejected.

    The observations will be transformed using a Kernel function of the form
    K(X, Y) = k(h(x,y)), where h(x, y) is a scalar similarity function, and k(z) is a
    scalar univariate kernel to be applied on the similarity metric.
    For example,
        - RBF kernel K(x, y): Set h(x, y) = ||x-y||^2 and k(z) = exp(Ɣ z^2/σ^2).
        - Polynomial kernel K(x, y): Set h(x, y) = x^T y and k(z) = (Ɣ z + r)^d.
        - Linear kernel K(x, y): Set h(x, y) = x^T y and k(z) = z.

    The observation from X, Y will be pooled into Z = [X; Y], and a Gram matrix K
    will be computed, such that K[i,j] = K(z_i, z_j).
    A test-statistic s(Z; n_x, n_y) will be applied to K and to permutations of K which
    mix between the groups X and Y, where n_x and n_y are the number of observations
    in X and Y respectively.

    :param X: (m, n_x) array containing a sample X, where n_x is the number of
        observations in the sample and m is the dimension of each observation.
    :param Y: (m, n_y) array containing sample Y with n_y observations of dimension m.
    :param k: number of permutations for significance evaluation
    :param similarity_fn: h(x, y), a scalar bivariate similarity function.
    :param kernel_fn: k(z), a scalar univariate kernel function.
        The full bivariate kernel will be K(x,y)=k(h(x,y)).
    :param statistic_fn: A callable describing the statistic.
    :param kernel_kwargs: Optional kwargs to pass to the kernel function.
    :return: Tuple containing the statistic value for (X, Y) and the p-value (
        significance) for the null-hypothesis that P_X = P_Y.
    """
    # sample sizes
    nx = X.shape[1]
    ny = Y.shape[1]

    # pooled vectors
    Z = np.hstack((X, Y))

    # pairwise distances
    D = squareform(pdist(Z.T, metric=similarity_fn))

    # inner products
    kernel_kwargs = kernel_kwargs or {}
    K = kernel_fn(D, **kernel_kwargs)

    return _two_sample_kernel_permutation_test_inner(K, nx, ny, k, statistic_fn)


@numba.jit(nopython=True, parallel=_NUMBA_PARALLEL)
def _two_sample_kernel_permutation_test_inner(
    K: ndarray, nx: int, ny: int, k: int, statistic_fn: Callable
) -> Tuple[float, float]:
    """
    Calculates p-value for an H0 of P_X=P_Y, based on permutation testing.

    :param K: Gram matrix of of two pooled samples (X and Y) of shape (nx+ny, nx+ny).
    :param nx: Number of observations from X.
    :param ny: Number of observations from Y.
    :param k: Number of permutations for significance evaluation.
    :return: Statistic value of un-permuted distances, and p-value for an H0 of P_X=P_Y.
    """
    if nx < 2 or ny < 2:
        raise ValueError("Tw2 test requires at least two observations in each sample")

    stat_val = statistic_fn(K, nx, ny)
    ss = np.zeros(k)
    for i in range(k):
        idx = permutation(nx + ny)
        stat_val_perm = statistic_fn(K[idx, :][:, idx], nx, ny)
        if stat_val <= stat_val_perm:
            ss[i] = 1

    p_val = float(np.mean(ss))
    return stat_val, p_val
