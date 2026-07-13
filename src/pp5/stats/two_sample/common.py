"""Shared permutation-test engine used by all the two-sample test statistics."""

from typing import Callable, Optional, Tuple, Union

import numba
import numpy as np
from numpy import ndarray
from scipy.spatial.distance import pdist, squareform

_NUMBA_PARALLEL = False


def two_sample_kernel_permutation_test(
    X: ndarray,
    Y: ndarray,
    k: int,
    statistic_fn: Callable[[ndarray, int, int], float],
    similarity_fn: Optional[Callable[[ndarray, ndarray], float]] = None,
    kernel_fn: Optional[Callable[[ndarray], ndarray]] = None,
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
        - RBF kernel K(x, y): Set h(x, y) = ||x-y|| and k(z) = exp(Ɣ z^2/σ^2).
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
        D = squareform(pdist(Z, metric=similarity_fn))  # type: ignore
        permute_pairs = True
    else:
        # D is (nx+ny, m)
        D = Z
        permute_pairs = False

    # inner products
    kernel_fn = kernel_fn if kernel_fn is not None else lambda x: x
    K = kernel_fn(D)  # in general can be (nx+ny, m')

    return two_sample_kernel_permutation_test_inner(
        K, nx, ny, k, statistic_fn, permute_pairs, k_min, k_th
    )


# @numba.jit(nopython=True, parallel=_NUMBA_PARALLEL)
def two_sample_kernel_permutation_test_inner(
    K: Union[ndarray, Tuple[ndarray, ...]],
    nx: int,
    ny: int,
    k: int,
    statistic_fn: Callable[..., float],
    permute_pairs: bool,
    k_min: int,
    k_th: float,
) -> Tuple[float, float, int]:
    """
    Calculates p-value for an H0 of P_X=P_Y, based on permutation testing.

    :param K: Observations matrix of of two pooled samples (X and Y) of shape
        (N, M, *) where '*' means any number of additional dims, and N=nx+ny the
        total number of observations in the sample X and sample Y combined.
        When ``permute_pairs=False``, K may also be an opaque tuple whose elements
        the statistic function knows how to index; the inner loop passes it
        through unchanged.
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
