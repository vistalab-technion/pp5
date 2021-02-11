import random
from typing import Any, Dict, List, Tuple, Union, Callable, Optional, Sequence
from itertools import product

import numba
import numpy as np
from numpy.random import permutation
from scipy.spatial.distance import pdist, squareform


def tw_test(
    X: np.ndarray,
    Y: np.ndarray,
    k: int = 1000,
    metric: Optional[Union[str, Callable]] = "sqeuclidean",
) -> Tuple[float, float]:
    """
    Calculates the Tw^2 Welch statistic based on distances.
    :param X: (n, Nx) array containing a sample X, where Nx is the number of
        observations in the sample and n is the dimension of each observation.
    :param Y: (n, Ny) array containing sample Y with Ny observations of dimension n.
    :param k: number of permutations for significance evaluation
    :param metric: A distance metric. Any of the distance metrics supported by
        :meth:`scipy.spatial.distance.pdist` can be used. Default is squared-euclidean.
        Can also be a callable that accepts two observations in order to use a custom
        metric.
    :return: Tw^2 statistic, p-value (significance).
    """
    # sample sizes
    nx = X.shape[1]
    ny = Y.shape[1]

    # pooled vectors
    Z = np.hstack((X, Y))

    # pairwise distances
    D = squareform(pdist(Z.T, metric=metric))

    t2, p = _tw_test_inner(D, nx, ny, k)
    return t2, p


@numba.jit(nopython=True, parallel=False)  # parallel doesn't seem to help
def _tw_test_inner(D: np.ndarray, nx: int, ny: int, k: int) -> Tuple[float, float]:
    """
    Calculates p-value based on Tw^2 test.
    :param D: Matrix of squared distanced of two pooled samples (X and Y) of
        shape (nx+ny, nx+ny).
    :param nx: Number of observations from X.
    :param ny: Number of observations from Y.
    :param k: Number of permutations for significance evaluation.
    :return: Tw^2 statistic of un-permuted distances, and p-value.
    """
    if nx < 2 or ny < 2:
        raise ValueError("Tw2 test requires at least two observations in each sample")

    t2 = tw2_statistic(D, nx, ny)
    ss = np.zeros(k)
    for i in range(k):
        idx = permutation(nx + ny)
        t2_perm = tw2_statistic(D[idx, :][:, idx], nx, ny)
        if t2 <= t2_perm:
            ss[i] = 1

    p = float(np.mean(ss))
    return t2, p


@numba.jit(nopython=True, parallel=False)
def tw2_statistic(D: np.ndarray, nx: int, ny: int) -> float:
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


def _histogram(
    samples: Sequence[Any], bins: Sequence[Any], normalized: bool = True
) -> Dict[Any, float]:
    counts = {b: 0.0 for b in bins}
    for s in samples:
        if s in counts:
            counts[s] += 1
    n = sum(counts.values()) if normalized else 1.0
    return {b: c / n for b, c in counts.items()}


def categorical_histogram(
    samples: Sequence[Any],
    bins: Sequence[Any],
    bootstraps: int = 1,
    normalized: bool = True,
) -> Dict[Any, float]:
    """
    Calculates a bootstrapped categorical histogram.
    :param samples: A sequence of discrete categorical values.
    :param bins: A sequence of categorical bins
    :param bootstraps: Number of boostrap iterations for uncertainty calculation.
    :param normalized: Whether to normalize the counts
    :return: A dictionary with the mapping bin: counts
    """
    hists: List[Dict[Any, float]] = []
    for n in range(bootstraps):
        hists.append(
            _histogram(
                samples=random.choices(samples, k=len(samples)),
                bins=bins,
                normalized=normalized,
            )
        )

    def _mean_std(a: np.array) -> Tuple[float, float]:
        sigma = np.std(a)
        return np.mean(a), sigma if not np.isnan(sigma) else 0.0

    return {
        b: _mean_std(np.array([hists[i][b] for i in range(len(hists))])) for b in bins
    }


def ratio(num: Tuple[float, float], den: Tuple[float, float]) -> Tuple[float, float]:
    """
    Calculates ratio with uncertainties.
    :param num: (E[X], Std[X])
    :param den: (E[Y], Std[Y])
    :return: Ratio (E[X]/E[Y], Std[X/Y])
    """
    ratio = num[0] / den[0]
    sigma = ratio * np.sqrt((num[1] / num[0]) ** 2 + (den[1] / den[0]) ** 2)
    return (ratio, sigma if not np.isnan(sigma) else 0.0)


def factor(*x: Tuple[float, float]) -> Tuple[float, float]:
    """
    Calculates ratio with uncertainties.
    :param *x: (E[X_i], Std[X_i])
    :return: product (E[X_1*...*X_n], Std[[X_1*...*X_n])
    """
    prod = np.product([x[0] for x in x])
    sigma = prod * np.sqrt(np.sum([(x[1] / x[0]) ** 2 for x in x]))
    return (prod, sigma if not np.isnan(sigma) else 0.0)


def relative_histogram(
    x_hist: Dict[Any, Tuple[float, float]],
    y_hist: Dict[Any, Tuple[float, float]],
    groups: Dict[Any, Any],
) -> Dict[Any, Dict[Any, Tuple[float, float]]]:
    """
    Calculates a relative histogram.
    :param x_hist: A dictionary containing x: (p(x), sigma)
    :param y_hist: A dictionary containing y: (p(y), sigma)
    :param groups: A dictionary with the mapping y: x
    :return: A dictionary of dictionaries containing x: y: (p(x|y), sigma)
    """
    return {
        yy: {
            xx: ratio(x_hist[xx], y_hist[yy])
            for xx in [x for x, y in groups.items() if y == yy]
        }
        for yy in y_hist.keys()
    }


def product_histogram(
    x_hist: Dict[Any, Tuple[float, float]], n: int = 1
) -> Dict[Any, Tuple[float, float]]:
    """
    Computes a separable product histogram for tuples
    :param x_hist: A dictionary containing x: (p(x), sigma)
    :param n: Tuple length to combine
    :return: A dictionary containing (x_1,...,x_n): (p(x_1)*...*p(x_n), sigma)
        for (x_1,...,x_n) in {x}^n
    """
    x = [*x_hist.keys()]
    combinations = tuple(a for a in product(*([x] * n)))
    return {xx: factor(*[x_hist[x] for x in xx]) for xx in combinations}
