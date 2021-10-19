from typing import Callable, Optional

import numpy as np
from numpy import ndarray
from scipy.ndimage import gaussian_filter


def kde_2d(
    x1: np.ndarray,
    x2: np.ndarray,
    kernel_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    n_bins: int,
    grid_low: float,
    grid_high: float,
    batch_size: Optional[int] = None,
    reduce: bool = True,
    dtype: np.dtype = np.float64,
):
    """
    Calculates a kernel-density estimate, evaluated on a discrete 2d grid.
    :param x1: First data variable, of shape (N,).
    :param x2: Second data variable, of shape (N,).
    :param kernel_fn: 2D kernel K(x1, x2) to apply to the samples.
    :param n_bins: Number M of discrete bins in each axis of the 2d grid.
    :param grid_low: Smallest value on the evaluation grid, inclusive.
    :param grid_high: Largest value on the evaluation grid, exclusive.
    :param batch_size: Maximal number of data points to process in a
        single batch. Increasing this will cause hgh memory usage.
        None or zero means no batching.
        Cannot be used with reduce=False.
    :param reduce: Whether to reduce and normalize the contribution of each sample.
        If True (the default), the regular 2D KDE of shape (M, M) will be returned.
        If False, the return value will be of shape (M, M, N), corresponding to the
        un-normalized contribution of each sample to each point on the estimation grid.
        Cannot be used with batch_size>0.
    :param dtype: Datatype of the result.
    :return: The KDE, as an array of shape (M, M) where M is the number of bins.
    """

    assert x1.ndim == x2.ndim == 1
    assert x1.shape == x2.shape
    if batch_size and not reduce:
        raise ValueError(f"Can't provide batch_size if reduce==False")

    # Create evaluation grid
    grid_width = grid_high - grid_low
    grid = np.linspace(grid_low, grid_high - (grid_width / n_bins), n_bins)
    grid = grid.reshape((-1, 1)).astype(dtype)  # (M, 1)

    # Reshape data and grid so they can be broadcast together
    x1 = np.ascontiguousarray(x1, dtype=dtype)
    x2 = np.ascontiguousarray(x2, dtype=dtype)
    x1 = x1.reshape((1, -1))  # (1, N)
    x2 = x2.reshape((1, -1))  # (1, N)

    # Process input in batches to avoid RAM explosion
    n = np.prod(np.array(x1.shape))

    # When reduce==False, batch size must be n.
    # Otherwise, use given or default to n.
    batch_size = (batch_size or n) if reduce else n

    n_chunks = int(np.ceil(n / batch_size))
    P_raw = np.zeros((n_bins, n_bins), dtype=dtype)

    for chunk_idx in range(n_chunks):
        start = chunk_idx * batch_size
        stop = np.minimum(n, (chunk_idx + 1) * batch_size)
        chunk = slice(start, stop)

        # Calculate grid minus data between each grid and data point
        dx1 = np.expand_dims(grid - x1[:, chunk], axis=1)  # (M, 1, N)
        dx2 = np.expand_dims(grid - x2[:, chunk], axis=0)  # (1, M, N)

        # Apply kernel pointwise, and sum all values at each grid location
        K = kernel_fn(dx1, dx2)
        K = np.nan_to_num(K, copy=False)

        # If reduce==False, we know batch_size=n so we can just return K from this
        # iteration.
        if not reduce:
            # Normalize contribution of each sample, i.e. each (M,M) grid
            K /= np.max(K, axis=(0, 1)).reshape((1, 1, -1))
            return K

        P_raw += np.sum(K, axis=2)  # (M, M, N) -> (M, M)

    # Normalize
    P = P_raw / np.sum(P_raw)
    return P


def gaussian_kernel(x: ndarray, sigma: float = 1):
    """
    Gaussian kernel function.
    """
    return np.exp(-0.5 * (x ** 2) / sigma ** 2)


def bvm_kernel(phi: np.ndarray, psi: np.ndarray, k1: float, k2: float, k3: float):
    """
    Bivariate von Mises (BvM) kernel function, cosine variant.
    All angles should be in radians.

    Uses the cosine-variant BvM distribution, with the means taken as zero.
    See:
    https://en.wikipedia.org/wiki/Bivariate_von_Mises_distribution

    :param phi: First angle values. Must be in [-pi, pi].
        Can be any shape, but needs to be broadcast-able together with psi.
    :param psi: Second angle values. Must be in [-pi, pi].
        Can be any shape, but needs to be broadcast-able together with phi.
    :param k1: First concentration parameter (for phi).
    :param k2: Second concentration parameter (for psi).
    :param k3: Correlation parameter.
    :return: BvM kernel evaluated pointwise on the given data.
        Output shape will the same as np.broadcast(phi, psi).
    """

    t1 = k1 * np.cos(phi)
    t2 = k2 * np.cos(psi)
    if k3 == 0.0:
        return np.exp(t1 + t2)

    t3 = k3 * np.cos(phi - psi)
    return np.exp(t1 + t2 + t3)


def torus_gaussian_kernel_2d(phi: np.ndarray, psi: np.ndarray, sigma: float):
    """
    Gaussian kernel function where the distance between points is calculated on the
    2d flat-torus (i.e. it has wraparound at ±pi).
    :param phi: First angle values. Must be in [-pi, pi].
        Can be any shape, but needs to be broadcast-able together with psi.
    :param psi: Second angle values. Must be in [-pi, pi].
        Can be any shape, but needs to be broadcast-able together with phi.
    :param sigma: Standard deviation of the kernel, in radians.
    :return: Value of the kernel evaluated at each (phi, psi).
    """
    d2 = np.arccos(np.cos(phi)) ** 2 + np.arccos(np.cos(psi)) ** 2
    return np.exp(-0.5 * d2 / sigma ** 2)


def w2_dist_sinkhorn(
    p: np.ndarray,
    q: np.ndarray,
    niter: int = 250,
    sigma: float = 1.0,
    eps: float = 1e-10,
):
    """
    Computes Wasserstein distance between two discrete distributions.
    :param p:
    :param q:
    :param niter:
    :param sigma:
    :param eps:
    :return:
    """
    assert p.shape == q.shape

    def _smooth(_w, _sigma):
        return np.maximum(gaussian_filter(_w, _sigma, mode="wrap", truncate=5), eps)
        # _w /= np.sum(_w)
        # return _w

    def _log(_x):
        return np.log(np.maximum(_x, eps))

    v = np.ones_like(p)
    w = np.ones_like(q)

    for i in range(0, niter):
        v = p / _smooth(w, sigma)
        w = q / _smooth(v, sigma)

    w2_dist = sigma * np.sum(p * _log(v) + q * _log(w))

    return (w2_dist, v, w)
