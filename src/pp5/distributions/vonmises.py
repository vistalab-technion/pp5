from typing import Tuple, Union, Optional
from functools import partial

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from pp5 import plot as plot
from pp5.distributions.kde import kde_2d, bvm_kernel


class BvMMixtureDiscreteDistribution(object):
    """
    A mixture of discrete Bivariate von Mises distributions.
    """

    def __init__(
        self,
        k1: Union[np.ndarray, list, tuple, float],
        k2: Union[np.ndarray, list, tuple, float],
        A: Union[np.ndarray, list, tuple, float],
        mu: Union[np.ndarray, list, tuple] = None,
        alpha: Union[np.ndarray, list, tuple] = None,
        gridsize: int = 512,
        two_pi: bool = False,
        dtype=np.float32,
    ):
        """
        Initializes the mixture distribution with K modes (single BvM distributions).

        :param k1: The phi concentration parameters. Should be a scalar or an array
        of shape (K,), one value for each distribution in the mixture..
        :param k2: The psi concentration parameters. Shape should be identical to k1.
        :param A: The correlation matrices, of shape (K, 2, 2) or (K,) or a scalar.
            Each element should be 2x2 matrix or a scalar, in which case it will be
            multiplied by identity.
        :param mu: The mean location of each distribution in the mixture. Should be
            an array of shape (K, 2) where row i is (mu_phi_i, mu_psi_i). None means
            use zeros.
        :param alpha: Mixture coefficients of shape (K,). Must sum to 1. If None,
            will be set to uniform weights.
        :param gridsize: Number of elements in each axis of the grid for the discrete
            PDF representing this mixture distribution. For example, if gridsize is
            512, the PDF will be calculated on a 512x512 grid representing [-pi,pi).
        :param two_pi: Whether the support of the distribution is [0, 2pi) (True) or
            [-pi,pi) (False, default).
        :param dtype: Numpy data type to use for generated samples.
        """
        assert k1 is not None and k2 is not None and A is not None and dtype is not None

        def _wrap(x) -> np.ndarray:
            if np.isscalar(x):
                return np.array([x], dtype=dtype)

            assert isinstance(x, (np.ndarray, list, tuple)), "Unexpected argument type"
            # In case it's a tuple/list
            return np.array(x, dtype=dtype)

        # In case these are scalars, and also to convert dtype
        k1, k2, A = (
            _wrap(k1),
            _wrap(k2),
            _wrap(A),
        )

        # Number of distributions in the mixture
        K = len(k1)

        if alpha is None:
            alpha = np.ones_like(k1) * (1 / K)
        else:
            alpha = _wrap(alpha)

        if mu is None:
            mu = np.zeros(shape=(K, 2), dtype=dtype)
        else:
            mu = _wrap(mu)

        # Make sure inputs are valid
        assert k1.shape == k2.shape == alpha.shape, "Inconsistent input shapes"
        assert np.ndim(k1) == 1, "Correlation parameters should be specified as vectors"
        assert mu.shape == (K, 2), f"Invalid shape for mu, should be {(K, 2)}"
        assert np.allclose(np.sum(alpha), 1.0), "alpha must sum to 1"

        if np.ndim(A) == 1:
            assert len(A) == len(k1), "Inconsistent number of correlation values"
            # If we have scalars as the correlation coefficients, convert to a
            # (K,2, 2) array, where each 2x2 matrix is I * k3_i
            A = np.stack([np.eye(2, dtype=dtype) * k3_i for k3_i in A], axis=0)
        elif np.ndim(A) == 2:
            # If A is a 2x2 matrix, it's only OK if K=1.
            # In such a case, add an extra dimension in the beginning.
            assert K == 1, "A has a shape inconsistent with k1, k2"
            A = np.expand_dims(A, axis=0)
        elif np.ndim(A) == 3:
            assert A.shape == (K, 2, 2), "A has a shape inconsistent with k1, k2"
        else:
            raise AssertionError(f"A has invalid dimensions ({A.shape})")

        # Generate the discrete grid on which we'll evaluate the PDF
        assert gridsize > 1
        grid_range = (0.0, 2 * np.pi) if two_pi else (-np.pi, np.pi)
        grid = np.linspace(*grid_range, num=gridsize, endpoint=False)
        phi = grid.reshape(-1, 1)  # (M, 1)
        psi = grid.reshape(1, -1)  # (1, M)

        # Generate the mixture PDF (actually it's a PMF, since it's discrete)
        pdf = np.array(0, dtype=dtype)
        for k in range(K):
            phi_k, psi_k = phi - mu[k, 0], psi - mu[k, 1]
            pdf_k = bvm_pdf(phi_k, psi_k, k1[k], k2[k], A[k])
            pdf = pdf + pdf_k * alpha[k]

        self.k1 = k1
        self.k2 = k2
        self.A = A
        self.alpha = alpha
        self.dtype = dtype
        self.n_modes = K
        self.two_pi = two_pi
        self.pdf = pdf
        self.grid = grid
        self.gridsize = gridsize

    def sample(self, n: int) -> np.ndarray:
        """
        Sample angles from the distribution. Work by sampling a bin from the grid
        from a multinomial distribution with probabilities defined by the discrete
        mixture PDF. Then for each sampled bin, a small amount of gaussian noise is
        added to move the sample around inside the bin.

        :param n: Number of angles to sample.
        :return: a (n,2) array of where is row i is sample i: (phi_i, psi_i).
        """

        # Use discrete PDF as weights to draw n samples from a multinomial distribution
        draw = np.random.multinomial(n, self.pdf.reshape(-1))
        # After reshaping back, the result is the number of sample drawn in each bin
        draw = np.reshape(draw, self.pdf.shape)

        # Convert from number of drawn samples per bin to a list of (phi, psi)
        # where (phi, psi) is repeated as many times as it was drawn
        ii, jj = np.nonzero(draw)
        samples = []
        for i, j in zip(ii, jj):
            n_samples = draw[i, j]
            phi_samples = self.grid[i]
            psi_samples = self.grid[j]
            samples.extend([(phi_samples, psi_samples)] * n_samples)

        samples = np.array(samples, dtype=self.dtype)

        # Create gaussian noise for each (phi, psi) which will move the sample around
        # inside it's bin.
        # Noise mu is half grid step in order to move them "to the middle" of each bin
        # in the grid
        # Noise sigma is set such that 3*sigma is half the bin width
        grid_delta = 2 * np.pi / self.gridsize
        eps = np.random.normal(grid_delta / 2, grid_delta / 2 / 3, size=(n, 2))
        samples = samples + eps

        # Now we need to make sure that the noised samples are inside the grid
        # if not, we must shift them circularly
        grid_min, grid_max = self.grid[0], self.grid[-1] + grid_delta
        samples[samples < grid_min] += 2 * np.pi
        samples[samples > grid_max] -= 2 * np.pi
        assert not np.any(samples < grid_min)
        assert not np.any(samples > grid_max)

        return samples

    def plot(
        self,
        samples: Union[np.ndarray, int] = None,
        ramachandran_kw: dict = {},
        scatter_kw: dict = {},
    ) -> Tuple[Figure, Axes]:
        """
        Plots the mixture distribution, optionally with samples from it plotted on top.
        :param samples: Either the number of samples to plot on top top the
            distribution, or a (N,2) array of the samples themselves to plot.
            By default, only the distribution is plotted.
        :param ramachandran_kw: Keyword args for configuring the PDF plot. See the
            documentation of `meth`:pp5.plot.ramachandran:.
        :param scatter_kw: Keyword args for matplotlib's `meth`:plt.scatter:.
        :return: The figure and axes of the plot.
        """

        if isinstance(samples, np.ndarray):
            if not (np.ndim(samples) == 2 and samples.shape[1] == 2):
                raise ValueError("Invalid shape for samples. Should be (N, 2).")
        elif isinstance(samples, int):
            if samples > 0:
                samples = self.sample(n=samples)
            else:
                samples = None
        else:
            if samples is not None:
                raise ValueError("Unsupported type for samples")

        # Set defaults and override with user prefs
        ramachandran_kw = {**dict(grid_2pi=self.two_pi, figsize=4), **ramachandran_kw}

        fig, ax = plot.ramachandran(
            [self.pdf],
            legend_label=["BvM Mixture"],
            samples=samples,
            **ramachandran_kw,
        )

        return fig, ax

    def __repr__(self):
        return f"BvM Mixture K={self.n_modes}"


class BvMKernelDensityEstimator(object):
    """
    Performs Kernel density estimation of a set of angles, evaluated on a
    discrete 2d grid.
    """

    def __init__(
        self,
        n_bins: int = 128,
        k1: float = 1.0,
        k2: float = 1.0,
        k3: float = 0.0,
        batchsize: Optional[int] = 64,
        dtype: np.dtype = np.float32,
    ):
        """
        :param n_bins: Number of discrete angle bins in each axis of the 2d grid.
        :param k1: Concentration parameter for phi
        :param k2: Concentration parameter for psi
        :param k3: Correlation parameter
        :param batchsize: Maximal number of datapoints to process in a
        single batch. Increasing this will cause hgh memory usage.
        :param dtype: Datatype of the result.
        """
        assert n_bins > 1
        assert not any(k is None for k in [k1, k2, k3])
        if batchsize is not None:
            assert batchsize >= 0

        self.n_bins = n_bins
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3
        self.batchsize = batchsize
        self.dtype = dtype

    def estimate(self, phi: np.ndarray, psi: np.ndarray):
        """
        Calculate KDE of set of angles, evaluated on a discrete 2d grid.
        :param phi: First angle values. Must be in [-pi, pi].
        :param psi: Second angle values. Must be in [-pi, pi].
        :return: The KDE, as an array of shape (M,M) where M is the number of bins.
        """
        # Apply general 2D KDE, but with a bivariate von-Mises kernel.
        return kde_2d(
            x1=phi,
            x2=psi,
            kernel_fn=partial(bvm_kernel, k1=self.k1, k2=self.k2, k3=self.k3),
            n_bins=self.n_bins,
            grid_low=-np.pi,
            grid_high=np.pi,
            batch_size=self.batchsize,
            dtype=self.dtype,
            reduce=True,
        )

    __call__ = estimate


def bvm_pdf(
    phi: np.ndarray,
    psi: np.ndarray,
    k1=1.0,
    k2=1.0,
    A: np.ndarray = None,
    skip_normalize=False,
):
    """
    PDF of a Bivariate von Mises (BvM) distribution.
    :param phi: Phi angle(s) to evaluate on. Can be scalar or vector.
    :param psi: Psi angle(s) to evaluate on. Can be scalar or vector.
    :param k1: The phi concentration parameters.
    :param k2: The psi concentration parameters.
    :param A: The correlation matrix. Should be 2x2 or a scalar,
        in which case it will be multiplied by identity.
    :param skip_normalize: Skip normalization of the PDF so that it sums to 1.
        Default is False, so the result sums to 1.
    :return: BvM kernel evaluated pointwise on the given data.

    Notes:
    - All angles should be in radians.
    - If phi and psi are 2d vectors of shape (1,M) and (M,1) then this
      function will evaluate the PDF on the 2d-grid defined by broadcasting these
      vectors together (like meshgrid(phi, psi)).

    Uses the full BvM distribution, with the means taken as zero.
    See: https://en.wikipedia.org/wiki/Bivariate_von_Mises_distribution
    """
    if A is None:
        A = np.zeros((2, 2), dtype=np.float32)
    elif np.isscalar(A):
        A = np.eye(2, dtype=np.float32) * A
    assert A.shape == (2, 2)

    def _check_dim(x):
        ndim = np.ndim(x)
        assert ndim in (0, 1, 2), "Input must be a scalar or vector"
        if ndim == 2:
            assert min(np.shape(x)) == 1, "Input must be a vector"
        return ndim

    ndim_phi, ndim_psi = _check_dim(phi), _check_dim(psi)
    assert ndim_phi == ndim_psi, "Input must have same number of dims"

    # Reshape phi, psi to (M, 1) and (1, M) respectively
    phi, psi = np.reshape(phi, (-1, 1)), np.reshape(psi, (1, -1))

    # Calculate concentration terms
    cos_phi, sin_phi = np.cos(phi), np.sin(phi)
    cos_psi, sin_psi = np.cos(psi), np.sin(psi)
    t1 = k1 * cos_phi
    t2 = k2 * cos_psi

    # Calculate correlation term
    cos_sin_phi = np.concatenate([cos_phi, sin_phi], axis=1)  # (M, 2)
    cos_sin_psi = np.concatenate([cos_psi, sin_psi], axis=0)  # (2, M)
    t3 = cos_sin_phi @ A @ cos_sin_psi  # (M, M)

    # Handle case of single input dim: output should also be single dim
    if ndim_phi == 1:
        t1 = np.reshape(t1, -1)
        t2 = np.reshape(t2, -1)
        t3 = np.diag(t3)

    pdf = np.exp(t1 + t2 + t3)
    if not skip_normalize:
        pdf /= np.sum(pdf)

    return pdf
