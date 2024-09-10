from typing import Optional

import numpy as np
from numpy import ndarray


def empirical_cdf(
    x_samples: ndarray, x_values: Optional[ndarray | float] = None
) -> ndarray:
    """
    Evaluates the empirical CDF of a some distribution from which x_samples were
    sampled. The empirical CDF will be evaluated at each value in x_values.
    Handles NaNs in both x_samples and x_values.

    The CDF value is estimated as the fraction of samples smaller than a value.
    For a single value x, we can compute np.nanmean(x_samples <= x) to get an
    empirical estimate of the true probability P(X <= x).

    This function just computes the above on multiple values and handles NaNs.
    NaNs in x_samples are ignored, in the sense that they will not be included in the
    frequency counts, and for NaNs in x_values a corresponding NaN will be present in
    the output.

    Note: not to be confused with `np.quantile()`, which given a quantile level e.g.
    q=0.7 calculates the value x such that P(X <= x) = q). This function is the
    inverse of the empirical quantile function: given some value x, it calculates the
    quantile **level** q such that P(X <= x) = q.

    :param x_samples: Samples from the distribution of some variable (X). Must be 1d.
    :param x_values: The values (x) at which to evaluate the empirical CDF. Must
    be 1d. If not provided, the CDF values will be evaluated at x_samples.
    :return: A 1d-array of CDF values corresponding to each value in x_values.
    """

    n_samples, *extra_dims = x_samples.shape
    if extra_dims:
        raise ValueError(f"Expected 1d samples, got {x_samples.shape=}")

    if x_values is None:
        x_values = x_samples
    elif isinstance(x_values, (int, float)):
        x_values = np.array([x_values], dtype=float)

    n_values, *extra_dims = x_values.shape
    if extra_dims:
        raise ValueError(f"Expected 1d x_values, got {x_values.shape=}")

    # Compare each value in x_samples to each value in x_values
    # (n_samples, n_values) boolean array: sample i < value j
    pairwise_comparison = x_samples.reshape(-1, 1) <= x_values.reshape(1, -1)
    assert pairwise_comparison.shape == (n_samples, n_values)

    # Set comparisons with NaN to NaN (since NaN > x is False for any x)
    nan_mask_samples = np.isnan(x_samples)
    nan_mask_values = np.isnan(x_values)
    pairwise_comparison = pairwise_comparison.astype(float)
    pairwise_comparison[nan_mask_samples, :] = np.nan
    pairwise_comparison[:, nan_mask_values] = np.nan

    # In case of a NaN in x_values, an entire column will be NaN, so nanmean(axis=0)
    # will warn about the mean of an empty slice. We can safely suppress this warning.
    with np.testing.suppress_warnings() as sup:
        sup.filter(RuntimeWarning, "Mean of empty slice")

        # Calculate the fraction of samples less than each value.
        ecdf_values = np.nanmean(pairwise_comparison, axis=0)

    assert ecdf_values.shape == (n_values,)
    return ecdf_values
