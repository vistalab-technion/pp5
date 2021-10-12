import numpy as np


def mht_bh(q: float, pvals: np.ndarray) -> float:
    """
    Multiple hypothesis testing with BH(q) method.
    :param q: The desired maximal FDR level (holds only in expectation across
        multiple realizations of the problem).
    :param pvals: 1d array of p-values corresponding to m different null hypotheses.
    :return: The threshold to use for determining which of the null hypotheses to
        reject (i.e., reject null where pvals <= mht_bh(q, pvals)).
    """
    if not 0.0 < q < 1.0:
        raise ValueError("q must be in (0, 1)")

    pvals = np.reshape(pvals, -1)
    m = len(pvals)
    if m < 2:
        raise ValueError("Need at least two hypotheses")

    # Sort the pvals from low to high
    idx_sorted = pvals.argsort()
    pvals_sorted = pvals[idx_sorted]

    # Calculate a different threshold for each pval based on it's rank
    bhq_thresh = (np.arange(m) + 1) * (q / m)

    # Find the index of the largest pval that's below its assigned threshold
    comp_lt = pvals_sorted <= bhq_thresh
    if not np.any(comp_lt):
        # All pvals are above threshold: return zero (can't reject any hypothesis)
        return 0.0
    else:
        i0 = m - 1 - np.argmax([*reversed(comp_lt)])

    # Sanity check
    assert pvals_sorted[i0] <= bhq_thresh[i0]
    if i0 + 1 < m:
        assert pvals_sorted[i0 + 1] > bhq_thresh[i0]

    # Return the corresponding threshold of that pval
    # This threshold should be used to determine significance among the
    # given hypotheses.
    return bhq_thresh[i0]
