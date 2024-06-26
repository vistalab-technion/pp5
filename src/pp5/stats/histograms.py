import random
from typing import Any, Dict, List, Tuple, Sequence
from itertools import product

import numpy as np


def histogram(
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
    if bootstraps > 0:
        for n in range(bootstraps):
            hists.append(
                histogram(
                    samples=random.choices(samples, k=len(samples)),
                    bins=bins,
                    normalized=normalized,
                )
            )
    else:
        hists.append(
            histogram(
                samples=samples,
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
