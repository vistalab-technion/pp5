from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from numpy import ndarray


def greedy_breakdown_k(
    n1: int,
    n2: int,
    stat_and_influence_fn: Callable[
        [List[int], List[int]], Tuple[float, ndarray, ndarray]
    ],
    pval_fn: Callable[[List[int], List[int]], Tuple[float, float]],
    thresh: float,
    k_grid: Sequence[int],
) -> Tuple[List[Dict], Optional[int], List[Tuple[str, int, float]]]:
    """
    Adversarial breakdown-k: greedily remove the single most influential
    remaining observation (re-ranked after every removal), recomputing the real
    p-value only at the checkpoints in `k_grid`, until the pair is no longer
    significant at `thresh`.

    breakdown-k is a greedy upper bound on the minimal kill-set: a more
    sophisticated "attack" could require fewer points. The removal ranking uses
    `stat_and_influence_fn`'s influence values as a proxy; the significance
    decision at each checkpoint always uses the real `pval_fn` recomputation.

    :param n1: Number of observations in group 1 (original, pre-removal).
    :param n2: Number of observations in group 2 (original, pre-removal).
    :param stat_and_influence_fn: Given the current (x_keep, y_keep) index
        lists (into the original group-1/group-2 observations), returns
        (base_stat, infl_1, infl_2), where infl_1[m] and infl_2[m] are the drop
        in the statistic if x_keep[m] (resp. y_keep[m]) were removed. Larger
        means more influential (removed first).
    :param pval_fn: Given the current (x_keep, y_keep), returns (ddist, pval)
        from a real recomputed significance test on the reduced samples.
    :param thresh: Significance threshold (e.g. a BH cutoff); the pair is
        significant while pval <= thresh.
    :param k_grid: Checkpoints (number of observations removed so far) at
        which to actually call pval_fn and check significance.
    :return: Tuple of:
        - rows: one dict per checkpoint in k_grid actually reached, with keys
          k, ddist, p, n1, n2, significant.
        - breakdown_k: the smallest k_grid checkpoint at which the pair is no
          longer significant, or None if it never breaks within k_grid.
        - removed_log: one (side, original_index, drop) tuple per removal
          actually performed, in removal order. side is "X" or "Y";
          original_index indexes into the original group-1/group-2 arrays.
    """
    x_keep = list(range(n1))
    y_keep = list(range(n2))
    rows: List[Dict] = []
    removed_log: List[Tuple[str, int, float]] = []
    breakdown_k: Optional[int] = None
    k_max = max(k_grid)

    for k in range(0, k_max + 1):
        if k in k_grid:
            ddist, pval = pval_fn(x_keep, y_keep)
            significant = pval <= thresh
            rows.append(
                dict(
                    k=k,
                    ddist=ddist,
                    p=pval,
                    n1=len(x_keep),
                    n2=len(y_keep),
                    significant=significant,
                )
            )
            if not significant and breakdown_k is None:
                breakdown_k = k

        if k == k_max or breakdown_k is not None:
            break

        # base_stat is not needed by the loop itself (stat_and_influence_fn
        # already folds it into infl_x/infl_y); exposed for callers that want
        # to log the ranking-time statistic.
        _base_stat, infl_x, infl_y = stat_and_influence_fn(x_keep, y_keep)

        # Find the single most influential remaining point. Strict ">" means
        # the first-encountered candidate wins exact ties: X before Y, and
        # within each side, the lower local index first.
        best_local_idx, best_drop, best_side = None, -np.inf, None
        for local_i, drop in enumerate(infl_x):
            if drop > best_drop:
                best_drop, best_local_idx, best_side = drop, local_i, "X"
        for local_j, drop in enumerate(infl_y):
            if drop > best_drop:
                best_drop, best_local_idx, best_side = drop, local_j, "Y"

        if best_side == "X":
            removed_original_idx = x_keep.pop(best_local_idx)
        else:
            removed_original_idx = y_keep.pop(best_local_idx)
        removed_log.append((best_side, removed_original_idx, float(best_drop)))

    return rows, breakdown_k, removed_log
