from typing import Tuple, Sequence

import numpy as np
from pandas import DataFrame


def extract_unmodelled_segments(
    df_prec: DataFrame,
) -> Sequence[Tuple[int, int, str]]:
    """
    :param df_prec: A prec dataframe.
    :return: A sequence of tuples, each representing an unmodelled segment:
    (start_index, length, type), where type is 'nterm', 'cterm', or 'inter'.
    """

    df_prec = df_prec.reset_index()

    # Ignore ligands for this purpose
    standard_aa_idx = df_prec["res_hflag"].isna()
    df_prec = df_prec[standard_aa_idx]
    last_idx = len(df_prec) - 1

    df_index = df_prec.index

    # True/False if residue is unmodelled.
    unmod_indicator = np.array(df_prec["res_icode"].astype(str).str.startswith("U_"))

    # 1 at start of unmodelled segment, -1 at end of unmodelled segment, 0 in between
    unmod_diff = np.diff(unmod_indicator.astype(int), prepend=0, append=0)
    assert len(unmod_indicator) + 1 == len(unmod_diff)

    # Indices of non-zero diffs (+1, and -1)
    nonzero_diff_idx = np.flatnonzero(unmod_diff)
    start_idxs, end_idxs = (
        nonzero_diff_idx[unmod_diff[nonzero_diff_idx] > 0],
        nonzero_diff_idx[unmod_diff[nonzero_diff_idx] < 0],
    )
    end_idxs = end_idxs - 1  # The -1 will be AFTER the segment
    assert len(start_idxs) == len(end_idxs)

    return tuple(
        (
            # Map to index in original dataframe (with ligands)
            df_index[start_idx],
            end_idx - start_idx + 1,
            (
                "nterm"
                if start_idx == 0
                else ("cterm" if end_idx == last_idx else "inter")
            ),
        )
        for start_idx, end_idx in zip(start_idxs, end_idxs)
    )
