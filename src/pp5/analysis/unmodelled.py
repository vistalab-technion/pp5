from typing import Tuple, Optional, Sequence

import numpy as np
from numpy import ndarray
from pandas import DataFrame
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

from pp5.stats.cdf import quantile_level


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


def extract_unmodelled_bfactor_context(
    df_prec, context_len: int = 5, quantiles: bool = True
) -> DataFrame:
    """
    Extracts bfactors in a context window around unmodelled segments.

    :param df_prec: The prec dataframe.
    :param context_len: The number of residues to include on each side of the unmodelled
    segment.
    :param quantiles: If True, the output contain the quantile levels of the bfactors
    relative to the entire structure. Otherwise, the raw bfactors will be used.
    :return: A dataframe array of n_segments rows, with 2*context_len columns,
    where each row represents a different unmodelled segment, and columns represent
    the bfactors of residues context_len residues before and after the segment.
    In case there are no residues before/after, the corresponding columns will be NaN.
    Additional columns will be added for the pdb_id and segment index.
    """
    if context_len < 1:
        raise ValueError("context_len must be at least 1")

    unmodelled_segs = extract_unmodelled_segments(df_prec)
    n_unmodelled = len(unmodelled_segs)

    bfactors = np.array(df_prec["bfactor"])
    if quantiles:
        bfactors = quantile_level(bfactors)

    # Create empty array to store the b-factors
    bfactors_ctx = np.full((n_unmodelled, 2 * context_len), np.nan)

    for idx_seg, (seg_start_idx, seg_len, _) in enumerate(unmodelled_segs):
        for j in range(context_len):
            # Pre-segment context
            pre_idx = seg_start_idx - context_len + j
            if pre_idx >= 0:
                bfactors_ctx[idx_seg, j] = bfactors[pre_idx]

            # Post-segment context
            post_idx = seg_start_idx + seg_len + j
            if post_idx < len(bfactors):
                bfactors_ctx[idx_seg, context_len + j] = bfactors[post_idx]

    # Create dataframe
    offsets = [
        *[-(i + 1) for i in reversed(range(context_len))],
        *[(i + 1) for i in range(context_len)],
    ]
    offset_cols = [str(o) for o in offsets]
    df_bfactors_ctx = DataFrame(bfactors_ctx, columns=offset_cols)
    df_bfactors_ctx["pdb_id"] = df_prec["pdb_id"].iloc[0]
    df_bfactors_ctx["seg_idx"] = np.arange(n_unmodelled)
    df_bfactors_ctx = df_bfactors_ctx[["pdb_id", "seg_idx", *offset_cols]]
    return df_bfactors_ctx


def plot_with_unmodelled(
    df_prec: DataFrame,
    col_name: str,
    col_data: Optional[np.ndarray] = None,
    ax: Optional[Axes] = None,
    figsize: Tuple[int, int] = (10, 5),
):
    """
    Plot a column of a prec dataframe with unmodelled segments highlighted.

    :param df_prec: The prec dataframe.
    :param col_name: The name of the column to plot.
    :param col_data: Data to plot, instead of df_prec[col_name]. If provided, must have
    the same length as df_prec, and col_name will be used as a label.
    :param ax: The axis to plot on. If None, a new figure will be created.
    :param figsize: The size of the figure to create if ax is None.
    """
    if col_data is None:
        col_data = np.array(df_prec.loc[:, col_name])
    elif [*col_data.shape] != [len(df_prec)]:
        raise ValueError(f"{col_data.shape=} does not match {len(df_prec)=}")

    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=figsize)
    ax.plot(np.arange(len(col_data)), col_data, label=col_name)

    # Plot a rectangle in gray between start, end residues of unmodelled segments
    unmodelled_segs = extract_unmodelled_segments(df_prec)
    for i, (seg_start_idx, seg_len, seg_type) in enumerate(unmodelled_segs):
        seg_end_idx = seg_start_idx + seg_len
        ax.axvspan(
            seg_start_idx,
            seg_end_idx,
            color="gray",
            alpha=0.5,
            label=f"Unmodelled" if i == 0 else None,
        )
    ax.set_xlabel("Residue index")
    ax.set_ylabel(col_name)
    ax.grid()
