from typing import Tuple, Optional, Sequence
from collections import defaultdict

import numpy as np
import pandas as pd
from numpy import ndarray
from pandas import DataFrame
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

from pp5.codons import ACIDS_1TO3
from pp5.stats.cdf import empirical_cdf

BFR_COL = "bfr"
BBR_COL = "bbr"
CTX_COL = "ctx"


def extract_unmodelled_segments(
    df_prec: DataFrame,
) -> Sequence[Tuple[int, int, str, str]]:
    """
    :param df_prec: A prec dataframe.
    :return: A sequence of tuples, each representing an unmodelled segment:
    - start_index: start index (iloc) of the segment in the dataframe.
    - length: length of the segment.
    - type: where type is 'nterm', 'cterm', or 'inter'.
    - sequence: The amino acid sequence of the segment.
    """

    # Make sure we have a simple integer index
    df_prec = df_prec.reset_index()

    # Create an index column without ligands
    non_ligand_idx = df_prec["res_hflag"].isna()
    df_index = df_prec.index
    df_index_non_ligand = df_index[non_ligand_idx]
    first_non_ligand_idx = df_index_non_ligand[0]
    last_non_ligand_idx = df_index_non_ligand[-1]

    # True/False if residue is unmodelled.
    # . . . U U U . . .
    # F F F T T T F F F
    unmod_indicator = np.array(df_prec["res_icode"].astype(str).str.startswith("U_"))

    # 1 at start of unmodelled segment, -1 after unmodelled segment, 0 in between
    # . . . U U U  . . .
    # 0 0 0 1 0 0 -1 0 0
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
            df_index[start_idx],
            end_idx - start_idx + 1,
            (
                "nterm"
                if start_idx == first_non_ligand_idx
                else ("cterm" if end_idx == last_non_ligand_idx else "inter")
            ),
            str.join("", df_prec.loc[start_idx:end_idx, "res_name"]),
        )
        for start_idx, end_idx in zip(start_idxs, end_idxs)
    )


def _resolve_column_data(
    df_prec: DataFrame,
    col_name: Optional[str] = None,
    col_data: Optional[ndarray] = None,
) -> ndarray:
    """
    Obtain data from a prec dataframe column, or confirm that a given data array has
    the appropriate length.

    :param df_prec: The prec dataframe.
    :param col_name: The name of the column to return data for.
    :param col_data: Data to use, instead of df_prec[col_name]. If provided,
    must have the same length as df_prec.
    :return: The data to use.
    """
    if col_data is None:
        if col_name is None:
            raise ValueError("Either col_name or col_data must be provided")
        return np.array(df_prec.loc[:, col_name])

    if [*col_data.shape] != [len(df_prec)]:
        raise ValueError(f"{col_data.shape=} does not match {len(df_prec)=}")

    return col_data


def extract_unmodelled_context(
    df_prec,
    context_len: int = 5,
    col_name: Optional[str] = None,
    col_data: Optional[np.ndarray] = None,
) -> DataFrame:
    """
    Extracts per-residue data in a context window around unmodelled segments.

    :param df_prec: The prec dataframe.
    :param context_len: The number of residues to include on each side of the unmodelled
    segment.
    :param col_name: The name of the column to extract data from.
    :param col_data: Data to use, instead of df_prec[col_name]. If provided,
    must have the same length as df_prec.
    :return: A dataframe array of n_segments rows, with 2*context_len columns,
    where each row represents a different unmodelled segment, and columns represent
    the col_data, with context_len residues before and after the segment.
    In case there are no residues before/after, the corresponding columns will be NaN.
    Additional columns will be added for the pdb_id and segment index.
    """
    if context_len < 1:
        raise ValueError("context_len must be at least 1")

    col_data = _resolve_column_data(df_prec, col_name, col_data)
    unmodelled_segs = extract_unmodelled_segments(df_prec)
    n_unmodelled = len(unmodelled_segs)

    # Create empty array to store the context around unmodelled segments
    unmodelled_ctx = np.full((n_unmodelled, 2 * context_len), np.nan, dtype=object)

    for idx_seg, (seg_start_idx, seg_len, *_) in enumerate(unmodelled_segs):
        for j in range(context_len):
            # Pre-segment context
            pre_idx = seg_start_idx - context_len + j
            if pre_idx >= 0:
                unmodelled_ctx[idx_seg, j] = col_data[pre_idx]

            # Post-segment context
            post_idx = seg_start_idx + seg_len + j
            if post_idx < len(col_data):
                unmodelled_ctx[idx_seg, context_len + j] = col_data[post_idx]

    # Create dataframe
    offsets = [
        *[-(i + 1) for i in reversed(range(context_len))],
        *[(i + 1) for i in range(context_len)],
    ]
    offset_cols = [str(o) for o in offsets]
    df_unmodelled_ctx = DataFrame(unmodelled_ctx, columns=offset_cols)
    df_unmodelled_ctx["pdb_id"] = df_prec["pdb_id"].iloc[0]
    df_unmodelled_ctx["seg_idx"] = np.arange(n_unmodelled)
    df_unmodelled_ctx = df_unmodelled_ctx[["pdb_id", "seg_idx", *offset_cols]]
    return df_unmodelled_ctx


def bfactor_ratios(df_prec: DataFrame, context_len: int = 0) -> DataFrame:
    """
    Compute B-factor forward and backward ratios for a given structure.

    Forward ratio (BFR):  bfr[i] = b[i] / b[i+1]
    Backward ratio (BBR): bbr[i] = b[i] / b[i-1]

    :param df_prec: A protein record dataframe.
    :param context_len: If >0, compute the ratios only in a context of this size around
    unmodelled segments, and an additional column will be added which combines the
    ratios. If 0, compute the ratios for the entire structure.
    :return: A dataframe with two or three columns:
    - 'bfr': the forward ratio of the B-factors.
    - 'bbr': the backward ratio of the B-factors.
    - 'ctx': Only exists when context_len>0. Contains BFR before each unmodelled
             segment, BBR after, NaN inside the segment.
    """
    df_bfactor = df_prec["bfactor"]

    # Use max bfactor where NaN (in unmodelled segments)
    nan_mask = df_bfactor.isna()
    bmax = df_bfactor.max()

    df_bfactor = df_bfactor.fillna(bmax)

    bfactor_forward_ratio = df_bfactor / df_bfactor.shift(-1)  # b[i] / b[i+1]
    bfactor_backward_ratio = df_bfactor / df_bfactor.shift(1)  # b[i] / b[i-1]

    df_ratios = pd.concat(
        [bfactor_forward_ratio, bfactor_backward_ratio], axis=1, keys=[BFR_COL, BBR_COL]
    )

    # Apply context mask
    if context_len:
        nan_mask_bfr = np.full_like(nan_mask, fill_value=True)
        nan_mask_bbr = np.full_like(nan_mask, fill_value=True)
        nan_mask_unmodelled = np.full_like(nan_mask, fill_value=False)

        # Compute NaN masks for context around unmodelled segments.
        # - BFR is non-NaN in the context window BEFORE each segment
        # - BBR is non-NaN in the context window AFTER each segment
        for seg_start_idx, seg_len, seg_type, _ in extract_unmodelled_segments(df_prec):
            seg_end_idx = seg_start_idx + seg_len
            pre_seg_context_idx = slice(
                max(0, seg_start_idx - context_len), seg_start_idx
            )
            post_seg_context_idx = slice(
                seg_end_idx, min(seg_end_idx + context_len, len(df_prec))
            )
            nan_mask_bfr[pre_seg_context_idx] = False
            nan_mask_bbr[post_seg_context_idx] = False
            nan_mask_unmodelled[seg_start_idx:seg_end_idx] = True

        # Apply NaN masks
        df_ratios.loc[nan_mask_bfr, BFR_COL] = np.nan
        df_ratios.loc[nan_mask_bbr, BBR_COL] = np.nan

        # Add ctx column combining the two ratios: BFR before each segment,
        # BBR after, NaN inside the segment.
        # Take BFR if not NaN, otherwise take BBR if not NaN, otherwise NaN
        df_ratios[CTX_COL] = np.where(
            ~nan_mask_bfr,
            df_ratios[BFR_COL],
            np.where(~nan_mask_bbr, df_ratios[BBR_COL], np.nan),
        )

        # Where there are overlaps between BFR and BBR (due to two unmodelled
        # segments closer than the context length), take the minimum to prevent
        # invalid (>1) values on the boundaries of unmodelled segments.
        idx_overlap = (~nan_mask_bfr) & (~nan_mask_bbr)
        df_ratios.loc[idx_overlap, CTX_COL] = np.minimum(
            df_ratios.loc[idx_overlap, BFR_COL], df_ratios.loc[idx_overlap, BBR_COL]
        )

    # Restore NaN values within unmodelled segments
    df_ratios[nan_mask] = np.nan

    return df_ratios


def extract_unmodelled_bfactor_context(
    df_prec, context_len: int = 5, quantiles: bool = True
) -> DataFrame:
    """
    Extracts bfactors in a context window around unmodelled segments.
    See `extract_unmodelled_context` for more details.

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
    bfactors = np.array(df_prec["bfactor"])
    if quantiles:
        bfactors = empirical_cdf(bfactors)

    return extract_unmodelled_context(
        df_prec, context_len=context_len, col_data=bfactors
    )


def extract_unmodelled_bfactor_ratio_context(
    df_prec, context_len: int = 5
) -> DataFrame:
    """
    Extracts bfactors in a context window around unmodelled segments.
    See `extract_unmodelled_context` for more details.

    :param df_prec: The prec dataframe.
    :param context_len: The number of residues to include on each side of the unmodelled
    segment.
    :return: A dataframe array of n_segments rows, with 2*context_len columns,
    where each row represents a different unmodelled segment, and columns represent
    the bfactors of residues context_len residues before and after the segment.
    In case there are no residues before/after, the corresponding columns will be NaN.
    Additional columns will be added for the pdb_id and segment index.
    """

    df_ratios = bfactor_ratios(df_prec, context_len=context_len)
    df_ratios_ctx = df_ratios["ctx"]
    return extract_unmodelled_context(
        df_prec, context_len=context_len, col_data=df_ratios_ctx
    )


def extract_aa_counts(df_prec: DataFrame) -> dict:
    """
    Count the occurrences of each amino acids and their pairs in a prec dataframe.
    :param df_prec: The prec dataframe.
    :return: A dictionary with the counts of each amino acid and their pairs.
    """
    counts = defaultdict(lambda: 0)
    prec_aas = df_prec["res_name"]

    # Filter out non-standard amino acids
    prec_aas_valid = (aa for aa in prec_aas if isinstance(aa, str))
    prec_aas_upper = (aa.upper() for aa in prec_aas_valid)
    prec_standard_aas = tuple(aa for aa in prec_aas_upper if aa in ACIDS_1TO3)

    for a1, a2 in zip(prec_standard_aas[:-1], prec_standard_aas[1:]):
        counts[a1] += 1
        counts[f"{a1}{a2}"] += 1

    last_aa = prec_standard_aas[-1]
    counts[last_aa] += 1

    return dict(counts)


def plot_with_unmodelled(
    df_prec: DataFrame,
    col_name: str,
    col_data: Optional[np.ndarray] = None,
    skip_unmodelled: bool = False,
    ax: Optional[Axes] = None,
    figsize: Tuple[int, int] = (10, 5),
):
    """
    Plot a column of a prec dataframe with unmodelled segments highlighted.

    :param df_prec: The prec dataframe.
    :param col_name: The name of the column to plot.
    :param col_data: Data to plot, instead of df_prec[col_name]. If provided, must have
    the same length as df_prec, and col_name will be used as a label.
    :param skip_unmodelled: If True, unmodelled segments will not be shown.
    :param ax: The axis to plot on. If None, a new figure will be created.
    :param figsize: The size of the figure to create if ax is None.
    """
    col_data = _resolve_column_data(df_prec, col_name, col_data)

    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=figsize)
    ax.plot(np.arange(len(col_data)), col_data, label=col_name)

    # Plot a rectangle in gray between start, end residues of unmodelled segments
    if not skip_unmodelled:
        unmodelled_segs = extract_unmodelled_segments(df_prec)
        for i, (seg_start_idx, seg_len, seg_type, _) in enumerate(unmodelled_segs):
            seg_end_idx = seg_start_idx + seg_len
            ax.axvspan(
                seg_start_idx,
                seg_end_idx - 1,  # because it's inclusive
                color="gray",
                alpha=0.5,
                label=f"Unmodelled" if i == 0 else None,
            )
    ax.set_xlabel("Residue index")
    ax.set_ylabel(col_name)
    ax.grid()
