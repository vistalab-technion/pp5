# %%
import sys
import logging
from pprint import pprint
from typing import Dict, Sequence
from itertools import product

import numpy as np
import pandas as pd
import networkx as nx
from Bio.Seq import Seq
from Bio.Align import PairwiseAligner
from tqdm.auto import tqdm
from Bio.SeqRecord import SeqRecord

from pp5.align import multiseq_align
from pp5.cache import cached_call_csv
from pp5.utils import logger_level_context
from pp5.collect import FolderDataset
from pp5.analysis.unmodelled.utils import extract_unmodelled_segments
from pp5.analysis.unmodelled.consts import (
    TO_CSV_KWARGS,
    READ_CSV_KWARGS,
    DATASET_DIR_PATH,
)
from pp5.analysis.unmodelled.consts import UNMODELLED_OUT_DIR as OUT_DIR
from pp5.analysis.unmodelled.consts import (
    UNMODELLED_OUTPUTS,
    OUTPUT_KEY_ALLSEGS_FILTERED,
)

# %%
print(f"{OUT_DIR=}")

# %%
dataset = FolderDataset(DATASET_DIR_PATH)

# %%

# Load metadata about each structure
df_meta = dataset.load_metadata()
pprint(df_meta.head())

# %%

# Load filtered segments
path_allsegs_filtered = OUT_DIR / UNMODELLED_OUTPUTS[OUTPUT_KEY_ALLSEGS_FILTERED]
df_allsegs_filtered = pd.read_csv(
    path_allsegs_filtered,
    **READ_CSV_KWARGS,
)
df_allsegs_filtered

# %%

# Add the unp_id column if its not there already
if "unp_id" not in df_allsegs_filtered.columns:
    # df_allsegs_filtered = pd.merge(
    df_allsegs_filtered = pd.merge(
        left=df_allsegs_filtered,
        right=df_meta[["pdb_id", "unp_id"]],
        how="left",
        left_on="pdb_id",
        right_on="pdb_id",
    )
    unp_id_col = df_allsegs_filtered.pop("unp_id")
    df_allsegs_filtered.insert(1, "unp_id", unp_id_col)


# %%


def find_distance_cliques(D: np.ndarray, d_thresh: float) -> Sequence[Sequence[int]]:
    """
    Given a symmetric distance matrix D (shape (n,n)) and a threshold d_thresh, return
    all *maximal* subsets of indices {i1, …, ik} such that D[ia, ib] > d_thresh for
    all a != b.
    """
    n = D.shape[0]
    G = nx.Graph()
    G.add_nodes_from(range(n))

    # add an edge whenever distance >= threshold, but only add each edge once (i<j)
    rows, cols = np.where(D >= d_thresh)
    for i, j in zip(rows, cols):
        if i < j:
            G.add_edge(i, j)

    # find all maximal cliques
    cliques = tuple(nx.find_cliques(G))
    return cliques


MIN_SEQUENCE_OVERLAP = 2
aligner = PairwiseAligner(
    mode="local",  # Local alignment to find best subsequence
    match_score=1.0,  # Positive score for matches
    mismatch_score=-1.0,  # Negative penalty for mismatches
    open_gap_score=-1000.0,  # Very large penalty to prevent gaps
    extend_gap_score=-1000.0,  # Very large penalty to prevent gaps
)

# Group chains by unp_id and segment, check for matches
df_groups_data = []
groups_iter = df_allsegs_filtered.groupby(
    by=["unp_id", "seg_idx", "seg_type"], as_index=False
)
with tqdm(total=groups_iter.ngroups, file=sys.stdout) as pbar:
    for i_group, ((unp_id, seg_idx, seg_type), df_group) in enumerate(groups_iter):
        pbar.update(1)
        pbar.set_description(f"{unp_id=} {seg_idx=} {seg_type=}")

        df_group = df_group.sort_values("pdb_id")

        pdb_ids = df_group["pdb_id"].values
        seg_start_idxs = df_group["seg_start_idx"].values

        group_size = len(df_group)
        assert len(pdb_ids) == len(set(pdb_ids)) == group_size

        # It's possible that some members of this group don't align to each other.
        # We'll split the group into further subgroups such that all members in the
        # subgroup have a sequence overlap of at least MIN_SEQUENCE_OVERLAP.

        # Calculate pairwise sequence-distance matrix
        sequences = df_group["seg_seq"].to_list()
        dist_mat = np.zeros((group_size, group_size))
        for i, seq1 in enumerate(sequences):
            for j, seq2 in enumerate(sequences):
                if j <= i:
                    continue
                dist_mat[i, j] = dist_mat[j, i] = aligner.score(seq1, seq2)

        # To find the subgroups of segments that overlap by at least
        # MIN_SEQUENCE_OVERLAP we'll construct a graph and find cliques
        subgroups = find_distance_cliques(dist_mat, float(MIN_SEQUENCE_OVERLAP))

        for j_subgroup, subgroup_idxs in enumerate(subgroups):
            df_subgroup = df_group.iloc[subgroup_idxs]
            subgroup_size = len(df_subgroup)

            subgroup_pdb_ids = df_subgroup["pdb_id"].values
            subgroup_seg_start_idxs = df_subgroup["seg_start_idx"].values

            if subgroup_size < 2:
                aligned_seq = df_subgroup.iloc[0]["seg_seq"]

            else:
                # Use multiseq alignment to find the most common segment in the group
                seq_records = [
                    SeqRecord(
                        Seq(row["seg_seq"]),
                        id=f"{row['pdb_id']}_{row['seg_idx']}",
                    )
                    for _, row in df_subgroup.iterrows()
                ]
                with logger_level_context("pp5.align", logging.WARNING):
                    try:
                        msa_result = multiseq_align(seq_records)

                        # Find the aligned seq with the most dashes ("-"). The parts of this
                        # sequence that are not dashed, must exist in all segments of this
                        # subgroup.
                        aligned_seqs = sorted(
                            list(msa_result.alignment), key=lambda x: x.count("-")
                        )
                        aligned_seq = aligned_seqs[-1]
                        aligned_seq = aligned_seq.replace("-", "_")  # excel compat
                    except Exception as e:
                        print(f"Failed to align @ {unp_id=} {seg_idx=} {seg_type=}")
                        aligned_seq = "?"

            df_groups_data.append(
                {
                    "unp_id": unp_id,
                    "seg_idx": seg_idx,
                    "subgroup_idx": j_subgroup,
                    "pdb_ids": str.join(";", subgroup_pdb_ids),
                    "start_idxs": str.join(";", [*map(str, subgroup_seg_start_idxs)]),
                    "seg_type": seg_type,
                    "seg_len_aligned": len(aligned_seq),
                    "seg_seq_aligned": aligned_seq,
                }
            )

        # if i_group > 1000:
        #     break

path_allsegs_grouped = OUT_DIR / f"{path_allsegs_filtered.stem}-grouped.csv"
df_groups = pd.DataFrame(df_groups_data)
df_groups.to_csv(path_allsegs_grouped, **TO_CSV_KWARGS)
