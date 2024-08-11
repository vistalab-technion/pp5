import itertools as it
from typing import Any, Tuple, Sequence
from concurrent.futures import ProcessPoolExecutor
from multiprocessing.context import SpawnContext

import numpy as np
import pandas as pd
from pandas import DataFrame
from tqdm.auto import tqdm

from pp5.collect import CollectedDataset


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


def pool_worker_fn(pdb_id: str, dataset: CollectedDataset) -> Any:
    df_prec = dataset.load_prec(pdb_id)
    return pdb_id, extract_unmodelled_segments(df_prec)


def load_unmodelled_segments(
    dataset: CollectedDataset, workers: int = 1, limit: int = None
):
    pdb_id_to_unmodelled = {}
    pdb_ids = dataset.pdb_ids

    if limit:
        pdb_ids = pdb_ids[:limit]

    with ProcessPoolExecutor(max_workers=workers, mp_context=SpawnContext()) as pool:
        map_results = pool.map(
            pool_worker_fn,
            pdb_ids,
            it.repeat(dataset, times=len(pdb_ids)),
            chunksize=100,
        )

        with tqdm(total=len(pdb_ids), desc="loading") as pbar:
            for pdb_id, result in map_results:
                pbar.set_postfix_str(pdb_id, refresh=False)
                pbar.update()

                pdb_id_to_unmodelled[pdb_id] = result

        return pdb_id_to_unmodelled
