# %%
from pprint import pprint
from typing import Dict

import pandas as pd

from pp5.cache import cached_call_csv
from pp5.collect import FolderDataset
from pp5.analysis.unmodelled.utils import extract_unmodelled_segments
from pp5.analysis.unmodelled.consts import (
    TO_CSV_KWARGS,
    READ_CSV_KWARGS,
    DATASET_DIR_PATH,
    OUTPUT_KEY_ALLSEGS,
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

# Load filtered segments and JOIN with context AAs
path_allsegs_filtered = OUT_DIR / UNMODELLED_OUTPUTS[OUTPUT_KEY_ALLSEGS_FILTERED]
df_allsegs_filtered = pd.read_csv(
    path_allsegs_filtered,
    **READ_CSV_KWARGS,
)
pprint(df_allsegs_filtered.head())

# %%

# Join seq_len column to the segments dataframe

df_allsegs_filtered_merged = pd.merge(
    left=df_allsegs_filtered,
    right=df_meta[["pdb_id", "seq_len"]],
    how="left",
    left_on="pdb_id",
    right_on="pdb_id",
)
assert len(df_allsegs_filtered) == len(df_allsegs_filtered_merged)

# %%

# Filter down according to some criteria: small proteins and segments.
MAX_AAS = 400
MIN_SEG_LEN = 5
MAX_SEG_LEN = 15

filter_idx = (
    (df_allsegs_filtered_merged["seq_len"] < 400)
    & (df_allsegs_filtered_merged["seg_len"] >= MIN_SEG_LEN)
    & (df_allsegs_filtered_merged["seg_len"] <= MAX_SEG_LEN)
)

df_allsegs_filtered_merged = df_allsegs_filtered_merged[filter_idx].sort_values(
    ["seq_len", "seg_len", "pdb_id"], ascending=[True, True, True]
)

df_allsegs_filtered_merged.to_csv(
    OUT_DIR / f"{path_allsegs_filtered.stem}-shortseq.csv", **TO_CSV_KWARGS
)
