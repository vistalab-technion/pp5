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

# Load filtered segments
path_allsegs_filtered = OUT_DIR / UNMODELLED_OUTPUTS[OUTPUT_KEY_ALLSEGS_FILTERED]
df_allsegs_filtered = pd.read_csv(
    path_allsegs_filtered,
    **READ_CSV_KWARGS,
)
pprint(df_allsegs_filtered.head())

# %%
