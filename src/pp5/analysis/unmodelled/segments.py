# %%
import logging
from pprint import pprint
from typing import Dict

import pandas as pd

from pp5 import OUT_DIR as PP5_OUT_DIR
from pp5.cache import cached_call_csv
from pp5.collect import FolderDataset
from pp5.analysis.unmodelled.utils import extract_unmodelled_segments

logging.basicConfig(level=logging.INFO)

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)
pd.set_option("display.max_colwidth", 20)

# %%

PREC_COLLECTED_OUT_DIR = PP5_OUT_DIR / "prec-collected"
DATASET_ZIPFILE_PATH = (
    PREC_COLLECTED_OUT_DIR / "20240615_185854-floria-unmod-r3.5-rc.zip"
)

DATASET_DIR_PATH = PREC_COLLECTED_OUT_DIR / "20240615_185854-floria-unmod-r3.5-rc"
dataset = FolderDataset(DATASET_DIR_PATH)

# %%
print(f"Loaded dataset: {dataset.name}")
print(f"Number of structure files in dataset: {len(dataset.pdb_ids)}")
print(f"Collection metadata: ")
pprint(dataset.collection_metadata, indent=4, width=120, compact=False)

# %% md
# # All unmodelled segments

# %%

df_prec = dataset.load_prec("2WUR:A")
ligand_idx = ~df_prec["res_hflag"].isna()
unmodelled_idx = df_prec["res_icode"].astype(str).str.startswith("U_")

pprint(df_prec[ligand_idx | unmodelled_idx])

print("Unmodelled segments in 2WUR:A:")
extract_unmodelled_segments(df_prec)

# %%

PDB_ID_SUBSET = [
    "2BP5:A",
    "1914:A",
    "8A4A:A",
    "4QN9:A",
    "3PV2:A",
    "1UZL:A",
    "4MSO:A",
    "7ZKX:A",
    "2B3P:A",
    "4M2Q:A",
]
PDB_ID_SUBSET = sorted(PDB_ID_SUBSET)

OUT_DIR = PP5_OUT_DIR / "unmodelled"
OUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"{OUT_DIR=}")

# %% md
# ## Load all segments

# %%


def load_unmodelled_segments(
    ds: FolderDataset, workers: int = 1, limit: int = None
) -> pd.DataFrame:
    pdb_id_to_segment_tuples: Dict[str, pd.DataFrame] = ds.apply_parallel(
        apply_fn=extract_unmodelled_segments,
        workers=workers,
        chunksize=100,
        limit=limit,
        context="fork",
    )
    # TODO: create a segment dataclass which can be converted to a dict
    segment_records = [
        {
            "pdb_id": pdb_id,
            "seg_idx": i,
            "seg_start_idx": seg[0],
            "seg_len": seg[1],
            "seg_type": seg[2],
            "seg_seq": seg[3],
        }
        for pdb_id, segs in pdb_id_to_segment_tuples.items()
        for i, seg in enumerate(segs)
    ]
    df_segments = pd.DataFrame(segment_records)
    df_segments = df_segments.sort_values(by=["pdb_id", "seg_idx"])
    return df_segments


df_allsegs_temp = load_unmodelled_segments(dataset, workers=1, limit=20)
df_allsegs_temp

# %%

UMODELLED_SEGMENTS_BASENAME = "unmodelled_segments"

df_allsegs, allsegs_file_path = cached_call_csv(
    target_fn=load_unmodelled_segments,
    cache_file_basename=UMODELLED_SEGMENTS_BASENAME,
    target_fn_args=dict(ds=dataset, workers=9, limit=None),
    hash_ignore_args=["workers"],
    cache_dir=OUT_DIR,
    clear_cache=False,
    return_cache_path=True,
    to_csv_kwargs=dict(index=False),
    read_csv_kwargs=dict(na_filter=True, keep_default_na=False, na_values=["", "NaN"]),
)

print(f"{allsegs_file_path=!s}")
df_allsegs

# %% md

## Filter out segments with Histidine tags

# %%

idx_his_tags = df_allsegs["seg_seq"].str.contains("H" * 6)

# Ignore inter segments because these aren't really His-tags
idx_inter_seg = df_allsegs["seg_type"] == "inter"
idx_his_tags = idx_his_tags & (~idx_inter_seg)

# Number of segments with Histidine tags by type
print("Number of segments with Histidine tags by type:")
pprint(
    df_allsegs.loc[idx_his_tags, "seg_type"].value_counts(),
)
pprint(
    f"Proportion of segments with histidine tags by type: "
    f"{idx_his_tags.sum()/len(df_allsegs):.2%}",
)

# %% md
# ## Filter out redundant segments

# We define redundant segments as the same segments in different chains of the same
# structure.

# Add pdb_base_id and chain columns
df_segs = pd.merge(
    left=df_allsegs,
    right=df_allsegs["pdb_id"]
    .str.split(":", expand=True)
    .rename(columns={0: "pdb_base_id", 1: "pdb_chain_id"}),
    left_index=True,
    right_index=True,
)

# Remove segments with histidine tags
orig_len = len(df_segs)
df_segs = df_segs[~idx_his_tags]
new_len = len(df_segs)
print(
    f"Dropped {orig_len - new_len} segments with His-tags "
    f"({100 * (1 - new_len / orig_len):.2f}%)"
)

# Remove duplicate segments
orig_len = len(df_segs)
df_segs = df_segs.drop_duplicates(subset=["pdb_base_id", "seg_start_idx", "seg_seq"])
new_len = len(df_segs)
print(
    f"Dropped {orig_len - new_len} duplicate segments "
    f"({100 * (1 - new_len / orig_len):.2f}%)"
)

# Remove pdb_base_id and chain columns
df_segs = df_segs.drop(columns=["pdb_base_id", "pdb_chain_id"])
pprint(df_segs)

# Write to file
filtered_segs_file_path = allsegs_file_path.with_stem(
    f"{allsegs_file_path.stem}-filtered"
)
df_segs.to_csv(filtered_segs_file_path, index=False)
print(f"Written to {filtered_segs_file_path}")
