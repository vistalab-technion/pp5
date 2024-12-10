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
    to_csv_kwargs=TO_CSV_KWARGS,
    read_csv_kwargs=READ_CSV_KWARGS,
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
df_segs.to_csv(filtered_segs_file_path, **TO_CSV_KWARGS)
print(f"Written to {filtered_segs_file_path}")

# %%
UNMODELLED_OUTPUTS.update(
    {
        OUTPUT_KEY_ALLSEGS: allsegs_file_path.name,
        OUTPUT_KEY_ALLSEGS_FILTERED: filtered_segs_file_path.name,
    }
)
