# %%
from pprint import pprint
from typing import Dict
from functools import partial
from collections import defaultdict

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from pp5.cache import cached_call_csv
from pp5.codons import ACIDS_1TO3
from pp5.collect import FolderDataset
from pp5.analysis.unmodelled.utils import (
    extract_aa_counts,
    extract_unmodelled_context,
    extract_unmodelled_segments,
)
from pp5.analysis.unmodelled.consts import (
    TO_CSV_KWARGS,
    READ_CSV_KWARGS,
    DATASET_DIR_PATH,
)
from pp5.analysis.unmodelled.consts import UNMODELLED_OUT_DIR as OUT_DIR
from pp5.analysis.unmodelled.consts import (
    OUTPUT_KEY_ALLSEGS_FILTERED,
    write_outputs,
    read_output_key,
)

# %%
print(f"{OUT_DIR.absolute()=!s}")

# %%
dataset = FolderDataset(DATASET_DIR_PATH)

print(f"Loaded dataset: {dataset.name}")
print(f"Number of structure files in dataset: {len(dataset.pdb_ids)}")
print(f"Collection metadata: ")
pprint(dataset.collection_metadata, indent=4, width=120, compact=False)

# %% md
# # Propensity analysis


# %%


def load_aa_counts(
    ds: FolderDataset, workers: int = 1, limit: int = None
) -> pd.DataFrame:
    pdb_id_to_aa_counts = ds.apply_parallel(
        apply_fn=extract_aa_counts,
        workers=workers,
        chunksize=100,
        limit=limit,
        context="fork",
    )

    total_counts = defaultdict(lambda: 0)
    for pdb_id, aa_counts in pdb_id_to_aa_counts.items():
        for aa, count in aa_counts.items():
            total_counts[aa] += count

    df_aa_counts = pd.DataFrame(
        [{"res_name": aa, "count": count} for aa, count in total_counts.items()]
    ).sort_values("res_name")

    return df_aa_counts


# %%

df_aa_counts, aa_counts_file_path = cached_call_csv(
    target_fn=load_aa_counts,
    cache_file_basename="aa_counts",
    target_fn_args=dict(ds=dataset, workers=9, limit=None),
    hash_ignore_args=["workers"],
    cache_dir=OUT_DIR,
    clear_cache=False,
    return_cache_path=True,
    to_csv_kwargs=TO_CSV_KWARGS.copy(),
    read_csv_kwargs=READ_CSV_KWARGS.copy(),
)
pprint(df_aa_counts)

# %% md
# ## Baseline frequencies

# %%

df_aa_counts_single_idx = df_aa_counts["res_name"].str.len() == 1

# Baseline counts
df_aa_counts_single = df_aa_counts[df_aa_counts_single_idx]
df_aa_counts_pairs = df_aa_counts[~df_aa_counts_single_idx]

# Baseline frequencies
df: pd.DataFrame
for df in [df_aa_counts_single, df_aa_counts_pairs]:
    total_count = df["count"].sum()
    df.loc[:, "freq"] = df["count"] / total_count

print(f"{len(df_aa_counts_single)=}, {len(df_aa_counts_pairs)=}")
print(f"{df_aa_counts_single['freq'].sum()=}, {df_aa_counts_pairs['freq'].sum()=}")

df_aa_counts_single.to_csv(
    OUT_DIR / f"{aa_counts_file_path.stem}-single.csv", **TO_CSV_KWARGS
)
df_aa_counts_pairs.to_csv(
    OUT_DIR / f"{aa_counts_file_path.stem}-pairs.csv", **TO_CSV_KWARGS
)


# %% md
# ## AA context around each unmodelled segment


# %%
def load_context_aas(
    ds: FolderDataset,
    workers: int = 1,
    context_len: int = 5,
    limit: int = None,
) -> pd.DataFrame:
    pdb_id_to_context_aas = ds.apply_parallel(
        apply_fn=partial(
            extract_unmodelled_context,
            context_len=context_len,
            col_name="res_name",
        ),
        workers=workers,
        chunksize=100,
        limit=limit,
        context="fork",
    )
    df_context_aas = pd.concat(pdb_id_to_context_aas.values(), axis=0)
    df_context_aas = df_context_aas.sort_values(by=["pdb_id", "seg_idx"])
    return df_context_aas


# %%

context_len = 5

df_context_aas, context_aas_file_path = cached_call_csv(
    target_fn=load_context_aas,
    cache_file_basename="context_aas",
    target_fn_args=dict(ds=dataset, workers=9, context_len=context_len, limit=None),
    hash_ignore_args=["workers"],
    cache_dir=OUT_DIR,
    clear_cache=False,
    return_cache_path=True,
    to_csv_kwargs=TO_CSV_KWARGS.copy(),
    read_csv_kwargs=READ_CSV_KWARGS.copy(),
)

pprint(df_context_aas)

# %%

# Load filtered segments and JOIN with context AAs
df_allsegs_filtered = pd.read_csv(
    read_output_key(OUTPUT_KEY_ALLSEGS_FILTERED),
    **READ_CSV_KWARGS,
)

# Join AA names in contex with unmodelled seg sequences, and add the first/last AA
# names of the sequence as new columns
df_segs_aa_context = pd.merge(
    left=df_allsegs_filtered,
    right=df_context_aas,
    how="inner",
    on=["pdb_id", "seg_idx"],
)
df_segs_aa_context["-0"] = df_segs_aa_context["seg_seq"].str.get(0)
df_segs_aa_context["0"] = df_segs_aa_context["seg_seq"].str.get(-1)

# Normalize and sort cols
for i in range(0, context_len + 1):
    df_segs_aa_context = df_segs_aa_context.rename(
        columns={f"{i}": f"res_name+{i}", f"-{i}": f"res_name-{i}"}
    )
res_name_ctx_cols = tuple(
    [
        *[f"res_name-{i}" for i in reversed(range(0, context_len + 1))],
        *[f"res_name+{i}" for i in range(0, context_len + 1)],
    ]
)
other_cols = [c for c in df_segs_aa_context.columns if c not in res_name_ctx_cols]
df_segs_aa_context = df_segs_aa_context[[*other_cols, *res_name_ctx_cols]]

pprint(df_segs_aa_context)
df_segs_aa_context.to_csv(
    OUT_DIR / f"{context_aas_file_path.stem}-segs-filtered.csv", **TO_CSV_KWARGS
)

# %% md
# ## Propensity of single AAs in unmodelled segments

aa_counts_segs = defaultdict(lambda: 0)
for seq in df_segs_aa_context["seg_seq"]:
    if not isinstance(seq, str):
        continue
    for aa in seq:
        if aa not in ACIDS_1TO3:
            continue
        aa_counts_segs[aa] += 1

df_aa_counts_seg_single = pd.DataFrame(
    [{"res_name": aa, "count": count} for aa, count in aa_counts_segs.items()]
).sort_values("res_name")

df_aa_counts_seg_single["freq"] = (
    df_aa_counts_seg_single["count"] / df_aa_counts_seg_single["count"].sum()
)

# Merge with baseline counts and freqs
df_aa_counts_seg_single = pd.merge(
    left=df_aa_counts_single,
    right=df_aa_counts_seg_single,
    on="res_name",
    suffixes=("_all", "_seg"),
)

# Calculate freq ratio
df_aa_counts_seg_single["freq_ratio"] = (
    df_aa_counts_seg_single["freq_seg"] / df_aa_counts_seg_single["freq_all"]
)
df_aa_counts_seg_single = df_aa_counts_seg_single.sort_values("freq_ratio")

# Write
aa_counts_single_seg_filepath = OUT_DIR / f"{aa_counts_file_path.stem}-single-seg.csv"
df_aa_counts_seg_single.to_csv(aa_counts_single_seg_filepath, **TO_CSV_KWARGS)
print(f"Written to {aa_counts_single_seg_filepath!s}")
pprint(df_aa_counts_seg_single.head())

# %%

# Plot the single AA propensities
fig, ax = plt.subplots(1, 1, figsize=(10, 5))

bar_x = np.arange(len(df_aa_counts_seg_single))
bar_width = 0.25
bar_mult = 0

labels = {"freq_all": "baseline", "freq_seg": "unmodelled", "freq_ratio": "ratio"}

for col in ["freq_all", "freq_seg"]:
    bar_offset = bar_width * bar_mult
    ax.bar(
        bar_x + bar_offset,
        df_aa_counts_seg_single[col],
        label=labels[col],
        width=bar_width,
    )
    bar_mult += 1
ax.legend(loc="upper left")

ax = ax.twinx()
bar_offset = bar_width * bar_mult
ax.bar(
    bar_x + bar_offset,
    df_aa_counts_seg_single["freq_ratio"],
    label=labels["freq_ratio"],
    color="C3",
    width=bar_width,
)
ax.axhline(y=1.0, linestyle="--", linewidth=3, color="C3")
ax.grid()
ax.legend(loc="upper right")
ax.set_xticks(bar_x + bar_width, df_aa_counts_seg_single["res_name"])
ax.set_title(f"AA propensity in unmodelled segments")
ax.set_ylim(0, 3.5)

# Write
_out_path = OUT_DIR / f"{aa_counts_single_seg_filepath.stem}.png"
fig.savefig(_out_path, bbox_inches="tight")
print(f"Wrote {_out_path}")
plt.show()
