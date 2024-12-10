# %%
from pprint import pprint
from typing import Dict
from functools import partial
from itertools import product
from collections import defaultdict

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from pp5.cache import cached_call_csv
from pp5.codons import ACIDS_1TO3
from pp5.collect import FolderDataset
from pp5.analysis.unmodelled.utils import extract_aa_counts, extract_unmodelled_context
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
plt.rcParams["font.family"] = "monospace"

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
df_aa_freqs_single = df_aa_counts[df_aa_counts_single_idx]
df_aa_freqs_pairs = df_aa_counts[~df_aa_counts_single_idx]

# Baseline frequencies
df: pd.DataFrame
for df in [df_aa_freqs_single, df_aa_freqs_pairs]:
    total_count = df["count"].sum()
    df.loc[:, "freq"] = df["count"] / total_count

print(f"{len(df_aa_freqs_single)=}, {len(df_aa_freqs_pairs)=}")
print(f"{df_aa_freqs_single['freq'].sum()=}, {df_aa_freqs_pairs['freq'].sum()=}")

filepath_aa_freqs_single = (
    OUT_DIR / f"{aa_counts_file_path.stem.replace('counts', 'freqs')}-single.csv"
)
df_aa_freqs_single.to_csv(filepath_aa_freqs_single, **TO_CSV_KWARGS)
print(f"Wrote to {filepath_aa_freqs_single!s}")

filepath_aa_freqs_pairs = (
    OUT_DIR / f"{aa_counts_file_path.stem.replace('counts', 'freqs')}-pairs.csv"
)
df_aa_freqs_pairs.to_csv(filepath_aa_freqs_pairs, **TO_CSV_KWARGS)
print(f"Wrote to {filepath_aa_freqs_pairs!s}")

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
    OUT_DIR / UNMODELLED_OUTPUTS[OUTPUT_KEY_ALLSEGS_FILTERED],
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

# %%

# Dataset filtering options

# Only 'inter' segments
idx_inter_seg = df_segs_aa_context["seg_type"] == "inter"

# Ignore segments with only one AA
idx_long_seg = df_segs_aa_context["seg_len"] > 1

label_filtered_aa_context_dfs = {
    "any": df_segs_aa_context,
    "any>1": df_segs_aa_context[idx_long_seg],
    "inter": df_segs_aa_context[idx_inter_seg],
    "inter>1": df_segs_aa_context[idx_inter_seg & idx_long_seg],
}

# Show the number of segments and the proportion of segments dropped by each filter
df_filter_stats = pd.DataFrame(
    [
        {
            "filter_label": filter_label,
            "n_segments": len(_df),
            "% dropped": (1 - len(_df) / len(df_segs_aa_context)) * 100,
        }
        for filter_label, _df in label_filtered_aa_context_dfs.items()
    ]
)
_out_path = OUT_DIR / f"{context_aas_file_path.stem}-filter_stats.csv"
df_filter_stats.to_csv(_out_path, **TO_CSV_KWARGS)
pprint(df_filter_stats)

# %% md
# ## Propensity of single AAs in unmodelled segments


def _calc_single_aa_frequencies(
    df_segs: pd.DataFrame, df_baseline: pd.DataFrame
) -> pd.DataFrame:
    """
    Calculate the propensity (empirical frequency estimate) of single AAs in unmodelled
    segments.

    :param df_segs: A DataFrame with unmodelled segments.
    :param df_baseline: A DataFrame with baseline AA counts and frequencies. Should
    have columns 'res_name', 'count', and 'freq'.
    :return: A DataFrame with columns 'res_name', 'count_[all,seg]', 'freq_[all,
    seg]', and  'freq_ratio'.
    """
    aa_counts_segs = defaultdict(lambda: 0)
    for seq in df_segs["seg_seq"]:
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
        left=df_baseline,
        right=df_aa_counts_seg_single,
        on="res_name",
        suffixes=("_all", "_seg"),
    ).sort_values("res_name")

    # Calculate freq ratio
    df_aa_counts_seg_single["freq_ratio"] = (
        df_aa_counts_seg_single["freq_seg"] / df_aa_counts_seg_single["freq_all"]
    )
    df_aa_counts_seg_single = df_aa_counts_seg_single
    return df_aa_counts_seg_single


# Write
label_aa_freqs_seg_single = {
    filter_label: _calc_single_aa_frequencies(_df, df_aa_freqs_single)
    for filter_label, _df in label_filtered_aa_context_dfs.items()
}
for filter_label, df_aa_freqs_seg_single in label_aa_freqs_seg_single.items():
    filepath_aa_freqs_seg_single = OUT_DIR / (
        f"{filepath_aa_freqs_single.stem}-seg_{filter_label}.csv"
    )
    df_aa_freqs_seg_single.to_csv(filepath_aa_freqs_seg_single, **TO_CSV_KWARGS)
    print(f"Written to {filepath_aa_freqs_seg_single!s}")
    pprint(df_aa_freqs_seg_single.head())

# %%

for filter_label, df_aa_freqs_seg_single in label_aa_freqs_seg_single.items():
    # Plot the single AA propensities
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    bar_x = np.arange(len(df_aa_freqs_seg_single))
    bar_width = 0.25
    bar_mult = 0

    # Sort bars by freq_ratio
    df_aa_freqs_seg_single = df_aa_freqs_seg_single.sort_values("freq_ratio")

    labels = {"freq_all": "baseline", "freq_seg": "unmodelled", "freq_ratio": "ratio"}

    for col in ["freq_all", "freq_seg"]:
        bar_offset = bar_width * bar_mult
        ax.bar(
            bar_x + bar_offset,
            df_aa_freqs_seg_single[col],
            label=labels[col],
            width=bar_width,
        )
        bar_mult += 1
    ax.legend(loc="upper left")

    ax = ax.twinx()
    bar_offset = bar_width * bar_mult
    ax.bar(
        bar_x + bar_offset,
        df_aa_freqs_seg_single["freq_ratio"],
        label=labels["freq_ratio"],
        color="C3",
        width=bar_width,
    )
    ax.axhline(y=1.0, linestyle="--", linewidth=3, color="C3")
    ax.grid()
    ax.legend(loc="upper right")
    ax.set_xticks(bar_x + bar_width, df_aa_freqs_seg_single["res_name"])
    ax.set_title(f"AA propensity in unmodelled segments ({filter_label})")
    ax.set_ylim(0, 3.5)

    # Write
    _out_path = OUT_DIR / f"{filepath_aa_freqs_single.stem}-seg-{filter_label}.png"
    fig.savefig(_out_path, bbox_inches="tight")
    print(f"Wrote {_out_path}")
    plt.show()

# %% md
# ## Propensity of AA pairs in unmodelled segments

# %%

OFFSETS = [
    # Controls where the border is. For example:
    # Zero means pairs are [-1, -0], and [+0, +1]
    # One means pairs are [-2, -1], and [+1, +2]
    0,
    1,
]

N_AAS = len(ACIDS_1TO3)


def _calc_aa_counts_pairs_seg(_df_segs_aa_context: pd.DataFrame) -> Dict:
    _aa_counts_pairs_seg = {}  # offset -> 'pre'/'post' -> pair -> count

    for _offset in OFFSETS:
        _aa_counts_pairs_pre_seg = defaultdict(lambda: 0)
        _aa_counts_pairs_post_seg = defaultdict(lambda: 0)

        for i, row in _df_segs_aa_context.iterrows():
            aa_pre_1, aa_pre_2 = (
                row[f"res_name-{_offset + 1}"],
                row[f"res_name-{_offset}"],
            )
            aa_post_1, aa_post_2 = (
                row[f"res_name+{_offset}"],
                row[f"res_name+{_offset + 1}"],
            )

            if aa_pre_1 in ACIDS_1TO3 and aa_pre_2 in ACIDS_1TO3:
                _aa_counts_pairs_pre_seg[f"{aa_pre_1}{aa_pre_2}"] += 1

            if aa_post_1 in ACIDS_1TO3 and aa_post_2 in ACIDS_1TO3:
                _aa_counts_pairs_post_seg[f"{aa_post_1}{aa_post_2}"] += 1

        _aa_counts_pairs_seg[_offset] = {}
        _aa_counts_pairs_seg[_offset]["pre"] = dict(_aa_counts_pairs_pre_seg)
        _aa_counts_pairs_seg[_offset]["post"] = dict(_aa_counts_pairs_post_seg)

    return _aa_counts_pairs_seg


# filter_label -> {offset -> {'pre'/'post' -> {pair -> count} } }
label_aa_counts_pairs_seg_inter = {
    filter_label: _calc_aa_counts_pairs_seg(_df)
    for filter_label, _df in label_filtered_aa_context_dfs.items()
}

# %%

# Create dataframes from counts:
# filter_label -> 'pre'/'post'-offset -> dataframe with ['res_name, 'count', 'freq']
label_aa_freqs_pairs_seg_dfs = {}

for filter_label, aa_counts_pairs_seg in label_aa_counts_pairs_seg_inter.items():
    label_aa_freqs_pairs_seg_dfs[filter_label] = {}

    for offset, pre_post_counts in aa_counts_pairs_seg.items():
        for pre_post in ["pre", "post"]:
            _pair_counts_dict = pre_post_counts[pre_post]  # aa pair -> count
            _df_pair_counts = pd.DataFrame(
                [
                    {"res_name": aa_pair, "count": count}
                    for aa_pair, count in _pair_counts_dict.items()
                ]
            )

            _df_pair_counts["freq"] = (
                _df_pair_counts["count"] / _df_pair_counts["count"].sum()
            )
            assert len(_df_pair_counts) == N_AAS**2

            _df_key = f"{pre_post}-o{offset}"
            label_aa_freqs_pairs_seg_dfs[filter_label][_df_key] = _df_pair_counts

# %%

# Merge into single df with key as suffix, and compute freq ratios against baseline
# frequencies.
# filter_label -> dataframe with ['res_name, 'count_all', 'freq_all',
# 'count_[pre/post]-o[offset]', 'freq_[pre/post]-o[offset]', 'freq_ratio_[pre/post]-o[offset]']
label_aa_freqs_pairs_seg_merged_dfs = {}

for filter_label, aa_freqs_pairs_seg_dfs in label_aa_freqs_pairs_seg_dfs.items():
    df_aa_freqs_pairs_seg_merged = df_aa_freqs_pairs.rename(  # 'all' is baseline freqs
        columns={"count": "count_all", "freq": "freq_all"}
    )

    for _df_key, _df in aa_freqs_pairs_seg_dfs.items():
        df_aa_freqs_pairs_seg_merged = pd.merge(
            left=df_aa_freqs_pairs_seg_merged,
            right=_df.rename(
                columns={"count": f"count_{_df_key}", "freq": f"freq_{_df_key}"}
            ),
            on="res_name",
        )

        # Add freq ratio
        df_aa_freqs_pairs_seg_merged[f"freq_ratio_{_df_key}"] = (
            df_aa_freqs_pairs_seg_merged[f"freq_{_df_key}"]
            / df_aa_freqs_pairs_seg_merged[f"freq_all"]
        )

    df_aa_freqs_pairs_seg_merged = df_aa_freqs_pairs_seg_merged.sort_values(
        "res_name"
    ).set_index("res_name")

    # Make sure we have all pairs
    assert len(df_aa_freqs_pairs_seg_merged) == N_AAS**2

    _out_path = OUT_DIR / f"{filepath_aa_freqs_pairs.stem}-seg_{filter_label}.csv"
    df_aa_freqs_pairs_seg_merged.to_csv(_out_path, index=True)
    print(f"Wrote {_out_path}")
    pprint(df_aa_freqs_pairs_seg_merged)

    label_aa_freqs_pairs_seg_merged_dfs[filter_label] = df_aa_freqs_pairs_seg_merged

# %%

nrows = len(OFFSETS)
ncols = 2

for (
    filter_label,
    df_aa_freqs_pairs_seg_merged,
) in label_aa_freqs_pairs_seg_merged_dfs.items():
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.25 * ncols, 5 * nrows))

    # Freq ratio matrices
    for j, (offset, pre_post) in enumerate(product(OFFSETS, ["pre", "post"])):
        ax = axes[j // ncols, j % ncols]

        _freq_ratios = df_aa_freqs_pairs_seg_merged[f"freq_ratio_{pre_post}-o{offset}"]
        _freq_ratios_mat = np.reshape(_freq_ratios.values, (N_AAS, N_AAS))

        im = ax.matshow(_freq_ratios_mat, vmin=0, vmax=np.ceil(_freq_ratios_mat.max()))
        ax.set_xticks(np.arange(N_AAS), sorted(ACIDS_1TO3))
        ax.set_yticks(np.arange(N_AAS), sorted(ACIDS_1TO3))
        ax.set_title(f"{pre_post}-seg, offset={offset}")
        ax.set_ylabel("AA1")
        ax.set_xlabel("AA2")
        fig.colorbar(im, ax=ax, location="right", fraction=0.1, shrink=0.9)

    fig.suptitle(f"Propensity ratio in unmodelled segments ({filter_label})")
    plt.show()

    _out_path = OUT_DIR / f"{filepath_aa_freqs_pairs.stem}-seg_{filter_label}.png"
    fig.savefig(_out_path, bbox_inches="tight")
    print(f"Wrote {_out_path}")

# %%
