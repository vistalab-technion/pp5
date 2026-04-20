#!/usr/bin/env python

import os
import subprocess
import sys
import time
from pathlib import Path

from pp5.utils import elapsed_seconds_to_dhms


def find_repo_root(max_levels=5):
    repo_root = Path(os.getcwd())
    level = 0
    while True:
        if repo_root.joinpath(".git").is_dir():
            break
        repo_root = repo_root.parent
        level += 1
        if level >= max_levels:
            raise RuntimeError("Can't find repo root")
    return str(repo_root)


REPO_ROOT = find_repo_root()
os.chdir(REPO_ROOT)

sys.path.append(REPO_ROOT)

PROCESSES = 4

TAG = None  # None means auto-generate based on settings below
OUT_DIR = None  # None means default location: dataset_dir/results/analysis_name-tag

# TAG = "TEMP"
OUT_DIR = "out/pnas-2026"

# Statistical test type
# Statistical test to use for quantifying significance of distances between
# distributions. Can be one of:
# - 'kde_v': Permutation test with KDE-L1 test statistic and with von Mises kernel
# - 'kde_g': As above, but with Gaussian kernel on torus
# - 'mmd': Permutation test with flat-torus distance, MMD test staistic and Gaussian
#   kernel.
# - 'tw': Permutation test with flat-torus distance, Welch t test-statistic.
# - 'torus_perm': Permutation test with distance based on S1 Wasserstein
#   distance after projecting torus data to S1.
# - 'torus_ub': Upper bound of pval based on torus Wasserstein distance.
#   Not a permutation test. ddist_k must be zero.
# - 'torus_p': Pval based on 1d Wasserstein distance on S1, after projecting.
#   Not a permutation test. ddist_k must be zero.
DDIST_STATISTIC = "kde_g"  # 'kde_g', 'torus_p', 'torus_perm'

# Statistical test settings
DDIST_BS_NITER = 1  # 1 to disable bootstrapping
DDIST_K = 5000  # permuations test iterations, 0 to disable
DDIST_K_MIN = 100  # Min permutations for early stopping
DDIST_K_TH = 100  # Threshold for early stopping if pval > K_TH * 1/(K+1) after K_MIN
DDIST_NMAX = 0  # Max (codon) group size, zero means no limit
DDIST_NMAX_AA = False  # Limit n_max per AA based on smallest codon
FDR = 0.05  # False discovery rate for BH multiple hypothesis correction

# Statistical test control (null) settings
# codon randomization options are:
# - 'none': don't randomize codons
# - 'aa': randomize codons within each amino acid
# - 'aa_ss': randomize codons within each amino acid and secondary structure group
RANDOMIZE_CODONS = "none"  # 'none', 'aa', 'aa_ss'
SELF_TEST = False  # whether to compare codons to themselves as a control

# KDE-based statistical test params (for kde_g)
DDIST_KERNEL_SIZE = 10.0

# Torustest params (for torus_p and torus_perm)
DDIST_TORUS_N_PROJECTIONS = 4  # number of geodesics to project onto
DDIST_TORUS_RANDOM_PROJECTIONS = False  # False to used fixed geodesics

# KDE params for plotting
KDE_NBINS = 128
KDE_WIDTH = 200

# Codon grouping
CODON_GROUPING_TYPE = ""  # "", "any", "last_nucleotide"
CODON_GROUPING_POSITION = "1"  # 0,1

# Other analysis options
TUPLE_LEN = 1  # Set to 2 to analyze codon pairs
MIN_GROUP = 1  # Minimum number of samples from a (unp, unp_idx) location to aggregate
COMPARISON_TYPES = [
    # "aa", # Compate AA distributions
    "cc",  # Compate codon distributions
]
SS_GROUP_ANY = False  # Include group of all SS?
IGNORE_OMEGA = True

DATASET_PATHS = [
    # Should point to a collected dataset, with 'data-precs.csv' and 'meta.json'
    Path("out/prec-collected/20211001_124553-aida-ex_EC-src_EC/")
]

DATASETS = {
    # Create a name for each dataset path
    path.name: path
    for path in DATASET_PATHS
}


for i, (dataset_name, dataset_path) in enumerate(DATASETS.items()):
    ddist_statistic_tag = f"{DDIST_STATISTIC}"
    if DDIST_STATISTIC.startswith("torus"):
        ddist_statistic_tag = f"{ddist_statistic_tag}_nproj={DDIST_TORUS_N_PROJECTIONS}_randproj={DDIST_TORUS_RANDOM_PROJECTIONS}"
    else:
        ddist_statistic_tag = f"{ddist_statistic_tag}_bw={DDIST_KERNEL_SIZE}"

    codon_grouping_tag = ""
    if CODON_GROUPING_TYPE:
        codon_grouping_tag = f"-cg={CODON_GROUPING_TYPE}_star{CODON_GROUPING_POSITION}"

    codon_randomization_tag = ""
    if RANDOMIZE_CODONS:
        codon_randomization_tag = f"-cr={RANDOMIZE_CODONS}"

    self_test_tag = ""
    if not SELF_TEST:
        self_test_tag = "-noself"

    tag = TAG or (
        f"t={TUPLE_LEN}-bs={DDIST_BS_NITER}-k={DDIST_K}-nmax={DDIST_NMAX}-"
        f"{ddist_statistic_tag}{codon_grouping_tag}{codon_randomization_tag}{self_test_tag}"
    )

    command_line = [
        "pp5",
        f"--processes={PROCESSES}",
        f"analyze-pointwise",
        f"--dataset-dir={dataset_path!s}",
        f"--aggregation-min-group-size={MIN_GROUP}",
        f"--tuple-len={TUPLE_LEN}",
        f"--codon-grouping-position={CODON_GROUPING_POSITION}" if TUPLE_LEN > 1 else "",
        f"--codon-grouping-type={CODON_GROUPING_TYPE}" if TUPLE_LEN > 1 else "",
        f"--kde-width={KDE_WIDTH}",
        f"--kde-nbins={KDE_NBINS}",
        f"--ddist-statistic={DDIST_STATISTIC}",
        f"--ddist-k={DDIST_K}",
        f"--ddist-k-min={DDIST_K_MIN}",
        f"--ddist-k-th={DDIST_K_TH}",
        f"--ddist-bs-niter={DDIST_BS_NITER}",
        f"--ddist-n-max={DDIST_NMAX}",
        f"--ddist-torus-n-projections={DDIST_TORUS_N_PROJECTIONS}",
        (
            f"--no-ddist-torus-random-projections"
            if not DDIST_TORUS_RANDOM_PROJECTIONS
            else ""
        ),
        f"--no-ddist-n-max-aa" if not DDIST_NMAX_AA else "",
        f"--ddist-kernel-size={DDIST_KERNEL_SIZE}",
        f"--fdr={FDR}",
        f"--comparison-types={str.join(',', COMPARISON_TYPES)}",
        f"--randomize-codons={RANDOMIZE_CODONS}",
        f"--no-self-test" if not SELF_TEST else "",
        f"--ss-group-any" if SS_GROUP_ANY else "",
        f"--ignore-omega" if IGNORE_OMEGA else "",
        f"--out-tag={tag}",
        (f"--out-dir={OUT_DIR!s}" if OUT_DIR else ""),
    ]

    command_line = [c for c in command_line if c]

    log_dir = Path(OUT_DIR) if OUT_DIR is not None else Path("out")
    log_file_path = log_dir.joinpath(f"analyze-pointwise_{dataset_name}-{tag}.log")
    log_dir.mkdir(parents=True, exist_ok=True)

    with open(log_file_path, "w") as out_file:
        # Write to log file and console
        for f in [out_file, sys.stdout]:
            print(f"### EXECUTING COMMAND:\n{str.join(' ', command_line)}", file=f)
            print(f"### LOG FILE: {log_file_path.absolute()}", file=f, end="\n\n")
            f.flush()  # So that it appears at the top

        start_time = time.time()

        process = subprocess.Popen(
            args=command_line,
            stdout=out_file,
            stderr=out_file,
            text=True,
            encoding="utf-8",
        )

        # Wait for the process to end and print a '.' every few seconds
        while True:
            try:
                return_code = process.wait(timeout=10.0)
                elapsed = elapsed_seconds_to_dhms(time.time() - start_time)
                print("", file=sys.stdout)
                print(
                    f"### DONE ({return_code=}), ELAPSED={elapsed} TAG={tag}",
                    file=sys.stdout,
                )
                break
            except subprocess.TimeoutExpired as e:
                print(".", end="", file=sys.stdout)
                sys.stdout.flush()
            except KeyboardInterrupt as e:
                print("### USER INTERRUPT, EXITING")
                sys.exit(1)
