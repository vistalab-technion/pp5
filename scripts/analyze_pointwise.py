#!/usr/bin/env python

import os
import sys
import time
import subprocess
from pathlib import Path


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
from pp5.utils import elapsed_seconds_to_dhms

PROCESSES = 30

TUPLE_LEN = 1
MIN_GROUP = 1
KDE_NBINS = 128
KDE_WIDTH = 200
DDIST_BS_NITER = 1 # 25
DDIST_K = 5000 # 200
DDIST_K_MIN = 100
DDIST_K_TH = 50
DDIST_NMAX = 1000 #200
DDIST_NMAX_AA = True # Limit n_max per AA based on smallest codon
DDIST_STATISTIC = "kde_g"  # 'tw', 'mmd', 'kde'
DDIST_KERNEL_SIZE = 16.0 # 2.0
CODON_GROUPING_TYPE = ""  # "", "any", "last_nucleotide"
CODON_GROUPING_POSITION = "1"  # 0,1
FDR = 0.05
COMPARISON_TYPES = [
    # "aa",
    "cc",
]
SS_GROUP_ANY = True # Include group of all SS?
IGNORE_OMEGA = True
RANDOMIZE_CODONS = "" # "aa", "aa_ss" or "" for no randomization
SELF_TEST = False # whether to compare codons to themselves

DATASET_PATHS = [
    # Path("out/prec-collected/20230730_063523-aida-ex_EC-src_EC-re/"),
    # Path("out/prec-collected/20211001_124553-aida-ex_EC-src_EC/")
    Path("out/prec-collected/cope_sim_w_ena/")
]

DATASETS = {
    # Create a name for each dataset path
    path.name: path
    for path in DATASET_PATHS
}

OUT_DIR = Path("out")

for i, (dataset_name, dataset_path) in enumerate(DATASETS.items()):

    ddist_statistic_tag = f"{DDIST_STATISTIC}_{DDIST_KERNEL_SIZE}"

    codon_grouping_tag = ""
    if CODON_GROUPING_TYPE:
        codon_grouping_tag = f"-g_{CODON_GROUPING_TYPE}_star{CODON_GROUPING_POSITION}"

    codon_randomization_tag = ""
    if RANDOMIZE_CODONS:
        codon_randomization_tag = f"-cr_{RANDOMIZE_CODONS}"

    self_test_tag = ""
    if not SELF_TEST:
        self_test_tag = "-noself"


    tag = f"t_{TUPLE_LEN}-bs_{DDIST_BS_NITER}-k_{DDIST_K}-nmax_{DDIST_NMAX}-{ddist_statistic_tag}{codon_grouping_tag}{codon_randomization_tag}{self_test_tag}"

    command_line = [
        "pp5",
        f"--processes={PROCESSES}",
        f"analyze-pointwise",
        f"--dataset-dir={dataset_path!s}",
        f"--min-group-size={MIN_GROUP}",
        f"--tuple-len={TUPLE_LEN}",
        f"--codon-grouping-position={CODON_GROUPING_POSITION}" if TUPLE_LEN>1 else "",
        f"--codon-grouping-type={CODON_GROUPING_TYPE}" if TUPLE_LEN>1 else "",
        f"--kde-width={KDE_WIDTH}",
        f"--kde-nbins={KDE_NBINS}",
        f"--ddist-statistic={DDIST_STATISTIC}",
        f"--ddist-k={DDIST_K}",
        f"--ddist-k-min={DDIST_K_MIN}",
        f"--ddist-k-th={DDIST_K_TH}",
        f"--ddist-bs-niter={DDIST_BS_NITER}",
        f"--ddist-n-max={DDIST_NMAX}",
        f"--no-ddist-n-max-aa" if not DDIST_NMAX_AA else "",
        f"--ddist-kernel-size={DDIST_KERNEL_SIZE}",
        f"--fdr={FDR}",
        f"--comparison-types={str.join(',',COMPARISON_TYPES)}",
        f"--randomize-codons={RANDOMIZE_CODONS}",
        f"--no-self-test" if not SELF_TEST else "",
        f"--ss-group-any" if SS_GROUP_ANY else "",
        f"--ignore-omega" if IGNORE_OMEGA else "",
        f"--out-tag={tag}",
    ]

    command_line = [c for c in command_line if c]

    out_file_path = OUT_DIR.joinpath(f"analyze-pointwise_{dataset_name}-{tag}.log")

    with open(out_file_path, "w") as out_file:

        # Write to output file and console
        for f in [out_file, sys.stdout]:
            print(f"### EXECUTING COMMAND:\n{str.join(' ', command_line)}", file=f)
            print(f"### OUTPUT FILE: {out_file_path.absolute()}", file=f, end="\n\n")
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
                print(f"### DONE ({return_code=}), ELAPSED={elapsed} TAG={tag}", file=sys.stdout)
                break
            except subprocess.TimeoutExpired as e:
                print(".", end="", file=sys.stdout)
                sys.stdout.flush()
            except KeyboardInterrupt as e:
                print("### USER INTERRUPT, EXITING")
                sys.exit(1)
