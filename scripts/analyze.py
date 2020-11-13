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

PROCESSES = 48

PREV_CONDITIONS = {
    "none": "",
    "acids": "aa",
    "codons": "codon",
}

PREV_TYPE = "none"

BS_NITER = 500
BS_FIXED = "min"
PARALLEL_KDES = 2  # was 15

DATASET_PATHS = [
    "out/pgroup-collected/20201102_180050-aida-ex_EC-src_HS",
    "out/pgroup-collected/20201102_201430-aida-ex_EC-src_ALL",
    "out/pgroup-collected/20201102_215425-aida-ex_SP-src_HS",
    "out/pgroup-collected/20201102_220145-aida-ex_SP-src_ALL",
]

DATASETS = {
    # Create a name for each dataset based on the tag
    str.join("-", path.split("-")[-2:]): path
    for path in DATASET_PATHS
}

OUT_DIR = Path("out")

for i, (dataset_name, dataset_path) in enumerate(DATASETS.items()):

    tag = f"n_{BS_FIXED}-prev_{PREV_TYPE}"

    command_line = [
        f"python",
        "pp5.py",
        f"-p={PROCESSES}",
        f"analyze-cdist",
        f"--dataset-dir={dataset_path}",
        f"--condition-on-prev={PREV_CONDITIONS[PREV_TYPE]}",
        f"--bs-niter={BS_NITER}",
        f"--bs-fixed-n={BS_FIXED}",
        f"--n-parallel-kdes={PARALLEL_KDES}",
        f"--out-tag={tag}",
    ]

    out_file_path = OUT_DIR.joinpath(f"analyze-cdist_{dataset_name}-{tag}.log")

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
                print(f"### DONE ({return_code=}), ELAPSED={elapsed}", file=sys.stdout)
                break
            except subprocess.TimeoutExpired as e:
                print(".", end="", file=sys.stdout)
                sys.stdout.flush()
            except KeyboardInterrupt as e:
                print("### USER INTERRUPT, EXITING")
                sys.exit(1)
