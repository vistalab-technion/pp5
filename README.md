# pp5

This repo contains an implementation of a toolkit for analysis of protein backbone
structure, specifically for estimating the distribution of dihedral angles and
quantifying the differences between such distributions.

It contains all the code required to collect the data and reproduce the results of
our paper:

    Aviv A. Rosenberg, Ailie Marx and Alex M. Bronstein.
    "Codon-specific Ramachandran plots show amino acid backbone conformation depends on
    identity of the translated codon".
    Nature Communications (2022).

When using this code, please cite the above work.

## Initial set-up

1. Install the python3 version of [miniforge](https://github.com/conda-forge/miniforge).
   Follow the installation instructions for your platform. Note that it's
   recommended to install the `mambaforge` variant as using `mamba` is much faster than
   `conda`.
2. Install `conda-lock` by running
   ```shell
   conda install -n base -c conda-forge conda-lock
   ```
3. Use `conda-lock` to create a virtual environment for the project based on the
   supplied lock file. From the project root directory, run
   ```shell
   conda run --no-capture-output -n base conda-lock install -n pp5 environment.conda-lock.yml
   ```
   This will install all the necessary packages into a new conda virtual
   environment named `pp5`.
4. Activate the new environment by running
   ```shell
   conda activate pp5
   ```
5. Install the `pp5` package itself: `pip install -e .` (make sure to note the `.`).
6. To make sure everything is working, simply run all the tests by typing `pytest`.

## Using the CLI

To see available commands:
```shell script
python pp5.py --help
```

To see available options for one command (e.g. pgroup):
```shell script
python pp5.py prec --help
```

To create a protein record with default options:
```shell script
python pp5.py prec --pdb-id 2WUR:A
```

### Reproducing the results

The data collection can be performed by running`pp5 collect-prec` (with appropriate
options), and the analysis can be performed by running `pp5 analyze-pointwise` (with
appropriate options).

## Running the analysis

To run the analysis with the same configuration as in the paper, use the following
bash script. You may point the `DATASET_DIR` to the folder containing the dataset
published along with the paper.

```shell
#!/bin/bash

# Edit these to suit your needs
PROCESSES=90
DATASET_DIR="out/prec-collected/20211001_124553-aida-ex_EC-src_EC/"
TAG="natcom"

# Values used in the paper results
MIN_GROUP=1
KDE_NBINS=128
KDE_WIDTH=200
DDIST_BS_NITER=25
DDIST_K=200
DDIST_K_MIN=100
DDIST_K_TH=50
DDIST_NMAX=200
DDIST_STATISTIC="kde_g"
DDIST_KERNEL_SIZE=2.0
FDR=0.05

set -eux
pp5 -p="$PROCESSES" \
 analyze-pointwise \
 --dataset-dir="$DATASET_DIR" \
 --min-group-size="$MIN_GROUP" \
 --kde-width="$KDE_WIDTH" \
 --kde-nbins="$KDE_NBINS" \
 --ddist-statistic="$DDIST_STATISTIC" \
 --ddist-k="$DDIST_K" \
 --ddist-k-min="$DDIST_K_MIN" \
 --ddist-k-th="$DDIST_K_TH" \
 --ddist-bs-niter="$DDIST_BS_NITER" \
 --ddist-n-max="$DDIST_NMAX" \
 --ddist-kernel-size="$DDIST_KERNEL_SIZE" \
 --fdr="$FDR" \
 --comparison-types aa cc \
 --ignore-omega \
 --out-tag="$TAG"

```

Alternatively, a comparable python script is available in `scripts/analyze_pointwise.py`
which can be used as a wrapper to reproduce the results.

## Re-collecting the data

To re-collect the data used for the analysis, use the following bash script.
Note that due to updates on the PDB servers over time, re-collecting the data will not
produce exactly the same dataset as was analyzed in the paper.

```shell
#!/bin/bash

PROCESSES=64
TAG="r${RESOLUTION}_s${SIMILARITY}"

# Data collection parameters used in the paper.
EXPR_ECOLI="Escherichia Coli"
SRC_ECOLI="562"
RESOLUTION="1.2"
SIMILARITY="0.7"
TIMEOUT="240"

set -eux
pp5 -p="$PROCESSES" \
 collect-prec \
 --expr-sys="$EXPR_ECOLI" \
 --source-taxid="$SRC_ECOLI" \
 --resolution="$RESOLUTION" \
 --seq-similarity-thresh="$SIMILARITY" \
 --out-tag="ex_EC-src_EC-$TAG" \
 --async-timeout="$TIMEOUT" \
 --no-write-csv
```
