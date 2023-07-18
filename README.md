# pp5

This repo contains an implementation of a toolkit for analysis of protein backbone
structure, specifically for: (i) estimating the distribution of dihedral angles and
quantifying the differences between such distributions; (ii) finding matched pairs
of proteins with regions of identical sequence and contacts but different backbone
structure.

It contains the code required to collect the data and reproduce the results of
these papers:

    Aviv A. Rosenberg, Alex M. Bronstein, Ailie Marx.
    "Does one sequence always translate to one structure?"
    Unpublished (2023).

    Aviv A. Rosenberg, Nitsan Yehishalom, Ailie Marx, Alex Bronstein.
    "An amino domino model described by a cross peptide bond Ramachandran plot
    defines amino acid pairs as local structural units"
    Unpublished (2023).

    Aviv A. Rosenberg, Ailie Marx and Alex M. Bronstein.
    "Codon-specific Ramachandran plots show amino acid backbone conformation depends on
    identity of the translated codon".
    Nature Communications (2022).

When using this code, please cite the relevant work.

## Initial set-up

This package was developed and tested on both Linux and macOS.
It might work on Windows, however this was not tested and is not supported.

1. Install the python3 version of mamba (or conda).
   If installing from scratch, follow the installation instructions
   [here](https://github.com/conda-forge/miniforge).
   Note that it's strongly recommended to use `mamba` instead of
   `conda` for this project, since it's much faster to solve the environment.
   In case you have a pre-existing installation of `conda`, you can  install
   `mamba` in addition by running `conda install mamba -n base -c conda-forge`.
2. If on Apple slicon hardware (M1/M2 mac) run `export CONDA_SUBDIR=osx-64`.
   before installing the environments.
2. Install the `arpeggio` environment by running
   ```shell
   mamba env create -n arpeggio -f environment-arpeggio.yml
   ```
   This is only required for tertiary contact analysis. Note that arpeggio is
   installed into a separate environment because it requires packages which are
   incompatible with the main pp5 environment.
3. Install the `pp5` environment by running
   ```shell
   mamba env create -n pp5 -f environment.yml
   ```
4. Test the arpeggio installation by running
   ```shell
   mamba run -n arpeggio arpeggio --help
   ```
   You should see an arpeggio help message and usage info.
4. Activate the main environment by running
   ```shell
   mamba activate pp5
   ```
5. Install the `pp5` package itself: `pip install -e .` (make sure to note the `.`).
6. To make sure everything is working, run all the tests by running `pytest`.

## Using the CLI

Some examples of using the CLI are provided below. Use the `--help` flag to see all
options. For example, to see available commands:
```shell script
pp5 --help
```
To see available options for one command (e.g. pgroup):
```shell script
pp5 pgroup --help
```

To collect a single protein record with default options:
```shell script
pp5 prec --pdb-id 2WUR:A
```
This will generate output CSV files in the `out/prec` directory.

To collect a single protein group, where a reference protein is matched by sequence
and structure to query structures and the potential-contact environments are compared:
```shell script
pp5 pgroup --ref-pdb-id 2WUR:A --match-len 2 --context-len 1 --compare-contacts
```
This will generate output CSV files in the `out/prgroup` directory.

## Reproducing "Does one sequence always translate to one structure?"

The data collection and structure pair matching can be performed by running `pp5
collect-pgroup`, with appropriate options provided as explained below.

### Running contact analysis and structure pair matching

To re-collect the data used for the analysis and generate the raw list of protein
structure pairs with matching sequence and contacts but different structure, use the
following bash script.

```shell
#!/bin/bash

PROCESSES=90
EXPR_ECOLI="Escherichia Coli"
SRC_ALL=""
RESOLUTION="1.8"
REJECTION_ARGS="--b-max=50 --plddt-min=70 --sa-outlier-cutoff=2.5 --angle-aggregation=max_res"
MATCH_ARGS="--match-len=2 --context-len=1"
PDB_SOURCE="re" # rc, re, af

pp5 \
    -p="$PROCESSES" collect-pgroup \
    --expr-sys="$EXPR_ECOLI" \
    --source-taxid="$SRC_ALL" \
    --resolution="$RESOLUTION" \
    $REJECTION_ARGS \
    $MATCH_ARGS \
    --no-strict-codons \
    --pdb-source=$PDB_SOURCE \
    --out-tag "ex_EC-src_ALL-${RESOLUTION/./}-$PDB_SOURCE"
```

Note that the `PROCESSES` variable controls the number of concurrent processes
used for the collection and analysis. It can generally be set close to the
number of cores available on the machine. Running this analysis on the entire
PDB can take several days, depending on the number of available cores.  To run
on smaller subsets of the PDB, you can restrict the search using the supplied
options, or even collect just a single protein group as shown in the previous
section.

## Reproducing "An Amino Domino Model"

To generate the clusters and sub-clusters shown in the paper, use the `notebooks/generate_clusters.ipynb`. Point the `DATASET_FILE` in the notebook to the dataset file from the supplementary data (`precs-non-redundant.csv`). Alternatively, you may re-collect the dataset as described below.

### Re-collecting the data

To re-collect the data used for the analysis, use the following bash script,
which will collect the relevant non-redundant protein structures from the PDB,
and extract their backbone atom positions.  Note that due to updates on the PDB
servers over time, re-collecting the data will not produce exactly the same
dataset as was analyzed in the paper.

```shell
#!/bin/bash

PROCESSES=64
TAG="r${RESOLUTION}_s${SIMILARITY}"
EXPR_ECOLI="Escherichia Coli"
SRC_ALL=""
TIMEOUT="240"
RESOLUTION="1.5"
SIMILARITY="0.7"
PDB_SOURCE="re"

set -eux
pp5 -p="$PROCESSES" collect-prec \
  --expr-sys="$EXPR_ECOLI" \
  --source-taxid="$SRC_ALL" \
  --resolution="$RESOLUTION" \
  --seq-similarity-thresh="$SIMILARITY" \
  --pdb-source=$PDB_SOURCE \
  --out-tag="ex_EC-src_ALL-$TAG" \
  --async-timeout="$TIMEOUT" \
  --with-backbone \
  --no-write-csv
```

The data will be collected to a subfolder with a name containing the `out-tag`,
within the `out/` folder (which will be created in the `pwd`). Within the
collected data folder, the relevant dataset file is `data-precs.csv`.

## Reproducing "Codon Specific Ramachandran Plots"

The data collection can be performed by running`pp5 collect-prec` (with appropriate
options), and the analysis can be performed by running `pp5 analyze-pointwise` (with
appropriate options).

### Running the analysis

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

### Re-collecting the data

To re-collect the data used for the analysis, use the following bash script.
Note that due to updates on the PDB servers over time, re-collecting the data will not
produce exactly the same dataset as was analyzed in the paper.

```shell
#!/bin/bash

PROCESSES=64
TAG="r${RESOLUTION}_s${SIMILARITY}"
EXPR_ECOLI="Escherichia Coli"
SRC_ECOLI="562"
RESOLUTION="1.8"
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

The data will be collected to a subfolder with a name containing the `out-tag`, within the `out/` folder (which will be created in the `pwd`). The analysis command should then be pointed to the collected data folder.
