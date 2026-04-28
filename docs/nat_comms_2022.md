# Codon-Specific Ramachandran Plots (Nature Communications, 2022)

This document describes how to reproduce the data collection and analysis for:

> Aviv A. Rosenberg, Ailie Marx, Alex M. Bronstein.
> *Codon-specific Ramachandran plots show amino acid backbone conformation depends on identity of the translated codon.*
> Nature Communications, 2022.

**Note:** A revisited and statistically corrected version of this analysis appears in our
2026 PNAS Brief Report. See `docs/pnas_2026.md` for that pipeline. The instructions
below reproduce the original 2022 results.

The data collection is performed by `pp5 collect-prec` and the pointwise analysis by
`pp5 analyze-pointwise`, each with appropriate options.

## Running the analysis

To run the analysis with the same configuration as the paper, use the following bash
script. Point `DATASET_DIR` to the folder containing the dataset published with the
paper.

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

Alternatively, a comparable Python script is available at
`scripts/analyze_pointwise.py` and can be used as a wrapper to reproduce the results.

## Re-collecting the data

To re-collect the data used for the analysis, use the following bash script. Note that
due to updates on the PDB servers over time, re-collecting the data will not produce
exactly the same dataset as was analyzed in the paper.

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

The data will be collected to a subfolder with a name containing the `out-tag`, within
the `out/` folder (created in the working directory). The analysis command should then
be pointed to the collected data folder.
