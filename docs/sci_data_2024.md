# Alternately Located Segments in Protein Crystal Structures (Scientific Data, 2024)

This document describes how to reproduce the data collection and contact-based structure
pair matching for:

> Aviv A. Rosenberg, Ailie Marx, Alex M. Bronstein.
> *A dataset of alternately located segments in protein crystal structures.*
> Scientific Data, 2024.

The data collection and structure pair matching are performed by `pp5 collect-pgroup`,
with options as explained below.

## Running contact analysis and structure pair matching

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

`PROCESSES` controls the number of concurrent processes used for the collection and
analysis. It can generally be set close to the number of cores available on the machine.
Running this analysis on the entire PDB can take several days, depending on the number
of available cores. To run on smaller subsets of the PDB, restrict the search using the
supplied options, or collect just a single protein group as shown in the
"Using the CLI" section of the top-level `README.md`.
