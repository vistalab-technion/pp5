#!/bin/bash

PROCESSES=90

EXPR_HUMAN="Homo sapiens"
EXPR_ECOLI="Escherichia Coli"
EXPR_INSECT="Spodoptera"
EXPR_ANY=""

SRC_HUMAN="9606"
SRC_ECOLI="562"
SRC_BACTERIA="2"
SRC_ALL=""

RESOLUTION="1.8"
REJECTION_ARGS="--b-max=50 --plddt-min=70 --sa-outlier-cutoff=2.5 --angle-aggregation=max_res"
MATCH_ARGS="--match-len=2 --context-len=1"
PDB_SOURCE="re" # rc, re, af

set -eux

# Clear prec CSV output dir
rm -rf out/pgroup

# Clear global pp5 cache
rm -rf /tmp/pp5_data

# No restriction on expr system
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
