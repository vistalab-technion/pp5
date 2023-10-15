#!/bin/bash

PROCESSES=64

EXPR_HUMAN="Homo sapiens"
EXPR_ECOLI="Escherichia Coli"
EXPR_INSECT="Spodoptera"

SRC_HUMAN="9606"
SRC_ECOLI="562"
SRC_BACTERIA="2"
SRC_ALL=""

RESOLUTION="1.5"
SIMILARITY="0.7"
PDB_SOURCE="re" # rc, re, af

TIMEOUT="1200"

TAG="r${RESOLUTION}_s${SIMILARITY}-${PDB_SOURCE}"

set -eux

# For amino-domino paper
# All in EC, pdb-redo, 1.5A, with contacts and backbone
# pp5 \
#   -p="$PROCESSES" collect-prec \
#   --expr-sys="$EXPR_ECOLI" \
#   --source-taxid="$SRC_ALL" \
#   --resolution=1.5 \
#   --seq-similarity-thresh=0.7 \
#   --pdb-source=$PDB_SOURCE \
#   --out-tag="ex_EC-src_ALL-$TAG" \
#   --async-timeout="$TIMEOUT" \
#   --with-backbone \
#   --with-contacts \
#   --no-write-csv

# For natcom paper
# EC in EC, 1.8A, similarity 0.7, regular pdb
# pp5 \
#     -p="$PROCESSES" collect-prec \
#     --expr-sys="$EXPR_ECOLI" \
#     --source-taxid="$SRC_ECOLI" \
#     --resolution=1.8 \
#     --seq-similarity-thresh=0.7 \
#     --pdb-source=rc \
#     --out-tag="ex_EC-src_EC-natcom" \
#     --async-timeout="$TIMEOUT" \
#     --no-write-csv

# New collection for codon pairs
pp5 \
    -p="$PROCESSES" collect-prec \
    --expr-sys="$EXPR_ECOLI" \
    --source-taxid="$SRC_ECOLI" \
    --resolution=1.8 \
    --seq-similarity-thresh=0.7 \
    --pdb-source=re \
    --out-tag="ex_EC-src_EC-re" \
    --async-timeout="$TIMEOUT" \
    --no-write-csv

