#!/bin/bash

PROCESSES=84

EXPR_HUMAN="Homo sapiens"
EXPR_ECOLI="Escherichia Coli"
EXPR_INSECT="Spodoptera"
EXPR_ALL=""

SRC_HUMAN="9606"
SRC_ECOLI="562"
SRC_BACTERIA="2"
SRC_ALL=""

ASYNC_TIMEOUT="3600"
ASYNC_RETRY_DELTA="5"


set -eux

# Clear prec CSV output dir
rm -rf out/prec

# Clear global pp5 cache
rm -rf /tmp/pp5_data

# Altlocs
RESOLUTION="3.5"
RFREE="0.33"
SIMILARITY="1.0"
PDB_SOURCE="rc" # rc, re, af
MAX_CHAINS="20"
TAG="r${RESOLUTION}-${PDB_SOURCE}"
pp5 \
  -p="$PROCESSES" collect-prec \
  --async-timeout="$ASYNC_TIMEOUT" \
  --async-retry-delta="$ASYNC_RETRY_DELTA" \
  --expr-sys="$EXPR_ALL" \
  --source-taxid="$SRC_ALL" \
  --resolution="$RESOLUTION" \
  --r-free="$RFREE" \
  --query-max-chains="$MAX_CHAINS" \
  --seq-similarity-thresh="$SIMILARITY" \
  --pdb-source="$PDB_SOURCE" \
  --out-tag="altlocs-$TAG" \
  --with-altlocs \
  --with-backbone \
  --with-contacts \
  --write-zip

# For amino-domino paper
# All in EC, pdb-redo, 1.5A, with contacts and backbone
# RESOLUTION="1.5"
# SIMILARITY="0.7"
# PDB_SOURCE="re" # rc, re, af
# TAG="r${RESOLUTION}_s${SIMILARITY}-${PDB_SOURCE}"
# pp5 \
#   -p="$PROCESSES" collect-prec \
#   --expr-sys="$EXPR_ECOLI" \
#   --source-taxid="$SRC_ALL" \
#   --resolution="$RESOLUTION" \
#   --seq-similarity-thresh="$SIMILARITY" \
#   --pdb-source="$PDB_SOURCE" \
#   --out-tag="ex_EC-src_ALL-$TAG" \
#   --async-timeout="$ASYNC_TIMEOUT" \
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
#     --async-timeout="$ASYNC_TIMEOUT" \
#     --no-write-csv
