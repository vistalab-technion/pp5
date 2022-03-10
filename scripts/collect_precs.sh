#!/bin/bash

PROCESSES=64

EXPR_HUMAN="Homo sapiens"
EXPR_ECOLI="Escherichia Coli"
EXPR_INSECT="Spodoptera"

SRC_HUMAN="9606"
SRC_ECOLI="562"
SRC_ALL=""

RESOLUTION="1.2"
SIMILARITY="0.7"

TIMEOUT="240"

TAG="r${RESOLUTION}_s${SIMILARITY}"

set -eux

pp5 \
    -p="$PROCESSES" collect-prec \
    --expr-sys="$EXPR_ECOLI" \
    --source-taxid="$SRC_ECOLI" \
    --resolution="$RESOLUTION" \
    --seq-similarity-thresh="$SIMILARITY" \
    --out-tag="ex_EC-src_EC-$TAG" \
    --async-timeout="$TIMEOUT" \
    --no-write-csv

pp5 \
    -p="$PROCESSES" collect-prec \
    --expr-sys="$EXPR_ECOLI" \
    --source-taxid="$SRC_HUMAN" \
    --resolution="$RESOLUTION" \
    --seq-similarity-thresh="$SIMILARITY" \
    --out-tag="ex_EC-src_HS-$TAG" \
    --async-timeout="$TIMEOUT" \
    --no-write-csv

5 \
  -p="$PROCESSES" collect-prec \
  --expr-sys="$EXPR_ECOLI" \
  --source-taxid="$SRC_ALL" \
  --resolution="$RESOLUTION" \
  --seq-similarity-thresh="$SIMILARITY" \
  --out-tag="ex_EC-src_ALL-$TAG" \
  --async-timeout="$TIMEOUT" \
  --no-write-csv

pp5 \
    -p="$PROCESSES" collect-prec \
    --expr-sys="$EXPR_INSECT" \
    --source-taxid="$SRC_HUMAN" \
    --resolution="$RESOLUTION" \
    --seq-similarity-thresh="$SIMILARITY" \
    --out-tag="ex_SP-src_HS-$TAG" \
    --async-timeout="$TIMEOUT" \
    --no-write-csv

pp5 \
    -p="$PROCESSES" collect-prec \
    --expr-sys="$EXPR_INSECT" \
    --source-taxid="$SRC_ALL" \
    --resolution="$RESOLUTION" \
    --seq-similarity-thresh="$SIMILARITY" \
    --out-tag="ex_SP-src_ALL-$TAG" \
    --async-timeout="$TIMEOUT" \
    --no-write-csv
