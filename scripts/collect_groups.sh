#!/bin/bash

PROCESSES=16

EXPR_HUMAN="Homo sapiens"
EXPR_ECOLI="Escherichia Coli"
EXPR_INSECT="Spodoptera"

SRC_HUMAN="9606"
SRC_ECOLI="562"
SRC_ALL=""

RESOLUTION="1.8"
REJECTION_ARGS="--b-max=30"

set -x

python pp5.py \
    -p="$PROCESSES" collect-pgroup \
    --expr-sys="$EXPR_ECOLI" \
    --source-taxid="$SRC_HUMAN" \
    --resolution="$RESOLUTION" $REJECTION_ARGS \
    --out-tag ex_EC-src_HS

python pp5.py \
    -p="$PROCESSES" collect-pgroup \
    --expr-sys="$EXPR_ECOLI" \
    --source-taxid="$SRC_ALL" \
    --resolution="$RESOLUTION" $REJECTION_ARGS \
    --out-tag ex_EC-src_ALL

python pp5.py \
    -p="$PROCESSES" collect-pgroup \
    --expr-sys="$EXPR_INSECT" \
    --source-taxid="$SRC_HUMAN" \
    --resolution="$RESOLUTION" $REJECTION_ARGS \
    --out-tag ex_SP-src_HS

python pp5.py \
    -p="$PROCESSES" collect-pgroup \
    --expr-sys="$EXPR_INSECT" \
    --source-taxid="$SRC_ALL" \
    --resolution="$RESOLUTION" $REJECTION_ARGS \
    --out-tag ex_SP-src_ALL
