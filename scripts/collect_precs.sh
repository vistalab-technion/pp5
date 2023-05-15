#!/bin/bash

PROCESSES=90

EXPR_HUMAN="Homo sapiens"
EXPR_ECOLI="Escherichia Coli"
EXPR_INSECT="Spodoptera"

SRC_HUMAN="9606"
SRC_ECOLI="562"
SRC_BACTERIA="2"
SRC_ALL=""

RESOLUTION="1.8"
SIMILARITY="0.7"

TIMEOUT="1200"

TAG="r${RESOLUTION}_s${SIMILARITY}"

set -eux

# EC in EC
# pp5 \
#     -p="$PROCESSES" collect-prec \
#     --expr-sys="$EXPR_ECOLI" \
#     --source-taxid="$SRC_ECOLI" \
#     --resolution="$RESOLUTION" \
#     --seq-similarity-thresh="$SIMILARITY" \
#     --out-tag="ex_EC-src_EC-$TAG" \
#     --async-timeout="$TIMEOUT" \
#     --with-contacts \
#     --with-backbone \
#     --no-write-csv

# Bacteria in EC (Deposited before 2007)
# pp5 \
#     -p="$PROCESSES" collect-prec \
#     --expr-sys="$EXPR_ECOLI" \
#     --source-taxid="$SRC_BACTERIA" \
#     --resolution="$RESOLUTION" \
#     --seq-similarity-thresh="$SIMILARITY" \
#     --out-tag="ex_EC-src_BAC-t2007-$TAG" \
#     --async-timeout="$TIMEOUT" \
#     --with-contacts \
#     --with-backbone \
#     --deposition-min-date="" \
#     --deposition-max-date="2007-01-01" \
#     --no-write-csv

# Bacteria in EC (Deposited between 2019 and 2022)
# pp5 \
#     -p="$PROCESSES" collect-prec \
#     --expr-sys="$EXPR_ECOLI" \
#     --source-taxid="$SRC_BACTERIA" \
#     --resolution="$RESOLUTION" \
#     --seq-similarity-thresh="$SIMILARITY" \
#     --out-tag="ex_EC-src_BAC-f2019t2022-$TAG" \
#     --async-timeout="$TIMEOUT" \
#     --with-contacts \
#     --with-backbone \
#     --deposition-min-date="2019-01-01" \
#     --deposition-max-date="2022-01-01" \
#     --no-write-csv

# Human in EC (Deposited between 2019 and 2022)
#pp5 \
#    -p="$PROCESSES" collect-prec \
#    --expr-sys="$EXPR_ECOLI" \
#    --source-taxid="$SRC_HUMAN" \
#    --resolution="$RESOLUTION" \
#    --seq-similarity-thresh="$SIMILARITY" \
#    --out-tag="ex_EC-src_HS-f2019t2022-$TAG" \
#    --async-timeout="$TIMEOUT" \
#    --with-contacts \
#    --with-backbone \
#    --deposition-min-date="2019-01-01" \
#    --deposition-max-date="2022-01-01" \
#    --no-write-csv


# # All in EC
# pp5 \
#   -p="$PROCESSES" collect-prec \
#   --expr-sys="$EXPR_ECOLI" \
#   --pdb-source="re" \
#   --source-taxid="$SRC_ALL" \
#   --out-tag="ex_EC-src_ALL-$TAG-re-no-res-filter" \
#   --async-timeout="$TIMEOUT" \
#   --with-backbone
##   --no-write-csv

 pp5 \
   -p="$PROCESSES" collect-prec \
   --expr-sys="$EXPR_ECOLI" \
   --pdb-source="af" \
   --resolution="100.0" \
   --source-taxid="$SRC_ALL" \
   --out-tag="ex_EC-src_ALL-$TAG-af-no-res-filter" \
   --async-timeout="$TIMEOUT" \
   --with-backbone
#   --no-write-csv

# # Human in EC
# pp5 \
#     -p="$PROCESSES" collect-prec \
#     --expr-sys="$EXPR_ECOLI" \
#     --source-taxid="$SRC_HUMAN" \
#     --resolution="$RESOLUTION" \
#     --seq-similarity-thresh="$SIMILARITY" \
#     --out-tag="ex_EC-src_HS-$TAG" \
#     --async-timeout="$TIMEOUT" \
#     --no-write-csv

# # Human in Insect
# pp5 \
#     -p="$PROCESSES" collect-prec \
#     --expr-sys="$EXPR_INSECT" \
#     --source-taxid="$SRC_HUMAN" \
#     --resolution="$RESOLUTION" \
#     --seq-similarity-thresh="$SIMILARITY" \
#     --out-tag="ex_SP-src_HS-$TAG" \
#     --async-timeout="$TIMEOUT" \
#     --no-write-csv

# # All in Insect
# pp5 \
#     -p="$PROCESSES" collect-prec \
#     --expr-sys="$EXPR_INSECT" \
#     --source-taxid="$SRC_ALL" \
#     --resolution="$RESOLUTION" \
#     --seq-similarity-thresh="$SIMILARITY" \
#     --out-tag="ex_SP-src_ALL-$TAG" \
#     --async-timeout="$TIMEOUT" \
#     --no-write-csv
