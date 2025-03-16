import logging
from pprint import pprint

import pandas as pd

from pp5 import OUT_DIR as PP5_OUT_DIR
from pp5.utils import YamlDict

LOGGER = logging.getLogger(__name__)

PREC_COLLECTED_OUT_DIR = PP5_OUT_DIR / "prec-collected"
DATASET_ZIPFILE_PATH = (
    PREC_COLLECTED_OUT_DIR / "20240615_185854-floria-unmod-r3.5-rc.zip"
)

DATASET_DIR_PATH = PREC_COLLECTED_OUT_DIR / "20240615_185854-floria-unmod-r3.5-rc"

PDB_ID_SUBSET = [
    "2BP5:A",
    "1914:A",
    "8A4A:A",
    "4QN9:A",
    "3PV2:A",
    "1UZL:A",
    "4MSO:A",
    "7ZKX:A",
    "2B3P:A",
    "4M2Q:A",
]
PDB_ID_SUBSET = sorted(PDB_ID_SUBSET)

UNMODELLED_OUT_DIR = PP5_OUT_DIR / "unmodelled"
UNMODELLED_OUT_DIR.mkdir(parents=True, exist_ok=True)

UNMODELLED_OUTPUTS_FILE_PATH = UNMODELLED_OUT_DIR / "outputs.yml"
UNMODELLED_OUTPUTS = YamlDict(UNMODELLED_OUTPUTS_FILE_PATH)
OUTPUT_KEY_ALLSEGS = "all_segments"
OUTPUT_KEY_ALLSEGS_FILTERED = "all_segments_filtered"

TO_CSV_KWARGS = dict(index=False)
READ_CSV_KWARGS = dict(na_filter=True, keep_default_na=False, na_values=["", "NaN"])

logging.basicConfig(level=logging.INFO)

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)
pd.set_option("display.max_colwidth", 20)
