import logging

import pandas as pd

from pp5 import OUT_DIR as PP5_OUT_DIR

PREC_COLLECTED_OUT_DIR = PP5_OUT_DIR / "prec-collected"
DATASET_ZIPFILE_PATH = (
    PREC_COLLECTED_OUT_DIR / "20240615_185854-floria-unmod-r3.5-rc.zip"
)

DATASET_DIR_PATH = PREC_COLLECTED_OUT_DIR / "20240615_185854-floria-unmod-r3.5-rc"

logging.basicConfig(level=logging.INFO)

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)
pd.set_option("display.max_colwidth", 20)
