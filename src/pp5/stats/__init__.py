from pathlib import Path

import rpy2.robjects as robjects

from pp5.stats.mht import mht_bh
from pp5.stats.two_sample import (
    tw_test,
    mmd_test,
    kde2d_test,
    torus_w2_ub_test,
    two_sample_kernel_permutation_test,
)

# Import all R code files from the torustest R package.
_TORUSTEST_PATH = Path(__file__).parent.joinpath("torustest")
for r_file_path in _TORUSTEST_PATH.glob("**/*.[rR]"):
    robjects.r.source(str(r_file_path))
