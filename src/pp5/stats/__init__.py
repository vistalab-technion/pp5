from pathlib import Path

import rpy2.robjects as robjects

from pp5.stats.mht import mht_bh
from pp5.stats.breakdown import breakdown_k_kde, breakdown_k_mmd, greedy_breakdown_k
from pp5.stats.controls import (
    gen_aa_ss_control_replicates,
    gen_pooled_shuffle_replicates,
    null_control_summary,
    randomized_codon_column,
)
from pp5.stats.two_sample import (
    tw_test,
    mmd_test,
    mmd_test_fast,
    mmd_statistic,
    mmd_statistic_unbiased,
    mmd_permutation_test_from_kernel,
    kde2d_test,
    kde2d_test_pergroup,
    kde2d_test_pergroup_fast,
    kde_l1_permutation_test_from_slabs,
    torus_w2_ub_test,
    two_sample_kernel_permutation_test,
    two_sample_kernel_permutation_test_inner,
)

# Import all R code files from the torustest R package.
_TORUSTEST_PATH = Path(__file__).parent.joinpath("torustest")
for r_file_path in _TORUSTEST_PATH.glob("**/*.[rR]"):
    robjects.r.source(str(r_file_path))
