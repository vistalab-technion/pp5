"""Two-sample statistical tests, split by test statistic into submodules:
:mod:`.common` (shared permutation-test engine), :mod:`.tw` (Welch T-squared),
:mod:`.mmd` (MMD), :mod:`.kde` (KDE-L1), and :mod:`.torustests` (flat-torus
Wasserstein tests). Re-exported here so ``pp5.stats.two_sample`` keeps working
exactly as it did as a single module.
"""

from pp5.stats.two_sample.common import (
    two_sample_kernel_permutation_test,
    two_sample_kernel_permutation_test_inner,
)
from pp5.stats.two_sample.kde import (
    kde_2d_slab_stacks,
    _kde_statistic,
    _kde_statistic_pergroup,
    kde2d_test,
    kde2d_test_pergroup,
    kde2d_test_pergroup_fast,
    kde_l1_permutation_test_from_slabs,
)
from pp5.stats.two_sample.mmd import (
    mmd_permutation_test_from_kernel,
    mmd_statistic,
    mmd_statistic_unbiased,
    mmd_test,
    mmd_test_fast,
)
from pp5.stats.two_sample.torustests import (
    PY2R_CONVERTER,
    R_TORUSTEST_GEODESIC,
    R_TORUSTEST_SIM_NULL_STAT,
    R_TORUSTEST_UBOUND,
    torus_projection_permutation_test,
    torus_projection_test,
    torus_projection_test_null_samples,
    torus_w2_ub_test,
)
from pp5.stats.two_sample.tw import _tw2_statistic, tw_test
