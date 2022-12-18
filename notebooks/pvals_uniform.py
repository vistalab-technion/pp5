import logging
from math import ceil
from functools import partial
from itertools import product

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from pp5.parallel import pool, yield_async_results
from pp5.stats.two_sample import tw_test, kde2d_test
from pp5.distributions.kde import torus_gaussian_kernel_2d
from pp5.analysis.pointwise import (
    _subgroup_permutation_test as subgroup_permutation_test,
)

matplotlib.rcParams["font.size"] = 22

# logging.getLogger("pp5").level=logging.WARNING

kde_n_bins = 128
sigma = 2.0

# bs_n_iter = 25  # 25
ddist_n_max = 200
# ddist_k = 200
ddist_k_min = 100  # ddist_k  # 100
ddist_k_th = 50.0  # float("inf")  # 50.0
randstate = 4242

ddist_statistic_fn = partial(
    # kde2d_test,
    tw_test,
    # n_bins=kde_n_bins,
    # grid_low=-np.pi,
    # grid_high=np.pi,
    # dtype=np.float64,
    # kernel_fn=partial(
    #     torus_gaussian_kernel_2d, sigma=np.deg2rad(sigma),
    # ),
)


def _calc_single_pval(
    n_observations: int,
    bs_n_iter: int,
    ddist_k: int,
    early_stop: bool,
    d=2,
    *args,
    **kwargs,
):
    rng = np.random.default_rng()
    loc, scale = 0, 1

    bootstrap_permutation_test = lambda x, y: subgroup_permutation_test(
        group_idx="group_idx",
        subgroup1_idx="X",
        subgroup2_idx="Y",
        subgroup1_data=x,
        subgroup2_data=y,
        randstate=randstate,
        ddist_statistic_fn=ddist_statistic_fn,
        ddist_statistic_fn_name="",
        ddist_bs_niter=bs_n_iter,
        ddist_n_max=ddist_n_max,
        ddist_k=ddist_k,
        ddist_k_min=ddist_k_min if early_stop else ddist_k,
        ddist_k_th=ddist_k_th if early_stop else float("inf"),
    )

    X = rng.normal(loc, scale, (n_observations, d))
    Y = rng.normal(loc, scale, (n_observations, d))
    ddist, pval = bootstrap_permutation_test(X, Y)
    return pval


def _calc_all_pvals():

    n_observations_vals = [100]
    n_samples_vals = [200]
    early_stop_vals = [False, True]

    params = list(product(n_observations_vals, n_samples_vals, early_stop_vals))
    n_params = len(params)
    fig_rows, fig_cols = ceil(n_params / 2), 2

    fig, axes = plt.subplots(
        fig_rows,
        fig_cols,
        squeeze=True,
        figsize=(10 * fig_cols, 7 * fig_rows),
        layout="constrained",
    )

    with pool("pval", processes=8, context="spawn") as p:
        for i, (n_observations, n_samples, early_stop) in enumerate(params):
            ax = axes[i]

            for B, K in zip([1, 10, 25], [5000, 500, 200]):
                pvals = p.starmap(
                    _calc_single_pval, [(n_observations, B, K, early_stop)] * n_samples
                )

                ax.hist(pvals, bins=50, alpha=0.5, label=f"{B=}, {K=}")

            ax.set_xlim([0, 1])
            ax.set_title(f"{early_stop=}, N={n_observations}, M={n_samples}")
            ax.legend()

    fig.savefig(f"out/pvals-synth_control.png", dpi=150)


if __name__ == "__main__":
    _calc_all_pvals()
    plt.show()
