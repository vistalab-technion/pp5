import re
import time
import logging
import itertools as it
import multiprocessing as mp
from typing import Dict, List, Tuple, Union, Callable, Optional
from pathlib import Path
from multiprocessing.pool import AsyncResult

import numpy as np
import pandas as pd
import matplotlib as mpl

import pp5.plot
from pp5.plot import PP5_MPL_STYLE
from pp5.utils import sort_dict
from pp5.codons import ACIDS, N_CODONS, AA_CODONS, SYN_CODON_IDX, codon2aac
from pp5.analysis import SS_TYPE_ANY, CODON_TYPE_ANY, DSSP_TO_SS_TYPE
from pp5.parallel import yield_async_results
from pp5.vonmises import BvMKernelDensityEstimator
from pp5.analysis.base import ParallelAnalyzer

LOGGER = logging.getLogger(__name__)


class PointwiseCodonDistanceAnalyzer(ParallelAnalyzer):
    def __init__(
        self,
        dataset_dir: Union[str, Path],
        out_dir: Union[str, Path] = None,
        pointwise_filename: str = "data-pointwise.csv",
        condition_on_prev="codon",
        condition_on_ss=True,
        consolidate_ss=DSSP_TO_SS_TYPE.copy(),
        strict_ss=True,
        strict_codons=True,
        cross_residue=False,
        kde_nbins=128,
        kde_k1=30.0,
        kde_k2=30.0,
        kde_k3=0.0,
        bs_niter=1,
        bs_randstate=None,
        bs_fixed_n: str = None,
        n_parallel_kdes: int = 8,
        out_tag: str = None,
    ):
        """
        Analyzes a pointwise dataset (dihedral angles with residue information)
        to produce a matrix of distances between codons Dij.
        Each entry ij in Dij corresponds to codons i and j, and the value is a
        distance metric between the distributions of dihedral angles coming
        from these codons.

        :param dataset_dir: Path to directory with the pointwise collector
        output.
        :param out_dir: Path to output directory. Defaults to <dataset_dir>/results.
        :param pointwise_filename: Filename of the pointwise dataset.
        :param consolidate_ss: Dict mapping from DSSP secondary structure to
        the consolidated SS types used in this analysis.
        :param condition_on_prev: What to condition on from previous residue of
        each sample. Options are 'codon', 'aa', or None/''.
        :param condition_on_ss: Whether to condition on secondary structure
        (of two consecutive residues, after consolidation).
        :param strict_ss: Enforce no ambiguous secondary structure in any residue.
        :param strict_codons: Enforce only one known codon per residue
        (reject residues where DNA matching was ambiguous).
        :param cross_residue: Whether to calculate pointwise distributions and codon
            distances for cross-residue angles, i.e. (phi+0,phi-1).
        :param kde_nbins: Number of angle binds for KDE estimation.
        :param kde_k1: KDE concentration parameter for phi.
        :param kde_k2: KDE concentration parameter for psi.
        :param kde_k3: KDE joint concentration parameter.
        :param bs_niter: Number of bootstrap iterations.
        :param bs_randstate: Random state for bootstrap.
        :param bs_fixed_n: Whether to fix number of samples in each
        bootstrap iteration for each subgroup to the number of samples in the
        smallest subgroup ('min'), largest subgroup ('max') or no limit ('').
        :param n_parallel_kdes: Number of parallel bootstrapped KDE
        calculations to run simultaneously.
        By default it will be equal to the number of available CPU cores.
        Setting this to a high number together with a high bs_niter will cause
        excessive memory usage.
        :param out_tag: Tag for output.
        """
        super().__init__(
            "pointwise_cdist",
            dataset_dir,
            pointwise_filename,
            out_dir,
            out_tag,
            clear_intermediate=False,
        )

        condition_on_prev = (
            "" if condition_on_prev is None else condition_on_prev.lower()
        )
        if condition_on_prev not in {"codon", "aa", ""}:
            raise ValueError(f"invalid condition_on_prev: {condition_on_prev}")

        if bs_fixed_n not in {None, "", "min", "max"}:
            raise ValueError(f"invalid bs_fixed_n: {bs_fixed_n}")

        self.condition_on_prev = condition_on_prev
        self.condition_on_ss = condition_on_ss
        self.consolidate_ss = consolidate_ss
        self.strict_ss = strict_ss
        self.strict_codons = strict_codons

        self.angle_pairs = [(f"phi+0", f"psi+0")]
        if cross_residue:
            self.angle_pairs += [(f"phi+0", f"psi-1")]
        self.angle_cols = sorted(set(it.chain(*self.angle_pairs)))
        self.codon_cols = [f"codon-1", f"codon+0"]
        self.codon_opts_cols = [f"codon_opts-1", f"codon_opts+0"]
        self.secondary_cols = [f"secondary-1", f"secondary+0"]
        self.secondary_col = "secondary"
        self.condition_col = "condition_group"

        self.kde_args = dict(n_bins=kde_nbins, k1=kde_k1, k2=kde_k2, k3=kde_k3)
        self.kde_dist_metric = "l2"

        self.bs_niter = bs_niter
        self.bs_randstate = bs_randstate
        self.bs_fixed_n = bs_fixed_n
        self.n_parallel_kdes = n_parallel_kdes

    def _collection_functions(
        self,
    ) -> Dict[str, Callable[[mp.pool.Pool], Optional[Dict]]]:
        return {
            "preprocess-dataset": self._preprocess_dataset,
            "dataset-stats": self._dataset_stats,
            "dihedral-kde-full": self._dihedral_kde_full,
            "codon-dists": self._codons_dists,
            "codon-dists-expected": self._codon_dists_expected,
            "plot-results": self._plot_results,
        }

    def _preprocess_dataset(self, pool: mp.pool.Pool) -> dict:
        # Specifies which columns to read from the CSV
        def col_filter(col_name: str):
            # Keep only columns from prev and current
            if col_name.endswith("-1") or col_name.endswith("+0"):
                return True
            return False

        df_pointwise_reader = pd.read_csv(
            str(self.input_file), usecols=col_filter, chunksize=10_000,
        )

        # Parallelize loading and preprocessing
        sub_dfs = pool.map(self._preprocess_subframe, df_pointwise_reader)

        # Dump processed dataset
        df_preproc = pd.concat(sub_dfs, axis=0, ignore_index=True)
        LOGGER.info(
            f"Loaded {self.input_file}: {len(df_preproc)} rows\n"
            f"{df_preproc}\n{df_preproc.dtypes}"
        )

        self._dump_intermediate("dataset", df_preproc)
        return {
            "n_TOTAL": len(df_preproc),
        }

    def _preprocess_subframe(self, df_sub: pd.DataFrame):
        # Logic for consolidating secondary structure between a pair of curr
        # and prev residues
        def ss_consolidator(row: pd.Series):
            ss_m1 = row[self.secondary_cols[0]]  # e.g. 'H' or e.g. 'H/G'
            ss_p0 = row[self.secondary_cols[1]]

            # In strict mode we require that all group members had the same SS,
            # i.e. we don't allow groups with more than one type
            # TODO: this is no longer relevant since each group has one SS given by
            #  majority vote
            if self.strict_ss and (len(ss_p0) != 1 or len(ss_m1) != 1):
                return None

            # The first SS type is always the reference SS type,
            # so we compare those. If they match, this is the SS type of
            # the pair, otherwise this row is useless for us
            if not self.consolidate_ss:
                ss_m1 = ss_m1[0]
                ss_p0 = ss_p0[0]
            else:
                ss_m1 = self.consolidate_ss.get(ss_m1[0])
                ss_p0 = self.consolidate_ss.get(ss_p0[0])

            if ss_m1 == ss_p0:
                return ss_m1

            # TODO: only reject if we condition on prev?
            return None

        # Based on the configuration, we create a column that represents
        # the group a sample belongs to when conditioning
        def assign_condition_group(row: pd.Series):
            prev_aac = row[self.codon_cols[0]]  # AA-CODON

            cond_groups = []

            if self.condition_on_prev == "codon":
                # Keep AA-CODON
                cond_groups.append(prev_aac)
            elif self.condition_on_prev == "aa":
                # Keep only AA
                cond_groups.append(prev_aac[0])
            else:
                cond_groups.append(CODON_TYPE_ANY)

            if self.condition_on_ss:
                cond_groups.append(row[self.secondary_col])
            else:
                cond_groups.append(SS_TYPE_ANY)

            return str.join("_", cond_groups)

        # Remove rows where ambiguous codons exist
        if self.strict_codons:
            idx_single_codon = df_sub[self.codon_opts_cols[0]].isnull()
            idx_single_codon &= df_sub[self.codon_opts_cols[1]].isnull()
            df_sub = df_sub[idx_single_codon]

        ss_consolidated = df_sub.apply(ss_consolidator, axis=1)

        # Keep only angle and codon columns from the full dataset
        df_pointwise = pd.DataFrame(data=df_sub[self.angle_cols + self.codon_cols],)

        # Add consolidated SS
        df_pointwise[self.secondary_col] = ss_consolidated

        # Remove rows without consolidated SS (this means the residues
        # pairs didn't have the same SS)
        has_ss = ~df_pointwise[self.secondary_col].isnull()
        df_pointwise = df_pointwise[has_ss]

        # Convert angles to radians
        angles_rad = df_pointwise[self.angle_cols].applymap(np.deg2rad)
        df_pointwise[self.angle_cols] = angles_rad

        # Convert angle columns to float32
        dtype = {col: np.float32 for col in self.angle_cols}
        df_pointwise = df_pointwise.astype(dtype)

        # Convert codon columns to AA-CODON
        aac = df_pointwise[self.codon_cols].applymap(codon2aac)
        df_pointwise[self.codon_cols] = aac

        # Add a column representing what we condition on
        condition_groups = df_pointwise.apply(assign_condition_group, axis=1)
        df_pointwise[self.condition_col] = condition_groups

        return df_pointwise

    def _dataset_stats(self, pool: mp.pool.Pool) -> dict:
        # Calculate likelihood distribution of prev codon, separated by SS
        codon_likelihoods = {}
        prev_codon, curr_codon = self.codon_cols
        df_processed: pd.DataFrame = self._load_intermediate("dataset")

        df_ss_groups = df_processed.groupby(self.secondary_col)
        for ss_type, df_ss_group in df_ss_groups:
            n_ss = len(df_ss_group)
            df_codon_groups = df_ss_group.groupby(prev_codon)
            df_codon_group_names = df_codon_groups.groups.keys()

            codon_likelihood = np.array(
                [
                    0.0
                    if codon not in df_codon_group_names
                    else len(df_codon_groups.get_group(codon)) / n_ss
                    for codon in AA_CODONS  # ensure consistent order
                ],
                dtype=np.float32,
            )
            assert np.isclose(np.sum(codon_likelihood), 1.0)

            # Save a map from codon name to it's likelihood
            codon_likelihoods[ss_type] = {
                c: codon_likelihood[i] for i, c in enumerate(AA_CODONS)
            }

        # Calculate AA likelihoods based on the codon likelihoods
        aa_likelihoods = {}
        for ss_type, codons in codon_likelihoods.items():
            aa_likelihoods[ss_type] = {aac[0]: 0.0 for aac in codons.keys()}
            for aac, likelihood in codons.items():
                aa = aac[0]
                aa_likelihoods[ss_type][aa] += likelihood

            assert np.isclose(sum(aa_likelihoods[ss_type].values()), 1.0)

        # Calculate SS likelihoods (ss->probability)
        ss_likelihoods = {}
        n_total = len(df_processed)
        for ss_type, df_ss_group in df_ss_groups:
            n_ss = len(df_ss_group)
            ss_likelihoods[ss_type] = n_ss / n_total
        assert np.isclose(sum(ss_likelihoods.values()), 1.0)

        # Add 'ANY' SS type to all likelihoods dicts
        for l in [codon_likelihoods, aa_likelihoods]:
            l[SS_TYPE_ANY] = {}

            for ss_type, aac_likelihoods in l.items():
                if ss_type == SS_TYPE_ANY:
                    continue

                p = ss_likelihoods[ss_type]

                for aac, likelihood in aac_likelihoods.items():
                    l[SS_TYPE_ANY].setdefault(aac, 0)
                    l[SS_TYPE_ANY][aac] += p * likelihood

            assert np.isclose(sum(l[SS_TYPE_ANY].values()), 1.0)

        self._dump_intermediate("codon-likelihoods", codon_likelihoods)
        self._dump_intermediate("aa-likelihoods", aa_likelihoods)
        self._dump_intermediate("ss-likelihoods", ss_likelihoods)

        # Calculate group and subgroup sizes
        group_sizes = {}
        _, curr_codon_col = self.codon_cols
        df_groups = df_processed.groupby(by=self.condition_col)
        for group_idx, df_group in df_groups:
            df_subgroups = df_group.groupby(curr_codon_col)

            # Not all codon may exist as subgroups. Default to zero and count each
            # existing subgroup.
            subgroup_sizes = {aac: 0 for aac in AA_CODONS}
            for aac, df_sub in df_subgroups:
                subgroup_sizes[aac] = len(df_sub)

            # Count size of each AA subgroup
            for aa in ACIDS:
                n_aa_samples = sum(
                    [size for aac, size in subgroup_sizes.items() if aac[0] == aa]
                )
                subgroup_sizes[aa] = n_aa_samples

            # Count size of each codon subgroup
            group_sizes[group_idx] = {
                "total": len(df_group),
                "subgroups": sort_dict(subgroup_sizes),
            }

        group_sizes = sort_dict(group_sizes, selector=lambda g: g["total"])
        self._dump_intermediate("group-sizes", group_sizes)

        return {"group_sizes": group_sizes}

    def _dihedral_kde_full(self, pool: mp.pool.Pool) -> dict:
        df_processed: pd.DataFrame = self._load_intermediate("dataset")
        df_groups = df_processed.groupby(by=self.secondary_col)
        df_groups_count: pd.DataFrame = df_groups.count()
        ss_counts = {
            f"n_{ss_type}": count
            for ss_type, count in df_groups_count.max(axis=1).to_dict().items()
        }

        LOGGER.info(f"Secondary-structure groups:\n{ss_counts})")
        LOGGER.info(f"Calculating dihedral distribution per SS type...")

        args = (
            (group_idx, df_group, self.angle_pairs, self.kde_args)
            for group_idx, df_group in df_groups
        )

        map_result = pool.starmap(_dihedral_kde_single_group, args)

        # maps from group (SS) to a list, containing a dihedral KDE
        # matrix for each angle-pair.
        map_result = {group_idx: dkdes for group_idx, dkdes in map_result}
        self._dump_intermediate("full-dkde", map_result)

        return {**ss_counts}

    def _codons_dists(self, pool: mp.pool.Pool) -> dict:
        group_sizes = self._load_intermediate("group-sizes")
        prev_codon_col, curr_codon_col = self.codon_cols

        # We currently only support one type of metric
        dist_metrics = {"l2": _kde_dist_metric_l2}
        dist_metric = dist_metrics[self.kde_dist_metric.lower()]

        # Calculate chunk-size for parallel mapping.
        # (Num groups in parallel) * (Num subgroups) / (num processors)
        n_procs = pp5.get_config("MAX_PROCESSES")
        chunksize = self.n_parallel_kdes * N_CODONS / n_procs
        chunksize = max(int(chunksize), 1)

        df_processed: pd.DataFrame = self._load_intermediate("dataset")
        df_groups = df_processed.groupby(by=self.condition_col)

        LOGGER.info(
            f"Calculating subgroup KDEs "
            f"(n_parallel_kdes={self.n_parallel_kdes}, "
            f"chunksize={chunksize})..."
        )

        codon_dists, codon_dkdes = {}, {}
        aa_dists, aa_dkdes = {}, {}
        dkde_asyncs: Dict[str, AsyncResult] = {}
        dist_asyncs: Dict[str, AsyncResult] = {}
        for i, (group_idx, df_group) in enumerate(df_groups):
            last_group = i == len(df_groups) - 1

            # In each pre-condition group, group by current codon.
            # These subgroups are where we estimate the dihedral angle KDEs.
            df_subgroups = df_group.groupby(curr_codon_col)

            # Find smallest subgroup
            subgroup_lens = [len(df_s) for _, df_s in df_subgroups]
            min_idx, max_idx = np.argmin(subgroup_lens), np.argmax(subgroup_lens)
            min_len, max_len = subgroup_lens[min_idx], subgroup_lens[max_idx]

            # Calculates number of samples in each bootstrap iteration:
            # We either take all samples in each subgroup, or the number of
            # samples in the smallest subgroup.
            if not self.bs_fixed_n:
                bs_nsamples = subgroup_lens
            elif self.bs_fixed_n == "min":
                bs_nsamples = [min_len] * len(subgroup_lens)
            else:
                bs_nsamples = [max_len] * len(subgroup_lens)

            # Run bootstrapped KDE estimation for all subgroups in parallel
            subprocess_args = (
                (
                    group_idx,
                    subgroup_idx,
                    df_subgroup,
                    self.angle_pairs,
                    self.kde_args,
                    self.bs_niter,
                    bs_nsamples[j],
                    self.bs_randstate,
                )
                for j, (subgroup_idx, df_subgroup) in enumerate(df_subgroups)
            )
            dkde_asyncs[group_idx] = pool.starmap_async(
                _codon_dkdes_single_subgroup, subprocess_args, chunksize=chunksize
            )

            # Allow limited number of simultaneous group KDE calculations
            # The limit is needed due to the very high memory required when
            # bs_niter is large.
            if not last_group and len(dkde_asyncs) < self.n_parallel_kdes:
                continue

            # If we already have enough simultaneous KDE calculations, collect
            # one (or all if this is the last group) of their results.
            # Note that we collect the first one that's ready.
            dkde_results_iter = yield_async_results(dkde_asyncs.copy())
            collected_dkde_results = {}
            LOGGER.info(
                f"[{i}] Waiting to collect KDEs "
                f"(#async_results={len(dkde_asyncs)})..."
            )
            for result_group_idx, group_dkde_result in dkde_results_iter:
                if group_dkde_result is None:
                    LOGGER.error(f"[{i}] No KDE result in {result_group_idx}")
                    continue

                LOGGER.info(f"[{i}] Collected KDEs for {result_group_idx}")
                collected_dkde_results[result_group_idx] = group_dkde_result

                # Remove async result so we dont see it next time (mark as collected)
                del dkde_asyncs[result_group_idx]

                # We only collect one result here, unless we are in the last
                # group. In that case we wait for all of the remaining results
                # since there's no next group.
                if not last_group:
                    break

            # Go over collected KDE results (usually only one) and start
            # the distance matrix calculation in parallel.
            for result_group_idx in collected_dkde_results:
                group_dkde_result = collected_dkde_results[result_group_idx]

                # bs_codon_dkdes maps from codon to a list of bootstrapped
                # KDEs of shape (B,N,N), one for each angle pair
                # Initialize in advance to obtain consistent order of codons
                bs_codon_dkdes = {c: None for c in AA_CODONS}
                for subgroup_idx, subgroup_bs_dkdes in group_dkde_result:
                    bs_codon_dkdes[subgroup_idx] = subgroup_bs_dkdes

                subgroup_sizes = group_sizes[result_group_idx]["subgroups"]

                # Run distance matrix calculation in parallel
                dist_asyncs[result_group_idx] = pool.apply_async(
                    _dkde_dists_single_group,
                    args=(
                        result_group_idx,
                        bs_codon_dkdes,
                        subgroup_sizes,
                        self.angle_pairs,
                        dist_metric,
                    ),
                )
                LOGGER.info(f"[{i}] Submitted cdist task {result_group_idx}")

            # Allow limited number of simultaneous distance matrix calculations
            if not last_group and len(dist_asyncs) < self.n_parallel_kdes:
                continue

            # Wait for one of the distance matrix calculations, or all of
            # them if it's the last group
            dists_results_iter = yield_async_results(dist_asyncs.copy())
            LOGGER.info(
                f"[{i}] Waiting to collect cdist matrices "
                f"(#async_results={len(dist_asyncs)})..."
            )
            for result_group_idx, group_dist_result in dists_results_iter:
                if group_dist_result is None:
                    LOGGER.error(f"[{i}] No dists in {result_group_idx}")
                    continue

                LOGGER.info(f"[{i}] Collected dist matrices in {result_group_idx}")
                (
                    group_codon_d2s,
                    group_codon_dkdes,
                    group_aa_d2s,
                    group_aa_dkdes,
                ) = group_dist_result
                codon_dists[result_group_idx] = group_codon_d2s
                codon_dkdes[result_group_idx] = group_codon_dkdes
                aa_dists[result_group_idx] = group_aa_d2s
                aa_dkdes[result_group_idx] = group_aa_dkdes

                # Remove async result so we dont see it next time (mark as collected)
                del dist_asyncs[result_group_idx]

                # If we're in the last group, collect everything
                if not last_group:
                    break

        # Make sure everything was collected
        assert len(dkde_asyncs) == 0, "Not all KDEs collected"
        assert len(dist_asyncs) == 0, "Not all dist matrices collected"
        LOGGER.info(f"Completed distance matrix collection")

        # codon_dists: maps from group (codon, SS) to a list, containing a
        # codon-distance matrix for each angle-pair.
        # The codon distance matrix is complex, where real is mu
        # and imag is sigma
        self._dump_intermediate("codon-dists", codon_dists)

        # aa_dists: Same as codon dists, but keys are AAs
        self._dump_intermediate("aa-dists", aa_dists)

        # codon_dkdes: maps from group to a dict where keys are codons.
        # For each codon we have a list of KDEs, one for each angle pair.
        self._dump_intermediate("codon-dkdes", codon_dkdes)

        # aa_dkdes: Same as codon_dkdes, but keys are AAs
        self._dump_intermediate("aa-dkdes", aa_dkdes)

        return {}

    def _codon_dists_expected(self, pool: mp.pool.Pool) -> dict:
        # To obtain the final expected distance matrices, we calculate the expectation
        # using the likelihood of the prev codon or aa so that conditioning is only
        # on SS. Then we also take the expectation over the different SS types to
        # obtain a distance matrix of any SS type.

        # Load map of likelihoods, depending on the type of previous
        # conditioning (ss->codon or AA->probability)
        if self.condition_on_prev == "codon":
            likelihoods: dict = self._load_intermediate("codon-likelihoods")
        elif self.condition_on_prev == "aa":
            likelihoods: dict = self._load_intermediate("aa-likelihoods")
        else:
            likelihoods = None

        # Load map of likelihoods of secondary structures.
        ss_likelihoods: dict = self._load_intermediate("ss-likelihoods")

        def _dists_expected(dists):
            # dists maps from group index to a list of distance matrices (one per
            # ange pair)

            # This dict will hold the final expected distance matrices (i.e. we
            # calculate the expectation using the likelihood of the prev codon
            # or aa). Note that we actually have one matrix per angle pair.
            dists_exp = {}
            for group_idx, d2_matrices in dists.items():
                assert len(d2_matrices) == len(self.angle_pairs)
                codon_or_aa, ss_type = group_idx.split("_")

                if ss_type not in dists_exp:
                    defaults = [np.zeros_like(d2) for d2 in d2_matrices]
                    dists_exp[ss_type] = defaults

                for i, d2 in enumerate(d2_matrices):
                    p = likelihoods[ss_type][codon_or_aa] if likelihoods else 1.0
                    # Don't sum nan's because they kill the entire cell
                    d2 = np.nan_to_num(d2, copy=False, nan=0.0)
                    dists_exp[ss_type][i] += p * d2

            # Now we also take the expectation over all SS types to produce one
            # averaged distance matrix, but only if we actually conditioned on it
            if self.condition_on_ss:
                defaults = [np.zeros_like(d2) for d2 in d2_matrices]
                dists_exp[SS_TYPE_ANY] = defaults
                for ss_type, d2_matrices in dists_exp.items():
                    if ss_type == SS_TYPE_ANY:
                        continue
                    p = ss_likelihoods[ss_type]
                    for i, d2 in enumerate(d2_matrices):
                        dists_exp[SS_TYPE_ANY][i] += p * d2

            return dists_exp

        for aa_codon in {"codon", "aa"}:
            # Load the calculated dists matrices. The loaded dict maps from group
            # index (prev_codon/AA, SS) to a list of distance matrices for each
            # angle pair.
            dists: dict = self._load_intermediate(f"{aa_codon}-dists")
            dists_exp = _dists_expected(dists)
            del dists
            self._dump_intermediate(f"{aa_codon}-dists-exp", dists_exp)

        return {}

    def _plot_results(self, pool: mp.pool.Pool):
        LOGGER.info(f"Plotting results...")

        ap_labels = [
            _cols2label(phi_col, psi_col) for phi_col, psi_col in self.angle_pairs
        ]

        async_results = []

        # Codon likelihoods
        codon_likelihoods = self._load_intermediate("codon-likelihoods", True)
        if codon_likelihoods is not None:
            async_results.append(
                pool.apply_async(
                    _plot_likelihoods,
                    args=(codon_likelihoods, "codon"),
                    kwds=dict(out_dir=self.out_dir,),
                )
            )
            del codon_likelihoods

        # AA likelihoods
        aa_likelihoods = self._load_intermediate("aa-likelihoods", True)
        if aa_likelihoods is not None:
            async_results.append(
                pool.apply_async(
                    _plot_likelihoods,
                    args=(aa_likelihoods, "aa"),
                    kwds=dict(out_dir=self.out_dir,),
                )
            )
            del aa_likelihoods

        # Dihedral KDEs of full dataset
        full_dkde: dict = self._load_intermediate("full-dkde", True)
        if full_dkde is not None:
            async_results.append(
                pool.apply_async(
                    _plot_full_dkdes,
                    args=(full_dkde,),
                    kwds=dict(out_dir=self.out_dir, angle_pair_labels=ap_labels),
                )
            )
            del full_dkde

        # Expected dists: we have four type of distance matrices: per codon or per AA
        # and regular (per group=prev-codon/AA+SS) or expected (per SS).
        for aa_codon, dist_type in it.product(["aa", "codon"], ["", "-exp"]):
            dists = self._load_intermediate(f"{aa_codon}-dists{dist_type}", True)
            if dists is None:
                continue

            out_dir = self.out_dir.joinpath(f"{aa_codon}-dists{dist_type}")
            labels = AA_CODONS if aa_codon == "codon" else ACIDS
            block_diagonal = SYN_CODON_IDX if aa_codon == "codon" else None
            for group_idx, d2_matrices in dists.items():
                async_results.append(
                    pool.apply_async(
                        _plot_d2_matrices,
                        kwds=dict(
                            group_idx=group_idx,
                            d2_matrices=d2_matrices,
                            out_dir=out_dir,
                            titles=ap_labels,
                            labels=labels,
                            vmin=None,
                            vmax=None,  # should consider scale
                            annotate_mu=True,
                            plot_std=False,
                            block_diagonal=block_diagonal,
                        ),
                    )
                )
            del dists, d2_matrices

        # Averaged Dihedral KDEs of each codon in each group
        group_sizes: dict = self._load_intermediate("group-sizes", True)
        for aa_codon in ["aa", "codon"]:
            avg_dkdes: dict = self._load_intermediate(f"{aa_codon}-dkdes", True)
            if avg_dkdes is None:
                continue
            for group_idx, dkdes in avg_dkdes.items():
                subgroup_sizes = group_sizes[group_idx]["subgroups"]

                async_results.append(
                    pool.apply_async(
                        _plot_dkdes,
                        kwds=dict(
                            group_idx=group_idx,
                            subgroup_sizes=subgroup_sizes,
                            dkdes=dkdes,
                            out_dir=self.out_dir.joinpath(f"{aa_codon}-dkdes"),
                            angle_pair_labels=ap_labels,
                        ),
                    )
                )
            del avg_dkdes, dkdes

        # Wait for plotting to complete. Each function returns a path
        fig_paths = self._handle_async_results(async_results, collect=True)


def _plot_likelihoods(
    likelihoods: dict, codon_or_aa: str, out_dir: Path,
):
    if codon_or_aa == "codon":
        fig_filename = out_dir.joinpath(f"codon-likelihoods.pdf")
        xlabel, ylabel = r"$c$", r"$\Pr(CODON=c)$"
    elif codon_or_aa == "aa":
        fig_filename = out_dir.joinpath(f"aa-likelihoods.pdf")
        xlabel, ylabel = r"$a$", r"$\Pr(AA=a)$"
    else:
        raise ValueError("Invalid type")

    # Convert from ss_type -> codon -> p, ss_type -> array
    for ss_type in likelihoods.keys():
        xticklabels = likelihoods[ss_type].keys()
        a = np.array([p for p in likelihoods[ss_type].values()], dtype=np.float32)
        likelihoods[ss_type] = a

    pp5.plot.multi_bar(
        likelihoods,
        xticklabels=xticklabels,
        xlabel=xlabel,
        ylabel=ylabel,
        fig_size=(20, 5),
        single_width=1.0,
        total_width=0.7,
        outfile=fig_filename,
    )

    return str(fig_filename)


def _plot_full_dkdes(full_dkde: dict, angle_pair_labels: List[str], out_dir: Path):
    fig_filename = out_dir.joinpath("full-dkdes.pdf")
    with mpl.style.context(PP5_MPL_STYLE):
        fig_rows, fig_cols = len(full_dkde) // 2, 2
        fig, ax = mpl.pyplot.subplots(
            fig_rows,
            fig_cols,
            figsize=(5 * fig_cols, 5 * fig_rows),
            sharex="all",
            sharey="all",
        )
        fig: mpl.pyplot.Figure
        ax: np.ndarray[mpl.pyplot.Axes] = ax.reshape(-1)

        vmin, vmax = 0.0, 5e-4
        for i, (group_idx, dkdes) in enumerate(full_dkde.items()):
            pp5.plot.ramachandran(
                dkdes,
                angle_pair_labels,
                title=group_idx,
                ax=ax[i],
                vmin=vmin,
                vmax=vmax,
            )

        pp5.plot.savefig(fig, fig_filename, close=True)

    return str(fig_filename)


def _plot_d2_matrices(
    group_idx: str,
    d2_matrices: List[np.ndarray],
    titles: List[str],
    labels: List[str],
    out_dir: Path,
    vmin: float = None,
    vmax: float = None,
    annotate_mu=True,
    plot_std=False,
    block_diagonal=None,
):
    LOGGER.info(f"Plotting distances for {group_idx}")

    # d2_matrices contains a matrix for each analyzed angle pair.
    # Split the mu and sigma from the complex d2 matrices
    d2_mu_sigma = [(np.real(d2), np.imag(d2)) for d2 in d2_matrices]
    d2_mu, d2_sigma = list(zip(*d2_mu_sigma))

    # Instead of plotting the std matrix separately, we can also use
    # annotations to denote the value of the std in each cell.
    # We'll annotate according to the quartiles of the std.
    pct = [np.nanpercentile(s, [25, 50, 75]) for s in d2_sigma]

    def quartile_ann_fn(ap_idx, j, k):
        std = d2_sigma[ap_idx][j, k]
        p25, p50, p75 = pct[ap_idx]
        if std < p25:
            return "*"
        elif std < p50:
            return ":"
        elif std < p75:
            return "."
        return ""

    # Here we plot a separate distance matrix for mu and for sigma.
    fig_filenames = []
    for avg_std, d2 in zip(("avg", "std"), (d2_mu, d2_sigma)):

        # Use annotations for the standard deviation
        if avg_std == "std":
            ann_fn = None
            if not plot_std:
                continue
        else:
            ann_fn = quartile_ann_fn if annotate_mu else None

        # Plot only the block-diagonal structure, e.g. synonymous codons.
        # The block_diagonal variable contains a list of tuples with the indices
        # of the desired block diagonal structure. The rest is set to nan.
        if block_diagonal:
            ii, jj = zip(*block_diagonal)
            mask = np.full_like(d2[0], fill_value=np.nan)
            mask[ii, jj] = 1.0
            d2 = [mask * d2i for d2i in d2]

        fig_filename = out_dir.joinpath(f"{group_idx}-{avg_std}.png")

        pp5.plot.multi_heatmap(
            d2,
            row_labels=labels,
            col_labels=labels,
            titles=titles,
            fig_size=20,
            vmin=vmin,
            vmax=vmax,
            fig_rows=1,
            outfile=fig_filename,
            data_annotation_fn=ann_fn,
        )

        fig_filenames.append(str(fig_filename))

    return fig_filenames


def _plot_dkdes(
    group_idx: str,
    subgroup_sizes: Dict[str, int],
    dkdes: Dict[str, List[np.ndarray]],
    angle_pair_labels: List[str],
    out_dir: Path,
):
    # Plot the kdes and distance matrices
    LOGGER.info(f"Plotting KDEs for {group_idx}")

    N = len(dkdes)
    fig_cols = int(np.ceil(np.sqrt(N)))
    fig_rows = int(np.ceil(N / fig_cols))

    with mpl.style.context(PP5_MPL_STYLE):
        vmin, vmax = 0.0, 5e-4
        fig, ax = mpl.pyplot.subplots(
            fig_rows,
            fig_cols,
            figsize=(5 * fig_cols, 5 * fig_rows),
            sharex="all",
            sharey="all",
        )
        ax: np.ndarray[mpl.pyplot.Axes] = np.reshape(ax, -1)

        for i, (codon, dkdes) in enumerate(dkdes.items()):
            title = f"{codon} ({subgroup_sizes.get(codon, 0)})"
            if not dkdes:
                ax[i].set_title(title)
                continue

            # Remove the std of the DKDE
            dkdes = [np.real(dkde) for dkde in dkdes]

            pp5.plot.ramachandran(
                dkdes, angle_pair_labels, title=title, ax=ax[i], vmin=vmin, vmax=vmax,
            )

        fig_filename = out_dir.joinpath(f"{group_idx}.png")
        pp5.plot.savefig(fig, fig_filename, close=True)

    return str(fig_filename)


def _dihedral_kde_single_group(group_idx, df_group, angle_pairs, kde_args):
    kde = BvMKernelDensityEstimator(**kde_args)

    # Creates 2D KDE for each angle pair
    dkdes = []
    for phi_col, psi_col in angle_pairs:
        phi = df_group[phi_col].values
        psi = df_group[psi_col].values
        dkde = kde(phi, psi)
        dkdes.append(dkde)

    return group_idx, dkdes


def _codon_dkdes_single_subgroup(
    group_idx: str,
    subgroup_idx: str,
    df_subgroup: pd.DataFrame,
    angle_pairs: list,
    kde_args: dict,
    bs_niter: int,
    bs_nsamples: int,
    bs_randstate: Optional[int],
) -> Tuple[str, List[np.ndarray]]:
    # Create a 3D tensor to hold the bootstrapped KDEs (for each angle
    # pair), of shape (B,N,N)
    bs_kde_shape = (bs_niter, kde_args["n_bins"], kde_args["n_bins"])
    bs_dkdes = [np.empty(bs_kde_shape, np.float32) for _ in angle_pairs]

    # We want a different random state in each subgroup, but still
    # should be reproducible
    if bs_randstate is not None:
        seed = (hash(group_idx + subgroup_idx) + bs_randstate) % (2 ** 31)
        np.random.seed(seed)

    t_start = time.time()

    for bootstrap_idx in range(bs_niter):
        # Sample from dataset with replacement, the same number of
        # elements as it's size. This is our bootstrap sample.
        df_subgroup_sampled = df_subgroup.sample(
            axis=0, replace=bs_niter > 1, n=bs_nsamples,
        )

        # dkdes contains one KDE for each pair in angle_pairs
        _, dkdes = _dihedral_kde_single_group(
            subgroup_idx, df_subgroup_sampled, angle_pairs, kde_args
        )

        # Save the current iteration's KDE into the results tensor
        for angle_pair_idx, dkde in enumerate(dkdes):
            bs_dkdes[angle_pair_idx][bootstrap_idx, ...] = dkde

    t_elapsed = time.time() - t_start
    bs_rate_iter = bs_niter / t_elapsed
    LOGGER.info(
        f"Completed {bs_niter} bootstrap iterations for "
        f"{group_idx}_{subgroup_idx}, size={len(df_subgroup)}, "
        f"bs_nsamples={bs_nsamples}, "
        f"rate={bs_rate_iter:.1f} iter/sec "
        f"elapsed={t_elapsed:.1f} sec"
    )

    return subgroup_idx, bs_dkdes


def _dkde_dists_single_group(
    group_idx: str,
    bs_codon_dkdes: Dict[str, List[np.ndarray]],
    subgroup_sizes: Dict[str, int],
    angle_pairs: list,
    kde_dist_metric: Callable,
):
    # Calculates the distance matrix for codons and AAs in each group.
    # Also averages the bootstrapped KDEs to obtain a single KDE estimate per
    # codon or AA in the group.

    tstart = time.time()
    # Codon distance matrix and average codon KDEs
    codon_d2s = _dkde_dists_pairwise(bs_codon_dkdes, angle_pairs, kde_dist_metric)
    codon_dkdes = _dkde_average(bs_codon_dkdes, angle_pairs,)
    LOGGER.info(
        f"Calculated codon distance matrix and average KDE for {group_idx} "
        f"({time.time() - tstart:.1f}s)..."
    )

    # AA distance matrix and average AA KDEs
    # First, we compute a weighted sum of the codon dkdes to obtain AA dkdes.
    tstart = time.time()
    bs_aa_dkdes = {aac[0]: None for aac in bs_codon_dkdes.keys()}
    for aa in ACIDS:
        # Total number of samples in from all subgroups (codons) of this AA
        n_aa_samples = subgroup_sizes[aa]
        for aac, bs_dkdes in bs_codon_dkdes.items():
            if aac[0] != aa:
                continue
            if subgroup_sizes[aac] == 0:
                continue

            # Empirical probability of this codon subgroup within it's AA
            p = subgroup_sizes[aac] / n_aa_samples

            # Initialize KDE of each angle pair to zero so we can accumulate
            if bs_aa_dkdes[aa] is None:
                bs_aa_dkdes[aa] = [0] * len(angle_pairs)

            # Weighted sum of codon dkdes belonging to the current AA
            for pair_idx in range(len(angle_pairs)):
                bs_aa_dkdes[aa][pair_idx] += p * bs_codon_dkdes[aac][pair_idx]

    # Now, we apply the pairwise distance between pairs of AA KDEs, as for the
    # codons
    aa_d2s = _dkde_dists_pairwise(bs_aa_dkdes, angle_pairs, kde_dist_metric)
    aa_dkdes = _dkde_average(bs_aa_dkdes, angle_pairs,)
    LOGGER.info(
        f"Calculated AA distance matrix and average KDE for {group_idx} "
        f"({time.time() - tstart:.1f}s)..."
    )

    return codon_d2s, codon_dkdes, aa_d2s, aa_dkdes


def _dkde_dists_pairwise(
    bs_dkdes: Dict[str, List[np.ndarray]],
    angle_pairs: List[str],
    kde_dist_metric: Callable,
):
    # Calculate a distance matrix based on the distance between each pair of
    # subgroups (codons or AAs) for which we have a multiple bootstrapped KDEs.

    dkde_names = list(bs_dkdes.keys())
    N = len(dkde_names)

    # Calculate distance matrix
    d2_matrices = []
    for pair_idx in range(len(angle_pairs)):
        # For each angle pair we have N dkde matrices,
        # so we compute the distance between each such pair.
        # We use a complex array to store mu as the real part and sigma
        # as the imaginary part in a single array.
        d2_mat = np.full((N, N), np.nan, np.complex64)

        dkde_pairs = it.product(enumerate(dkde_names), enumerate(dkde_names))
        for (i, ci), (j, cj) in dkde_pairs:
            if bs_dkdes[ci] is None:
                continue

            if j < i or bs_dkdes[cj] is None:
                continue

            # Get the two dihedral KDEs arrays to compare, each is of
            # shape (B, N, N) due to bootstrapping B times
            dkde1 = bs_dkdes[ci][pair_idx]
            dkde2 = bs_dkdes[cj][pair_idx]

            # If ci is cj, randomize the order of the KDEs when
            # comparing, so that different bootstrapped KDEs are
            # compared
            if i == j:
                # This permutes the order along the first axis
                dkde2 = np.random.permutation(dkde2)

            # Compute the distances, of shape (B,)
            d2 = kde_dist_metric(dkde1, dkde2)
            d2_mu = np.nanmean(d2)
            d2_sigma = np.nanstd(d2)

            # Store distance mu and std as a complex number
            d2_mat[i, j] = d2_mat[j, i] = d2_mu + 1j * d2_sigma

        d2_matrices.append(d2_mat)

    return d2_matrices


def _dkde_average(
    bs_dkdes: Dict[str, List[np.ndarray]], angle_pairs: List[str],
):
    # Average the KDEs from all bootstraps, so that we can save a single KDE
    # per subgroup (codon or AA). Note that this KDE will also
    # include the variance in each bin, so we'll save as a complex matrix where
    # real is the KDE value and imag is the std.
    avg_dkdes = {c: [] for c in bs_dkdes.keys()}
    for subgroup, bs_dkde in bs_dkdes.items():
        if bs_dkde is None:
            continue
        for pair_idx in range(len(angle_pairs)):
            # bs_dkde[pair_idx] here is of shape (B, N, N) due to
            # bootstrapping. Average it over the bootstrap dimension
            mean_dkde = np.nanmean(bs_dkde[pair_idx], axis=0, dtype=np.float32)
            std_dkde = np.nanstd(bs_dkde[pair_idx], axis=0, dtype=np.float32)
            avg_dkdes[subgroup].append(mean_dkde + 1j * std_dkde)

    return avg_dkdes


def _kde_dist_metric_l2(kde1: np.ndarray, kde2: np.ndarray):
    # We expect kde1 and kde2 to be of shape (B, N, N)
    # We calculate distance between each 2D NxN plane and return B
    # distances
    assert kde1.ndim == 3 and kde2.ndim == 3
    return np.nansum((kde1 - kde2) ** 2, axis=(1, 2))


def _cols2label(phi_col: str, psi_col: str):
    def rep(col: str):
        col = col.replace("phi", r"\varphi")
        col = col.replace("psi", r"\psi")
        col = re.sub(r"([+-][01])", r"_{\1}", col)
        return col

    return rf"${rep(phi_col)}, {rep(psi_col)}$"
