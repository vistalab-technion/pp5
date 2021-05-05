import math
import time
import logging
import itertools as it
import multiprocessing as mp
from typing import Dict, List, Tuple, Union, Callable, Iterable, Optional
from pathlib import Path

import numpy as np
import pandas as pd

from pp5.utils import sort_dict
from pp5.codons import N_CODONS, AA_CODONS, codon2aac
from pp5.analysis import SS_TYPE_ANY, DSSP_TO_SS_TYPE
from pp5.dihedral import Dihedral
from pp5.parallel import yield_async_results
from pp5.analysis.base import ParallelAnalyzer

LOGGER = logging.getLogger(__name__)


class PairwiseCodonDistanceAnalyzer(ParallelAnalyzer):
    def __init__(
        self,
        dataset_dir: Union[str, Path],
        out_dir: Union[str, Path] = None,
        pairwise_filename: str = "data-pairwise.csv",
        condition_on_ss=True,
        consolidate_ss=DSSP_TO_SS_TYPE.copy(),
        strict_codons=True,
        bs_niter=1,
        bs_randstate=None,
        out_tag: str = None,
    ):
        """
        Analyzes a pairwise dataset (pairs of match-groups of a residue in a
        specific context, from different structures)
        to produce a matrix of distances between codons Dij.
        Each entry ij in Dij corresponds to codons i and j, and the value is a
        distance metric between the distributions of dihedral angles coming
        from these codons.

        :param dataset_dir: Path to directory with the pairwise collector
        output.
        :param out_dir: Path to output directory. Defaults to <dataset_dir>/results.
        :param pairwise_filename: Filename of the pairwise dataset.
        :param condition_on_ss: Whether to group pairwise matches by secondary
        structure and analyse each SS group separately.
        :param consolidate_ss: Dict mapping from DSSP secondary structure to
        the consolidated SS types used in this analysis.
        :param strict_codons:  If True, matches where one of the groups has more than
        one codon option will be discarded.
        :param bs_niter: Number of bootstrap iterations.
        :param bs_randstate: Random state for bootstrap.
        :param out_tag: Tag for output.
        """
        super().__init__(
            "pairwise_cdist",
            dataset_dir,
            pairwise_filename,
            out_dir,
            out_tag,
            clear_intermediate=False,
        )

        self.condition_on_ss = condition_on_ss
        self.consolidate_ss = consolidate_ss
        self.strict_codons = strict_codons

        self.ref_prefix = "ref_"
        col_prefixes = (self.ref_prefix, "")

        self.codon_opts_cols = [f"{p}codon_opts" for p in col_prefixes]
        self.secondary_cols = [f"{p}secondary" for p in col_prefixes]
        self.codon_cols = [f"{p}codon" for p in col_prefixes]

        angle_cols = ("curr_phis", "curr_psis")
        self.ref_angle_cols = [f"{self.ref_prefix}{p}" for p in angle_cols]
        self.query_angle_cols = list(angle_cols)

        self.condition_col = "condition_group"
        self.type_col = "type"

        self.allowed_match_types = ["SAME", "SILENT"]
        self.bs_niter = bs_niter
        self.bs_randstate = bs_randstate

    def _collection_functions(
        self,
    ) -> Dict[str, Callable[[mp.pool.Pool], Optional[Dict]]]:
        return {
            "preprocess-dataset": self._preprocess_dataset,
            "codon-dists": self._codon_dists,
            "plot-results": self._plot_results,
        }

    def _preprocess_dataset(self, pool: mp.pool.Pool) -> dict:
        def any_in(queries: Iterable[str], target: str):
            # Are any of the query strings contained in the target?
            return any(filter(lambda q: q in target, queries))

        # Load just a single row to get all column names, then group them by
        # their types
        all_cols = list(pd.read_csv(self.input_file, nrows=1).columns)
        float_list_cols = [c for c in all_cols if any_in(("phis", "psis"), c)]
        int_list_cols = [c for c in all_cols if any_in(("contexts", "idxs",), c)]
        str_list_cols = [c for c in all_cols if any_in(("res_ids", "codons"), c)]
        list_cols = float_list_cols + int_list_cols + str_list_cols
        float_cols = [
            c
            for c in all_cols
            if any_in(("phi", "psi", "norm"), c) and c not in float_list_cols
        ]
        str_cols = [c for c in all_cols if c not in list_cols and c not in float_cols]

        # Make sure we got all columns
        if len(all_cols) != len(list_cols) + len(float_cols) + len(str_cols):
            raise ValueError("Not all columns accounted for")

        # We'll initially load the columns containing list as strings for
        # faster parsing
        col_dtypes = {
            **{c: str for c in list_cols + str_cols},
            **{c: np.float32 for c in float_cols},
        }

        # Load the data lazily and in chunks
        df_pointwise_reader = pd.read_csv(
            str(self.input_file), dtype=col_dtypes, chunksize=10_000,
        )

        # Parallelize loading and preprocessing
        async_results = []
        for df_sub in df_pointwise_reader:
            async_results.append(
                pool.apply_async(
                    self._preprocess_subframe,
                    args=(df_sub, float_list_cols, int_list_cols, str_list_cols),
                )
            )

        _, _, results = self._handle_async_results(async_results, collect=True)
        sub_dfs, orig_counts = zip(*results)

        # Dump processed dataset
        df_preproc = pd.concat(sub_dfs, axis=0, ignore_index=True)
        LOGGER.info(f"Loaded {self.input_file}: {len(df_preproc)} rows")
        self._dump_intermediate("dataset", df_preproc)

        return {
            "n_ORIG": sum(orig_counts),
            "n_TOTAL": len(df_preproc),
        }

    def _preprocess_subframe(
        self, df_sub: pd.DataFrame, float_list_cols, int_list_cols, str_list_cols
    ):
        pd.set_option("mode.chained_assignment", "raise")

        orig_len = len(df_sub)

        # Based on the configuration, we create a column that represents
        # the group a sample belongs to when conditioning
        def assign_condition_group(row: pd.Series):
            cond_groups = []

            # Don't consider matches with an ignored type
            if row[self.type_col] not in self.allowed_match_types:
                return None

            # Don't consider matches with ambiguous codons
            if self.strict_codons:
                if not all(pd.isna(row[c]) for c in self.codon_opts_cols):
                    return None

            # Don't consider matches without same SS for ref and queries
            secondaries = set(row[c] for c in self.secondary_cols)
            if len(secondaries) != 1:
                return None

            # Consolidate SS and add to condition groups
            if not self.condition_on_ss:
                cond_groups.append(SS_TYPE_ANY)
            else:
                ss = self.consolidate_ss.get(secondaries.pop())
                # Dont consider matches without SS
                if ss is None:
                    return None
                cond_groups.append(ss)

            return str.join("_", cond_groups)

        # Calculate condition group and only keep rows with a valid group
        cond_col = df_sub.apply(assign_condition_group, axis=1)
        df_sub[self.condition_col] = cond_col
        has_cond = ~df_sub[self.condition_col].isnull()
        df_sub = df_sub[has_cond].copy()

        # Drop unnecessary columns
        drop_columns = self.codon_opts_cols + self.secondary_cols
        df_sub = df_sub.drop(drop_columns, axis=1)

        # Convert codon columns to AA-CODON
        aac = df_sub[self.codon_cols].applymap(codon2aac)
        df_sub[self.codon_cols] = aac

        return df_sub, orig_len

    def _codon_dists(self, pool: mp.pool.Pool) -> dict:
        # Load processed dataset
        df_processed: pd.DataFrame = self._load_intermediate("dataset")

        # Group by condition column
        df_groups = df_processed.groupby(by=self.condition_col)

        # Create unique codon pairs
        codon_pairs = sorted(
            set(map(lambda t: tuple(sorted(t)), it.product(AA_CODONS, AA_CODONS)))
        )

        cdist_angle_cols = self.ref_angle_cols + self.query_angle_cols

        group_sizes = {}

        async_results = {}
        for group_idx, df_group in df_groups:
            df_group: pd.DataFrame

            df_subgroups = df_group.groupby(self.codon_cols)
            subgroup_indices = set(df_subgroups.groups.keys())

            group_sizes[group_idx] = {}

            # Loop over unique pairs of codons
            for c1, c2 in codon_pairs:
                subgroup_idx = f"{c1}_{c2}"

                # Find subgroups with codon pair in any order
                dfs = []
                if (c1, c2) in subgroup_indices:
                    dfs.append(df_subgroups.get_group((c1, c2)))
                if c1 != c2 and (c2, c1) in subgroup_indices:
                    dfs.append(df_subgroups.get_group((c2, c1)))

                if len(dfs) == 0:
                    LOGGER.debug(
                        f"Skipping codon-dists for {group_idx=}, "
                        f"{subgroup_idx=}, no matches"
                    )
                    continue

                # Combine matches from this codon pair, in any order
                df_subgroup = pd.concat(dfs, axis=0)

                # Keep only relevant rows and columns
                df_subgroup = df_subgroup[cdist_angle_cols]

                group_sizes[group_idx][subgroup_idx] = len(df_subgroup)

                # Parallelize over the sub-groups
                async_results[(group_idx, subgroup_idx)] = pool.apply_async(
                    self._codon_dists_single_subgroup,
                    args=(
                        group_idx,
                        subgroup_idx,
                        df_subgroup,
                        self.ref_angle_cols,
                        self.query_angle_cols,
                        self.bs_niter,
                        self.bs_randstate,
                    ),
                )

            group_sizes[group_idx] = sort_dict(group_sizes[group_idx])

        # Construct empty distance matrix per group
        # We use a complex matrix to store mu and std as real and imag.
        codon_dists = {}
        for group_idx, _ in df_groups:
            d2_mat = np.full((N_CODONS, N_CODONS), np.nan + 1j * np.nan, np.complex64)
            codon_dists[group_idx] = d2_mat

        # Populate distance matrix per group as the results come in
        codon_to_idx = {c: i for i, c in enumerate(AA_CODONS)}
        results_iter = yield_async_results(async_results)
        for (group_idx, subgroup_idx), (mu, sigma) in results_iter:
            LOGGER.info(
                f"Codon-dist for {group_idx=}, {subgroup_idx=}: "
                f"({mu:.2f}±{sigma:.2f})"
            )
            c1, c2 = subgroup_idx.split("_")
            i, j = codon_to_idx[c1], codon_to_idx[c2]
            d2_mat = codon_dists[group_idx]
            d2_mat[i, j] = d2_mat[j, i] = mu + 1j * sigma

        # codon_dists: maps from group to a codon-distance matrix.
        # The codon distance matrix is complex, where real is mu
        # and imag is sigma.
        self._dump_intermediate("codon-dists", codon_dists)

        # Apply correction to distance estimates based on the measured variance.
        ii, jj = np.triu_indices(N_CODONS, k=1)  # k=1 means don't include main diagonal
        for group_idx, _ in df_groups:
            d2_mat = codon_dists[group_idx]
            d2_mu, d2_sigma = np.real(d2_mat), np.imag(d2_mat)

            # Correction: delta12 = d12 - 0.5 * (d11 + d22)
            d2_mu[ii, jj] = d2_mu[ii, jj] - 0.5 * (d2_mu[ii, ii] + d2_mu[jj, jj])
            d2_mu[jj, ii] = d2_mu[ii, jj]

            codon_dists[group_idx] = d2_mu + 1j * d2_sigma

            # TODO: This correction procedure is naïve, we'll need to replace it with
            #       fitting a metric.
            if np.any(d2_mu < 0):
                LOGGER.warning(
                    f"NEGATIVE corrected d^2 for {group_idx}, min={np.nanmin(d2_mu)}, "
                    f"neg_avg={np.nanmean(d2_mu[d2_mu<0])}"
                )

        # Dump corrected version
        self._dump_intermediate("codon-dists-corrected", codon_dists)

        return {"group_sizes": group_sizes}

    def _plot_results(self, pool: mp.pool.Pool):
        LOGGER.info(f"Plotting results...")
        async_results = []

        # Expected codon dists
        for result_name in ("codon-dists", "codon-dists-corrected"):
            codon_dists = self._load_intermediate(result_name, True)
            if codon_dists is None:
                continue
            for group_idx, d2_matrix in codon_dists.items():
                d2_mu, d2_sigma = np.real(d2_matrix), np.imag(d2_matrix)
                # When plotting, use d instead of d^2 for better dynamic range.
                d_mu = np.sqrt(d2_mu)

                async_results.append(
                    pool.apply_async(
                        _plot_d2_matrices,
                        kwds=dict(
                            group_idx=group_idx,
                            d2_matrices=[d_mu + 1j * d2_sigma],
                            out_dir=self.out_dir.joinpath(result_name),
                            titles=[""],
                            labels=AA_CODONS,
                            vmin=None,
                            vmax=None,  # should consider scale
                            annotate_mu=True,
                            plot_std=False,
                        ),
                    )
                )

        # Wait for plotting to complete. Each function returns a path
        _ = self._handle_async_results(async_results, collect=True)

    @staticmethod
    def _codon_dists_single_subgroup(
        group_idx: str,
        subgroup_idx: str,
        df_subgroup: pd.DataFrame,
        ref_angle_cols: list,
        query_angle_cols: list,
        bs_niter,
        bs_randstate,
    ):
        tstart = time.time()

        # Reference and query angle columns
        ref_phi_col, ref_psi_col = ref_angle_cols
        query_phi_col, query_psi_col = query_angle_cols

        def convert_angles_col(col_val: Union[str, float]):
            # Col val is a string like '-123.456;-12.34' or nan
            angles = map(float, str.split(str(col_val), ";"))
            return map(math.radians, angles)

        def extract_dihedrals(row: pd.Series) -> Tuple[List[Dihedral], List[Dihedral]]:
            # row contains phi,psi columns for ref and query. Each column
            # contains multiple values (but the same number).
            refs = it.starmap(
                Dihedral.from_rad,
                zip(
                    convert_angles_col(row[ref_phi_col]),
                    convert_angles_col(row[ref_psi_col]),
                ),
            )
            queries = it.starmap(
                Dihedral.from_rad,
                zip(
                    convert_angles_col(row[query_phi_col]),
                    convert_angles_col(row[query_psi_col]),
                ),
            )

            return list(refs), list(queries)

        # Transform into a dataframe with two columns,
        # each column containing a list of reference or query Dihedrals angles.
        df_angles = df_subgroup.apply(
            extract_dihedrals, axis=1, raw=False, result_type="expand"
        )

        # Now we have for each match, a list of dihedrals of ref and query
        # groups.
        ref_groups, query_groups = df_angles[0].values, df_angles[1].values
        assert len(ref_groups) == len(query_groups) > 0
        n_matches = len(ref_groups)

        # We want a different random state in each subgroup, but still
        # should be reproducible
        if bs_randstate is not None:
            seed = (hash(group_idx + subgroup_idx) + bs_randstate) % (2 ** 31)
            np.random.seed(seed)

        # Create a K*B matrix for bootstrapped squared-distances
        dists2 = np.empty(shape=(n_matches, bs_niter), dtype=np.float32)
        for m_idx in range(n_matches):
            ref_sample = np.random.choice(ref_groups[m_idx], bs_niter)
            query_sample = np.random.choice(query_groups[m_idx], bs_niter)
            zip_rq = zip(ref_sample, query_sample)

            dists2[m_idx, :] = [
                Dihedral.flat_torus_distance(r, q, degrees=True, squared=True)
                for r, q in zip_rq
            ]

        mean = np.nanmean(dists2)
        std = np.nanstd(dists2)
        elapsed = time.time() - tstart

        LOGGER.info(
            f"Calculated codon-dists for {group_idx=}, "
            f"{subgroup_idx=}, n_matches={len(df_subgroup)}, "
            f"{elapsed=:.2f}s"
        )

        return mean, std
