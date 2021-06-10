import re
import time
import logging
import itertools as it
import multiprocessing as mp
from math import ceil, floor
from typing import Any, Dict, List, Tuple, Union, Callable, Optional, Sequence
from pathlib import Path
from multiprocessing.pool import AsyncResult

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.style
from matplotlib.axes import Axes
from matplotlib.figure import Figure

import pp5.plot
from pp5.plot import PP5_MPL_STYLE
from pp5.stats import tw_test
from pp5.utils import sort_dict
from pp5.codons import (
    ACIDS,
    AAC_SEP,
    N_ACIDS,
    N_CODONS,
    AA_CODONS,
    AAC_TUPLE_SEP,
    SYN_CODON_IDX,
    aac2aa,
    aact2aat,
    codon2aac,
    aac_tuples,
)
from pp5.analysis import SS_TYPE_ANY, SS_TYPE_MIXED, DSSP_TO_SS_TYPE
from pp5.dihedral import Dihedral, flat_torus_distance
from pp5.parallel import yield_async_results
from pp5.vonmises import BvMKernelDensityEstimator
from pp5.analysis.base import ParallelAnalyzer

LOGGER = logging.getLogger(__name__)

PDB_ID_COL = "pdb_id"
UNP_ID_COL = "unp_id"
UNP_IDX_COL = "unp_idx"
AA_COL = "AA"
CODON_COL = "codon"
CODON_SCORE_COL = "codon_score"
SECONDARY_COL = "secondary"
PHI_COL = "phi"
PSI_COL = "psi"
ANGLE_COLS = (PHI_COL, PSI_COL)
CONDITION_COL = "condition_group"
GROUP_SIZE_COL = "group_size"


class PointwiseCodonDistanceAnalyzer(ParallelAnalyzer):
    def __init__(
        self,
        dataset_dir: Union[str, Path],
        out_dir: Union[str, Path] = None,
        pointwise_filename: str = "data-precs.csv",
        condition_on_ss: bool = True,
        consolidate_ss: dict = DSSP_TO_SS_TYPE.copy(),
        codon_tuple_len: int = 1,
        min_group_size: int = 1,
        strict_codons: bool = True,
        kde_nbins: int = 128,
        kde_k1: float = 30.0,
        kde_k2: float = 30.0,
        kde_k3: float = 0.0,
        bs_niter: int = 1,
        bs_randstate: Optional[int] = None,
        bs_fixed_n: Optional[Union[str, int]] = None,
        n_parallel_groups: int = 2,
        t2_n_max: Optional[int] = 1000,
        t2_permutations: int = 1000,
        out_tag: str = None,
    ):
        """
        Analyzes a dataset of protein records to produce a matrix of distances between
        codons Dij.
        Each entry ij in Dij corresponds to codons i and j, and the value is a
        distance metric between the distributions of dihedral angles coming
        from these codons.

        :param dataset_dir: Path to directory with the pointwise collector
            output.
        :param out_dir: Path to output directory. Defaults to <dataset_dir>/results.
        :param pointwise_filename: Filename of the pointwise dataset.
        :param consolidate_ss: Dict mapping from DSSP secondary structure to
            the consolidated SS types used in this analysis.
        :param condition_on_ss: Whether to condition on secondary structure
            (of two consecutive residues, after consolidation).
        :param codon_tuple_len: Number of consecutive codons to analyze as a tuple.
            Set 1 for single codons, 2 for codon pairs. Other values are not supported.
        :param min_group_size: Minimal number of angle-pairs from different
            structures belonging to the same Uniprot ID, location and codon in order to
            consider the group of angles for analysis.
        :param strict_codons: Enforce only one known codon per residue
            (reject residues where DNA matching was ambiguous).
        :param kde_nbins: Number of angle binds for KDE estimation.
        :param kde_k1: KDE concentration parameter for phi.
        :param kde_k2: KDE concentration parameter for psi.
        :param kde_k3: KDE joint concentration parameter.
        :param bs_niter: Number of bootstrap iterations.
        :param bs_randstate: Random state for bootstrap.
        :param bs_fixed_n: Whether to fix number of samples in each
            bootstrap iteration for each subgroup.
            Values can be "min": fix number of samples to equal the smallest
            subgroup; "max": fix to size of largest subgroup; ""/None: no limit;
            integer value: fix to the given number, and zero also means no limit.
        :param n_parallel_groups: Number of groups to schedule for running in
            parallel. Note that the sub-groups in each group will be parallelized over
            all available cores.
        :param t2_n_max: Maximal sample-size to use when calculating
            p-value of distances with the T^2 statistic. If there are larger samples,
            bootstrap sampling with the given maximal sample size will be performed.
            If None or zero, sample size wont be limited.
        :param t2_permutations: Number of permutations to use when calculating
            p-value of distances with the T^2 statistic.
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

        bs_fixed_n = bs_fixed_n or 0
        if isinstance(bs_fixed_n, str) and bs_fixed_n not in {"", "min", "max"}:
            raise ValueError(f"invalid bs_fixed_n: {bs_fixed_n}, must be min/max/''")
        elif isinstance(bs_fixed_n, int) and bs_fixed_n < 0:
            raise ValueError(f"invalid bs_fixed_n: {bs_fixed_n}, must be > 0")

        if codon_tuple_len < 1:
            raise ValueError(f"invalid {codon_tuple_len=}, must be >= 1")

        self.condition_on_ss = condition_on_ss
        self.consolidate_ss = consolidate_ss
        self.codon_tuple_len = codon_tuple_len
        self.min_group_size = min_group_size
        self.strict_codons = strict_codons
        self.condition_on_prev = None

        self.kde_args = dict(n_bins=kde_nbins, k1=kde_k1, k2=kde_k2, k3=kde_k3)
        self.kde_dist_metric = "l2"

        self.bs_niter = bs_niter
        self.bs_randstate = bs_randstate
        self.bs_fixed_n = bs_fixed_n
        self.n_parallel_groups = n_parallel_groups
        self.t2_n_max = t2_n_max
        self.t2_permutations = t2_permutations

        # Initialize codon tuple names and corresponding indices
        tuples = list(aac_tuples(k=self.codon_tuple_len))
        self._codon_tuple_to_idx = {
            str.join(AAC_TUPLE_SEP, aac_tuple): idx
            for idx, aac_tuple in enumerate(tuples)
        }
        self._aa_tuple_to_idx = {
            str.join(AAC_TUPLE_SEP, aat): idx
            for idx, aat in enumerate(sorted(set(aact2aat(aact) for aact in tuples)))
        }
        self._n_codon_tuples = len(self._codon_tuple_to_idx)
        self._n_aa_tuples = len(self._aa_tuple_to_idx)

    def _collection_functions(
        self,
    ) -> Dict[str, Callable[[mp.pool.Pool], Optional[Dict]]]:
        return {
            "preprocess_dataset": self._preprocess_dataset,
            "tuples_dataset": self._create_tuples_dataset,
            "dataset_stats": self._dataset_stats,
            "pointwise_dists_dihedral": self._pointwise_dists_dihedral,
            # "dihedral_kde_groups": self._dihedral_kde_groups,
            # "pointwise_dists_kde": self._pointwise_dists_kde,
            # "codon_dists_expected": self._codon_dists_expected,
            # "plot_results": self._plot_results,
        }

    def _preprocess_dataset(self, pool: mp.pool.Pool) -> dict:
        """
        Converts the input raw data to an intermediate data frame which we use for
        analysis.
        """

        input_cols = (
            PDB_ID_COL,
            UNP_ID_COL,
            UNP_IDX_COL,
            CODON_COL,
            CODON_SCORE_COL,
            SECONDARY_COL,
            *ANGLE_COLS,
        )

        # Specifying this dtype allows an integer column with missing values
        dtype = {UNP_IDX_COL: "Int64"}
        dtype = {**dtype, **{ac: "float32" for ac in ANGLE_COLS}}

        # Consolidate different SS types into the ones we support
        converters = {SECONDARY_COL: lambda ss: self.consolidate_ss.get(ss, "")}

        start = time.time()

        # Load the data
        df_pointwise = pd.read_csv(
            str(self.input_file),
            usecols=input_cols,
            dtype=dtype,
            header=0,
            converters=converters,
        )

        # Filter out rows with missing SS, unp_idx or with ambiguous codons
        idx_filter = ~df_pointwise.secondary.isnull()
        idx_filter &= ~df_pointwise.unp_idx.isnull()
        if self.strict_codons:
            idx_filter &= df_pointwise.codon_score == 1.0

        df_pointwise = df_pointwise[idx_filter]

        # Set a column for the condition group (what we condition on)
        if self.condition_on_ss:
            condition_groups = df_pointwise.secondary
        else:
            condition_groups = "ANY"
        df_pointwise[CONDITION_COL] = condition_groups

        # Convert codon columns to AA-CODON
        df_pointwise[CODON_COL] = df_pointwise[[CODON_COL]].applymap(codon2aac)

        async_results = []

        # Process groups in parallel.
        # Add additional conditioning on codon just to break it into many more groups
        # so that it parallelizes better.
        for group_idx, df_group in df_pointwise.groupby(by=[CONDITION_COL, CODON_COL]):
            condition_group_id, _ = group_idx
            async_results.append(
                pool.apply_async(
                    self._preprocess_group, args=(condition_group_id, df_group),
                )
            )

        _, _, results = self._handle_async_results(
            async_results, collect=True, flatten=True
        )

        df_processed = pd.DataFrame(data=results)
        LOGGER.info(f"preprocessing done, elapsed={time.time()-start:.2f}s")
        LOGGER.info(f"{df_processed}")

        # Sort rows using the order of residues in each protein
        df_processed = df_processed.sort_values(by=[UNP_ID_COL, UNP_IDX_COL])

        self._dump_intermediate("dataset", df_processed, debug=True)
        return {
            "n_TOTAL": len(df_processed),
        }

    def _preprocess_group(self, group_id: str, df_group: pd.DataFrame):
        """
        Applies pre-processing to a single group (e.g. SS) in the dataset.
        """
        processed_subgroups = []

        # Group by each unique codon at a unique location in a unique protein
        for subgroup_idx, df_subgroup in df_group.groupby(
            [UNP_ID_COL, UNP_IDX_COL, CODON_COL]
        ):
            if len(df_subgroup) < self.min_group_size:
                continue

            # There shouldn't be more than one SS type since all members of this
            # subgroup come from the same residue in the same protein
            secondaries = set(df_subgroup[SECONDARY_COL])
            if len(secondaries) > 1:
                LOGGER.warning(
                    f"Ambiguous secondary structure in {group_id=} {subgroup_idx=}"
                )
                continue

            unp_id, unp_idx, aa_codon = subgroup_idx
            subgroup_ss = secondaries.pop()

            # Make sure all angles have a value
            if np.any(df_subgroup[[*ANGLE_COLS]].isnull()):
                continue

            # Calculate average angle from the different structures in this sub group
            angles = (
                Dihedral.from_deg(phi, psi)
                for phi, psi in df_subgroup[[*ANGLE_COLS]].values
            )
            centroid = Dihedral.circular_centroid(*angles)
            processed_subgroups.append(
                {
                    UNP_ID_COL: unp_id,
                    UNP_IDX_COL: unp_idx,
                    AA_COL: str.split(aa_codon, AAC_SEP)[0],
                    CODON_COL: aa_codon,
                    CONDITION_COL: group_id,
                    SECONDARY_COL: subgroup_ss,
                    PHI_COL: centroid.phi,
                    PSI_COL: centroid.psi,
                    GROUP_SIZE_COL: len(df_subgroup),
                }
            )
        return processed_subgroups

    def _create_tuples_dataset(self, pool: mp.pool.Pool) -> dict:
        curr_codon_col = CODON_COL
        async_results: List[AsyncResult] = []

        start = time.time()
        df_processed: pd.DataFrame = self._load_intermediate("dataset")

        # Process groups in parallel.
        for group_idx, df_group in df_processed.groupby(by=[UNP_ID_COL]):
            unp_id = group_idx
            async_results.append(
                pool.apply_async(
                    self._create_group_tuples, args=(df_group, self.codon_tuple_len),
                )
            )

        # Results is a list of dataframes
        _, _, results = self._handle_async_results(
            async_results, collect=True, flatten=False
        )

        df_tuples = pd.concat((r for r in results if r is not None), axis=0)
        LOGGER.info(f"tuples dataset created, elapsed={time.time()-start:.2f}s")
        LOGGER.info(f"{df_tuples}")

        self._dump_intermediate("dataset-tuples", df_tuples, debug=True)
        return {
            "n_tuples_TOTAL": len(df_tuples),
        }

    def _create_group_tuples(
        self, df_group: pd.DataFrame, tuple_len: int,
    ) -> Optional[pd.DataFrame]:

        # Sort rows using the order of residues in each protein
        df_group = df_group.sort_values(by=[UNP_ID_COL, UNP_IDX_COL])

        # Create a shifted version of the data and concatenate is next to the original
        # Each row will have the current and next residue.
        prefixes = [""]
        shifted_dfs = [df_group]
        for i in range(1, tuple_len):
            p = f"next_{i}_"
            prefixes.append(p)

            # Shift and fix type due to NaNs appearing
            df_shifted = df_group.shift(-i)
            df_shifted[f"{UNP_IDX_COL}"] = df_shifted[f"{UNP_IDX_COL}"].astype("Int64")
            df_shifted = df_shifted.add_prefix(p)

            shifted_dfs.append(df_shifted)

        df_m = pd.concat(shifted_dfs, axis=1)

        # Only keep rows where the next residue is in the same protein and has the
        # successive index.
        queries = []
        for i, prefix in enumerate(prefixes[1:], start=1):
            queries.append(f"{UNP_ID_COL} == {prefix}{UNP_ID_COL}")
            queries.append(f"{UNP_IDX_COL} + {i} == {prefix}{UNP_IDX_COL}")

        query = str.join(" and ", queries)
        if query:
            df_m = df_m.query(query)

        if len(df_m) == 0:
            return None

        # Function to map rows in the merged dataframe to the final rows we'll use.
        def _row_mapper(row: pd.Series):

            codons, aas, sss, group_sizes = [], [], [], []
            for i, p in enumerate(prefixes):
                codons.append(row[f"{p}{CODON_COL}"])
                aas.append(row[f"{p}{AA_COL}"])
                sss.append(row[f"{p}{SECONDARY_COL}"])
                group_sizes.append(row[f"{p}{GROUP_SIZE_COL}"])

            codon_tuple = str.join(AAC_TUPLE_SEP, codons)
            aa_tuple = str.join(AAC_TUPLE_SEP, aas)
            ss_tuple = str.join(AAC_TUPLE_SEP, sss)
            if all(ss == sss[0] for ss in sss):
                condition_group = sss[0]
            else:
                condition_group = SS_TYPE_MIXED

            return {
                UNP_ID_COL: row[UNP_ID_COL],
                UNP_IDX_COL: row[UNP_IDX_COL],
                AA_COL: aa_tuple,
                CODON_COL: codon_tuple,
                SECONDARY_COL: ss_tuple,
                CONDITION_COL: condition_group,
                # Use Psi0, Phi1 (current psi, next (or last) phi)
                PHI_COL: row[f"{prefixes[-1]}{PHI_COL}"],
                PSI_COL: row[PSI_COL],
                GROUP_SIZE_COL: int(min(group_sizes)),
            }

        df_tuples = df_m.apply(_row_mapper, axis=1, raw=False, result_type="expand")
        return df_tuples

    def _dataset_stats(self, pool: mp.pool.Pool) -> dict:
        """
        Extracts various statistics from the dataset.
        """
        # Calculate likelihood distribution of prev codon, separated by SS
        codon_likelihoods = {}
        codon_col = CODON_COL
        df_processed: pd.DataFrame = self._load_intermediate("dataset")

        df_ss_groups = df_processed.groupby(SECONDARY_COL)
        for ss_type, df_ss_group in df_ss_groups:
            n_ss = len(df_ss_group)
            df_codon_groups = df_ss_group.groupby(codon_col)
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
        df_processed: pd.DataFrame = self._load_intermediate("dataset-tuples")
        group_sizes = {}
        curr_codon_col = CODON_COL
        df_groups = df_processed.groupby(by=CONDITION_COL)
        for group_idx, df_group in df_groups:
            df_subgroups = df_group.groupby(curr_codon_col)

            # Not all codon may exist as subgroups. Default to zero and count each
            # existing subgroup.
            subgroup_sizes = {
                **{aac: 0 for aac in self._codon_tuple_to_idx.keys()},
                **{aa: 0 for aa in self._aa_tuple_to_idx.keys()},
            }
            for aac, df_sub in df_subgroups:
                subgroup_sizes[aac] = len(df_sub)

                aa = str.join(
                    AAC_TUPLE_SEP, [aac2aa(aac) for aac in aac.split(AAC_TUPLE_SEP)]
                )
                subgroup_sizes[aa] += len(df_sub)

            # Count size of each codon subgroup
            group_sizes[group_idx] = {
                "total": len(df_group),
                "subgroups": sort_dict(subgroup_sizes),
            }

        group_sizes = sort_dict(group_sizes, selector=lambda g: g["total"])
        self._dump_intermediate("group-sizes", group_sizes)

        return {"group_sizes": group_sizes}

    def _pointwise_dists_dihedral(self, pool: mp.pool.Pool):
        """
        Calculate pointwise-distances between pairs of codon-tuples (sub-groups).
        Each codon-tuple is represented by the set of all dihedral angles coming from it.

        The distance between two codon-tuple sub-groups is calculated using the T2
        statistic as a distance metric between sets of angle-pairs. The distance
        between two angle-pairs is calculated on the torus.
        """
        df_processed: pd.DataFrame = self._load_intermediate("dataset-tuples")

        result_types_to_subgroup_pairs = {
            "aa": (AA_COL, AA_COL),
            "aac": (AA_COL, CODON_COL),
            "codon": (CODON_COL, CODON_COL),
        }
        for (
            result_name,
            (subgroup1_col, subgroup2_col),
        ) in result_types_to_subgroup_pairs.items():

            sub1_names_to_idx, sub2_names_to_idx = (
                self._aa_tuple_to_idx
                if subgroup_col == AA_COL
                else self._codon_tuple_to_idx
                for subgroup_col in (subgroup1_col, subgroup2_col)
            )

            # Collect results
            default = dict(
                shape=(len(sub1_names_to_idx), len(sub2_names_to_idx)),
                fill_value=np.nan,
                dtype=np.float32,
            )

            group_names = sorted(set(df_processed[CONDITION_COL]))
            collected_t2s: Dict[str, np.ndarray] = {
                g: np.full(**default) for g in group_names
            }
            collected_pvals: Dict[str, np.ndarray] = {
                g: np.full(**default) for g in group_names
            }

            async_results = self._pointwise_dists_dihedral_subgroup_pairs(
                pool, df_processed, result_name, subgroup1_col, subgroup2_col
            )

            for (group, sub1, sub2), result in yield_async_results(async_results):
                i, j = (
                    sub1_names_to_idx[sub1],
                    sub2_names_to_idx[sub2],
                )
                LOGGER.info(
                    f"Collected {result_name} pairwise-pval {group=}, {sub1=} ({i=}), "
                    f"{sub2=} ({j=}): (t2, p)={result}"
                )
                t2, pval = result
                t2s = collected_t2s[group]
                pvals = collected_pvals[group]
                t2s[i, j] = t2
                pvals[i, j] = pval
                if subgroup1_col == subgroup2_col:
                    t2s[j, i] = t2
                    pvals[j, i] = pval

            self._dump_intermediate(f"{result_name}-dihedral-t2s", collected_t2s)
            self._dump_intermediate(f"{result_name}-dihedral-pvals", collected_pvals)

    def _pointwise_dists_dihedral_subgroup_pairs(
        self,
        pool: mp.pool.Pool,
        df_processed: pd.DataFrame,
        result_name: str,
        subgroup1_col: str,
        subgroup2_col: str,
    ):
        """
        Helper function that submits pairs of subgroups for pointwise angle-based
        analysis.
        """
        df_groups = df_processed.groupby(by=CONDITION_COL)
        async_results: Dict[Tuple[str, str, str], AsyncResult] = {}

        LOGGER.info(
            f"Calculating dihedral angle differences between pairs of "
            f"{result_name}-tuples..."
        )

        # Group by the conditioning criteria, e.g. SS
        for group_idx, (group, df_group) in enumerate(df_groups):

            # Group by subgroup1 (e.g. AA  or codon)
            df_sub1_groups = df_group.groupby(subgroup1_col)
            for i, (sub1, df_sub1) in enumerate(df_sub1_groups):
                # Need at least 2 observations in each statistical sample
                if len(df_sub1) < 2:
                    continue

                # Group by subgroup2 (e.g. codon)
                if subgroup2_col == subgroup1_col:
                    # If the subgroup columns are the same, interpret as sub1 and
                    # sub2 are independent.
                    df_sub2_groups = df_group.groupby(subgroup2_col)
                else:
                    # If the subgroup columns are the distinct, interpret as sub2 is a
                    # subgroup of sub1
                    df_sub2_groups = df_sub1.groupby(subgroup2_col)

                for j, (sub2, df_sub2) in enumerate(df_sub2_groups):
                    # Need at least 2 observations in each statistical sample
                    if len(df_sub2) < 2:
                        continue

                    # Skip redundant calculations: identical pairs with h different
                    # order
                    if subgroup2_col == subgroup1_col and j < i:
                        continue

                    # Analyze the angles of subgroup1 and subgroup2
                    angles1 = df_sub1[[*ANGLE_COLS]].values
                    angles2 = df_sub2[[*ANGLE_COLS]].values
                    args = (
                        group,
                        sub1,
                        sub2,
                        angles1,
                        angles2,
                        self.bs_randstate,
                        self.t2_n_max,
                        self.t2_permutations,
                        flat_torus_distance,
                    )
                    res = pool.apply_async(_subgroup_tw2_test, args=args)
                    async_results[(group, sub1, sub2)] = res

        return async_results

    def _dihedral_kde_groups(self, pool: mp.pool.Pool) -> dict:
        """
        Estimates the dihedral angle distribution of each group (e.g. SS) as a
        Ramachandran plot, using kernel density estimation (KDE).
        """
        df_processed: pd.DataFrame = self._load_intermediate("dataset")
        df_groups = df_processed.groupby(by=SECONDARY_COL)
        df_groups_count: pd.DataFrame = df_groups.count()
        ss_counts = {
            f"n_{ss_type}": count
            for ss_type, count in df_groups_count.max(axis=1).to_dict().items()
        }

        LOGGER.info(f"Secondary-structure groups:\n{ss_counts})")
        LOGGER.info(f"Calculating dihedral distribution per SS type...")

        args = (
            (group_idx, df_group, ANGLE_COLS, self.kde_args)
            for group_idx, df_group in df_groups
        )

        map_result = pool.starmap(_dihedral_kde_single_group, args)

        # maps from group (SS) to a dihedral KDE matrix
        map_result = {group_idx: dkde for (group_idx, dkde) in map_result}
        self._dump_intermediate("full-dkde", map_result)

        return {**ss_counts}

    def _pointwise_dists_kde(self, pool: mp.pool.Pool) -> dict:
        """
        Calculate pointwise-distances between pairs of codon (sub-groups).
        Each codon is represented by the KDE (estimate distribution) of the dihedral
        angles coming from it.

        The distance between two codon sub-groups is calculated using both euclidean
        distance between their KDEs (using bootstrapping to estimate multiple KDEs
        for the same codon), and also using the T2 statistic as a distance.
        """
        group_sizes = self._load_intermediate("group-sizes")
        curr_codon_col = CODON_COL

        # We currently only support one type of metric
        dist_metrics = {"l2": _kde_dist_metric_l2}
        dist_metric = dist_metrics[self.kde_dist_metric.lower()]

        # Set chunk-size for parallel mapping.
        chunksize = 1

        df_processed: pd.DataFrame = self._load_intermediate("dataset")
        df_groups = df_processed.groupby(by=CONDITION_COL)

        LOGGER.info(
            f"Calculating subgroup KDEs "
            f"(n_parallel_groups={self.n_parallel_groups}, "
            f"chunksize={chunksize})..."
        )

        codon_d2s, codon_t2s, codon_pvals, codon_dkdes = {}, {}, {}, {}
        aa_d2s, aa_t2s, aa_pvals, aa_dkdes = {}, {}, {}, {}
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
            # samples in the smallest subgroup, or a fixed number.
            if not self.bs_fixed_n:  # 0 or ""
                bs_nsamples = subgroup_lens
            elif self.bs_fixed_n == "min":
                bs_nsamples = [min_len] * len(subgroup_lens)
            elif self.bs_fixed_n == "max":
                bs_nsamples = [max_len] * len(subgroup_lens)
            else:  # some positive integer
                bs_nsamples = [self.bs_fixed_n] * len(subgroup_lens)

            # Run bootstrapped KDE estimation for all subgroups in parallel
            subprocess_args = (
                (
                    group_idx,
                    subgroup_idx,
                    df_subgroup,
                    ANGLE_COLS,
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
            if not last_group and len(dkde_asyncs) < self.n_parallel_groups:
                continue  # submit another group fo KDE analysis

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

                # bs_codon_dkdes maps from codon to a bootstrapped KDE of shape (B,N,N),
                # Initialize in advance to obtain consistent order of codons
                bs_codon_dkdes: Dict[str, Optional[np.ndarray]] = {
                    c: None for c in AA_CODONS
                }
                for subgroup_idx, subgroup_bs_dkde in group_dkde_result:
                    bs_codon_dkdes[subgroup_idx] = subgroup_bs_dkde

                subgroup_sizes = group_sizes[result_group_idx]["subgroups"]

                # Run distance matrix calculation in parallel
                dist_asyncs[result_group_idx] = pool.apply_async(
                    _dkde_dists_single_group,
                    args=(
                        result_group_idx,
                        bs_codon_dkdes,
                        subgroup_sizes,
                        self.bs_randstate,
                        self.t2_n_max,
                        self.t2_permutations,
                        dist_metric,
                    ),
                )
                LOGGER.info(f"[{i}] Submitted cdist task {result_group_idx}")

            # Allow limited number of simultaneous distance matrix calculations
            if not last_group and len(dist_asyncs) < self.n_parallel_groups:
                continue  # submit another group fo KDE analysis

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
                    group_codon_t2s,
                    group_codon_pvals,
                    group_codon_dkdes,
                    group_aa_d2s,
                    group_aa_t2s,
                    group_aa_pvals,
                    group_aa_dkdes,
                ) = group_dist_result
                codon_d2s[result_group_idx] = group_codon_d2s
                codon_t2s[result_group_idx] = group_codon_t2s
                codon_pvals[result_group_idx] = group_codon_pvals
                codon_dkdes[result_group_idx] = group_codon_dkdes
                aa_d2s[result_group_idx] = group_aa_d2s
                aa_t2s[result_group_idx] = group_aa_t2s
                aa_pvals[result_group_idx] = group_aa_pvals
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

        # codon_dkde_d2s: maps from group (e.g. SS) a codon-distance matrix.
        # The codon distance matrix is complex, where real is mu and imag is sigma
        self._dump_intermediate("codon-dkde-d2s", codon_d2s)

        # codon_dkde_t2s: same as codon_dkde_d2s, but distances are the t2 statistic.
        self._dump_intermediate("codon-dkde-t2s", codon_t2s)

        # codon_pvals: maps from group (e.g. SS) to a pairwise pvalue matrix
        self._dump_intermediate("codon-dkde-pvals", codon_pvals)

        # aa_dkde_d2s: Same as codon_dkde_d2s, but keys are AAs
        self._dump_intermediate("aa-dkde-d2s", aa_d2s)

        # aa_dkde_t2s: Same as codon_dkde_t2s, but keys are AAs
        self._dump_intermediate("aa-dkde-t2s", aa_t2s)

        # aa_pvals: Same as codon pvals, but keys are AAs
        self._dump_intermediate("aa-dkde-pvals", aa_pvals)

        # codon_dkdes: maps from group to a dict where keys are codons.
        # For each codon we have it's average KDE.
        self._dump_intermediate("codon-dkdes", codon_dkdes)

        # aa_dkdes: Same as codon_dkdes, but keys are AAs
        self._dump_intermediate("aa-dkdes", aa_dkdes)

        return {}

    def _codon_dists_expected(self, pool: mp.pool.Pool) -> dict:
        """
        Takes the expectation of the various distances between pairs of codons
        (sub-groups) over the groups (e.g. SS+prev codon).
        """

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

        def _dists_expected(
            dists: Dict[str, Optional[np.ndarray]]
        ) -> Dict[str, Optional[np.ndarray]]:
            # dists maps from group index to a distance matrix

            # This dict will hold the final expected distance matrix (i.e. we
            # calculate the expectation using the likelihood of the prev codon
            # or aa).
            dists_exp: Dict[str, Optional[np.ndarray]] = {}
            for group_idx, d2 in dists.items():

                if len(group_idx.split("_")) == 1:
                    codon_or_aa, ss_type = None, group_idx
                else:
                    codon_or_aa, ss_type = group_idx.split("_")

                if ss_type not in dists_exp:
                    dists_exp[ss_type] = np.zeros_like(d2)

                p = likelihoods[ss_type][codon_or_aa] if likelihoods else 1.0
                # Don't sum nan's because they kill the entire cell
                d2 = np.nan_to_num(d2, copy=False, nan=0.0)
                dists_exp[ss_type] += p * d2

            # Now we also take the expectation over all SS types to produce one
            # averaged distance matrix, but only if we actually conditioned on it
            if self.condition_on_ss:
                dists_exp[SS_TYPE_ANY] = np.zeros_like(d2)
                for ss_type, d2 in dists_exp.items():
                    if ss_type == SS_TYPE_ANY:
                        continue
                    p = ss_likelihoods[ss_type]
                    dists_exp[SS_TYPE_ANY] += p * d2

            return dists_exp

        for aa_codon, dihedral_dkde, d2_t2 in it.product(
            {"codon", "aa"}, {"dihedral", "dkde"}, {"d2s", "t2s"}
        ):
            # Load the calculated dists matrices. The loaded dict maps from group
            # index (prev_codon/AA, SS) to a distance matrix.
            dists: Dict[str, Optional[np.ndarray]] = self._load_intermediate(
                f"{aa_codon}" f"-{dihedral_dkde}-{d2_t2}"
            )
            if dists is None:
                continue

            dists_exp: Dict[str, Optional[np.ndarray]] = _dists_expected(dists)
            self._dump_intermediate(
                f"{aa_codon}-{dihedral_dkde}-{d2_t2}-exp", dists_exp
            )

        return {}

    def _plot_results(self, pool: mp.pool.Pool):
        """
        Loads the intermediate results that were generated during the analysis and
        plots them.
        """
        LOGGER.info(f"Plotting results...")

        ap_label = _cols2label(PHI_COL, PSI_COL)

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
                    kwds=dict(
                        out_dir=self.out_dir,
                        angle_pair_label=ap_label,
                        vmin=0.0,
                        vmax=5e-4,
                    ),
                )
            )
            del full_dkde

        # Distance matrices and p-vals
        distance_types = it.product(
            ["aa", "codon", "aac"],
            ["dkde", "dihedral"],
            ["d2s", "t2s", "pvals"],
            ["", "exp"],
            ["block", ""],
        )
        for aa_codon, dkde_dihedral, dist_type, exp, block_diag in distance_types:
            result_name = str.join(
                "-", filter(None, [aa_codon, dkde_dihedral, dist_type, exp])
            )
            dists: Dict[str, Optional[np.ndarray]] = self._load_intermediate(
                result_name, True
            )
            if dists is None:
                continue

            out_dir = self.out_dir.joinpath(result_name)
            if aa_codon == "aa":
                row_labels = col_labels = ACIDS
            elif aa_codon == "codon":
                row_labels = col_labels = AA_CODONS
            else:  # aac
                row_labels = ACIDS
                col_labels = AA_CODONS

            block_diagonal_pairs = None
            if block_diag:
                if aa_codon == "codon":
                    block_diagonal_pairs = SYN_CODON_IDX
                else:
                    continue  # Prevent plotting twice

            vmin, vmax = None, None
            if dist_type == "pvals":
                vmin, vmax = 0.0, 1.0

            for group_idx, d2 in dists.items():
                async_results.append(
                    pool.apply_async(
                        _plot_dist_matrix,
                        kwds=dict(
                            group_idx=group_idx,
                            d2=d2,
                            out_dir=out_dir,
                            title=None,
                            row_labels=row_labels,
                            col_labels=col_labels,
                            vmin=vmin,
                            vmax=vmax,
                            annotate_mu=False,
                            plot_std=False,
                            block_diagonal=block_diagonal_pairs,
                            tag=block_diag,
                        ),
                    )
                )
            del dists, d2

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
                            angle_pair_labels=ap_label,
                            vmin=0.0,
                            vmax=5e-4,
                        ),
                    )
                )
            del avg_dkdes, dkdes

        # Wait for plotting to complete. Each function returns a path
        fig_paths = self._handle_async_results(async_results, collect=True)


def _subgroup_tw2_test(
    group_idx: str,
    subgroup1_idx: str,
    subgroup2_idx: str,
    subgroup1_data: np.ndarray,
    subgroup2_data: np.ndarray,
    randstate: int,
    t2_n_max: Optional[int],
    t2_permutations: int,
    t2_metric: Union[str, Callable],
) -> Tuple[float, float]:
    """
    Calculates Tw^2 statistic and p-value to determine whether the dihedral angles of a
    codon are significantly different then another.
    :param group_idx: Name of the group.
    :param subgroup1_idx: Name of subgroup.
    :param subgroup2_idx: Name of second subgroup.
    :param subgroup1_data: Observations of first subgroup.
        Should be a (N, M) array where N is the number of M-dimensional observations.
    :param subgroup2_data: Observations of second subgroup.
        Should be a (N, M) array where N is the number of M-dimensional observations.
    :param randstate: Random state for bootstrapping.
    :param t2_n_max: Max sample size. If None or zero then no limit.
    :param t2_permutations: Number of permutations for computing significance.
    :param t2_metric: Distance metric to use between observations.
    :return: A Tuple (t2, pval) containing the value of the T2 statistic and the p-value.
    """
    t_start = time.time()

    # We want a different random state in each subgroup, but reproducible
    seed = None
    if randstate is not None:
        seed = (hash(group_idx + subgroup1_idx + subgroup2_idx) + randstate) % (2 ** 31)
        np.random.seed(seed)
    random = np.random.default_rng(seed)

    # We use bootstrapping if at least one of the samples is larger than t2_n_max.
    n1, n2 = len(subgroup1_data), len(subgroup2_data)
    n_iter = max(ceil(n1 / t2_n_max), ceil(n2 / t2_n_max)) if t2_n_max else 1

    # For a sample larger than t2_n_max, we create a new sample from it by sampling with
    # replacement.
    def _bootstrap_sample(angles: np.ndarray):
        n = len(angles)
        if t2_n_max and n > t2_n_max:
            sample_idxs = random.choice(n, t2_n_max, replace=True)
        else:
            sample_idxs = np.arange(n)
        return angles[sample_idxs]

    # Run bootstrapped tests
    t2s, pvals = np.empty(n_iter, dtype=np.float32), np.empty(n_iter, dtype=np.float32)
    for i in range(n_iter):
        t2s[i], pvals[i] = tw_test(
            X=_bootstrap_sample(subgroup1_data).transpose(),
            Y=_bootstrap_sample(subgroup2_data).transpose(),
            k=t2_permutations,
            metric=t2_metric,
        )

    # Calculate the t2 corresponding to the maximal (worst) p-value, and that p-value.
    argmax_p = np.argmax(pvals)
    max_t2, max_p = t2s[argmax_p], pvals[argmax_p]

    # Calculate the t2 corresponding to the median p-value.
    sort_idx = np.argsort(pvals)
    median_idx = sort_idx[floor(n_iter / 2)]  # note: floor is correct since indices
    # are zero based
    med_t2, med_p = t2s[median_idx], pvals[median_idx]

    t_elapsed = time.time() - t_start
    LOGGER.info(
        f"Calculated (t2, pval) {group_idx=}, {subgroup1_idx=} (n={n1}), "
        f"{subgroup2_idx=} (n={n2}), using {n_iter=}: "
        f"(med_p, max_p)=({med_p:.3f},{max_p:.3f}),"
        f"(med_t2, max_t2)=({med_t2:.2f},{max_t2:.2f}), "
        f"elapsed={t_elapsed:.2f}s"
    )
    return med_t2, med_p


def _dihedral_kde_single_group(
    group_idx: str,
    df_group: pd.DataFrame,
    angle_cols: Tuple[str, str],
    kde_args: Dict[str, Any],
) -> Tuple[str, np.ndarray]:
    """
    Computes kernel density estimation for a dataframe of angles belonging to a
        condition group (e.g same SS).
    :param group_idx: Identifier of the group.
    :param df_group: Dataframe containing the group's data, specifically all the
        angle columns.
    :param angle_cols: A 2-tuple containing the angle column names.
    :param kde_args: arguments for KDE.
    :return: A tuple:
        - The group id
        - A list of KDEs, each an array of shape (M, M).
    """
    kde = BvMKernelDensityEstimator(**kde_args)

    phi_col, psi_col = angle_cols
    phi = df_group[phi_col].values
    psi = df_group[psi_col].values
    dkde = kde(phi, psi)

    return group_idx, dkde


def _codon_dkdes_single_subgroup(
    group_idx: str,
    subgroup_idx: str,
    df_subgroup: pd.DataFrame,
    angle_cols: Tuple[str, str],
    kde_args: dict,
    bs_niter: int,
    bs_nsamples: int,
    bs_randstate: Optional[int],
) -> Tuple[str, np.ndarray]:
    """
    Calculates (bootstrapped) KDEs of angle distributions for a subgroup of data
    (e.g. a specific codon).
    :param group_idx: Identifier of the given data's group (e.g. SS).
    :param subgroup_idx: Identifier of the given data's subgroup (e.g. codon).
    :param df_subgroup: Dataframe with the angle data.
    :param angle_cols: A two-tuple with names of angle columns.
    :param kde_args: Arguments for density estimation.
    :param bs_niter: Number of bootstrap iterations.
    :param bs_nsamples: Number of times to sample the group in each bootstrap iteration.
    :param bs_randstate: Random state for bootstrapping.
    :return: A tuple:
        - The subgroup id.
        - A KDE of shape (B, M, M)
    """

    # Create a 3D tensor to hold the bootstrapped KDE, of shape (B,M,M)
    bs_kde_shape = (bs_niter, kde_args["n_bins"], kde_args["n_bins"])
    bs_dkde = np.empty(bs_kde_shape, np.float32)

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

        _, dkde = _dihedral_kde_single_group(
            subgroup_idx, df_subgroup_sampled, angle_cols, kde_args
        )

        # Save the current iteration's KDE into the results tensor
        bs_dkde[bootstrap_idx, ...] = dkde

    t_elapsed = time.time() - t_start
    bs_rate_iter = bs_niter / t_elapsed
    LOGGER.info(
        f"Completed {bs_niter} bootstrap iterations for "
        f"{group_idx}_{subgroup_idx}, size={len(df_subgroup)}, "
        f"bs_nsamples={bs_nsamples}, "
        f"rate={bs_rate_iter:.1f} iter/sec "
        f"elapsed={t_elapsed:.1f} sec"
    )

    return subgroup_idx, bs_dkde


def _dkde_dists_single_group(
    group_idx: str,
    bs_codon_dkdes: Dict[str, Optional[np.ndarray]],
    subgroup_sizes: Dict[str, int],
    bs_randstate: int,
    t2_n_max: int,
    t2_permutations: int,
    kde_dist_metric: Callable,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    Dict[str, np.ndarray],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    Dict[str, np.ndarray],
]:
    """
    Calculates the pairwise distance matrix for codons and AAs based on permutation
    tests between bootstrapped KDEs from different codons.

    Also averages the bootstrapped KDEs to obtain a single KDE estimate per
    codon or AA in the group.

    :param group_idx: Identifier of the group this data is from.
    :param bs_codon_dkdes: A dict mapping from an AA-codon to the bootstrapped KDE
       of that codon.
    :param subgroup_sizes: A dict mapping from subgroup name (AA or AA-codon) to the
        number of samples (dataset rows) available for that AA or AA-codon.
    :param bs_randstate: Random state for bootstrapping.
    :param t2_n_max: Maximum sample size for t-test.
    :param t2_permutations: Number of permutations to use when calculating T^2
        statistic.
    :param kde_dist_metric: Function to compute distance between two KDEs.
    :return: A tuple:
        - Codon pairwise KDE-distance matrix
        - Codon pairwise t2-distance matrix
        - Codon pairwise-pvalue matrix
        - Codon averaged KDE
        - AA pairwise KDE-distance matrix
        - AA pairwise t2-distance matrix
        - AA pairwise-pvalue matrix
        - AA averaged KDE
    """

    tstart = time.time()
    # Codon distance matrix and average codon KDEs
    codon_d2s, codon_t2s, codon_pvals = _dkde_dists_pairwise(
        group_idx,
        bs_codon_dkdes,
        bs_randstate,
        t2_n_max,
        t2_permutations,
        kde_dist_metric,
    )
    codon_dkdes = _dkde_average(bs_codon_dkdes)
    LOGGER.info(
        f"Calculated codon distance matrix and average KDE for {group_idx} "
        f"({time.time() - tstart:.1f}s)..."
    )

    # AA distance matrix and average AA KDEs
    # First, we compute a weighted sum of the codon dkdes to obtain AA dkdes.
    tstart = time.time()
    bs_aa_dkdes: Dict[str, Optional[np.ndarray]] = {
        aac[0]: None for aac, dkde in bs_codon_dkdes.items()
    }
    for aa in ACIDS:
        # Total number of samples from all subgroups (codons) of this AA
        n_aa_samples = subgroup_sizes[aa]
        for aac, bs_dkde in bs_codon_dkdes.items():
            if aac[0] != aa:
                continue
            if subgroup_sizes[aac] == 0:
                continue
            if bs_dkde is None:
                continue

            # Empirical probability of this codon subgroup within its AA
            p = subgroup_sizes[aac] / n_aa_samples

            # Initialize KDE of current AA to zero so we can accumulate
            if bs_aa_dkdes[aa] is None:
                bs_aa_dkdes[aa] = np.zeros_like(bs_dkde)

            # Weighted sum of codon dkdes belonging to the current AA
            bs_aa_dkdes[aa] += p * bs_dkde

    # Now, we apply the pairwise distance between pairs of AA KDEs, as for the
    # codons
    aa_d2s, aa_t2s, aa_pvals = _dkde_dists_pairwise(
        group_idx, bs_aa_dkdes, bs_randstate, t2_n_max, t2_permutations, kde_dist_metric
    )
    aa_dkdes = _dkde_average(bs_aa_dkdes)
    LOGGER.info(
        f"Calculated AA distance matrix and average KDE for {group_idx} "
        f"({time.time() - tstart:.1f}s)..."
    )
    return (
        codon_d2s,
        codon_t2s,
        codon_pvals,
        codon_dkdes,
        aa_d2s,
        aa_t2s,
        aa_pvals,
        aa_dkdes,
    )


def _dkde_dists_pairwise(
    group_idx: str,
    bs_dkdes: Dict[str, Optional[np.ndarray]],
    bs_randstate: int,
    t2_n_max: int,
    t2_permutations: int,
    kde_dist_metric: Callable[[np.ndarray, np.ndarray], np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate a distance matrix based on the distance between each pair of
    subgroups (codons or AAs) for which we have a bootstrapped KDE.

    :param group_idx: Name of the current group.
    :param bs_dkdes: A dict from a subgroup identifier to a bootstrapped KDE.
    :param bs_randstate: Random state for bootstrapping.
    :param t2_n_max: Maximum sample size for t-test.
    :param t2_permutations: Number of permutations to use when calculating T^2
        statistic.
    :param kde_dist_metric: A function to compute the distance between two KDEs.
    :return: A tuple:
        - Pairwise KDE-distance matrix between subgroups.
          The distance matrix will be complex, where the real value is the average
          distance over the bootstrapped samples and the imag is the std.
        - Pairwise t2-distance matrix.
        - Pairwise p-value matrix. Contains the p-value, i.e the
          probability that the real distance between the distributions is zero between
          each pair of subgroups.
    """

    def _dkde_to_tw_observation(dkde: np.ndarray):
        # Converts batch of KDEs to an observations matrix for a Tw^2 test.
        B, M, M = dkde.shape
        # Flatten to (B, M * M, B): We have B observations and the dimension of each
        # observation is M*M.
        X = dkde.reshape((B, M * M))
        # Apply log in such a way as to scale dynamic range to [-100, 0].
        X = np.log(X + 1e-43)
        return X

    dkde_names = list(bs_dkdes.keys())
    N = len(dkde_names)

    # We have N dkde matrices, so we compute the distance between each such pair.
    # We use a complex array to store mu as the real part and sigma
    # as the imaginary part in a single array.
    d2_mat = np.full((N, N), np.nan, np.complex64)
    t2_mat = np.full((N, N), np.nan, np.float32)
    pval_mat = np.full((N, N), np.nan, np.float32)

    dkde_pairs = it.product(enumerate(dkde_names), enumerate(dkde_names))
    for (i, ci), (j, cj) in dkde_pairs:
        if bs_dkdes[ci] is None:
            continue

        if j < i or bs_dkdes[cj] is None:
            continue

        t_start = time.time()

        # Get the two dihedral KDEs arrays to compare, each is of
        # shape (B, M, M) due to bootstrapping B times
        dkde1 = bs_dkdes[ci]
        dkde2 = bs_dkdes[cj]

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

        t_elapsed = time.time() - t_start
        LOGGER.info(
            f"Calculated d2 {group_idx=}, {ci=}, {cj=}, value={d2_mu:.3e}±"
            f"{d2_sigma:.3e}, elapsed={t_elapsed:.2f}s"
        )

        #  Calculate statistical significance of the distance based on T_w^2 metric
        t2, pval = _subgroup_tw2_test(
            group_idx=group_idx,
            subgroup1_idx=ci,
            subgroup2_idx=cj,
            subgroup1_data=_dkde_to_tw_observation(dkde1),
            subgroup2_data=_dkde_to_tw_observation(dkde2),
            t2_n_max=t2_n_max,
            randstate=bs_randstate,
            t2_permutations=t2_permutations,
            t2_metric="sqeuclidean",
        )
        t2_mat[i, j] = t2_mat[j, i] = t2
        pval_mat[i, j] = pval_mat[j, i] = pval

    return d2_mat, t2_mat, pval_mat


def _dkde_average(bs_dkdes: Dict[str, Optional[np.ndarray]]) -> Dict[str, np.ndarray]:
    """
    Averages the KDEs from all bootstraps, to obtain a single KDE per subgroup
    (codon or AA). Note that this KDE will also include the variance in each bin,
    so we'll save as a complex matrix where real is the KDE value and imag is the std.

    :param bs_dkdes: A dict from a subgroup identifier to a bootstrapped KDE.
    :return: A dict mapping from a group identifier (codon or AA) to an
        averaged KDE for that subgroup. The averaged KDE will actually be a
        complex matrix where the real part is average and the imag is std.
    """

    avg_dkdes = {c: None for c in bs_dkdes.keys()}
    for subgroup, bs_dkde in bs_dkdes.items():
        if bs_dkde is None:
            continue

        # bs_dkde here is of shape (B, N, N) due to bootstrapping.
        # Average it over the bootstrap dimension
        mean_dkde = np.nanmean(bs_dkde, axis=0, dtype=np.float32)
        std_dkde = np.nanstd(bs_dkde, axis=0, dtype=np.float32)
        avg_dkdes[subgroup] = mean_dkde + 1j * std_dkde

    return avg_dkdes


def _kde_dist_metric_l2(kde1: np.ndarray, kde2: np.ndarray):
    """
    L2 distance between two bootstrapped KDEs.
    We expect kde1 and kde2 to be of shape (B, N, N)
    We calculate distance between each 2D NxN plane and return B distances.
    :param kde1: First bootstrapped KDE.
    :param kde2: Second bootstrapped KDE.
    :return: Distances, an array of shape (B,)
    """
    assert kde1.ndim == 3 and kde2.ndim == 3
    return np.nansum((kde1 - kde2) ** 2, axis=(1, 2))


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


def _plot_full_dkdes(
    full_dkde: Dict[str, Optional[np.ndarray]],
    angle_pair_label: str,
    out_dir: Path,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
):
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
        fig: Figure
        axes: Sequence[Axes] = ax.reshape(-1)

        for i, (group_idx, dkde) in enumerate(full_dkde.items()):
            pp5.plot.ramachandran(
                dkde,
                angle_pair_label,
                title=group_idx,
                ax=axes[i],
                vmin=vmin,
                vmax=vmax,
            )

        pp5.plot.savefig(fig, fig_filename, close=True)

    return str(fig_filename)


def _plot_dist_matrix(
    group_idx: str,
    d2: np.ndarray,
    title: Optional[str],
    row_labels: List[str],
    col_labels: List[str],
    out_dir: Path,
    vmin: float = None,
    vmax: float = None,
    annotate_mu: bool = True,
    plot_std: bool = False,
    block_diagonal: Sequence[Tuple[int, int]] = None,
    tag: str = None,
):
    # Split the mu and sigma from the complex d2 matrix
    d2_mu, d2_sigma = np.real(d2), np.imag(d2)

    # Instead of plotting the std matrix separately, we can also use
    # annotations to denote the value of the std in each cell.
    # We'll annotate according to the quartiles of the std.
    pct = np.nanpercentile(d2_sigma, [25, 50, 75])

    def quartile_ann_fn(_, j, k):
        std = d2_sigma[j, k]
        p25, p50, p75 = pct
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
            mask = np.full_like(d2, fill_value=np.nan)
            mask[ii, jj] = 1.0
            d2 = mask * d2

        filename = str.join(
            "-", filter(None, [group_idx, avg_std if plot_std else None, tag])
        )
        fig_filepath = out_dir.joinpath(f"{filename}.png")

        pp5.plot.multi_heatmap(
            [d2],
            row_labels=row_labels,
            col_labels=col_labels,
            titles=[title] if title else None,
            fig_size=20,
            vmin=vmin,
            vmax=vmax,
            fig_rows=1,
            outfile=fig_filepath,
            data_annotation_fn=ann_fn,
        )

        fig_filenames.append(str(fig_filepath))

    return fig_filenames


def _plot_dkdes(
    group_idx: str,
    subgroup_sizes: Dict[str, int],
    dkdes: Dict[str, Optional[np.ndarray]],
    angle_pair_label: Optional[str],
    out_dir: Path,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
):
    # Plot the kdes and distance matrices
    LOGGER.info(f"Plotting KDEs for {group_idx}")

    N = len(dkdes)
    fig_cols = int(np.ceil(np.sqrt(N)))
    fig_rows = int(np.ceil(N / fig_cols))

    with matplotlib.style.context(PP5_MPL_STYLE):
        fig, ax = mpl.pyplot.subplots(
            fig_rows,
            fig_cols,
            figsize=(5 * fig_cols, 5 * fig_rows),
            sharex="all",
            sharey="all",
        )
        axes: Sequence[Axes] = np.reshape(ax, -1)

        for i, (subgroup_idx, d2) in enumerate(dkdes.items()):
            title = f"{subgroup_idx} ({subgroup_sizes.get(subgroup_idx, 0)})"
            if d2 is None:
                ax[i].set_title(title)
                continue

            # Remove the std of the DKDE
            d2_real = np.real(d2)

            pp5.plot.ramachandran(
                d2_real,
                angle_pair_label,
                title=title,
                ax=axes[i],
                vmin=vmin,
                vmax=vmax,
            )

        fig_filename = out_dir.joinpath(f"{group_idx}.png")
        pp5.plot.savefig(fig, fig_filename, close=True)

    return str(fig_filename)


def _cols2label(phi_col: str, psi_col: str):
    def rep(col: str):
        col = col.replace("phi", r"\varphi")
        col = col.replace("psi", r"\psi")
        col = re.sub(r"([+-][01])", r"_{\1}", col)
        return col

    return rf"${rep(phi_col)}, {rep(psi_col)}$"
