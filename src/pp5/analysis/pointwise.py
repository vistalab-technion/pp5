import os
import re
import time
import logging
import multiprocessing as mp
from math import ceil, floor
from typing import (
    Any,
    Dict,
    List,
    Tuple,
    Union,
    Literal,
    Callable,
    Iterator,
    Optional,
    Sequence,
)
from pathlib import Path
from multiprocessing.pool import AsyncResult

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.style
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

import pp5.plot
from pp5.plot import PP5_MPL_STYLE
from pp5.stats import mht_bh, tw_test, mmd_test
from pp5.utils import sort_dict
from pp5.codons import (
    ACIDS,
    AAC_SEP,
    AA_CODONS,
    UNKNOWN_AA,
    AAC_TUPLE_SEP,
    UNKNOWN_CODON,
    UNKNOWN_NUCLEOTIDE,
    aac2aa,
    aac_join,
    aact2aat,
    aac_split,
    codon2aac,
    aac_tuples,
    aact_str2tuple,
    aact_tuple2str,
    aac_index_pairs,
    is_synonymous_tuple,
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
SUBGROUP_COL = "subgroup"
GROUP_SIZE_COL = "group_size"
PVAL_COL = "pval"
T2_COL = "t2"
SIGNIFICANT_COL = "significant"
TEST_STATISTICS = {"mmd": mmd_test, "tw": tw_test}

CODON_TUPLE_GROUP_ANY = "any"
CODON_TUPLE_GROUP_LAST_NUCL = "last_nucleotide"
CODON_TUPLE_GROUPINGS = {None, CODON_TUPLE_GROUP_ANY, CODON_TUPLE_GROUP_LAST_NUCL}


class PointwiseCodonDistanceAnalyzer(ParallelAnalyzer):
    def __init__(
        self,
        dataset_dir: Union[str, Path],
        out_dir: Union[str, Path] = None,
        pointwise_filename: str = "data-precs.csv",
        condition_on_ss: bool = True,
        consolidate_ss: dict = DSSP_TO_SS_TYPE.copy(),
        tuple_len: int = 1,
        codon_grouping_type: str = None,
        codon_grouping_position: int = 0,
        min_group_size: int = 1,
        strict_codons: bool = True,
        kde_nbins: int = 128,
        kde_width: float = 30.0,
        bs_niter: int = 1,
        bs_randstate: Optional[int] = None,
        t2_statistic: Union[Literal["mmd"], Literal["tw"]] = "mmd",
        t2_n_max: int = 1000,
        t2_permutations: int = 1000,
        t2_kernel_size: float = 10.0,
        fdr: float = 0.1,
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
        :param tuple_len: Number of consecutive residues to analyze as a tuple.
            Set 1 for single, 2 for pairs. Higher values are not recommended.

        :param codon_grouping_type:
        :param codon_grouping_position:

        :param min_group_size: Minimal number of angle-pairs from different
            structures belonging to the same Uniprot ID, location and codon in order to
            consider the group of angles for analysis.
        :param strict_codons: Enforce only one known codon per residue
            (reject residues where DNA matching was ambiguous).
        :param kde_nbins: Number of angle binds for KDE estimation.
        :param kde_width: KDE concentration parameter (will use same for phi and psi).
        :param bs_niter: Number of bootstrap iterations.
        :param bs_randstate: Random state for bootstrap.
        :param t2_statistic: Statistic to use for statistical tests. Can be either
            'mmd' or 'tw'.
        :param t2_n_max: Maximal sample-size to use when calculating
            p-value of distances with a statistical test. If there are larger samples,
            bootstrap sampling with the given maximal sample size will be performed.
            If None or zero, sample size wont be limited.
        :param t2_permutations: Number of permutations to use when calculating
            p-value of distances with a statistical test.
        :param t2_kernel_size: Size of kernel used in MMD-based permutation test (
            ignored if the test statistic is not MMD).
        :param fdr: False discovery rate for multiple hypothesis testing using
            Benjamini-Hochberg method.
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

        if tuple_len < 1:
            raise ValueError(f"invalid {tuple_len=}, must be >= 1")

        if codon_grouping_type not in CODON_TUPLE_GROUPINGS:
            raise ValueError(
                f"invalid {codon_grouping_type=}, must be in {CODON_TUPLE_GROUPINGS}"
            )

        if codon_grouping_position >= tuple_len:
            raise ValueError(
                f"invalid {codon_grouping_position=}, must be < {tuple_len=}"
            )

        if t2_statistic not in TEST_STATISTICS:
            raise ValueError(
                f"t2_statistic must be one of {tuple(TEST_STATISTICS.keys())}"
            )

        if not 0.0 < fdr < 1.0:
            raise ValueError("FDR should be between 0 and 1, exclusive")

        self.condition_on_ss = condition_on_ss
        self.consolidate_ss = consolidate_ss
        self.tuple_len = tuple_len
        self.codon_grouping_type = codon_grouping_type
        self.codon_grouping_position = codon_grouping_position
        self.min_group_size = min_group_size
        self.strict_codons = strict_codons
        self.condition_on_prev = None

        self.kde_args = dict(n_bins=kde_nbins, k1=kde_width, k2=kde_width, k3=0)
        self.kde_dist_metric = "l2"

        self.bs_niter = bs_niter
        self.bs_randstate = bs_randstate
        self.t2_statistic_fn = TEST_STATISTICS[t2_statistic]
        self.t2_n_max = t2_n_max
        self.t2_permutations = t2_permutations
        self.t2_kernel_size = t2_kernel_size
        self.fdr = fdr

        # Initialize codon tuple names and corresponding indices
        tuples = list(aac_tuples(k=self.tuple_len))
        self._codon_tuple_to_idx = {
            aact_tuple2str(aac_tuple): idx for idx, aac_tuple in enumerate(tuples)
        }
        self._idx_to_codon_tuple = {i: t for t, i in self._codon_tuple_to_idx.items()}
        self._aa_tuple_to_idx = {
            aact_tuple2str(aat): idx
            for idx, aat in enumerate(sorted(set(aact2aat(aact) for aact in tuples)))
        }
        self._idx_to_aa_tuple = {i: t for t, i in self._aa_tuple_to_idx.items()}
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
            "kde_groups": self._kde_groups,
            "kde_subgroups": self._kde_subgroups,
            "write_pvals": self._write_pvals,
            "plot_results": self._plot_results,
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
            condition_groups = SS_TYPE_ANY
        df_pointwise[CONDITION_COL] = condition_groups

        # Convert codon columns to AA-CODON
        idx_no_codon = df_pointwise[CODON_COL] == UNKNOWN_CODON
        df_pointwise = df_pointwise[~idx_no_codon]
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
            centroid = _subgroup_centroid(df_subgroup, input_degrees=True)
            processed_subgroups.append(
                {
                    UNP_ID_COL: unp_id,
                    UNP_IDX_COL: unp_idx,
                    AA_COL: str.split(aa_codon, AAC_SEP)[0],
                    CODON_COL: aa_codon,
                    CONDITION_COL: group_id,
                    SECONDARY_COL: subgroup_ss,
                    PHI_COL: centroid.phi_deg,
                    PSI_COL: centroid.psi_deg,
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
                    self._create_group_tuples, args=(df_group, self.tuple_len),
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

        # Converts codons so they can be grouped in a several ways
        def _codon_converter(aacs: Sequence[str]) -> Sequence[str]:
            if not self.codon_grouping_type:
                return aacs

            aacs = list(aacs)
            aac = aacs[self.codon_grouping_position]
            aa, c = aac_split(aac)

            if self.codon_grouping_type == CODON_TUPLE_GROUP_ANY:
                c = UNKNOWN_CODON
                aa = UNKNOWN_AA

            if self.codon_grouping_type == CODON_TUPLE_GROUP_LAST_NUCL:
                c = f"{UNKNOWN_NUCLEOTIDE*2}{c[-1]}"
                aa = UNKNOWN_AA

            aac = aac_join(aa, c, validate=False)
            aacs[self.codon_grouping_position] = aac
            return aacs

        # Function to map rows in the merged dataframe to the final rows we'll use.
        def _row_mapper(row: pd.Series):

            codons, aas, sss, group_sizes = [], [], [], []
            for i, p in enumerate(prefixes):
                codons.append(row[f"{p}{CODON_COL}"])
                aas.append(row[f"{p}{AA_COL}"])
                sss.append(row[f"{p}{SECONDARY_COL}"])
                group_sizes.append(row[f"{p}{GROUP_SIZE_COL}"])

            codons = _codon_converter(codons)
            codon_tuple = aact_tuple2str(codons)
            aa_tuple = aact_tuple2str(aas)
            ss_tuple = aact_tuple2str(sss)
            if not self.condition_on_ss:
                condition_group = SS_TYPE_ANY
            elif all(ss == sss[0] for ss in sss):
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
        df_groups = groupby_with_full_group(
            df_processed, full_group_name=SS_TYPE_ANY, full_first=True, by=CONDITION_COL
        )
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
                SUBGROUP_COL: sort_dict(subgroup_sizes),
            }

        group_sizes = sort_dict(group_sizes, selector=lambda g: g["total"])
        self._dump_intermediate("group-sizes", group_sizes)

        return {"group_sizes": group_sizes}

    def _pointwise_dists_dihedral(self, pool: mp.pool.Pool) -> dict:
        """
        Calculate pointwise-distances between pairs of codon-tuples (sub-groups).
        Each codon-tuple is represented by the set of all dihedral angles coming from it.

        The distance between two codon-tuple sub-groups is calculated using the T2
        statistic as a distance metric between sets of angle-pairs. The distance
        between two angle-pairs is calculated on the torus.
        """
        df_processed: pd.DataFrame = self._load_intermediate("dataset-tuples")

        def _non_syn_codons_pair_filter(group: str, aact1: str, aact2: str):
            # Returns True if aact1 and aact2 are not synonymous (therefore should be
            # filtered out).
            return not is_synonymous_tuple(aact_str2tuple(aact1), aact_str2tuple(aact2))

        totals = {}

        result_types_to_subgroup_pairs = {
            "aa": (AA_COL, AA_COL, None),
            "aac": (AA_COL, CODON_COL, None),
            "codon": (CODON_COL, CODON_COL, _non_syn_codons_pair_filter),
        }
        for (
            result_name,
            (subgroup1_col, subgroup2_col, pair_filter_fn),
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

            group_names = sorted(set(df_processed[CONDITION_COL]).union({SS_TYPE_ANY}))
            collected_t2s: Dict[str, np.ndarray] = {
                g: np.full(**default) for g in group_names
            }
            collected_pvals: Dict[str, np.ndarray] = {
                g: np.full(**default) for g in group_names
            }

            async_results = self._pointwise_dists_dihedral_subgroup_pairs(
                pool,
                df_processed,
                result_name,
                subgroup1_col,
                subgroup2_col,
                pair_filter_fn,
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

            self._dump_intermediate(f"{result_name}-dihedral-t2s", collected_t2s)
            self._dump_intermediate(f"{result_name}-dihedral-pvals", collected_pvals)

            totals[result_name] = {
                g: np.sum(~np.isnan(pvals)) for g, pvals in collected_pvals.items()
            }

        LOGGER.info(f"Total number of unique tuple-pairwise pvals: {totals}")
        return {"pval_counts": totals}

    def _pointwise_dists_dihedral_subgroup_pairs(
        self,
        pool: mp.pool.Pool,
        df_processed: pd.DataFrame,
        result_name: str,
        subgroup1_col: str,
        subgroup2_col: str,
        pair_filter_fn: Optional[Callable[[str, str, str], bool]],
    ):
        """
        Helper function that submits pairs of subgroups for pointwise angle-based
        analysis.
        """
        df_groups = groupby_with_full_group(
            df_processed, full_group_name=SS_TYPE_ANY, full_first=True, by=CONDITION_COL
        )
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

                    # Skip redundant calculations: identical pairs with a different
                    # order
                    if subgroup2_col == subgroup1_col and j <= i:
                        continue

                    # Skip based on custom pair filtering logic
                    if pair_filter_fn is not None and pair_filter_fn(group, sub1, sub2):
                        continue

                    # Analyze the angles of subgroup1 and subgroup2
                    angles1 = np.deg2rad(df_sub1[[*ANGLE_COLS]].values)
                    angles2 = np.deg2rad(df_sub2[[*ANGLE_COLS]].values)
                    args = (
                        group,
                        sub1,
                        sub2,
                        angles1,
                        angles2,
                        self.bs_randstate,
                        self.t2_statistic_fn,
                        self.t2_n_max,
                        self.t2_permutations,
                        self.t2_kernel_size,
                        flat_torus_distance,
                    )

                    res = pool.apply_async(_subgroup_permutation_test, args=args)
                    async_results[(group, sub1, sub2)] = res

        return async_results

    def _kde_groups(self, pool: mp.pool.Pool) -> dict:
        """
        Estimates the dihedral angle distribution of each group (e.g. SS) as a
        Ramachandran plot, using kernel density estimation (KDE).
        """
        df_processed: pd.DataFrame = self._load_intermediate("dataset-tuples")
        df_groups = tuple(
            groupby_with_full_group(
                df_processed,
                full_group_name=SS_TYPE_ANY,
                full_first=True,
                by=CONDITION_COL,
            )
        )
        ss_counts = {f"n_{ss_type}": len(df_group) for ss_type, df_group in df_groups}

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

    def _kde_subgroups(self, pool: mp.pool.Pool):
        """
        Estimates dihedral angle distributions of subgroups (AA and codon tuples).
        """
        df_processed: pd.DataFrame = self._load_intermediate("dataset-tuples")
        df_groups = tuple(
            groupby_with_full_group(
                df_processed,
                full_group_name=SS_TYPE_ANY,
                full_first=True,
                by=CONDITION_COL,
            )
        )

        result_types_to_subgroup_pairs = {
            "aa": (AA_COL, self._aa_tuple_to_idx),
            "codon": (CODON_COL, self._codon_tuple_to_idx),
        }
        for (
            result_name,
            (subgroup_col, sub_names_to_idx),
        ) in result_types_to_subgroup_pairs.items():

            LOGGER.info(
                f"Calculating dihedral angle distributions for {result_name}-tuples..."
            )

            async_results: Dict[Tuple[str, str], AsyncResult] = {}

            # Group by the conditioning criteria, e.g. SS
            for group_idx, (group, df_group) in enumerate(df_groups):

                # Group by subgroup (e.g. AA or codon tuple)
                df_sub_groups = df_group.groupby(subgroup_col)
                for i, (sub, df_sub) in enumerate(df_sub_groups):

                    args = (
                        group_idx,
                        df_sub,
                        ANGLE_COLS,
                        self.kde_args,
                    )

                    async_results[(group, sub)] = pool.apply_async(
                        _dihedral_kde_single_group, args=args
                    )

            group_names = sorted(set(df_processed[CONDITION_COL]).union({SS_TYPE_ANY}))
            collected_kdes: Dict[str, Dict[str, np.ndarray]] = {
                g: {} for g in group_names
            }
            for ((group, sub), result) in yield_async_results(async_results):
                i = sub_names_to_idx[sub]
                _, kde = result
                collected_kdes[group][sub] = kde
                LOGGER.info(f"Collected {result_name} KDEs {group=}, {sub=} ({i=})")

            self._dump_intermediate(f"{result_name}-dihedral-kdes", collected_kdes)

    def _write_pvals(self, pool: mp.pool.Pool) -> dict:
        df_processed: pd.DataFrame = self._load_intermediate("dataset-tuples")
        df_groups = tuple(
            groupby_with_full_group(
                df_processed,
                full_group_name=SS_TYPE_ANY,
                full_first=True,
                by=CONDITION_COL,
            )
        )

        out_dir = self.out_dir.joinpath("pvals")
        os.makedirs(out_dir, exist_ok=True)

        result_types = [
            ("aa", AA_COL, AA_COL, self._idx_to_aa_tuple, self._idx_to_aa_tuple),
            (
                "codon",
                CODON_COL,
                CODON_COL,
                self._idx_to_codon_tuple,
                self._idx_to_codon_tuple,
            ),
            ("aac", AA_COL, CODON_COL, self._idx_to_aa_tuple, self._idx_to_codon_tuple),
        ]

        significance_metadata = {}
        df_data = {}
        async_results = {}

        for (result_type, i_col, j_col, i_tuples, j_tuples) in result_types:
            # pvals and t2s are dicts from a group name to a dict from a
            # subgroup-pair to a pval/t2.
            pvals = self._load_intermediate(
                f"{result_type}-dihedral-pvals", True, raise_if_missing=True
            )
            t2s = self._load_intermediate(
                f"{result_type}-dihedral-t2s", True, raise_if_missing=True
            )

            significance_metadata[result_type] = {}
            df_data[result_type] = []
            group: str
            df_group: pd.DataFrame
            for idx_group, (group, df_group) in enumerate(df_groups):
                group_pvals = pvals[group]
                group_t2s = t2s[group]

                async_results[(result_type, group)] = pool.apply_async(
                    self._write_pvals_inner,
                    kwds=dict(
                        result_type=result_type,
                        i_col=i_col,
                        j_col=j_col,
                        i_tuples=i_tuples,
                        j_tuples=j_tuples,
                        group=group,
                        df_group=df_group,
                        group_pvals=group_pvals,
                        group_t2s=group_t2s,
                    ),
                )

        # Collect results
        for (
            (result_type, group),
            (group_significance_meta, group_df_data),
        ) in yield_async_results(async_results):
            significance_metadata[result_type][group] = group_significance_meta
            df_data[result_type].extend(group_df_data)

        # Write output
        self._dump_intermediate("significance", significance_metadata)
        for result_type, result_df_data in df_data.items():
            df_pvals = pd.DataFrame(data=result_df_data)
            df_pvals.sort_values(
                by=[CONDITION_COL, PVAL_COL, T2_COL],
                ascending=[True, True, False],
                inplace=True,
            )
            csv_path = str(out_dir.joinpath(f"{result_type}-pvals.csv"))
            df_pvals.to_csv(csv_path, index=False)
            LOGGER.info(f"Wrote {csv_path}")

        return {"significance": significance_metadata}

    def _write_pvals_inner(
        self,
        result_type: str,
        i_col: str,
        j_col,
        i_tuples: dict,
        j_tuples: dict,
        group: str,
        df_group: pd.DataFrame,
        group_pvals: np.ndarray,
        group_t2s: np.ndarray,
    ):

        # Get all indices of non-null pvals
        idx_valid = np.argwhere(~np.isnan(group_pvals))

        # Calculate significance threshold for pvalues for  multiple-hypothesis
        # testing.
        group_pvals_flat = group_pvals[idx_valid[:, 0], idx_valid[:, 1]]
        significance_thresh = mht_bh(q=self.fdr, pvals=group_pvals_flat)

        # Metadata about the significance of members in this group
        significance_meta = {
            "pval_thresh": significance_thresh,
            "num_hypotheses": len(group_pvals_flat),
            "num_rejections": np.sum(group_pvals_flat <= significance_thresh).item(),
        }

        # Loop over the indices of subgroups for which we computed pvalues
        df_data = []
        for i, j in idx_valid:
            subgroup_i: str = i_tuples[i]
            subgroup_j: str = j_tuples[j]

            df_subgroup_i = df_group[df_group[i_col] == subgroup_i]
            df_subgroup_j = df_group[df_group[j_col] == subgroup_j]

            mu1: Dihedral = _subgroup_centroid(df_subgroup_i, input_degrees=True)
            mu2: Dihedral = _subgroup_centroid(df_subgroup_j, input_degrees=True)
            d12 = Dihedral.flat_torus_distance(mu1, mu2, degrees=True, squared=False)

            df_data.append(
                {
                    CONDITION_COL: group,
                    f"{SUBGROUP_COL}1": subgroup_i,
                    f"{SUBGROUP_COL}2": subgroup_j,
                    "n1": len(df_subgroup_i),
                    "n2": len(df_subgroup_j),
                    "phi1_mean": mu1.phi_deg,
                    "psi1_mean": mu1.psi_deg,
                    "phi2_mean": mu2.phi_deg,
                    "psi2_mean": mu2.psi_deg,
                    "d12": d12,
                    PVAL_COL: group_pvals[i, j],
                    T2_COL: group_t2s[i, j],
                    SIGNIFICANT_COL: group_pvals[i, j] <= significance_thresh,
                }
            )

        LOGGER.info(
            f"Computed significance for {result_type=} {group=}: {significance_meta}"
        )
        return significance_meta, df_data

    def _plot_results(self, pool: mp.pool.Pool):
        """
        Loads the intermediate results that were generated during the analysis and
        plots them.
        """
        LOGGER.info(f"Plotting results...")

        ap_label = _cols2label(
            PHI_COL + ("+0" if self.tuple_len == 1 else "+1"), PSI_COL + "+0"
        )

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

        # Dihedral KDEs of each codon in each group
        df_processed: pd.DataFrame = self._load_intermediate("dataset-tuples")
        df_groups = {
            group_name: df_group
            for group_name, df_group in groupby_with_full_group(
                df_processed,
                full_group_name=SS_TYPE_ANY,
                full_first=True,
                by=CONDITION_COL,
            )
        }
        group_sizes: dict = self._load_intermediate("group-sizes", True)
        for aa_codon in ["aa", "codon"]:

            avg_dkdes: dict = self._load_intermediate(f"{aa_codon}-dihedral-kdes", True)
            if avg_dkdes is None:
                continue

            for group_idx, dkdes in avg_dkdes.items():
                subgroup_sizes = group_sizes[group_idx][SUBGROUP_COL]

                # Create glob patterns to define which ramachandran plots will go
                # into the same figure
                split_subgroups_glob = None
                if self.tuple_len > 1:
                    tuple_elements = ACIDS if aa_codon == "aa" else AA_CODONS
                    split_subgroups_glob = [f"{s}*" for s in tuple_elements]
                    # For AAs, also include the reverse glob
                    if aa_codon == "aa":
                        split_subgroups_glob.extend([f"*{s}" for s in tuple_elements])

                # Get the samples (angles) of all subgroups in this group
                subgroup_col = AA_COL if aa_codon == "aa" else CODON_COL
                df_group: pd.DataFrame = df_groups[group_idx]
                df_group_samples: pd.DataFrame = df_group[[subgroup_col, *ANGLE_COLS]]
                df_group_samples = df_group_samples.rename(
                    columns={subgroup_col: SUBGROUP_COL}
                )

                async_results.append(
                    pool.apply_async(
                        _plot_dkdes,
                        kwds=dict(
                            group_idx=group_idx,
                            subgroup_sizes=subgroup_sizes,
                            split_subgroups_glob=split_subgroups_glob,
                            dkdes=dkdes,
                            df_group_samples=df_group_samples,
                            out_dir=self.out_dir.joinpath(f"{aa_codon}-dkdes"),
                            angle_pair_label=ap_label,
                            vmin=0.0,
                            vmax=5e-4,
                        ),
                    )
                )
            del avg_dkdes, dkdes

        # pvalues
        significance_meta: dict = self._load_intermediate("significance", True)
        group_sizes: dict = self._load_intermediate("group-sizes", True)
        type_to_pvals = {
            result_type: self._load_intermediate(
                f"{result_type}-dihedral-pvals", True, raise_if_missing=True
            )
            for result_type in ["aa", "codon", "aac"]
        }
        async_results.append(
            pool.apply_async(
                _plot_pvals_hist,
                kwds=dict(
                    pvals=type_to_pvals,
                    group_sizes=group_sizes,
                    significance_meta=significance_meta,
                    out_dir=self.out_dir.joinpath(f"pvals"),
                ),
            )
        )

        # Wait for plotting to complete. Each function returns a path
        fig_paths = self._handle_async_results(async_results, collect=True)


def _subgroup_centroid(
    df_subgroup: pd.DataFrame, input_degrees: bool = False
) -> Dihedral:
    """
    Calculates centroid angle from a subgroup dataframe containing phi,psi dihedral
    angles in degrees under the columns ANGLE_COLS.
    :param df_subgroup: The dataframe.
    :param input_degrees: Whether the input data in the ANGLE_COLS is in degrees.
    :return: A Dihedral angles object containing the result.
    """
    raw_angles = df_subgroup[[*ANGLE_COLS]].values
    if input_degrees:
        raw_angles = np.deg2rad(raw_angles)

    angles = [Dihedral.from_rad(phi, psi) for phi, psi in raw_angles]
    centroid = Dihedral.circular_centroid(*angles)
    return centroid


def _subgroup_permutation_test(
    group_idx: str,
    subgroup1_idx: str,
    subgroup2_idx: str,
    subgroup1_data: np.ndarray,
    subgroup2_data: np.ndarray,
    randstate: int,
    t2_statistic_fn: Callable,
    t2_n_max: Optional[int],
    t2_permutations: int,
    t2_kernel_size: float,
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
        t2s[i], pvals[i] = t2_statistic_fn(
            X=_bootstrap_sample(subgroup1_data).transpose(),
            Y=_bootstrap_sample(subgroup2_data).transpose(),
            k=t2_permutations,
            similarity_fn=t2_metric,
            kernel_kwargs=dict(sigma=t2_kernel_size),
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
    t2_statistic_fn_name = [
        k for k, v in TEST_STATISTICS.items() if v == t2_statistic_fn
    ][0]
    LOGGER.info(
        f"Calculated (t2, pval) [{t2_statistic_fn_name}] {group_idx=}, "
        f"{subgroup1_idx=} (n={n1}), "
        f"{subgroup2_idx=} (n={n2}), "
        f"using {n_iter=}: "
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
        - A KDE of shape (M, M) where M is the number of bins.
    """
    kde = BvMKernelDensityEstimator(**kde_args)

    phi_col, psi_col = angle_cols
    phi = np.deg2rad(df_group[phi_col].values)
    psi = np.deg2rad(df_group[psi_col].values)
    dkde = kde(phi, psi)

    return group_idx, dkde


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
        fig_rows, fig_cols = len(full_dkde) // 2 + len(full_dkde) % 2, 2
        fig, ax = plt.subplots(
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

        while i + 1 < len(axes):
            i += 1
            axes[i].set_axis_off()

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
    split_subgroups_glob: Sequence[str],
    dkdes: Dict[str, Optional[np.ndarray]],
    df_group_samples: Optional[pd.DataFrame],
    angle_pair_label: Optional[str],
    out_dir: Path,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
):
    """
    :param group_idx: Name of group.
    :param subgroup_sizes: Dict from subgroup name to number of samples.
    :param split_subgroups_glob: Sequence of glob patterns to use for splitting sub
        groups into separate figures
    :param dkdes: The data to plot, maps from subgroup name to a KDE
    :param df_group_samples: The data of the current group. Optional, if provided,
        samples will be plotted on top of each ramachandran plot.
    :param angle_pair_label: Label for plot legend.
    :param out_dir: Output directory.
    :param vmin: Normalization min value.
    :param vmax: Normalization max value.
    """
    # Plot the kdes and distance matrices
    LOGGER.info(f"Plotting KDEs for {group_idx} into {out_dir!s}")

    from fnmatch import fnmatchcase

    if not split_subgroups_glob:
        split_subgroups_glob = ["*"]

    fig_filenames = []
    for subgroup_glob in split_subgroups_glob:

        filtered_dkdes = {
            sub: dkde for sub, dkde in dkdes.items() if fnmatchcase(sub, subgroup_glob)
        }

        # Sort the filtered dict by key
        filtered_dkdes = dict(sorted(filtered_dkdes.items(), key=lambda item: item[0]))

        N = len(filtered_dkdes)
        if N < 1:
            continue

        fig_cols = int(np.ceil(np.sqrt(N)))
        fig_rows = int(np.ceil(N / fig_cols))

        with matplotlib.style.context(PP5_MPL_STYLE):
            fig, ax = plt.subplots(
                fig_rows,
                fig_cols,
                figsize=(5 * fig_cols, 5 * fig_rows),
                sharex="all",
                sharey="all",
            )
            axes: Sequence[Axes] = np.reshape(ax, -1)

            for i, (subgroup_idx, d2) in enumerate(filtered_dkdes.items()):
                title = f"{subgroup_idx} ({subgroup_sizes.get(subgroup_idx, 0)})"
                if d2 is None:
                    axes[i].set_title(title)
                    continue

                # Remove the std of the DKDE
                d2_real = np.real(d2)

                samples = None
                if df_group_samples is not None:
                    idx_samples = df_group_samples[SUBGROUP_COL] == subgroup_idx
                    samples = np.deg2rad(
                        df_group_samples[idx_samples][[*ANGLE_COLS]].values
                    )

                pp5.plot.ramachandran(
                    d2_real,
                    angle_pair_label,
                    title=title,
                    ax=axes[i],
                    samples=samples,
                    vmin=vmin,
                    vmax=vmax,
                )

            while i + 1 < len(axes):
                i += 1
                axes[i].set_axis_off()

            if subgroup_glob != "*":
                fig_filename = out_dir.joinpath(f"{group_idx}").joinpath(
                    f"{subgroup_glob.replace('*', '_')}.png"
                )
            else:
                fig_filename = out_dir.joinpath(f"{group_idx}.png")

            pp5.plot.savefig(fig, fig_filename, close=True)
            fig_filenames.append(fig_filename)

    return str(fig_filenames)


def _plot_pvals_hist(
    pvals: Dict[str, np.ndarray],
    group_sizes: Dict[str, int],
    significance_meta: Dict[str, dict],
    out_dir: Path,
    n_bins: int = 50,
):
    """
    Plots pvalue histogram.
    :param pvals: Dict mapping from result_type to an array of pvalues.
    :param group_sizes: Dict from group_name to size ('total') and subgroup sizes ('subgroup').
    :param significance_meta: A dict mapping from result_type to group_name to a dict
        containing the significance information.
    :return: Path of output figure.
    """

    n_groups = len(group_sizes.keys())

    fig_filename = out_dir.joinpath("pvals_hist.pdf")
    with mpl.style.context(PP5_MPL_STYLE):
        fig_cols = 2 if n_groups > 1 else 1
        fig_rows = n_groups // 2 + n_groups % 2
        fig, ax = plt.subplots(
            fig_rows, fig_cols, figsize=(5 * fig_cols, 5 * fig_rows), squeeze=False,
        )
        fig: Figure
        axes: Sequence[Axes] = ax.reshape(-1)

        for i, (group_name, group_sizes) in enumerate(group_sizes.items()):
            ax = axes[i]

            for result_type, group_to_pvals in pvals.items():
                pvals_2d = group_to_pvals[group_name]
                pvals_flat = pvals_2d[~np.isnan(pvals_2d)]
                meta = significance_meta[result_type][group_name]
                ax.hist(
                    pvals_flat,
                    bins=n_bins,
                    density=True,
                    log=True,
                    alpha=0.5,
                    label=(
                        f"{result_type} t={meta['pval_thresh']:.4f}, "
                        f"({meta['num_rejections']}/{meta['num_hypotheses']})"
                    ),
                )
            ax.set_title(f"{group_name} ({group_sizes['total']})")
            ax.set_ylabel("log-density")
            ax.set_xlabel("pval")
            ax.legend()

        while i + 1 < len(axes):
            i += 1
            axes[i].set_axis_off()

        return pp5.plot.savefig(fig, fig_filename, close=True)


def _cols2label(phi_col: str, psi_col: str):
    def rep(col: str):
        col = col.replace("phi", r"\varphi")
        col = col.replace("psi", r"\psi")
        col = re.sub(r"([+-][01])", r"_{\1}", col)
        return col

    return rf"${rep(phi_col)}, {rep(psi_col)}$"


def groupby_with_full_group(
    df: pd.DataFrame,
    full_group_name: Union[str, Tuple[str, ...]],
    full_first: bool = False,
    **groupby_kwargs,
) -> Iterator[Tuple[Union[str, Tuple[str, ...]], pd.DataFrame]]:
    """
    Performs a groupby on a pandas dataframe, yields all groups and then yields the
    "full" group, i.e. the entire dataframe.
    :param df: The data frame.
    :param full_group_name: Name of the full group. Should be a column name or a
        tuple of column names, depending on the "by" argument of groupby.
    :param full_first: Whether to add the full-group first (True) or last (False).
    :param groupby_kwargs: kwargs to be passed as-is to groupby.
    :return: Yields tuples of (group_name, group_df). The first or last yielded tuple
        will contain the full dataframe. If there's another group with the same name
        as full_group_name, it will not be yielded.
    """
    yielded_group_names = set()

    if full_first:
        yielded_group_names.add(full_group_name)
        yield (full_group_name, df)

    for group_name, df_group in df.groupby(**groupby_kwargs):
        if group_name not in yielded_group_names:
            yielded_group_names.add(group_name)
            yield group_name, df_group

    if not full_first and (full_group_name not in yielded_group_names):
        yield (full_group_name, df)
