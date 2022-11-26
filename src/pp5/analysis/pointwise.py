import os
import re
import time
import logging
import multiprocessing as mp
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
from functools import partial
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
from pp5.stats import mht_bh, tw_test, mmd_test, kde2d_test
from pp5.utils import sort_dict
from pp5.codons import (
    ACIDS,
    AAC_SEP,
    AA_CODONS,
    UNKNOWN_AA,
    AAC_TUPLE_SEP,
    MISSING_CODON,
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
    is_synonymous_tuple,
)
from pp5.analysis import SS_TYPE_ANY, SS_TYPE_MIXED, DSSP_TO_SS_TYPE
from pp5.dihedral import Dihedral, wraparound_mean, flat_torus_distance_sq
from pp5.parallel import yield_async_results
from pp5.analysis.base import ParallelAnalyzer
from pp5.distributions.kde import bvm_kernel, gaussian_kernel, torus_gaussian_kernel_2d
from pp5.distributions.vonmises import BvMKernelDensityEstimator

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
OMEGA_COL = "omega"
PHI_PSI_COLS = (PHI_COL, PSI_COL)
ANGLE_COLS = (PHI_COL, PSI_COL, OMEGA_COL)
CONDITION_COL = "condition_group"
SUBGROUP_COL = "subgroup"
GROUP_SIZE_COL = "group_size"
GROUP_STD_COL = "group_std"
PVAL_COL = "pval"
DDIST_COL = "ddist"
SIGNIFICANT_COL = "significant"
TEST_STATISTICS = {"mmd", "tw", "kde", "kde_g"}

COMP_TYPE_AA = "aa"
COMP_TYPE_CC = "cc"
COMP_TYPE_AAC = "aac"
COMP_TYPES = (COMP_TYPE_AA, COMP_TYPE_CC, COMP_TYPE_AAC)

CODON_TUPLE_GROUP_ANY = "any"
CODON_TUPLE_GROUP_LAST_NUCL = "last_nucleotide"
CODON_TUPLE_GROUPINGS = {
    None,
    "",
    CODON_TUPLE_GROUP_ANY,
    CODON_TUPLE_GROUP_LAST_NUCL,
}


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
        bs_randstate: Optional[int] = None,
        ddist_statistic: str = "mmd",
        ddist_bs_niter: int = 1,
        ddist_n_max: Optional[int] = None,
        ddist_k: int = 1000,
        ddist_k_min: Optional[int] = None,
        ddist_k_th: float = 50.0,
        ddist_kernel_size: float = 1.0,
        fdr: float = 0.1,
        comparison_types: Sequence[str] = COMP_TYPES,
        ss_group_any: bool = False,
        ignore_omega: bool = False,
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
        :param codon_grouping_type: When analyzing tuples with length > 1,
            allows grouping certain codon together at a position in the tuple.
            Can be either None (default), 'last_nucleotide' or 'any'.
            Cannot be used with tuple_len==1.
        :param codon_grouping_position: The (zero-indexed) position in the tuple in
            which to group. Default is zero, which means first tuple element.
            No effect if codon_grouping_type==None.
        :param min_group_size: Minimal number of angle-pairs from different
            structures belonging to the same Uniprot ID, location and codon in order to
            consider the group of angles for analysis.
        :param strict_codons: Enforce only one known codon per residue
            (reject residues where DNA matching was ambiguous).
        :param kde_nbins: Number of angle binds for KDE estimation.
        :param kde_width: KDE concentration parameter for von Mises distribution (will
            use same for phi and psi). Used for visualization and for kernel of
            the 'kde_v' statistic.
        :param bs_randstate: Random state for bootstrap.
        :param ddist_statistic: Statistical test to use for quantifying significance
            of  distances between distributions (ddists).
            Can be either 'kde_v' (KDE with von Mises kernel), 'kde_g' (KDE with
            Gaussian kernel), 'mmd' (MMD with Gaussian kernel) or 'tw' (Welch t-test).
        :param ddist_bs_niter: Number of bootstrap iterations when resampling data
            for permutation tests.
        :param ddist_n_max: Maximal sample-size to use when calculating
            p-value of distances with a statistical test. If there are larger samples,
            bootstrap sampling with the given maximal sample size will be performed.
            If None or zero, sample size wont be limited.
        :param ddist_k: Number of permutations to use when calculating
            p-value of distances with a statistical test.
        :param ddist_k_min: Minimal number of permutations to run. Setting this to a
            truthy value enables early termination: when the number of permutations k
            exceeds this number and pvalue >= ddist_k_th * 1/(k+1), no more
            permutations will be performed.
        :param ddist_k_th: Early termination threshold for permutation test. Can be
            thought of as a factor of the smallest pvalue 1/(k+1). I.e. if k_th=50,
            then if after k_min permutations the pvalue is 50 times larger than it's
            smallest possible value - terminate.
        :param ddist_kernel_size: Size of kernel used in 'kde_g' and 'mmd' type
            permutation tests. Should be in degrees.
        :param fdr: False discovery rate for multiple hypothesis testing using
            Benjamini-Hochberg method.
        :param comparison_types: One or more types of entities to compare pointwise.
            Values can be "aa" for comparing amino-acid pairs, "cc" for comparins codon
            pairs, or "aac" for comparing pairs of (a, c) where c is a codon coding for
            the amino acid a. None or empy means all comparison types will be used.
        :param ss_group_any: Whether to add an ANY group to the analysis which contains
            all SS types, even when conditioning by SS type.
        :param ignore_omega: Whether to ignore the omega angle or process it.
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

        if codon_grouping_type and tuple_len == 1:
            raise ValueError(f"codon_grouping_type, can't be used with {tuple_len=}")

        if codon_grouping_position >= tuple_len:
            raise ValueError(
                f"invalid {codon_grouping_position=}, must be < {tuple_len=}"
            )

        if ddist_bs_niter < 1:
            raise ValueError(f"invalid {ddist_bs_niter=}, must be >= 1")

        if ddist_statistic not in TEST_STATISTICS:
            raise ValueError(
                f"ddist_statistic must be one of {tuple(TEST_STATISTICS.keys())}"
            )

        if not 0.0 < fdr < 1.0:
            raise ValueError("FDR should be between 0 and 1, exclusive")

        comparison_types = [c for c in comparison_types if c] or COMP_TYPES
        if any(ct not in COMP_TYPES for ct in comparison_types):
            raise ValueError(
                f"One or more invalid {comparison_types}, must be one of {COMP_TYPES}"
            )

        self.condition_on_ss = condition_on_ss
        self.consolidate_ss = consolidate_ss
        self.tuple_len = tuple_len
        self.codon_grouping_type = codon_grouping_type
        self.codon_grouping_position = codon_grouping_position
        self.min_group_size = min_group_size
        self.strict_codons = strict_codons
        self.condition_on_prev = None

        self.kde_args = dict(
            n_bins=kde_nbins, k1=kde_width, k2=kde_width, k3=0, dtype=np.float64
        )

        self.bs_randstate = bs_randstate
        self.ddist_bs_niter = ddist_bs_niter
        self.ddist_n_max = int(ddist_n_max) if ddist_n_max else 0
        self.ddist_k = ddist_k
        self.ddist_k_min = int(ddist_k_min) if ddist_k_min else 0
        self.ddist_k_th = ddist_k_th
        self.ddist_kernel_size = ddist_kernel_size
        self.fdr = fdr
        self.comparison_types = comparison_types
        self.ss_group_any = ss_group_any
        self.ignore_omega = ignore_omega

        if condition_on_ss:
            consolidated_ss_types = [ss for ss in consolidate_ss.values() if ss]
            if self.ss_group_any:
                consolidated_ss_types.append(SS_TYPE_ANY)
            self.ss_group_names = tuple(sorted(set(consolidated_ss_types)))
        else:
            self.ss_group_names = (SS_TYPE_ANY,)

        # Setup parameters for statistical tests
        if ddist_statistic == "kde_v":
            self.ddist_statistic_fn = partial(
                kde2d_test,
                n_bins=self.kde_args["n_bins"],
                grid_low=-np.pi,
                grid_high=np.pi,
                dtype=self.kde_args["dtype"],
                kernel_fn=partial(
                    bvm_kernel,
                    k1=self.kde_args["k1"],
                    k2=self.kde_args["k2"],
                    k3=self.kde_args["k3"],
                ),
            )
        elif ddist_statistic == "kde_g":
            self.ddist_statistic_fn = partial(
                kde2d_test,
                n_bins=self.kde_args["n_bins"],
                grid_low=-np.pi,
                grid_high=np.pi,
                dtype=self.kde_args["dtype"],
                kernel_fn=partial(
                    torus_gaussian_kernel_2d,
                    sigma=np.deg2rad(self.ddist_kernel_size),
                ),
            )

        elif ddist_statistic == "mmd":
            self.ddist_statistic_fn = partial(
                mmd_test,
                similarity_fn=flat_torus_distance_sq,
                kernel_fn=partial(
                    gaussian_kernel, sigma=np.deg2rad(self.ddist_kernel_size)
                ),
            )
        elif ddist_statistic == "tw":
            self.ddist_statistic_fn = partial(
                tw_test,
                similarity_fn=flat_torus_distance_sq,
            )
        else:
            raise ValueError(f"Unexpected {ddist_statistic=}")
        self.ddist_statistic_fn_name = ddist_statistic

        # Initialize codon tuple names and corresponding indices
        tuples = aac_tuples(k=self.tuple_len)
        tuples = sorted(set(map(self._map_codon_tuple, tuples)))
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

    def _map_codon_tuple(self, aacs: Sequence[str]) -> Sequence[str]:
        """
        Maps codons in a tuple so that they can be grouped by the grouping
        options of this analysis.

        :param aacs: A tuple (aac1, aac2).
        :return: A tuple (aac1, aac2) after conversion.
        """
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
        return tuple(aacs)

    def _preprocess_dataset(self, pool: mp.pool.Pool) -> dict:
        """
        Converts the input raw data to an intermediate data frame which we use for
        analysis.
        """

        angle_cols = ANGLE_COLS if not self.ignore_omega else (PHI_COL, PSI_COL)

        input_cols = (
            PDB_ID_COL,
            UNP_ID_COL,
            UNP_IDX_COL,
            CODON_COL,
            CODON_SCORE_COL,
            SECONDARY_COL,
            *angle_cols,
        )

        # Specifying this dtype allows an integer column with missing values
        dtype = {UNP_IDX_COL: "Int64"}
        dtype = {**dtype, **{ac: "float32" for ac in angle_cols}}

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

        # Drop rows without a codon, and convert codons to AA-CODON
        idx_no_codon = df_pointwise[CODON_COL].isin([UNKNOWN_CODON, MISSING_CODON])
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
                    self._preprocess_group,
                    args=(condition_group_id, df_group),
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
            if np.any(df_subgroup[[*PHI_PSI_COLS]].isnull()):
                continue

            # Calculate average angle from the different structures in this sub group
            angles_centroid = _subgroup_centroid(df_subgroup, input_degrees=True)

            # Calculate variance of the angles around the centroid: average of
            # squared distances to the centroid
            angles_variance_rad = np.mean(
                flat_torus_distance_sq(
                    np.array([[angles_centroid.phi, angles_centroid.psi]]),
                    np.deg2rad(df_subgroup[[*PHI_PSI_COLS]].values),
                )
            )
            # Convert to standard deviation in degrees
            angle_std_deg = np.rad2deg(np.sqrt(angles_variance_rad)).item()

            processed_subgroups.append(
                {
                    UNP_ID_COL: unp_id,
                    UNP_IDX_COL: unp_idx,
                    AA_COL: str.split(aa_codon, AAC_SEP)[0],
                    CODON_COL: aa_codon,
                    CONDITION_COL: group_id,
                    SECONDARY_COL: subgroup_ss,
                    PHI_COL: angles_centroid.phi_deg,
                    PSI_COL: angles_centroid.psi_deg,
                    OMEGA_COL: wraparound_mean(
                        np.array(df_subgroup[OMEGA_COL]), deg=True
                    )
                    if not self.ignore_omega
                    else np.nan,
                    GROUP_SIZE_COL: len(df_subgroup),
                    GROUP_STD_COL: angle_std_deg,
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
                    self._create_group_tuples,
                    args=(df_group, self.tuple_len),
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
        self,
        df_group: pd.DataFrame,
        tuple_len: int,
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

        # Construct a dataframe query that only keeps rows where:
        # 1. The next residue is in the same protein (unp id).
        # 2. The next residue has the successive index (i+1).
        # 3. The next residue has the same secondary structure.
        queries = []
        for i, prefix in enumerate(prefixes[1:], start=1):
            queries.append(f"{UNP_ID_COL} == {prefix}{UNP_ID_COL}")
            queries.append(f"{UNP_IDX_COL} + {i} == {prefix}{UNP_IDX_COL}")
            queries.append(f"{SECONDARY_COL} == {prefix}{SECONDARY_COL}")

        query = str.join(" and ", queries)
        if query:
            df_m = df_m.query(query)

        if len(df_m) == 0:
            return None

        # Function to map rows in the merged dataframe to the final rows we'll use.
        def _row_mapper(row: pd.Series):

            aa_codons, sss, group_sizes, group_stds = [], [], [], []
            for i, p in enumerate(prefixes):
                aa_codons.append(row[f"{p}{CODON_COL}"])
                sss.append(row[f"{p}{SECONDARY_COL}"])
                group_sizes.append(row[f"{p}{GROUP_SIZE_COL}"])
                group_stds.append(row[f"{p}{GROUP_STD_COL}"])

            aa_codons = self._map_codon_tuple(aa_codons)
            aas = tuple(aac2aa(aac) for aac in aa_codons)

            codon_tuple = aact_tuple2str(aa_codons)
            aa_tuple = aact_tuple2str(aas)
            ss_tuple = aact_tuple2str(sss)

            # Make sure there's no ambiguity in the SS type of the tuple
            if not all(ss == sss[0] for ss in sss):
                raise RuntimeError(
                    f"Expecting all members of tuple to have the same SS type. "
                    f"Offending row: {row}"
                )
            condition_group = sss[0]

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
                OMEGA_COL: row[OMEGA_COL] if not self.ignore_omega else np.nan,
                GROUP_SIZE_COL: int(min(group_sizes)),
                GROUP_STD_COL: max(group_stds),  # use worst-case
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
            df_processed,
            full_group_name=SS_TYPE_ANY,
            full_first=True,
            by=CONDITION_COL,
            skip_full_group=not self.ss_group_any,
        )
        for group_idx, df_group in df_groups:
            df_subgroups = df_group.groupby(curr_codon_col)

            # Not all codon may exist as subgroups. Default to zero and count each
            # existing subgroup.
            subgroup_sizes = {}
            for aac, df_sub in df_subgroups:
                subgroup_sizes[aac] = len(df_sub)

                aa = str.join(
                    AAC_TUPLE_SEP, [aac2aa(aac) for aac in aac.split(AAC_TUPLE_SEP)]
                )
                subgroup_sizes.setdefault(aa, 0)
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

        The distance between two codon-tuple sub-groups is calculated using some chosen
        statistic as a distance metric between sets of angle-pairs. The distance
        between two angle-pairs is calculated on the torus.
        """
        df_processed: pd.DataFrame = self._load_intermediate("dataset-tuples")
        group_sizes: dict = self._load_intermediate("group-sizes")

        def _non_syn_codons_pair_filter_fn(group: str, aact1: str, aact2: str) -> bool:
            # Returns True if aact1 and aact2 are synonymous (therefore should be
            # analyzed).
            return is_synonymous_tuple(aact_str2tuple(aact1), aact_str2tuple(aact2))

        def _aa_tuples_filter_fn(group: str, aat1: str, aat2: str):
            # Returns True if aat1 and aat2 are either single AAs or AA tuples with
            # the same first AA.
            aa11, *aa1_ = aact_str2tuple(aat1)
            aa21, *aa2_ = aact_str2tuple(aat2)

            # don't filter any singleton tuples
            if len(aa1_) == 0:
                return True

            # Only analyze tuples where the first AA matches, e.g. A_X and A_Y.
            return aa11 == aa21

        def _syn_codon_pair_nmax_fn(group: str, aact1: str, aact2: str) -> int:
            # Returns the maximal sample size to use when comparing two synonymous
            # codons. We're selecting the smallest sample size of all codons from the
            # same AA.
            aat = _aact_to_aat(aact1)
            assert aat == _aact_to_aat(aact2)  # sanity check
            codon_counts = {
                aact: count
                for aact, count in group_sizes[group][SUBGROUP_COL].items()
                if _aact_to_aat(aact) == aat
            }
            # Codon counts also include the entire group, remove it
            codon_counts.pop(aat)
            # Use minimal group size for all codons of this AA
            min_group_size = np.min([*codon_counts.values()]).item()
            # Make sure we have at least two samples (statistical analysis requires it)
            if min_group_size < 2:
                LOGGER.warning(
                    f"Only one sample of smallest codon group for {group=},"
                    f" {aact1=}, {aact2=}."
                )
            return max(min_group_size, 2)

        totals = {}

        comp_types_to_subgroup_pairs = {
            COMP_TYPE_AA: (AA_COL, AA_COL, _aa_tuples_filter_fn, None),
            COMP_TYPE_AAC: (
                AA_COL,
                CODON_COL,
                None,
                _syn_codon_pair_nmax_fn,
            ),
            COMP_TYPE_CC: (
                CODON_COL,
                CODON_COL,
                _non_syn_codons_pair_filter_fn,
                _syn_codon_pair_nmax_fn,
            ),
        }

        for (
            comp_type,
            (subgroup1_col, subgroup2_col, pair_filter_fn, pair_nmax_fn),
        ) in comp_types_to_subgroup_pairs.items():

            # Only run the requested comparison types
            if comp_type not in self.comparison_types:
                LOGGER.info(f"Skipping {comp_type=} in pairwise analysis")
                continue

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

            collected_ddists: Dict[str, np.ndarray] = {
                g: np.full(**default) for g in self.ss_group_names
            }
            collected_pvals: Dict[str, np.ndarray] = {
                g: np.full(**default) for g in self.ss_group_names
            }

            async_results = self._pointwise_dists_dihedral_subgroup_pairs(
                pool,
                df_processed,
                comp_type,
                subgroup1_col,
                subgroup2_col,
                pair_filter_fn,
                pair_nmax_fn,
            )

            for (group, sub1, sub2), result in yield_async_results(async_results):
                i, j = (
                    sub1_names_to_idx[sub1],
                    sub2_names_to_idx[sub2],
                )
                LOGGER.info(
                    f"Collected {comp_type} pairwise-pval {group=}, {sub1=} ({i=}), "
                    f"{sub2=} ({j=})"
                )
                ddist, pval = result
                ddists = collected_ddists[group]
                pvals = collected_pvals[group]
                ddists[i, j] = ddist
                pvals[i, j] = pval

            self._dump_intermediate(f"{comp_type}-dihedral-ddists", collected_ddists)
            self._dump_intermediate(f"{comp_type}-dihedral-pvals", collected_pvals)

            totals[comp_type] = {
                g: np.sum(~np.isnan(pvals)) for g, pvals in collected_pvals.items()
            }

        LOGGER.info(f"Total number of unique tuple-pairwise pvals: {totals}")
        return {"pval_counts": totals}

    def _pointwise_dists_dihedral_subgroup_pairs(
        self,
        pool: mp.pool.Pool,
        df_processed: pd.DataFrame,
        comp_type: str,
        subgroup1_col: str,
        subgroup2_col: str,
        pair_filter_fn: Optional[Callable[[str, str, str], bool]],
        pair_nmax_fn: Optional[Callable[[str, str, str], int]],
    ):
        """
        Helper function that submits pairs of subgroups for pointwise angle-based
        analysis.
        """
        df_groups = groupby_with_full_group(
            df_processed,
            full_group_name=SS_TYPE_ANY,
            full_first=True,
            by=CONDITION_COL,
            skip_full_group=not self.ss_group_any,
        )
        async_results: Dict[Tuple[str, str, str], AsyncResult] = {}

        LOGGER.info(
            f"Calculating dihedral angle differences between pairs of "
            f"{comp_type}-tuples..."
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
                    # order. Allow comparing subgroup to itself as a sanity check.
                    if subgroup2_col == subgroup1_col and j < i:
                        continue

                    # Skip based on custom pair filtering logic
                    if pair_filter_fn is not None and (
                        not pair_filter_fn(group, sub1, sub2)
                    ):
                        continue

                    # Calculate ddist_nmax for pair based on custom logic
                    if pair_nmax_fn is not None:
                        ddist_n_max = pair_nmax_fn(group, sub1, sub2)
                        # Never use more than the maximum if it was set
                        if self.ddist_n_max:
                            ddist_n_max = min(ddist_n_max, self.ddist_n_max)
                    else:
                        ddist_n_max = self.ddist_n_max

                    # Analyze the angles of subgroup1 and subgroup2
                    angles1 = np.deg2rad(df_sub1[[*PHI_PSI_COLS]].values)
                    angles2 = np.deg2rad(df_sub2[[*PHI_PSI_COLS]].values)
                    res = pool.apply_async(
                        _subgroup_permutation_test,
                        kwds=dict(
                            group_idx=group,
                            subgroup1_idx=sub1,
                            subgroup2_idx=sub2,
                            subgroup1_data=angles1,
                            subgroup2_data=angles2,
                            randstate=self.bs_randstate,
                            ddist_statistic_fn=self.ddist_statistic_fn,
                            ddist_statistic_fn_name=self.ddist_statistic_fn_name,
                            ddist_bs_niter=self.ddist_bs_niter,
                            ddist_n_max=ddist_n_max,
                            ddist_k=self.ddist_k,
                            ddist_k_min=self.ddist_k_min,
                            ddist_k_th=self.ddist_k_th,
                        ),
                    )
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
                skip_full_group=not self.ss_group_any,
            )
        )
        ss_counts = {f"n_{ss_type}": len(df_group) for ss_type, df_group in df_groups}

        LOGGER.info(f"Secondary-structure groups:\n{ss_counts})")
        LOGGER.info(f"Calculating dihedral distribution per SS type...")

        args = (
            (group_idx, df_group, PHI_PSI_COLS, self.kde_args)
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
                skip_full_group=not self.ss_group_any,
            )
        )

        comp_types_to_subgroup_pairs = {
            COMP_TYPE_AA: (AA_COL, self._aa_tuple_to_idx),
            COMP_TYPE_CC: (CODON_COL, self._codon_tuple_to_idx),
        }

        for (
            comp_type,
            (subgroup_col, sub_names_to_idx),
        ) in comp_types_to_subgroup_pairs.items():

            # Only run the requested comparison types
            if comp_type not in self.comparison_types:
                LOGGER.info(f"Skipping {comp_type=} in dihedral distributions")
                continue

            LOGGER.info(
                f"Calculating dihedral angle distributions for {comp_type}-tuples..."
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
                        PHI_PSI_COLS,
                        self.kde_args,
                    )

                    async_results[(group, sub)] = pool.apply_async(
                        _dihedral_kde_single_group, args=args
                    )

            collected_kdes: Dict[str, Dict[str, np.ndarray]] = {
                g: {} for g in self.ss_group_names
            }
            for ((group, sub), result) in yield_async_results(async_results):
                i = sub_names_to_idx[sub]
                _, kde = result
                collected_kdes[group][sub] = kde
                LOGGER.info(f"Collected {comp_type} KDEs {group=}, {sub=} ({i=})")

            self._dump_intermediate(f"{comp_type}-dihedral-kdes", collected_kdes)

    def _write_pvals(self, pool: mp.pool.Pool) -> dict:
        df_processed: pd.DataFrame = self._load_intermediate("dataset-tuples")
        df_groups = tuple(
            groupby_with_full_group(
                df_processed,
                full_group_name=SS_TYPE_ANY,
                full_first=True,
                by=CONDITION_COL,
                skip_full_group=not self.ss_group_any,
            )
        )

        out_dir = self.out_dir.joinpath("pvals")
        os.makedirs(out_dir, exist_ok=True)

        comp_types = [
            (
                COMP_TYPE_AA,
                AA_COL,
                AA_COL,
                self._idx_to_aa_tuple,
                self._idx_to_aa_tuple,
            ),
            (
                COMP_TYPE_CC,
                CODON_COL,
                CODON_COL,
                self._idx_to_codon_tuple,
                self._idx_to_codon_tuple,
            ),
            (
                COMP_TYPE_AAC,
                AA_COL,
                CODON_COL,
                self._idx_to_aa_tuple,
                self._idx_to_codon_tuple,
            ),
        ]

        significance_metadata = {}
        df_data = {}
        async_results = {}

        for (comp_type, i_col, j_col, i_tuples, j_tuples) in comp_types:
            if comp_type not in self.comparison_types:
                continue

            # pvals and ddists are dicts from a group name to a dict from a
            # subgroup-pair to a pval/ddist.
            pvals = self._load_intermediate(
                f"{comp_type}-dihedral-pvals", True, raise_if_missing=True
            )
            ddists = self._load_intermediate(
                f"{comp_type}-dihedral-ddists", True, raise_if_missing=True
            )

            significance_metadata[comp_type] = {}
            df_data[comp_type] = []
            group: str
            df_group: pd.DataFrame
            for idx_group, (group, df_group) in enumerate(df_groups):
                group_pvals = pvals[group]
                group_ddists = ddists[group]

                async_results[(comp_type, group)] = pool.apply_async(
                    self._write_pvals_inner,
                    kwds=dict(
                        comp_type=comp_type,
                        i_col=i_col,
                        j_col=j_col,
                        i_tuples=i_tuples,
                        j_tuples=j_tuples,
                        group=group,
                        df_group=df_group,
                        group_pvals=group_pvals,
                        group_ddists=group_ddists,
                    ),
                )

        # Collect results
        for (
            (comp_type, group),
            (group_significance_meta, group_df_data),
        ) in yield_async_results(async_results):
            significance_metadata[comp_type][group] = group_significance_meta
            df_data[comp_type].extend(group_df_data)

        # Write output
        self._dump_intermediate("significance", significance_metadata)
        for comp_type, result_df_data in df_data.items():
            df_pvals = pd.DataFrame(data=result_df_data)
            df_pvals.sort_values(
                by=[CONDITION_COL, PVAL_COL, DDIST_COL],
                ascending=[True, True, False],
                inplace=True,
            )
            csv_path = str(out_dir.joinpath(f"{comp_type}-pvals.csv"))
            df_pvals.to_csv(csv_path, index=False)
            LOGGER.info(f"Wrote {csv_path}")

        return {"significance": significance_metadata}

    def _write_pvals_inner(
        self,
        comp_type: str,
        i_col: str,
        j_col,
        i_tuples: dict,
        j_tuples: dict,
        group: str,
        df_group: pd.DataFrame,
        group_pvals: np.ndarray,
        group_ddists: np.ndarray,
    ):

        # Get all indices of non-null pvals
        idx_valid = np.argwhere(~np.isnan(group_pvals))

        # Calculate significance threshold for pvalues for  multiple-hypothesis
        # testing.
        group_pvals_flat = group_pvals[idx_valid[:, 0], idx_valid[:, 1]]
        if len(group_pvals_flat) > 1:
            significance_thresh = mht_bh(q=self.fdr, pvals=group_pvals_flat)
        else:
            significance_thresh = 0.0

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
                    DDIST_COL: group_ddists[i, j],
                    SIGNIFICANT_COL: group_pvals[i, j] <= significance_thresh,
                }
            )

        LOGGER.info(
            f"Computed significance for {comp_type=} {group=}: {significance_meta}"
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
                    kwds=dict(
                        out_dir=self.out_dir,
                    ),
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
                    kwds=dict(
                        out_dir=self.out_dir,
                    ),
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
                skip_full_group=not self.ss_group_any,
            )
        }
        group_sizes: dict = self._load_intermediate("group-sizes", True)

        def _glob_mapper(aact: str):
            return str.join(AAC_TUPLE_SEP, aact_str2tuple(aact)[:-1])

        for aa_codon in [COMP_TYPE_AA, COMP_TYPE_CC]:

            avg_dkdes: dict = self._load_intermediate(f"{aa_codon}-dihedral-kdes", True)
            if avg_dkdes is None:
                continue

            for group_idx, dkdes in avg_dkdes.items():
                subgroup_sizes = group_sizes[group_idx][SUBGROUP_COL]

                # Create glob patterns to define which ramachandran plots will go
                # into the same figure
                split_subgroups_glob = None
                if self.tuple_len > 1:
                    if aa_codon == COMP_TYPE_CC:
                        glob_source = self._codon_tuple_to_idx.keys()
                    else:  # "aa"
                        glob_source = self._aa_tuple_to_idx.keys()
                    glob_elements = sorted(set(map(_glob_mapper, glob_source)))
                    split_subgroups_glob = [f"{s}*" for s in glob_elements]
                    # For AAs, also include the reverse glob
                    if aa_codon == COMP_TYPE_AA:
                        split_subgroups_glob.extend([f"*{s}" for s in glob_elements])

                # Get the samples (angles) of all subgroups in this group
                subgroup_col = AA_COL if aa_codon == COMP_TYPE_AA else CODON_COL
                df_group: pd.DataFrame = df_groups[group_idx]
                df_group_samples: pd.DataFrame = df_group[[subgroup_col, *PHI_PSI_COLS]]
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

        type_to_pvals, type_to_ddists = {}, {}
        for comp_type in self.comparison_types:
            type_to_pvals[comp_type] = self._load_intermediate(
                f"{comp_type}-dihedral-pvals", True, raise_if_missing=True
            )
            type_to_ddists[comp_type] = self._load_intermediate(
                f"{comp_type}-dihedral-ddists", True, raise_if_missing=True
            )

        async_results.append(
            pool.apply_async(
                _plot_pvals_hist,
                kwds=dict(
                    pvals=type_to_pvals,
                    group_sizes=group_sizes,
                    significance_meta=significance_meta,
                    fdr=self.fdr,
                    out_dir=self.out_dir.joinpath(f"pvals"),
                ),
            )
        )

        # ddists
        async_results.append(
            pool.apply_async(
                _plot_pvals_ddists,
                kwds=dict(
                    type_to_pvals=type_to_pvals,
                    type_to_ddists=type_to_ddists,
                    idx_to_name={
                        COMP_TYPE_AA: (self._idx_to_aa_tuple, self._idx_to_aa_tuple),
                        COMP_TYPE_CC: (
                            self._idx_to_codon_tuple,
                            self._idx_to_codon_tuple,
                        ),
                        COMP_TYPE_AAC: (
                            self._idx_to_aa_tuple,
                            self._idx_to_codon_tuple,
                        ),
                    },
                    group_sizes=group_sizes,
                    significance_meta=significance_meta,
                    out_dir=self.out_dir.joinpath(f"ddists"),
                    normalize=True,
                ),
            )
        )

        # Wait for plotting to complete. Each function returns a path
        fig_paths = self._handle_async_results(async_results, collect=True)


def _aact_to_aat(aact: str) -> str:
    """
    Converts an AAC or AA tuple string to an AA tuple string. For example:
        I -> I
        I_J -> I_J
        I-ATA -> I
        I-ATA_I-ATT -> I_I

    :param aact: The tuple string to convert, containing only AAs or AAs and codons.
    :return: The converted tuple, containing only the AAs.
    """
    return aact_tuple2str(aact2aat(aact_str2tuple(aact)))


def _subgroup_centroid(
    df_subgroup: pd.DataFrame, input_degrees: bool = False
) -> Dihedral:
    """
    Calculates centroid angle from a subgroup dataframe containing phi,psi dihedral
    angles in degrees under the columns PHI_PSI_COLS.
    :param df_subgroup: The dataframe.
    :param input_degrees: Whether the input data in the PHI_PSI_COLS is in degrees.
    :return: A Dihedral angles object containing the result.
    """
    raw_angles = df_subgroup[[*PHI_PSI_COLS]].values
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
    ddist_statistic_fn: Callable,
    ddist_statistic_fn_name: str,
    ddist_bs_niter: int,
    ddist_n_max: Optional[int],
    ddist_k: int,
    ddist_k_min: int,
    ddist_k_th: float,
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
    :param ddist_bs_niter: Number of iterations for bootstrap sampling.
    :param ddist_n_max: Max sample size. If None or zero then no limit.
    :param ddist_k: Number of permutations for computing significance.
    :param ddist_statistic_fn: Permutation test function to use for comparing the
        samples.
    :return: A Tuple (ddist, pval) containing the value of the ddist statistic and the p-value.
    """
    t_start = time.time()

    # We want a different random state in each subgroup, but reproducible
    seed = None
    if randstate is not None:
        seed = (hash(group_idx + subgroup1_idx + subgroup2_idx) + randstate) % (2**31)
        np.random.seed(seed)
    random = np.random.default_rng(seed)

    # We use bootstrapping if requested more than one iteration or if we need to
    # limit the sample sizes
    bootstrap_enabled = (ddist_bs_niter > 1) or bool(ddist_n_max)
    n1, n2 = len(subgroup1_data), len(subgroup2_data)
    if not ddist_n_max:
        ddist_n_max = max(n1, n2)

    # Bootstrapping sample size is based on the minimum size of each group
    bs_nsample = min(min(n1, n2), ddist_n_max)

    # For a sample larger than ddist_n_max, we create a new sample from it by sampling
    # with replacement.
    def _bootstrap_resample(angles: np.ndarray):
        n = len(angles)
        if bootstrap_enabled:
            sample_idxs = random.choice(n, bs_nsample, replace=True)
        else:
            sample_idxs = np.arange(n)
        return angles[sample_idxs]

    # Run bootstrapped tests
    ddists, pvals, ks = [], [], []
    k_total, count_total, pval = 0, 0, 0.0
    k_max = ddist_bs_niter * ddist_k  # maximal number of permutations in all resamples
    bs_idx = 0
    for bs_idx in range(ddist_bs_niter):
        curr_ddist, curr_pval, curr_k = ddist_statistic_fn(
            X=_bootstrap_resample(subgroup1_data),
            Y=_bootstrap_resample(subgroup2_data),
            k=ddist_k,
            # Disable early termination inside?
            k_min=ddist_k_min,  # if not bootstrap_enabled else 0,
            k_th=ddist_k_th,
        )
        ddists.append(curr_ddist)
        pvals.append(curr_pval)
        ks.append(curr_k)

        # Aggregate total number of permutations and counts to compute pval based
        # on all permutations across resampling. This is equivalent to the pval we'd
        # get if we'd run all the samples together in the same test for k_total
        # permutations.
        k_total += curr_k
        count_total += max(int(curr_pval * (curr_k + 1)) - 1, 0)
        pval = (count_total + 1) / (k_total + 1)

        # Early termination criterion: minimal number of permutations reached,
        # and pval is larger than some factor (k_th) times the smallest possible pvalue.
        if (k_total >= ddist_k_min) and (pval >= ddist_k_th * 1 / (k_total + 1)):
            break

    ddist = np.mean(ddists).item()
    ddist_std = np.std(ddists).item()

    t_elapsed = time.time() - t_start
    LOGGER.info(
        f"[{ddist_statistic_fn_name}] "
        f"{group_idx}, "
        f"sub1={subgroup1_idx} (n={n1}), "
        f"sub2={subgroup2_idx} (n={n2}), "
        f"bs_niter={bs_idx+1}/{ddist_bs_niter}, {bs_nsample=}, "
        f"k={k_total}/{k_max}: "
        f"(pval, ddist)=({pval:.3f},{ddist:.3f}±{ddist_std:.2f}), "
        f"elapsed={t_elapsed:.2f}s"
    )
    return ddist, pval


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
    likelihoods: dict,
    codon_or_aa: str,
    out_dir: Path,
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
                        df_group_samples[idx_samples][[*PHI_PSI_COLS]].values
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


def _plot_pvals_ddists(
    type_to_pvals: Dict[str, Dict[str, np.ndarray]],
    type_to_ddists: Dict[str, Dict[str, np.ndarray]],
    idx_to_name: Dict[str, Tuple[Dict[int, str], Dict[int, str]]],
    group_sizes: Dict[str, int],
    significance_meta: Dict[str, dict],
    out_dir: Path,
    normalize: bool = False,
):
    """
    Plots distances between AAs and codons, and highlights significant ones based on pval.
    :param type_to_pvals: Dict mapping from comparison type to a dict from a group name
        to an array of pvalues.
    :param type_to_ddists: Dict mapping from comparison type to a dict from a group name to
        an array of ddists.
    :param idx_to_name: Maps from comparison type to two dicts, one for mapping from
        index i to a name and the second for mapping from index j to a name.
    :param group_sizes: Dict from group_name to size ('total') and subgroup sizes ('subgroup').
    :param significance_meta: A dict mapping from comparison type to group_name to a dict
        containing the significance information.
    :param out_dir: Where to write the figures to.
    :param normalize: Whether to normalize each distance matrix so that the main
        diagonal is 1, i.e. Dij -> Dij/sqrt(Dii * Djj).
    :return: Path of output figure.
    """
    out_files = []

    comp_types = tuple(type_to_pvals.keys())
    assert comp_types == tuple(type_to_ddists.keys())

    for comp_type in comp_types:
        group_to_pvals = type_to_pvals[comp_type]
        group_to_ddists = type_to_ddists[comp_type]
        i_to_name, j_to_name = idx_to_name[comp_type]
        tuple_len = len(aact_str2tuple(i_to_name[0]))

        group_names = tuple(group_to_pvals.keys())
        assert group_names == tuple(group_to_ddists.keys())

        for group_name in group_names:
            pvals: np.ndarray = group_to_pvals[group_name]  # 2d array
            ddists: np.ndarray = group_to_ddists[group_name]
            ij_valid = np.argwhere(~np.isnan(pvals))

            pval_thresh = significance_meta[comp_type][group_name]["pval_thresh"]

            plot_datas = []
            plot_titles = []
            plot_row_labels = []
            plot_col_labels = []
            plot_data_annotations = []

            single_aa_to_ijs = {}
            aat_to_ijs = {}
            for i, j in ij_valid:
                i_name, j_name = i_to_name[i], j_to_name[j]

                aat_i, aat_j = _aact_to_aat(i_name), _aact_to_aat(j_name)
                aa_i, aa_j = aact_str2tuple(aat_i)[0], aact_str2tuple(aat_j)[0]

                # single_aa_to_ijs: A mapping from each single AA name to the AA-tuple
                # indices in the distance matrix
                if aa_i == aa_j:
                    single_aa_to_ijs.setdefault(aa_i, [])
                    single_aa_to_ijs[aa_i].append((i, j))

                # aat_to_ijs: A mapping from each AA-tuple name to the synonymous
                # indices in the distance matrix
                if aat_i == aat_j:
                    aat_name = aat_i
                    aat_to_ijs.setdefault(aat_name, [])
                    aat_to_ijs[aat_name].append((i, j))

            if comp_type == COMP_TYPE_AA and tuple_len == 1:
                plot_datas.append(ddists)
                plot_titles.append("")
                plot_row_labels.append(tuple(i_to_name.values()))
                plot_col_labels.append(tuple(j_to_name.values()))
                plot_data_annotations.append(pvals <= pval_thresh)

            else:  # CC or AC
                # For AA, t>1: Plot a separate distance matrix for every single AA
                # For CC, AC: Plot a separate distance matrix for every AA-tuple
                # Choose one of the above mappings accordingly
                selected_aa_to_ijs = (
                    single_aa_to_ijs if comp_type == COMP_TYPE_AA else aat_to_ijs
                )

                # Populate plot data based on indices from mapping
                for aat_name, ijs in selected_aa_to_ijs.items():
                    # Extract square ddist sub-matrix
                    ijs = np.array(ijs)  # (n,2)
                    i_slice = slice(np.min(ijs[:, 0]), np.max(ijs[:, 0]) + 1)
                    j_slice = slice(np.min(ijs[:, 1]), np.max(ijs[:, 1]) + 1)
                    plot_datas.append(ddists[i_slice, j_slice])
                    plot_titles.append(aat_name)
                    plot_row_labels.append(
                        [i_to_name[i] for i in range(i_slice.start, i_slice.stop)]
                    )
                    plot_col_labels.append(
                        [j_to_name[j] for j in range(j_slice.start, j_slice.stop)]
                    )
                    plot_data_annotations.append(pvals[i_slice, j_slice] <= pval_thresh)

            if normalize:
                # Normalizing: Dij -> Dij/sqrt(Dii * Djj)
                # TODO: how to normalize the AAC case?
                for i, d in enumerate(plot_datas):
                    normalization = np.sqrt(
                        np.diag(d).reshape(-1, 1) * np.diag(d).reshape(1, -1)
                    )
                    plot_datas[i] = d / (normalization + 1e-12)

            fig, _ = pp5.plot.multi_heatmap(
                datas=plot_datas,
                titles=plot_titles,
                row_labels=plot_row_labels,
                col_labels=plot_col_labels,
                fig_rows=int(np.ceil(np.sqrt(len(plot_datas)))),
                fig_size=6,
                data_annotation_locations=plot_data_annotations,
                data_annotation_fn=lambda *a: dict(
                    s="*",
                    ha="center",
                    va="center",
                    color="darkred",
                    fontdict={"size": "medium"},
                ),
            )
            fig_filename = out_dir.joinpath(f"{comp_type}-{group_name}.pdf")
            out_files.append(pp5.plot.savefig(fig, fig_filename, close=True))


def _plot_pvals_hist(
    pvals: Dict[str, np.ndarray],
    group_sizes: Dict[str, int],
    significance_meta: Dict[str, dict],
    fdr: float,
    out_dir: Path,
    n_bins: int = 25,
):
    """
    Plots pvalue histogram.
    :param pvals: Dict mapping from comparison type to a dict from a group name
        to an array of pvalues.
    :param group_sizes: Dict from group_name to size ('total') and subgroup sizes ('subgroup').
    :param significance_meta: A dict mapping from comparison type to group_name to a dict
        containing the significance information.
    :return: Path of output figure.
    """

    n_groups = len(group_sizes.keys())

    out_files = []

    with mpl.style.context(PP5_MPL_STYLE):

        for comp_type, group_to_pvals in pvals.items():

            hist_fig_filename = out_dir.joinpath(f"pvals_hist-{comp_type}.pdf")
            pvals_fig_filename = out_dir.joinpath(f"pvals-{comp_type}.pdf")

            fig_cols = 2 if n_groups > 1 else 1
            fig_rows = n_groups // 2 + n_groups % 2

            # Histogram fig and axes
            fig_hist, ax_hist = plt.subplots(
                fig_rows,
                fig_cols,
                figsize=(5 * fig_cols, 5 * fig_rows),
                squeeze=False,
            )
            fig_hist: Figure
            axes_hist: Sequence[Axes] = ax_hist.reshape(-1)

            # pvals fig and axes
            fig_pvals, ax_pvals = plt.subplots(
                fig_rows,
                fig_cols,
                figsize=(5 * fig_cols, 5 * fig_rows),
                squeeze=False,
            )
            fig_pvals: Figure
            axes_pvals: Sequence[Axes] = ax_pvals.reshape(-1)

            for i, (group_name, subgroup_sizes) in enumerate(group_sizes.items()):
                pvals_2d = group_to_pvals[group_name]
                pvals_flat = pvals_2d[~np.isnan(pvals_2d)]

                meta = significance_meta[comp_type][group_name]
                pval_thresh = meta["pval_thresh"]
                num_rejections = meta["num_rejections"]
                num_hypotheses = meta["num_hypotheses"]
                assert len(pvals_flat) == num_hypotheses  # sanity

                # Histograms
                ax_hist = axes_hist[i]
                ax_hist.hist(
                    pvals_flat,
                    bins=n_bins,
                    density=True,
                    log=True,
                    alpha=0.5,
                    label=(f"t={pval_thresh:.4f}, ({num_rejections}/{num_hypotheses})"),
                )
                ax_hist.set_title(f"{group_name} ({subgroup_sizes['total']})")
                ax_hist.set_ylabel("log-density")
                ax_hist.set_xlabel("pval")
                ax_hist.legend()

                # Pvals
                ax_pvals = axes_pvals[i]
                x_axis = np.arange(num_hypotheses)
                pvals_sorted = np.sort(pvals_flat)
                bhq_thresh_line = (x_axis + 1) * (fdr / num_hypotheses)
                idx_rejections = pvals_sorted <= pval_thresh

                ax_pvals.plot(
                    x_axis,
                    bhq_thresh_line,
                    label=f"BH(q={fdr})",
                )
                ax_pvals.plot(
                    x_axis[~idx_rejections],
                    pvals_sorted[~idx_rejections],
                    label=f"non-rejections ({np.sum(~idx_rejections)}/{num_hypotheses})",
                    marker="x",
                    linestyle="",
                )
                ax_pvals.plot(
                    x_axis[idx_rejections],
                    pvals_sorted[idx_rejections],
                    label=f"rejections ({np.sum(idx_rejections)}/{num_hypotheses})",
                    marker="*",
                    linestyle="",
                )
                ax_pvals.hlines(
                    pval_thresh,
                    x_axis[0],
                    x_axis[-1],
                    colors="black",
                    linestyles="--",
                    label=f"t={pval_thresh:.4f}",
                )

                ax_pvals.set_yscale("log")
                ax_pvals.set_title(f"{group_name} ({subgroup_sizes['total']})")
                ax_pvals.set_ylabel("pval")
                ax_pvals.set_xlabel("hypothesis number")
                ax_pvals.grid(True)
                ax_pvals.legend()

            fig_hist.tight_layout()
            fig_pvals.tight_layout()

            # Fix for empty axes in plots
            while i + 1 < len(axes_hist):
                i += 1
                axes_hist[i].set_axis_off()
                axes_pvals[i].set_axis_off()

            out_files.append(pp5.plot.savefig(fig_hist, hist_fig_filename, close=True))
            out_files.append(
                pp5.plot.savefig(fig_pvals, pvals_fig_filename, close=True)
            )

        return out_files


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
    skip_full_group: bool = True,
    **groupby_kwargs,
) -> Iterator[Tuple[Union[str, Tuple[str, ...]], pd.DataFrame]]:
    """
    Performs a groupby on a pandas dataframe, yields all groups and then yields the
    "full" group, i.e. the entire dataframe.
    :param df: The data frame.
    :param full_group_name: Name of the full group. Should be a column name or a
        tuple of column names, depending on the "by" argument of groupby.
    :param full_first: Whether to add the full-group first (True) or last (False).
    :param skip_full_group: If True, the full group will not be returned. Use this
        flag to conditionally turn this method into a regular group_by.
    :param groupby_kwargs: kwargs to be passed as-is to groupby.
    :return: Yields tuples of (group_name, group_df). The first or last yielded tuple
        will contain the full dataframe. If there's another group with the same name
        as full_group_name, it will not be yielded.
    """
    yielded_group_names = set()

    if not skip_full_group and full_first:
        yielded_group_names.add(full_group_name)
        yield (full_group_name, df)

    for group_name, df_group in df.groupby(**groupby_kwargs):
        if group_name not in yielded_group_names:
            yielded_group_names.add(group_name)
            yield group_name, df_group

    if (
        not skip_full_group
        and not full_first
        and (full_group_name not in yielded_group_names)
    ):
        yield (full_group_name, df)
