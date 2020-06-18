import os
import pickle
import re
import itertools as it
import logging
import multiprocessing as mp
import shutil
import time
from abc import ABC
from multiprocessing.pool import AsyncResult
from pathlib import Path
from typing import Union, Dict, Callable, Optional, List, Tuple

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.style
import matplotlib.pyplot
from Bio.Data.CodonTable import standard_dna_table as dna_table

from pp5.collect import ParallelDataCollector
from pp5.dihedral import DihedralKDE
import pp5.plot
from pp5.parallel import yield_async_results
from pp5.plot import PP5_MPL_STYLE
from pp5.utils import sort_dict

LOGGER = logging.getLogger(__name__)

SS_TYPE_ANY = 'ANY'
SS_TYPE_HELIX = 'HELIX'
SS_TYPE_SHEET = 'SHEET'
SS_TYPE_TURN = 'TURN'
SS_TYPE_OTHER = 'OTHER'
DSSP_TO_SS_TYPE = {
    # The DSSP codes for secondary structure used here are:
    # H        Alpha helix (4-12)
    # B        Isolated beta-bridge residue
    # E        Strand
    # G        3-10 helix
    # I        Pi helix
    # T        Turn
    # S        Bend
    # -        None
    'E': SS_TYPE_SHEET,
    'H': SS_TYPE_HELIX,
    'G': SS_TYPE_OTHER,  # maybe also helix?
    'I': SS_TYPE_OTHER,  # maybe also helix?
    'T': SS_TYPE_TURN,
    'S': SS_TYPE_OTHER,  # maybe also turn?
    'B': SS_TYPE_OTHER,  # maybe also sheet?
    '-': None,
    '': None,
}


def codon2aac(codon: str):
    """
    Converts codon to AA-CODON, which we will use as codon identifiers.
    :param codon: a codon string.
    :return: a string formatted AA-CODON where AA is the
    corresponding amino acid.
    """
    aa = dna_table.forward_table[codon]
    return f'{aa}-{codon}'.upper()


CODONS = sorted(codon2aac(c) for c in dna_table.forward_table)
ACIDS = sorted(set([aac[0] for aac in CODONS]))
N_CODONS = len(CODONS)
CODON_TYPE_ANY = 'ANY'


class ParallelAnalyzer(ParallelDataCollector, ABC):
    """
    Base class for analyzers.
    """

    def __init__(
            self,
            analysis_name,
            dataset_dir: Union[str, Path],
            out_tag: str = None,
            clear_intermediate=False,
    ):
        """

        :param analysis_name:
        :param dataset_dir: Path to directory with the dataset files.
        :param out_tag: Tag for output files.
        :param clear_intermediate: Whether to clear intermediate folder.
        """
        self.analysis_name = analysis_name
        self.dataset_dir = Path(dataset_dir)
        self.out_tag = out_tag

        if not self.dataset_dir.is_dir():
            raise ValueError(f'Dataset dir {self.dataset_dir} not found')

        tag = f'-{self.out_tag}' if self.out_tag else ''
        out_dir = self.dataset_dir.joinpath('results')
        super().__init__(id=f'{self.analysis_name}{tag}', out_dir=out_dir,
                         tag=out_tag, create_zip=False, )

        # Create clean directory for intermediate results between steps
        self.intermediate_dir = self.out_dir.joinpath('_intermediate_')
        if clear_intermediate and self.intermediate_dir.exists():
            shutil.rmtree(str(self.intermediate_dir))
        os.makedirs(str(self.intermediate_dir), exist_ok=True)

        # Create dict for storing paths of intermediate results
        self._intermediate_files: Dict[str, Path] = {}

    def _dump_intermediate(self, name: str, obj):
        # Update dict of intermediate files
        path = self.intermediate_dir.joinpath(f'{name}.pkl')
        self._intermediate_files[name] = path

        with open(str(path), 'wb') as f:
            pickle.dump(obj, f, protocol=4)

        LOGGER.info(f'Wrote intermediate file {path}')
        return path

    def _load_intermediate(self, name, allow_old=True, raise_if_missing=True):
        path = self.intermediate_dir.joinpath(f'{name}.pkl')

        if name not in self._intermediate_files:
            # Intermediate files might exist from a previous run we wish to
            # resume
            if allow_old:
                LOGGER.warning(f'Loading old intermediate file {path}')
            else:
                return None

        if not path.is_file():
            if raise_if_missing:
                raise ValueError(f"Can't find intermediate file {path}")
            else:
                return None

        self._intermediate_files[name] = path
        with open(str(path), 'rb') as f:
            obj = pickle.load(f)

        LOGGER.info(f'Loaded intermediate file {path}')
        return obj


class PointwiseCodonDistanceAnalyzer(ParallelAnalyzer):
    def __init__(
            self,
            dataset_dir: Union[str, Path],
            pointwise_filename: str = 'data-pointwise.csv',
            condition_on_prev='codon', condition_on_ss=True,
            consolidate_ss=DSSP_TO_SS_TYPE.copy(), strict_ss=True,
            kde_nbins=128, kde_k1=30., kde_k2=30., kde_k3=0.,
            bs_niter=1, bs_randstate=None, bs_limit_n=False,
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
        :param pointwise_filename: Filename of the pointwise daataset.
        :param consolidate_ss: Dict mapping from DSSP secondary structure to
        the consolidated SS types used in this analysis.
        :param condition_on_prev: What to condition on from previous residue of
        each sample. Options are 'codon', 'aa', or None/''.
        :param condition_on_ss: Whether to condition on secondary structure
        (of two consecutive residues, after consolidation).
        :param strict_ss: Enforce no ambiguous codons in any residue.
        :param kde_nbins: Number of angle binds for KDE estimation.
        :param kde_k1: KDE concentration parameter for phi.
        :param kde_k2: KDE concentration parameter for psi.
        :param kde_k3: KDE joint concentration parameter.
        :param bs_niter: Number of bootstrap iterations.
        :param bs_randstate: Random state for bootstrap.
        :param bs_limit_n: Whether to limit number of samples in each
        bootstrap iteration for each subgroup, to the number of samples in the
        smallest subgroup.
        :param n_parallel_kdes: Number of parallel bootstrapped KDE
        calculations to run simultaneously.
        By default it will be equal to the number of available CPU cores.
        Setting this to a high number together with a high bs_niter will cause
        excessive memory usage.
        :param out_tag: Tag for output.
        """
        super().__init__('pointwise_cdist', dataset_dir, out_tag,
                         clear_intermediate=False)

        self.input_file = self.dataset_dir.joinpath(pointwise_filename)
        if not self.input_file.is_file():
            raise ValueError(f'Dataset file {self.input_file} not found')

        condition_on_prev = '' if condition_on_prev is None \
            else condition_on_prev.lower()
        if condition_on_prev not in {'codon', 'aa', ''}:
            raise ValueError(f'invalid condition_on_prev: {condition_on_prev}')

        self.condition_on_prev = condition_on_prev
        self.condition_on_ss = condition_on_ss
        self.consolidate_ss = consolidate_ss
        self.strict_ss = strict_ss

        self.angle_pairs = [(f'phi+0', f'psi+0'), (f'phi+0', f'psi-1')]
        self.angle_cols = sorted(set(it.chain(*self.angle_pairs)))
        self.codon_cols = [f'codon-1', f'codon+0']
        self.secondary_cols = [f'secondary-1', f'secondary+0']
        self.secondary_col = 'secondary'
        self.condition_col = 'condition_group'

        self.kde_args = dict(n_bins=kde_nbins, k1=kde_k1, k2=kde_k2, k3=kde_k3)
        self.kde_dist_metric = 'l2'

        self.bs_niter = bs_niter
        self.bs_randstate = bs_randstate
        self.bs_limit_n = bs_limit_n
        self.n_parallel_kdes = n_parallel_kdes

    def _collection_functions(self) \
            -> Dict[str, Callable[[mp.pool.Pool], Optional[Dict]]]:
        return {
            'preprocess-dataset': self._preprocess_dataset,
            'dataset-stats': self._dataset_stats,
            'dihedral-kde-full': self._dihedral_kde_full,
            'codon-dists': self._codons_dists,
            'codon-dists-expected': self._codon_dists_expected,
            'plot-results': self._plot_results,
        }

    def _preprocess_dataset(self, pool: mp.pool.Pool) -> dict:
        # Specifies which columns to read from the CSV
        def col_filter(col_name: str):
            # Keep only columns from prev and current
            if col_name.endswith('-1') or col_name.endswith('+0'):
                return True
            return False

        df_pointwise_reader = pd.read_csv(
            str(self.input_file), usecols=col_filter, chunksize=10_000,
        )

        # Parallelize loading and preprocessing
        sub_dfs = pool.map(self._preprocess_subframe, df_pointwise_reader)

        # Dump processed dataset
        df_preproc = pd.concat(sub_dfs, axis=0, ignore_index=True)
        LOGGER.info(f'Loaded {self.input_file}: {len(df_preproc)} rows\n'
                    f'{df_preproc}\n{df_preproc.dtypes}')

        self._dump_intermediate('dataset', df_preproc)
        return {
            'n_TOTAL': len(df_preproc),
        }

    def _preprocess_subframe(self, df_sub: pd.DataFrame):
        # Logic for consolidating secondary structure between a pair of curr
        # and prev residues
        def ss_consolidator(row: pd.Series):
            ss_m1 = row[self.secondary_cols[0]]  # e.g. 'H' or e.g. 'H/G'
            ss_p0 = row[self.secondary_cols[1]]

            # In strict mode we require that all group members had the same SS,
            # i.e. we don't allow groups with more than one type (ambiguous
            # codons).
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

            return None

        # Based on the configuratrion, we create a column that represents
        # the group a sample belongs to when conditioning
        def assign_condition_group(row: pd.Series):
            prev_aac = row[self.codon_cols[0]]  # AA-CODON

            cond_groups = []

            if self.condition_on_prev == 'codon':
                # Keep AA-CODON
                cond_groups.append(prev_aac)
            elif self.condition_on_prev == 'aa':
                # Keep only AA
                cond_groups.append(prev_aac[0])
            else:
                cond_groups.append(CODON_TYPE_ANY)

            if self.condition_on_ss:
                cond_groups.append(row[self.secondary_col])
            else:
                cond_groups.append(SS_TYPE_ANY)

            return str.join('_', cond_groups)

        ss_consolidated = df_sub.apply(ss_consolidator, axis=1)

        # Keep only angle and codon columns from the full dataset
        df_pointwise = pd.DataFrame(
            data=df_sub[self.angle_cols + self.codon_cols],
        )

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
        df_processed: pd.DataFrame = self._load_intermediate('dataset')

        df_ss_groups = df_processed.groupby(self.secondary_col)
        for ss_type, df_ss_group in df_ss_groups:
            n_ss = len(df_ss_group)
            df_codon_groups = df_ss_group.groupby(prev_codon)
            df_codon_group_names = df_codon_groups.groups.keys()

            codon_likelihood = np.array([
                0.
                if codon not in df_codon_group_names
                else len(df_codon_groups.get_group(codon)) / n_ss
                for codon in CODONS  # ensure consistent order
            ], dtype=np.float32)
            assert np.isclose(np.sum(codon_likelihood), 1.)

            # Save a map from codon name to it's likelihood
            codon_likelihoods[ss_type] = {
                c: codon_likelihood[i] for i, c in enumerate(CODONS)
            }

        # Calculate AA likelihoods based on the codon likelihoods
        aa_likelihoods = {}
        for ss_type, codons in codon_likelihoods.items():
            aa_likelihoods[ss_type] = {aac[0]: 0. for aac in codons.keys()}
            for aac, likelihood in codons.items():
                aa = aac[0]
                aa_likelihoods[ss_type][aa] += likelihood

            assert np.isclose(sum(aa_likelihoods[ss_type].values()), 1.)

        # Calculate SS likelihoods (ss->probability)
        ss_likelihoods = {}
        n_total = len(df_processed)
        for ss_type, df_ss_group in df_ss_groups:
            n_ss = len(df_ss_group)
            ss_likelihoods[ss_type] = n_ss / n_total
        assert np.isclose(sum(ss_likelihoods.values()), 1.)

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

            assert np.isclose(sum(l[SS_TYPE_ANY].values()), 1.)

        self._dump_intermediate('codon-likelihoods', codon_likelihoods)
        self._dump_intermediate('aa-likelihoods', aa_likelihoods)
        self._dump_intermediate('ss-likelihoods', ss_likelihoods)

        # Calculate group and subgroup sizes
        group_sizes = {}
        _, curr_codon_col = self.codon_cols
        df_groups = df_processed.groupby(by=self.condition_col)
        for group_idx, df_group in df_groups:
            df_subgroups = df_group.groupby(curr_codon_col)
            group_sizes[group_idx] = {
                'total': len(df_group),
                'subgroups': sort_dict({
                    idx_sub: len(df_sub) for idx_sub, df_sub in df_subgroups
                })
            }
        group_sizes = sort_dict(group_sizes, selector=lambda g: g['total'])
        self._dump_intermediate('group-sizes', group_sizes)

        return {'group_sizes': group_sizes}

    def _dihedral_kde_full(self, pool: mp.pool.Pool) -> dict:
        df_processed: pd.DataFrame = self._load_intermediate('dataset')
        df_groups = df_processed.groupby(by=self.secondary_col)
        df_groups_count: pd.DataFrame = df_groups.count()
        ss_counts = {
            f'n_{ss_type}': count
            for ss_type, count in df_groups_count.max(axis=1).to_dict().items()
        }

        LOGGER.info(f'Secondary-structure groups:\n{ss_counts})')
        LOGGER.info(f'Calculating dihedral distribution per SS type...')

        args = ((group_idx, df_group, self.angle_pairs, self.kde_args)
                for group_idx, df_group in df_groups)

        map_result = pool.starmap(self._dihedral_kde_single_group, args)

        # maps from group (SS) to a list, containing a dihedral KDE
        # matrix for each angle-pair.
        map_result = {group_idx: dkdes for group_idx, dkdes in map_result}
        self._dump_intermediate('full-dkde', map_result)

        return {**ss_counts}

    def _codons_dists(self, pool: mp.pool.Pool) -> dict:
        prev_codon_col, curr_codon_col = self.codon_cols

        # We currently only support one type of metric
        dist_metrics = {
            'l2': self._kde_dist_metric_l2
        }
        dist_metric = dist_metrics[self.kde_dist_metric.lower()]

        # Calculate chunk-size for parallel mapping.
        # (Num groups in parallel) * (Num subgroups) / (num processors)
        n_procs = pp5.get_config('MAX_PROCESSES')
        chunksize = self.n_parallel_kdes * N_CODONS / n_procs
        chunksize = max(int(chunksize), 1)

        df_processed: pd.DataFrame = self._load_intermediate('dataset')
        df_groups = df_processed.groupby(by=self.condition_col)

        LOGGER.info(f'Calculating subgroup KDEs '
                    f'(n_parallel_kdes={self.n_parallel_kdes}, '
                    f'chunksize={chunksize})...')

        codon_dists, codon_dkdes = {}, {}
        dkde_asyncs: Dict[str, AsyncResult] = {}
        dist_asyncs: Dict[str, AsyncResult] = {}
        for i, (group_idx, df_group) in enumerate(df_groups):
            last_group = i == len(df_groups) - 1

            # In each pre-condition group, group by current codon.
            # These subgroups are where we estimate the dihedral angle KDEs.
            df_subgroups = df_group.groupby(curr_codon_col)

            # Find smallest subgroup
            subgroup_lens = [len(df_s) for _, df_s in df_subgroups]
            min_idx = np.argmin(subgroup_lens)
            min_len = subgroup_lens[min_idx]

            # Calculates number of samples in each bootstrap iteration:
            # We either take all samples in each subgroup, or the number of
            # samples in the smallest subgroup.
            if self.bs_limit_n:
                bs_nsamples = [min_len] * len(subgroup_lens)
            else:
                bs_nsamples = subgroup_lens

            # Run bootstrapped KDE estimation for all subgroups in parallel
            subprocess_args = (
                (
                    group_idx, subgroup_idx, df_subgroup,
                    self.angle_pairs, self.kde_args,
                    self.bs_niter, bs_nsamples[j], self.bs_randstate,
                )
                for j, (subgroup_idx, df_subgroup) in enumerate(df_subgroups)
            )
            dkde_asyncs[group_idx] = pool.starmap_async(
                self._codon_dkdes_single_subgroup, subprocess_args,
                chunksize=chunksize
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
            LOGGER.info(f'[{i}] Waiting to collect KDEs '
                        f'(#async_results={len(dkde_asyncs)})...')
            for result_group_idx, group_dkde_result in dkde_results_iter:
                if group_dkde_result is None:
                    LOGGER.error(f'[{i}] No KDE result in {result_group_idx}')
                    continue

                # Remove async result so we dont see it next time
                LOGGER.info(f'[{i}] Collected KDEs for {result_group_idx}')
                collected_dkde_results[result_group_idx] = group_dkde_result
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
                bs_codon_dkdes = {c: None for c in CODONS}
                for subgroup_idx, subgroup_bs_dkdes in group_dkde_result:
                    bs_codon_dkdes[subgroup_idx] = subgroup_bs_dkdes

                # Run distance matrix calculation in parallel
                dist_asyncs[result_group_idx] = pool.apply_async(
                    self._codon_dists_single_group,
                    args=(result_group_idx, bs_codon_dkdes, self.angle_pairs,
                          dist_metric)
                )
                LOGGER.info(f'[{i}] Submitted cdist task {result_group_idx}')

            # Allow limited number of simultaneous distance matrix calculations
            if not last_group and len(dist_asyncs) < self.n_parallel_kdes:
                continue

            # Wait for one of the distance matrix calculations, or all of
            # them if it's the last group
            dists_results_iter = yield_async_results(dist_asyncs.copy())
            LOGGER.info(f'[{i}] Waiting to collect cdist matrices '
                        f'(#async_results={len(dist_asyncs)})...')
            for result_group_idx, group_dist_result in dists_results_iter:
                if group_dist_result is None:
                    LOGGER.error(f'[{i}] No cdist in {result_group_idx}')
                    continue

                LOGGER.info(f'[{i}] Collected cdist matrix {result_group_idx}')
                group_d2_matrices, group_codon_dkdes = group_dist_result
                codon_dists[result_group_idx] = group_d2_matrices
                codon_dkdes[result_group_idx] = group_codon_dkdes
                del dist_asyncs[result_group_idx]

                # If we're in the last group, collect everything
                if not last_group:
                    break

        # Make sure everything was collected
        assert len(dkde_asyncs) == 0, 'Not all KDEs collected'
        assert len(dist_asyncs) == 0, 'Not all dist matrices collected'
        LOGGER.info(f'Completed distance matrix collection')

        # codon_dists: maps from group (codon, SS) to a list, containing a
        # codon-distance matrix for each angle-pair.
        # The codon distance matrix is complex, where real is mu
        # and imag is sigma
        self._dump_intermediate('codon-dists', codon_dists)

        # codon_dkdes: maps from group to a dict where keys are codons.
        # For each codon we have a list of KDEs, one for each angle pair.
        self._dump_intermediate('codon-dkdes', codon_dkdes)

        return {}

    def _codon_dists_expected(self, pool: mp.pool.Pool) -> dict:
        # Load map of likelihoods, depending on the type of previous
        # conditioning (ss->codon or AA->probability)
        if self.condition_on_prev == 'codon':
            likelihoods: dict = self._load_intermediate('codon-likelihoods')
        elif self.condition_on_prev == 'aa':
            likelihoods: dict = self._load_intermediate('aa-likelihoods')
        else:
            likelihoods = None

        # Load the calculated codon-dists matrices. The loaded dict
        # maps from  (prev_codon/AA, SS) to a list of distance matrices
        codon_dists: dict = self._load_intermediate('codon-dists')

        # This dict will hold the final expected distance matrices (i.e. we
        # calculate the expectation using the likelihood of the prev codon
        # or aa). Note that we actually have one matrix per angle pair.
        codon_dists_exp = {}
        for group_idx, d2_matrices in codon_dists.items():
            assert len(d2_matrices) == len(self.angle_pairs)
            codon_or_aa, ss_type = group_idx.split('_')

            if ss_type not in codon_dists_exp:
                defaults = [np.zeros_like(d2) for d2 in d2_matrices]
                codon_dists_exp[ss_type] = defaults

            for i, d2 in enumerate(d2_matrices):
                p = likelihoods[ss_type][codon_or_aa] if likelihoods else 1.
                # Don't sum nan's because they kill the entire cell
                d2 = np.nan_to_num(d2, copy=False, nan=0.)
                codon_dists_exp[ss_type][i] += p * d2

        # Now we also take the expectation over all SS types to produce one
        # averaged distance matrix, but only if we actually conditioned on it
        if self.condition_on_ss:
            ss_likelihoods: dict = self._load_intermediate('ss-likelihoods')
            defaults = [np.zeros_like(d2) for d2 in d2_matrices]
            codon_dists_exp[SS_TYPE_ANY] = defaults
            for ss_type, d2_matrices in codon_dists_exp.items():
                if ss_type == SS_TYPE_ANY:
                    continue
                p = ss_likelihoods[ss_type]
                for i, d2 in enumerate(d2_matrices):
                    codon_dists_exp[SS_TYPE_ANY][i] += p * d2

        self._dump_intermediate('codon-dists-exp', codon_dists_exp)
        return {}

    def _plot_results(self, pool: mp.pool.Pool):
        LOGGER.info(f'Plotting results...')

        ap_labels = [self._cols2label(phi_col, psi_col)
                     for phi_col, psi_col in self.angle_pairs]

        async_results = []

        # Expected codon dists
        codon_dists_exp = self._load_intermediate('codon-dists-exp', False)
        if codon_dists_exp is not None:
            for ss_type, d2_matrices in codon_dists_exp.items():
                args = (ss_type, d2_matrices)
                async_results.append(pool.apply_async(
                    self._plot_codon_distances, args=args,
                    kwds=dict(out_dir=self.out_dir.joinpath('codon-dists-exp'),
                              angle_pair_labels=ap_labels,
                              annotate_mu=True, plot_std=True)
                ))
            del codon_dists_exp, d2_matrices

        # Codon likelihoods
        codon_likelihoods = self._load_intermediate('codon-likelihoods', False)
        if codon_likelihoods is not None:
            async_results.append(pool.apply_async(
                self._plot_likelihoods, args=(codon_likelihoods, 'codon'),
                kwds=dict(out_dir=self.out_dir, )
            ))
            del codon_likelihoods

        # AA likelihoods
        aa_likelihoods = self._load_intermediate('aa-likelihoods', False)
        if aa_likelihoods is not None:
            async_results.append(pool.apply_async(
                self._plot_likelihoods, args=(aa_likelihoods, 'aa'),
                kwds=dict(out_dir=self.out_dir, )
            ))
            del aa_likelihoods

        # Dihedral KDEs of full dataset
        full_dkde: dict = self._load_intermediate('full-dkde', False)
        if full_dkde is not None:
            async_results.append(pool.apply_async(
                self._plot_full_dkdes,
                args=(full_dkde,),
                kwds=dict(out_dir=self.out_dir, angle_pair_labels=ap_labels)
            ))
            del full_dkde

        # Codon distance matrices
        codon_dists: dict = self._load_intermediate('codon-dists', False)
        if codon_dists is not None:
            for group_idx, d2_matrices in codon_dists.items():
                args = (group_idx, d2_matrices)
                async_results.append(pool.apply_async(
                    self._plot_codon_distances, args=args,
                    kwds=dict(out_dir=self.out_dir.joinpath('codon-dists'),
                              angle_pair_labels=ap_labels,
                              annotate_mu=True, plot_std=False)
                ))
            del codon_dists, d2_matrices

        # Dihedral KDEs of each codon in each group
        codon_dkdes: dict = self._load_intermediate('codon-dkdes', False)
        group_sizes: dict = self._load_intermediate('group-sizes', False)
        if codon_dkdes is not None:
            for group_idx, dkdes in codon_dkdes.items():
                subgroup_sizes = group_sizes[group_idx]['subgroups']
                args = (group_idx, subgroup_sizes, dkdes)
                async_results.append(pool.apply_async(
                    self._plot_codon_dkdes, args=args,
                    kwds=dict(out_dir=self.out_dir.joinpath('codon-dkdes'),
                              angle_pair_labels=ap_labels)
                ))
            del codon_dkdes, group_sizes, dkdes

        # Wait for plotting to complete. Each function returns a path
        fig_paths = self._handle_async_results(async_results, collect=True)

    @staticmethod
    def _plot_likelihoods(
            likelihoods: dict, codon_or_aa: str, out_dir: Path,
    ):
        if codon_or_aa == 'codon':
            fig_filename = out_dir.joinpath(f'codon-likelihoods.pdf')
            xlabel, ylabel = r'$c$', r'$\Pr(CODON=c)$'
        elif codon_or_aa == 'aa':
            fig_filename = out_dir.joinpath(f'aa-likelihoods.pdf')
            xlabel, ylabel = r'$a$', r'$\Pr(AA=a)$'
        else:
            raise ValueError('Invalid type')

        # Convert from ss_type -> codon -> p, ss_type -> array
        for ss_type in likelihoods.keys():
            xticklabels = likelihoods[ss_type].keys()
            a = np.array([p for p in likelihoods[ss_type].values()],
                         dtype=np.float32)
            likelihoods[ss_type] = a

        pp5.plot.multi_bar(
            likelihoods,
            xticklabels=xticklabels, xlabel=xlabel, ylabel=ylabel,
            fig_size=(20, 5), single_width=1., total_width=0.7,
            outfile=fig_filename,
        )

        return str(fig_filename)

    @staticmethod
    def _plot_full_dkdes(
            full_dkde: dict,
            angle_pair_labels: List[str], out_dir: Path
    ):
        fig_filename = out_dir.joinpath('full-dkdes.pdf')
        with mpl.style.context(PP5_MPL_STYLE):
            fig_rows, fig_cols = len(full_dkde) // 2, 2
            fig, ax = mpl.pyplot.subplots(
                fig_rows, fig_cols, figsize=(5 * fig_cols, 5 * fig_rows),
                sharex='all', sharey='all'
            )
            fig: mpl.pyplot.Figure
            ax: np.ndarray[mpl.pyplot.Axes] = ax.reshape(-1)

            vmin, vmax = 0., 5e-4
            for i, (group_idx, dkdes) in enumerate(full_dkde.items()):
                pp5.plot.ramachandran(
                    dkdes, angle_pair_labels, title=group_idx, ax=ax[i],
                    vmin=vmin, vmax=vmax
                )

            pp5.plot.savefig(fig, fig_filename, close=True)

        return str(fig_filename)

    @staticmethod
    def _plot_codon_distances(
            group_idx: str, d2_matrices: List[np.ndarray],
            angle_pair_labels: List[str], out_dir: Path,
            annotate_mu=True, plot_std=False,
    ):
        LOGGER.info(f'Plotting codon distances for {group_idx}')

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
            if std < p25: return '*'
            elif std < p50: return ':'
            elif std < p75: return '.'
            return ''

        # Here we plot a separate distance matrix for mu and for sigma.
        fig_filenames = []
        for avg_std, d2 in zip(('avg', 'std'), (d2_mu, d2_sigma)):
            if avg_std == 'std':
                ann_fn = None
                if not plot_std: continue
            else:
                ann_fn = quartile_ann_fn if annotate_mu else None

            fig_filename = out_dir.joinpath(f'{group_idx}-{avg_std}.png')

            pp5.plot.multi_heatmap(
                d2, CODONS, CODONS, titles=angle_pair_labels, fig_size=20,
                fig_rows=1, outfile=fig_filename, data_annotation_fn=ann_fn
            )

            fig_filenames.append(str(fig_filename))

        return fig_filenames

    @staticmethod
    def _plot_codon_dkdes(
            group_idx: str, subgroup_sizes: Dict[str, int],
            codon_dkdes: Dict[str, List[np.ndarray]],
            angle_pair_labels: List[str], out_dir: Path
    ):
        # Plot the kdes and distance matrices
        LOGGER.info(f'Plotting KDEs for {group_idx}')

        with mpl.style.context(PP5_MPL_STYLE):
            vmin, vmax = 0., 5e-4
            fig, ax = mpl.pyplot.subplots(8, 8, figsize=(40, 40),
                                          sharex='all', sharey='all')
            ax: np.ndarray[mpl.pyplot.Axes] = np.reshape(ax, -1)

            for i, (codon, dkdes) in enumerate(codon_dkdes.items()):
                title = f'{codon} ({subgroup_sizes.get(codon, 0)})'
                if not dkdes:
                    ax[i].set_title(title)
                    continue

                pp5.plot.ramachandran(
                    dkdes, angle_pair_labels, title=title, ax=ax[i],
                    vmin=vmin, vmax=vmax
                )

            fig_filename = out_dir.joinpath(f'{group_idx}.png')
            pp5.plot.savefig(fig, fig_filename, close=True)

        return str(fig_filename)

    @staticmethod
    def _dihedral_kde_single_group(group_idx, df_group, angle_pairs, kde_args):
        kde = DihedralKDE(**kde_args)

        # Creates 2D KDE for each angle pair
        dkdes = []
        for phi_col, psi_col in angle_pairs:
            phi = df_group[phi_col].values
            psi = df_group[psi_col].values
            dkde = kde(phi, psi)
            dkdes.append(dkde)

        return group_idx, dkdes

    @staticmethod
    def _codon_dkdes_single_subgroup(
            group_idx: str, subgroup_idx: str, df_subgroup: pd.DataFrame,
            angle_pairs: list, kde_args: dict,
            bs_niter: int, bs_nsamples: int, bs_randstate: Optional[int],
    ) -> Tuple[str, List[np.ndarray]]:
        # Create a 3D tensor to hold the bootstrapped KDEs (for each angle
        # pair), of shape (B,N,N)
        bs_kde_shape = (bs_niter, kde_args['n_bins'], kde_args['n_bins'])
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
            _, dkdes = PointwiseCodonDistanceAnalyzer._dihedral_kde_single_group(
                subgroup_idx, df_subgroup_sampled, angle_pairs, kde_args
            )

            # Save the current iteration's KDE into the results tensor
            for angle_pair_idx, dkde in enumerate(dkdes):
                bs_dkdes[angle_pair_idx][bootstrap_idx, ...] = dkde

        t_elapsed = time.time() - t_start
        bs_rate_iter = bs_niter / t_elapsed
        LOGGER.info(f'Completed {bs_niter} bootstrap iterations for '
                    f'{group_idx}_{subgroup_idx}, size={len(df_subgroup)}, '
                    f'bs_nsamples={bs_nsamples}, '
                    f'rate={bs_rate_iter:.1f} iter/sec '
                    f'elapsed={t_elapsed:.1f} sec')

        return subgroup_idx, bs_dkdes

    @staticmethod
    def _codon_dists_single_group(
            group_idx: str, bs_codon_dkdes: Dict[str, List[np.ndarray]],
            angle_pairs: list, kde_dist_metric: Callable,
    ):
        tstart = time.time()

        # Calculate distance matrix
        d2_matrices = []
        for pair_idx in range(len(angle_pairs)):
            # For each angle pair we have N_CODONS dkde matrices,
            # so we compute the distance between each such pair.
            # We use a complex array to store mu as the real part and sigma
            # as the imaginary part in a single array.
            d2_mat = np.full((N_CODONS, N_CODONS), np.nan, np.complex64)

            codon_pairs = it.product(enumerate(CODONS), enumerate(CODONS))
            for (i, ci), (j, cj) in codon_pairs:
                if bs_codon_dkdes[ci] is None:
                    continue

                if j < i or bs_codon_dkdes[cj] is None:
                    continue

                # Get the two dihedral KDEs arrays to compare, each is of
                # shape (B, N, N) due to bootstrapping B times
                dkde1 = bs_codon_dkdes[ci][pair_idx]
                dkde2 = bs_codon_dkdes[cj][pair_idx]

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

        # Average the codon KDEs from all bootstraps for each codon,
        # so that we can save a simple KDE per codon
        codon_dkdes = {c: [] for c in CODONS}
        for codon, bs_dkde in bs_codon_dkdes.items():
            if bs_dkde is None:
                continue
            for pair_idx in range(len(angle_pairs)):
                # bs_dkde[pair_idx] here is of shape (B, N, N) due to
                # bootstrapping. Average it over the bootstrap dimension
                mean_dkde = np.nanmean(bs_dkde[pair_idx], axis=0)
                codon_dkdes[codon].append(mean_dkde)

        tend = time.time()
        LOGGER.info(f'Calculated distance matrix for {group_idx} '
                    f'({tend - tstart:.1f}s)...')

        return d2_matrices, codon_dkdes

    @staticmethod
    def _kde_dist_metric_l2(kde1: np.ndarray, kde2: np.ndarray):
        # We expect kde1 and kde2 to be of shape (B, N, N)
        # We calculate distance between each 2D NxN plane and return B
        # distances
        assert kde1.ndim == 3 and kde2.ndim == 3
        return np.nansum((kde1 - kde2) ** 2, axis=(1, 2))

    @staticmethod
    def _cols2label(phi_col: str, psi_col: str):
        def rep(col: str):
            col = col.replace('phi', r'\varphi')
            col = col.replace('psi', r'\psi')
            col = re.sub(r'([+-][01])', r'_{\1}', col)
            return col

        return rf'${rep(phi_col)}, {rep(psi_col)}$'
