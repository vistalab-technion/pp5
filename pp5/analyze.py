import os
import pickle
import re
import itertools as it
import logging
import multiprocessing as mp
import shutil
import time
from pathlib import Path
from typing import Union, Dict, Callable, Optional, List

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.style
import matplotlib.pyplot
from Bio.Data.CodonTable import standard_dna_table as dna_table

from pp5.collect import ParallelDataCollector
from pp5.dihedral import DihedralKDE
import pp5.plot
from pp5.plot import PP5_MPL_STYLE

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


class PointwiseCodonDistance(ParallelDataCollector):

    def __init__(
            self, dataset_dir: Union[str, Path],
            pointwise_filename: str = 'data-pointwise.csv',
            condition_on_prev='codon', condition_on_ss=True,
            consolidate_ss=DSSP_TO_SS_TYPE.copy(), strict_ss=True,
            kde_nbins=128, kde_k1=30., kde_k2=30., kde_k3=0.,
            bs_niter=1, bs_randstate=None,
            out_tag: str = None,
            clear_intermediate=False,
    ):
        """
        Analyzes a pointwise dataset (dihedral angles with residue information)
        to produce a matrix of distances between codons Dij.
        Each entry ij in Dij corresponds to codons i and j, and teh value is a
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
        :param out_tag: Tag for output.
        :param clear_intermediate: Whether to clear intermediate folder.
        """
        self.dataset_dir = Path(dataset_dir)
        self.out_tag = out_tag
        self.input_file = self.dataset_dir.joinpath(pointwise_filename)
        if not self.dataset_dir.is_dir():
            raise ValueError(f'Dataset dir {self.dataset_dir} not found')
        if not self.input_file.is_file():
            raise ValueError(f'Dataset file {self.input_file} not found')
        condition_on_prev = '' if condition_on_prev is None \
            else condition_on_prev.lower()
        if condition_on_prev not in {'codon', 'aa', ''}:
            raise ValueError(f'invalid condition_on_prev: {condition_on_prev}')

        tag = f'-{self.out_tag}' if self.out_tag else ''
        out_dir = self.dataset_dir.joinpath('results')
        super().__init__(id=f'pointwise_cdist{tag}', out_dir=out_dir,
                         tag=out_tag, create_zip=False, )

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

        # Update metadata with current configuration
        state_dict = self.__getstate__()
        state_dict.pop('collection_meta')
        self.collection_meta.update(state_dict)

        # Create clean directory for intermediate results between steps
        self.intermediate_dir = self.out_dir.joinpath('_intermediate_')
        if clear_intermediate and self.intermediate_dir.exists():
            shutil.rmtree(str(self.intermediate_dir))
        os.makedirs(str(self.intermediate_dir), exist_ok=True)

        # Create dict for storing paths of intermediate results
        self.intermediate_files: Dict[str, Path] = {}

    def __getstate__(self):
        state = self.__dict__.copy()
        for k, v in state.items():
            if isinstance(v, (Path,)):
                state[k] = str(v)
        return state

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

    def _dump_intermediate(self, name: str, obj):
        # Update dict of intermediate files
        path = self.intermediate_dir.joinpath(f'{name}.pkl')
        self.intermediate_files[name] = path

        with open(str(path), 'wb') as f:
            pickle.dump(obj, f, protocol=4)

        LOGGER.info(f'Wrote intermediate file {path}')
        return path

    def _load_intermediate(self, name, allow_old=True, raise_if_missing=True):
        path = self.intermediate_dir.joinpath(f'{name}.pkl')

        if name not in self.intermediate_files:
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

        self.intermediate_files[name] = path
        with open(str(path), 'rb') as f:
            obj = pickle.load(f)

        LOGGER.info(f'Loaded intermediate file {path}')
        return obj

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

        return {}

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
        prev_codon, curr_codon = self.codon_cols

        df_processed: pd.DataFrame = self._load_intermediate('dataset')
        df_groups = df_processed.groupby(by=self.condition_col)

        LOGGER.info(f'Calculating codon-pair distance matrices...')

        # We currently only support one type of metric
        dist_metrics = {
            'l2': self._kde_dist_metric_l2
        }

        args = (
            (
                group_idx, df_group, curr_codon,
                self.angle_pairs, self.kde_args,
                dist_metrics[self.kde_dist_metric.lower()],
                self.bs_niter, self.bs_randstate,
            )
            for group_idx, df_group in df_groups
        )

        map_result = pool.starmap(self._codon_dists_single_group, args)

        codon_dists, codon_dkdes, group_sizes = {}, {}, {}
        for group_idx, group_size, d2_matrices, dkdes in map_result:
            codon_dists[group_idx] = d2_matrices
            codon_dkdes[group_idx] = dkdes
            group_sizes[group_idx] = group_size

        # codon_dists: maps from group (codon, SS) to a list, containing a
        # codon-distance matrix for each angle-pair.
        # The codon distance matrix is complex, where real is mu
        # and imag is sigma
        self._dump_intermediate('codon-dists', codon_dists)

        # codon_dkdes: maps from group to a dict where keys are codons.
        # For each codon we have a list of KDEs, one for each angle pair.
        self._dump_intermediate('codon-dkdes', codon_dkdes)

        return {'groups': group_sizes}

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
        if codon_dkdes is not None:
            for group_idx, dkdes in codon_dkdes.items():
                args = (group_idx, dkdes)
                async_results.append(pool.apply_async(
                    self._plot_codon_dkdes, args=args,
                    kwds=dict(out_dir=self.out_dir.joinpath('codon-dkdes'),
                              angle_pair_labels=ap_labels)
                ))
            del codon_dkdes, dkdes

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
            group_idx: str, codon_dkdes: Dict[str, List[np.ndarray]],
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
                if dkdes is None:
                    ax[i].set_title(codon)
                    continue

                pp5.plot.ramachandran(
                    dkdes, angle_pair_labels, title=codon, ax=ax[i],
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
    def _codon_dists_single_group(
            group_idx: str, df_codon_group: pd.DataFrame, curr_codon: str,
            angle_pairs: list, kde_args: dict, kde_dist_metric: Callable,
            bs_niter: int, bs_randstate: Optional[int],
    ):
        # Initialize in advance to obtain consistent order of codons
        codon_dkdes = {c: None for c in CODONS}

        # We want a different random state in each group, but still
        # shoud be reproducible
        if bs_randstate is not None:
            seed = (hash(group_idx) + bs_randstate) % (2 ** 31)
            np.random.seed(seed)

        group_size = len(df_codon_group)
        start_time = time.time()

        # df_codon_group is grouped by SS and prev codon.
        # Now we also group by the current codon so we can compute a
        # distance matrix between the distribution of each pair of current
        # codons.
        df_subgroups = df_codon_group.groupby(curr_codon)
        for subgroup_idx, df_subgroup in df_subgroups:
            for bootstrap_idx in range(bs_niter):
                # Sample from dataset with replacement, the same number of
                # elements as it's size. This is our bootstrap sample.
                df_subgroup_sampled = df_subgroup.sample(
                    axis=0, frac=1., replace=bs_niter > 1,
                )

                # dkdes contains one KDE for each pair in angle_pairs
                _, dkdes = PointwiseCodonDistance._dihedral_kde_single_group(
                    subgroup_idx, df_subgroup_sampled, angle_pairs, kde_args
                )

                if codon_dkdes[subgroup_idx] is None:
                    codon_dkdes[subgroup_idx] = [dkdes]
                else:
                    codon_dkdes[subgroup_idx].append(dkdes)

        bs_rate_iter = bs_niter / (time.time() - start_time)
        bs_rate_samples = group_size * bs_rate_iter
        LOGGER.info(f'Completed {bs_niter} bootstrap iterations for '
                    f'{group_idx}, size={group_size}, '
                    f'rate={bs_rate_iter:.1f} iter/sec '
                    f'({bs_rate_samples:.1f} samples/sec)')

        # Now we have the bootstrap results, we must consolidate them.
        # For each current codon, we'll create a 3D tensor containing
        # n_bootstraps 2D KDEs.
        for (codon, codon_dkdes_list) in codon_dkdes.items():
            if codon_dkdes_list is None:
                # Will happen if we didn't see this codon in the current group
                continue

            # We will replace the contents of this map with a list of
            # arrays, one for each angle pair, each array will contain all
            # bootstrapped KDEs
            codon_dkdes[codon] = []

            # codon_dkdes_list is e.g.
            # [(kde1_1, kde2_1), (kde1_2, kde2_2), ... (kde1_B, kde2_B)]
            # (when we have two angle pairs).
            # Convert it to [(kde1_1, kde1_2, ..., kde1_B), ...]
            for codon_angle_pair_dkdes in zip(*codon_dkdes_list):
                # stacked will be of shape (B, N, N) and contain all the
                # bootstrapped KDEs for this codon and current angle pair
                stacked = np.stack(codon_angle_pair_dkdes, axis=0)
                codon_dkdes[codon].append(stacked)

        LOGGER.info(f'Calculating codon-codon distance matrix in subgroup '
                    f'{group_idx}...')

        d2_matrices = []
        for pair_idx in range(len(angle_pairs)):
            # For each angle pair we have N_CODONS dkde matrices,
            # so we compute the distance between each such pair.
            # We use a complex array to store mu as the real part and sigma
            # as the imaginary part in a single array.
            d2_mat = np.full((N_CODONS, N_CODONS), np.nan, dtype=np.complex64)

            all_codon_pairs = it.product(enumerate(CODONS), enumerate(CODONS))
            for (i, ci), (j, cj) in all_codon_pairs:
                if codon_dkdes[ci] is None:
                    continue

                if j < i or codon_dkdes[cj] is None:
                    continue

                # Get the two dihedral KDEs arrays to compare, each is of
                # shape (B, N, N) due to bootstrapping B times
                dkde1 = codon_dkdes[ci][pair_idx]
                dkde2 = codon_dkdes[cj][pair_idx]

                # If ci is cj, randomize the order of the KDEs when
                # comparing, so that different bootstrapped KDEs are compared
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

        # Average the codon KDEs from all bootstraps for each codon, so that we
        # can return a simple KDE per codon
        for codon, dkde in codon_dkdes.items():
            if codon_dkdes[codon] is None:
                continue
            for pair_idx in range(len(angle_pairs)):
                # kde here is of shape (B, N, N) due to bootstrapping
                # Average it over the bootstrap dimension
                kde = codon_dkdes[codon][pair_idx]
                mean_kde = np.nanmean(kde, axis=0)
                codon_dkdes[codon][pair_idx] = mean_kde

        return group_idx, group_size, d2_matrices, codon_dkdes

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
