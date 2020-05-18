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
from Bio.Data.CodonTable import standard_dna_table

from pp5.collect import ParallelDataCollector
from pp5.dihedral import DihedralKDE
import pp5.plot
from pp5.plot import PP5_MPL_STYLE

LOGGER = logging.getLogger(__name__)

CODONS = sorted(standard_dna_table.forward_table)
N_CODONS = len(CODONS)

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
    'G': SS_TYPE_HELIX,
    'I': SS_TYPE_HELIX,
    'T': SS_TYPE_TURN,
    'S': SS_TYPE_OTHER,  # maybe also turn?
    'B': SS_TYPE_OTHER,  # maybe also sheet?
    '-': SS_TYPE_OTHER,
    '': SS_TYPE_OTHER,
}


class PointwiseCodonDistance(ParallelDataCollector):
    """
    Analyzes a pointwise dataset (dihedral angles with residue information)
    to produce a matrix of distances between codons Dij.
    Each entry ij in Dij corresponds to codons i and j, and teh value is a
    distance metric between the distributions of dihedral angles coming from
    these codons.
    """

    def __init__(
            self, dataset_dir: Union[str, Path],
            pointwise_filename: str = 'data-pointwise.csv',
            condition_on_prev_codon=True,
            condition_on_ss=True,
            consolidate_ss=DSSP_TO_SS_TYPE.copy(),
            strict_ss=True, angle_pairs=None,
            kde_nbins=128, kde_k1=30., kde_k2=30., kde_k3=0.,
            bs_niter=1, bs_randstate=None,
            clear_intermediate=False,
    ):
        self.dataset_dir = Path(dataset_dir)
        self.input_file = self.dataset_dir.joinpath(pointwise_filename)
        if not self.dataset_dir.is_dir():
            raise ValueError(f'Dataset dir {self.dataset_dir} not found')
        if not self.input_file.is_file():
            raise ValueError(f'Dataset file {self.input_file} not found')

        out_dir = self.dataset_dir.joinpath('results')
        super().__init__(id='pointwise-cdist', out_dir=out_dir,
                         create_zip=False)

        self.condition_on_prev_codon = condition_on_prev_codon
        self.condition_on_ss = condition_on_ss
        self.consolidate_ss = consolidate_ss
        self.strict_ss = strict_ss

        if not angle_pairs:
            self.angle_pairs = [
                (f'phi+0', f'psi+0'), (f'phi+0', f'psi-1'),
            ]
        else:
            self.angle_pairs = angle_pairs

        self.angle_cols = sorted(set(it.chain(*self.angle_pairs)))
        self.codon_cols = [f'codon-1', f'codon+0']
        self.secondary_cols = [f'secondary-1', f'secondary+0']
        self.secondary_col = 'secondary'

        self.kde_args = dict(n_bins=kde_nbins, k1=kde_k1, k2=kde_k2, k3=kde_k3)
        self.kde_dist_metric = self._kde_dist_metric_l2

        self.bs_niter = bs_niter
        self.bs_randstate = bs_randstate

        # Create clean directory for intermediate results between steps
        self.intermediate_dir = self.out_dir.joinpath('_intermediate_')
        if clear_intermediate and self.intermediate_dir.exists():
            shutil.rmtree(str(self.intermediate_dir))
        os.makedirs(str(self.intermediate_dir), exist_ok=True)

        # Create dict for storing paths of intermediate results
        self.intermediate_files: Dict[str, Path] = {}

    def _collection_functions(self) \
            -> Dict[str, Callable[[mp.pool.Pool], Optional[Dict]]]:
        return {
            'preprocess-dataset': self._preprocess_dataset,
            'dihedral-kde-full': self._dihedral_kde_full,
            'codon-dists': self._codons_dists,
            'plot-results': self._plot_results,
        }

    def _dump_intermediate_result(self, name: str, obj):
        # Update dict of intermediate files
        path = self.intermediate_dir.joinpath(f'{name}.pkl')
        self.intermediate_files[name] = path

        with open(str(path), 'wb') as f:
            pickle.dump(obj, f, protocol=4)

        LOGGER.info(f'Wrote intermediate file {path}')
        return path

    def _load_intermediate_result(self, name):
        path = self.intermediate_dir.joinpath(f'{name}.pkl')
        if not path.is_file():
            raise ValueError(f"Can't file intermediate file {path}")
        if name not in self.intermediate_files:
            # This is not a problem, might be from a previous run we wish to
            # resume
            LOGGER.warning(f'Unregistered intermediate file {path}')
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

        self._dump_intermediate_result('dataset', df_preproc)
        return {
            'input_file': str(self.input_file),
            'n_rows': len(df_preproc)
        }

    def _preprocess_subframe(self, df_sub: pd.DataFrame):
        # Logic for consolidating secondary structure between a pair of curr
        # and prev residues
        def ss_consolidator(row: pd.Series):
            ss_m1 = row[self.secondary_cols[0]]  # e.g. 'H' or e.g. 'H/G'
            ss_p0 = row[self.secondary_cols[1]]

            # In strict mode we require that all group members had the same SS,
            # i.e. we don't allow groups with more than one type.
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

        ss_consolidated = df_sub.apply(ss_consolidator, axis=1)

        # Keep only angle and codon columns from the full dataset
        df_pointwise = pd.DataFrame(
            data=df_sub[self.angle_cols + self.codon_cols],
        )

        # Add consolidated SS
        df_pointwise[self.secondary_col] = ss_consolidated

        # Remove rows without consolidated SS (this means the residues
        # pairs didn't have the same SS)
        has_ss = ~df_pointwise['secondary'].isnull()
        df_pointwise = df_pointwise[has_ss]

        # Convert angles to radians
        angles_rad = df_pointwise[self.angle_cols].applymap(np.deg2rad)
        df_pointwise[self.angle_cols] = angles_rad

        # Convert angle columns to float32
        dtype = {col: np.float32 for col in self.angle_cols}
        df_pointwise = df_pointwise.astype(dtype)

        return df_pointwise

    def _dihedral_kde_full(self, pool: mp.pool.Pool) -> dict:
        df_processed: pd.DataFrame = self._load_intermediate_result('dataset')
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
        self._dump_intermediate_result('full-dkde', map_result)

        return {
            **ss_counts,
        }

    def _codons_dists(self, pool: mp.pool.Pool) -> dict:
        prev_codon, curr_codon = self.codon_cols

        df_processed: pd.DataFrame = self._load_intermediate_result('dataset')
        df_groups = df_processed.groupby(by=[self.secondary_col, prev_codon])

        LOGGER.info(f'Calculating codon-pair distance matrices...')

        args = (
            (
                group_idx, df_group, curr_codon,
                self.angle_pairs, self.kde_args, self.kde_dist_metric,
                self.bs_niter, self.bs_randstate,
            )
            for group_idx, df_group in df_groups
        )
        map_result = pool.starmap(self._codon_dists_single_group, args)

        # maps from group (codon, SS) to a list, containing a codon-distance
        # matrix for each angle-pair. The codon distance matrix is complex,
        # where real is mu and imag is sigma
        codon_dists = {group_idx: d2 for group_idx, d2, _ in map_result}
        self._dump_intermediate_result('codon-dists', codon_dists)

        codon_dkdes = {group_idx: dkdes for group_idx, _, dkdes in map_result}
        self._dump_intermediate_result('codon-dkdes', codon_dkdes)

        return {}

    def _plot_results(self, pool: mp.pool.Pool):
        LOGGER.info(f'Plotting results...')

        angle_pair_labels = [self._cols2label(phi_col, psi_col)
                             for phi_col, psi_col in self.angle_pairs]

        plot_fn_kwargs = {
            'angle_pair_labels': angle_pair_labels,
            'out_dir': self.out_dir,
        }

        async_results = []

        # Dihedral KDEs of full dataset
        full_dkde: dict = self._load_intermediate_result('full-dkde')
        async_results.append(pool.apply_async(
            self._plot_full_dkde, args=(full_dkde,), kwds=plot_fn_kwargs
        ))
        del full_dkde

        # Codon distance matrices
        codon_dists: dict = self._load_intermediate_result('codon-dists')
        for group_idx, d2_matrices in codon_dists.items():
            args = (group_idx, d2_matrices)
            async_results.append(pool.apply_async(
                self._plot_codon_distances, args=args, kwds=plot_fn_kwargs
            ))
        del codon_dists, d2_matrices

        # Dihedral KDEs of each codon in each group
        codon_dkdes: dict = self._load_intermediate_result('codon-dkdes')
        for group_idx, dkdes in codon_dkdes.items():
            args = (group_idx, dkdes)
            async_results.append(pool.apply_async(
                self._plot_codon_dkdes, args=args, kwds=plot_fn_kwargs
            ))
        del codon_dkdes, dkdes

        # Wait for plotting to complete. Each function returns a path
        fig_paths = self._handle_async_results(async_results, collect=True)

        return {
            'figure_paths': fig_paths
        }

    @staticmethod
    def _plot_full_dkde(
            full_dkde: dict,
            angle_pair_labels: List[str], out_dir: Path
    ):
        fig_filename = out_dir.joinpath('full-dkde.pdf')
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

        return fig_filename

    @staticmethod
    def _plot_codon_distances(
            group_idx: tuple, d2_matrices: List[np.ndarray],
            angle_pair_labels: List[str], out_dir: Path
    ):
        LOGGER.info(f'Plotting codon distances for {group_idx}')

        # d2_matrices contains a matrix for each analyzed angle pair.
        # Split the mu and sigma from the complex d2 matrices
        d2_mu_sigma = [(np.real(d2), np.imag(d2)) for d2 in d2_matrices]
        d2_mu, d2_sigma = list(zip(*d2_mu_sigma))

        out_dir = out_dir.joinpath('codon-dist')
        fig_basename = f'{str.join("_", reversed(group_idx))}'

        fig_filenames = []
        for mu_sigma, d2 in zip(('mu', 'sigma'), (d2_mu, d2_sigma)):
            fig_filename = out_dir.joinpath(f'{fig_basename}-{mu_sigma}.png')

            pp5.plot.multi_heatmap(
                d2, CODONS, CODONS, titles=angle_pair_labels, fig_size=20,
                fig_rows=1, outfile=fig_filename
            )
            fig_filenames.append(fig_filename)

        return fig_filenames

    @staticmethod
    def _plot_codon_dkdes(
            group_idx: tuple, codon_dkdes: Dict[str, List[np.ndarray]],
            angle_pair_labels: List[str], out_dir: Path
    ):
        # Plot the kdes and distance matrices
        LOGGER.info(f'Plotting KDEs for {group_idx}')

        out_dir = out_dir.joinpath('codon-dkde')
        fig_basename = f'{str.join("_", reversed(group_idx))}'

        with mpl.style.context(PP5_MPL_STYLE):
            vmin, vmax = 0., 5e-4
            fig, ax = mpl.pyplot.subplots(8, 8, figsize=(40, 40),
                                          sharex='all', sharey='all')
            ax = np.reshape(ax, -1)

            for i, (codon, dkdes) in enumerate(codon_dkdes.items()):
                if dkdes is None:
                    continue

                pp5.plot.ramachandran(
                    dkdes, angle_pair_labels, title=codon, ax=ax[i],
                    vmin=vmin, vmax=vmax
                )

            fig_filename = out_dir.joinpath(f'{fig_basename}.png')
            pp5.plot.savefig(fig, fig_filename, close=True)

        return fig_filename

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
            group_idx: tuple, df_codon_group: pd.DataFrame, curr_codon: str,
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
        bs_rate_samples = group_size / (time.time() - start_time)
        LOGGER.info(f'Completed {bs_niter} bootstrap iterations for '
                    f'{group_idx}, size={group_size}, '
                    f'rate={bs_rate_iter:.1f} iter/sec '
                    f'({bs_rate_samples} samples/sec)')

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

        angle_pair_labels = [
            PointwiseCodonDistance._cols2label(phi_col, psi_col)
            for phi_col, psi_col in angle_pairs
        ]

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

        return group_idx, d2_matrices, codon_dkdes

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
