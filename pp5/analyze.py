import os
import re
import itertools as it
import logging
import multiprocessing as mp
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

        self.angle_cols = list(set(it.chain(*self.angle_pairs)))
        self.codon_cols = [f'codon-1', f'codon+0']
        self.secondary_cols = [f'secondary-1', f'secondary+0']
        self.secondary_col = 'secondary'

        self.kde_args = dict(n_bins=kde_nbins, k1=kde_k1, k2=kde_k2, k3=kde_k3)
        self.kde_dist_metric = self._kde_dist_metric_l2

        self.df_pointwise = None

    def _collection_functions(self) \
            -> Dict[str, Callable[[mp.pool.Pool], Optional[Dict]]]:
        return {
            'load-dataset': self._load_dataset,
            'dihedral-kde-full': self._dihedral_kde_full,
            'codon-dists': self._codons_dists,
        }

    def _load_dataset(self, pool: mp.pool.Pool) -> dict:
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
        sub_dfs = pool.map(self._load_dataset_subframe, df_pointwise_reader)
        df_pointwise = pd.concat(sub_dfs, axis=0, ignore_index=True)

        LOGGER.info(f'Loaded {self.input_file}: {len(df_pointwise)} rows\n'
                    f'{df_pointwise}')

        self.df_pointwise = df_pointwise
        return {
            'input_file': str(self.input_file),
            'n_rows': len(df_pointwise)
        }

    def _load_dataset_subframe(self, df_sub: pd.DataFrame):
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
                ss_m1 = self.consolidate_ss.get(ss_m1[0], SS_TYPE_OTHER)
                ss_p0 = self.consolidate_ss.get(ss_p0[0], SS_TYPE_OTHER)

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

        # Convert dtype of angle columns
        dtype = {}
        for c in self.angle_cols:
            dtype[c] = np.float32
        df_pointwise = df_pointwise.astype(dtype)

        # Convert angles to radians
        df_pointwise[self.angle_cols] = \
            df_pointwise[self.angle_cols].applymap(np.deg2rad)

        return df_pointwise

    def _dihedral_kde_full(self, pool: mp.pool.Pool) -> dict:
        df_groups = self.df_pointwise.groupby(by=self.secondary_col)

        df_groups_count: pd.DataFrame = df_groups.count()
        ss_counts = {
            f'n_{ss_type}': count
            for ss_type, count in df_groups_count.max(axis=1).to_dict().items()
        }

        LOGGER.info(f'Secondary-structure groups:\n{ss_counts})')
        LOGGER.info(f'Calculating dihedral distribution per SS type...')

        args = ((group_idx, df_group, self.angle_pairs, self.kde_args)
                for group_idx, df_group in df_groups)

        mapres = pool.starmap(self._dihedral_kde_single_group, args)

        LOGGER.info(f'Plotting dihedral distributions...')
        # Plot the results
        with mpl.style.context(PP5_MPL_STYLE):
            fig, ax = mpl.pyplot.subplots(
                len(df_groups) // 2, 2, figsize=(12, 12),
                sharex='all', sharey='all'
            )
            fig: mpl.pyplot.Figure
            ax: np.ndarray[mpl.pyplot.Axes] = ax.reshape(-1)

            legend_labels = [self._cols2label(phi_col, psi_col)
                             for phi_col, psi_col in self.angle_pairs]

            vmin, vmax = 0., 5e-4
            for i, (group_idx, dkdes) in enumerate(mapres):
                pp5.plot.ramachandran(
                    dkdes, legend_labels, title=group_idx, ax=ax[i],
                    vmin=vmin, vmax=vmax
                )

            fig_filename = self.out_dir.joinpath('dihedral-kde_full.pdf')
            pp5.plot.savefig(fig, fig_filename)

        LOGGER.info(f'Wrote {fig_filename}')

        return {
            **ss_counts,
            'dkde-fig': str(fig_filename)
        }

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

    def _codons_dists(self, pool: mp.pool.Pool) -> dict:
        prev_codon, curr_codon = self.codon_cols
        df_groups = self.df_pointwise.groupby(
            by=[self.secondary_col, prev_codon]
        )

        LOGGER.info(f'Calculating codon-pair distance matrices...')

        args = (
            (
                group_idx, df_group, curr_codon, self.out_dir,
                self.angle_pairs, self.kde_args, self.kde_dist_metric,
            )
            for group_idx, df_group in df_groups
        )
        mapres = pool.starmap(self._codon_dist_single_group, args)

        for group_idx, d2_matrices in mapres:
            LOGGER.info(f'Codon-pair dists for {group_idx}: '
                        f'{list(map(np.shape, d2_matrices))}')

        return {

        }

    @staticmethod
    def _codon_dist_single_group(
            group_idx: tuple, df_codon_group: pd.DataFrame, curr_codon: str,
            out_dir: Path, angle_pairs: list, kde_args: dict,
            kde_dist_metric: Callable
    ):
        # Initialize to obtain consistent order of codons
        codon_dkdes = {c: None for c in CODONS}

        LOGGER.info(f'Calculating dihedral distributions per codon in '
                    f'subgroup {group_idx}...')

        # df_codon_group is grouped by SS and prev codon. Now we also group by
        # the current codon so we can compute a distance matrix between the
        # distribution of each pair of current codons.
        df_subgroups = df_codon_group.groupby(curr_codon)
        for subgroup_idx, df_subgroup in df_subgroups:
            _, dkdes = PointwiseCodonDistance._dihedral_kde_single_group(
                subgroup_idx, df_subgroup, angle_pairs, kde_args
            )

            codon_dkdes[subgroup_idx] = dkdes

        LOGGER.info(f'Calculating codon-codon distance matrix in subgroup '
                    f'{group_idx}...')

        d2_matrices = []
        for pair_idx in range(len(angle_pairs)):
            # For each angle pair we have N_CODONS dkde matrices,
            # so we compute the distance between each such pair.
            d2_mat = np.full((N_CODONS, N_CODONS), np.nan, dtype=np.float32)

            all_codon_pairs = it.product(enumerate(CODONS), enumerate(CODONS))
            for (i, ci), (j, cj) in all_codon_pairs:
                if codon_dkdes[ci] is None:
                    continue

                if j < i or codon_dkdes[cj] is None:
                    continue

                # Get the two dihedral distributions to compare
                dkde1 = codon_dkdes[ci][pair_idx]
                dkde2 = codon_dkdes[cj][pair_idx]
                d2 = kde_dist_metric(dkde1, dkde2)
                d2_mat[i, j] = d2_mat[j, i] = d2

            d2_matrices.append(d2_mat)

        legend_labels = [
            PointwiseCodonDistance._cols2label(phi_col, psi_col)
            for phi_col, psi_col in angle_pairs
        ]
        # Plot the kdes and distance matrices
        # LOGGER.info(f'Plotting KDEs for {group_idx}')
        # with mpl.style.context(PP5_MPL_STYLE):
        #     vmin, vmax = 0., 5e-4
        #     fig, ax = mpl.pyplot.subplots(8, 8, figsize=(64, 64))
        #     ax = np.reshape(ax, -1)
        #     for i, (codon, dkdes) in enumerate(codon_dkdes.items()):
        #         ramachandran(dkdes, legend_labels, title=codon, ax=ax[i],
        #                      vmin=vmin, vmax=vmax)
        #
        #     fig_filename = out_dir.joinpath('dihedral-kde') \
        #         .joinpath(group_idx[1]).joinpath(f'{group_idx[0]}.pdf')
        #     os.makedirs(str(fig_filename.parent), exist_ok=True)
        #     fig.savefig(str(fig_filename), format='pdf')
        #     mpl.pyplot.close(fig)
        #     LOGGER.info(f'Wrote {fig_filename}')

        LOGGER.info(f'Plotting codon distances for {group_idx}')
        fig_filename = out_dir.joinpath('codon-dist') \
            .joinpath(f'{str.join("_", group_idx)}.pdf')
        pp5.plot.multi_heatmap(
            d2_matrices, CODONS, CODONS,
            titles=legend_labels, figsize=20, outfile=fig_filename
        )

        return group_idx, d2_matrices

    @staticmethod
    def _kde_dist_metric_l2(kde1, kde2):
        return np.nansum((kde1 - kde2) ** 2)

    @staticmethod
    def _cols2label(phi_col: str, psi_col: str):
        def rep(col: str):
            col = col.replace('phi', r'\varphi')
            col = col.replace('psi', r'\psi')
            col = re.sub(r'([+-][01])', r'_{\1}', col)
            return col

        return rf'${rep(phi_col)}, {rep(psi_col)}$'
