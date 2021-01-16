import logging
import itertools as it
import multiprocessing as mp
from typing import Dict, Union, Callable, Optional
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn import manifold as manifold

import pp5.plot
from pp5.codons import ACIDS, CODON_RE, N_CODONS, AA_CODONS, ACIDS_1TO1AND3
from pp5.analysis import SS_TYPES, DSSP_TO_SS_TYPE
from pp5.analysis.base import ParallelAnalyzer
from pp5.analysis.pairwise import PairwiseCodonDistanceAnalyzer
from pp5.analysis.pointwise import PGroupPointwiseCodonDistanceAnalyzer

LOGGER = logging.getLogger(__name__)


class CodonDistanceAnalyzer(ParallelAnalyzer):
    def __init__(
        self,
        dataset_dir: Union[str, Path],
        out_dir: Union[str, Path] = None,
        pointwise_filename: str = "data-pointwise.csv",
        pairwise_filename: str = "data-pairwise.csv",
        condition_on_ss=True,
        consolidate_ss=DSSP_TO_SS_TYPE.copy(),
        bs_niter=1,
        bs_randstate=None,
        out_tag: str = None,
        pointwise_extra_kw: dict = None,
        pairwise_extra_kw: dict = None,
    ):
        """
        Performs both pointwise and pairwise codon-distance analysis and produces a
        comparison between these two type of analysis.
        Uses two sub-analyzers PGroupPointwiseCodonDistanceAnalyzer and
        PairwiseCodonDistanceAnalyzer as sub analyzers, which should be configured
        with the *_extra_kw dicts.

        :param dataset_dir: Path to directory with the collector output.
        :param out_dir: Path to output directory. Defaults to <dataset_dir>/results.
        :param pointwise_filename: Filename of the pointwise dataset.
        :param pairwise_filename: Filename of the pairwise dataset.
        :param condition_on_ss: Whether to group pairwise matches by sencondary
        structure and analyse each SS group separately.
        :param consolidate_ss: Dict mapping from DSSP secondary structure to
        the consolidated SS types used in this analysis.
        :param bs_niter: Number of bootstrap iterations.
        :param bs_randstate: Random state for bootstrap.
        :param out_tag: Tag for output.
        :param pointwise_extra_kw: Extra args for PGroupPointwiseCodonDistanceAnalyzer.
        Will override the ones define in this function.
        :param pairwise_extra_kw: Extra args for PairwiseCodonDistanceAnalyzer.
        Will override the ones define in this function.
        """
        super().__init__(
            "codon_dist",
            dataset_dir,
            "meta.json",
            out_dir,
            out_tag,
            clear_intermediate=False,
        )

        common_kw = dict(
            dataset_dir=dataset_dir,
            out_dir=self.out_dir,
            condition_on_ss=condition_on_ss,
            consolidate_ss=consolidate_ss,
            bs_niter=bs_niter,
            bs_randstate=bs_randstate,
        )
        pointwise_extra_kw = pointwise_extra_kw or {}
        pairwise_extra_kw = pairwise_extra_kw or {}

        pointwise_kw = common_kw.copy()
        pointwise_kw.update(pointwise_filename=pointwise_filename, **pointwise_extra_kw)
        self.pointwise_analyzer = PGroupPointwiseCodonDistanceAnalyzer(**pointwise_kw)

        pairwise_kw = common_kw.copy()
        pairwise_kw.update(pairwise_filename=pairwise_filename, **pairwise_extra_kw)
        self.pairwise_analyzer = PairwiseCodonDistanceAnalyzer(**pairwise_kw)

    def _collection_functions(
        self,
    ) -> Dict[str, Callable[[mp.pool.Pool], Optional[Dict]]]:
        return {
            "run-analysis": self._run_analysis,
            "cdist-correlation": self._cdist_correlation,
            "plot-corr-rainbows": self._plot_corr_rainbows,
            "plot-mds-rainbows": self._plot_mds_rainbows,
        }

    def _run_analysis(self, pool: mp.pool.Pool) -> dict:
        analyzers = {
            "pointwise": self.pointwise_analyzer,
            "pairwise": self.pairwise_analyzer,
        }

        return_meta = {}
        for name, analyzer in analyzers.items():
            analyzer.collect()
            if any(s.result == "FAIL" for s in analyzer._collection_steps):
                raise RuntimeError(f"{name} analysis failed, stopping...")
            return_meta[name] = analyzer._collection_meta

        return return_meta

    def _cdist_correlation(self, pool: mp.pool.Pool) -> dict:
        cdists_pointwise = self.pointwise_analyzer._load_intermediate("codon-dists-exp")
        cdists_pairwise = self.pairwise_analyzer._load_intermediate("codon-dists")

        # Will hold a dataframe of corr coefficients for each SS type
        corr_dfs = []

        # Dict: SS -> AA or 'ALL' -> dists, acids, codons
        # where dists is (x,y,r1,r2) where x,y are pairwise and pointwise codon
        # distances for one codon of AA and r1, r2 are the variances.
        # Acids contains the AA names, and Labels contains the codon names.
        corr_data = {}
        for ss_type in SS_TYPES:
            LOGGER.info(f"Calculating pointwise-pairwise correlation for {ss_type}...")

            # Get codon-distance matrices (61, 61)
            pointwise = cdists_pointwise[ss_type][0]
            pairwise = cdists_pairwise[ss_type]

            # Loop over unique pairs of codons
            ii, jj = np.triu_indices(N_CODONS)
            ss_corr_data = {}
            for k in range(len(ii)):
                i, j = ii[k], jj[k]
                a1, c1 = CODON_RE.match(AA_CODONS[i]).groups()
                a2, c2 = CODON_RE.match(AA_CODONS[j]).groups()

                # We care only about synonymous codons
                if a1 != a2:
                    continue

                # Get the pairwise/pointwise d^2 avg and std for this codon pair
                x, y = np.real(pairwise[i, j]), np.real(pointwise[i, j])
                s1, s2 = np.imag(pairwise[i, j]), np.imag(pointwise[i, j])

                # Add to both current AA key and 'ALL' key
                for plot_data_key in ("ALL", a1):
                    curr_plot_data = ss_corr_data.setdefault(plot_data_key, {})
                    curr_plot_data.setdefault("dists", []).append((x, y, s1, s2))
                    curr_plot_data.setdefault("acids", []).append(a1)
                    label = c1 if c1 == c2 else f"{c1}-{c2}"
                    curr_plot_data.setdefault("codons", []).append(label)

            corr_data[ss_type] = ss_corr_data

            # Calculate the correlation coefficient in each AA and for ALL
            corr_coeffs = {}
            for aa in ACIDS + ["ALL"]:
                d = ss_corr_data[aa]
                xy = np.array(d["dists"])[:, :2]
                if len(xy) == 1:
                    r = np.nan
                else:
                    r = np.corrcoef(xy[:, 0], xy[:, 1])[0, 1]
                corr_coeffs[aa] = r

            corr_dfs.append(pd.DataFrame([corr_coeffs]))

        # Now we can create a full DataFrame with the corr coeffs of all SS types.
        df_corr = pd.concat(corr_dfs)
        df_corr.index = SS_TYPES
        df_corr.columns = list(ACIDS_1TO1AND3.values()) + ["ALL"]
        df_corr = df_corr.T

        # Write the CSV file
        outfile = self.out_dir.joinpath("corr-pairwise_pointwise.csv")
        df_corr.to_csv(outfile, float_format="%.3f")

        # Save the correlation data dict
        self._dump_intermediate("cdist-corr-data", corr_data)
        return {}

    def _plot_corr_rainbows(self, pool: mp.pool.Pool) -> dict:
        corr_data = self._load_intermediate("cdist-corr-data", allow_old=True)

        fig_size = (8, 8)
        err_scale, alpha = 0.1, 0.5

        for ss_type in SS_TYPES:
            LOGGER.info(
                f"Plotting pairwise-pointwise correlation rainbows for" f" {ss_type}..."
            )
            async_results = []

            for aa in ACIDS + ["ALL"]:
                d = corr_data[ss_type][aa]
                title = f"{aa}_{ss_type}" if aa != "ALL" else f"{ss_type}"
                fig_dir = "rainbow-dists-aa-codon" if aa != "ALL" else "rainbow-dists"
                fig_dir = self.out_dir.joinpath(fig_dir)
                fig_file = fig_dir.joinpath(f"{title}.pdf")

                async_results.append(
                    pool.apply_async(
                        pp5.plot.rainbow,
                        args=(d["dists"],),
                        kwds=dict(
                            group_labels=d["acids"],
                            point_labels=d["codons"],
                            all_groups=ACIDS_1TO1AND3,
                            alpha=alpha,
                            fig_size=fig_size,
                            err_scale=err_scale,
                            error_ellipse=False,
                            normalize=True,
                            xlabel="Pairwise",
                            ylabel="Pointwise",
                            with_regression=True,
                            title=title,
                            outfile=fig_file,
                        ),
                    )
                )

            # Wait for plotting to complete
            self._handle_async_results(async_results)
        return {}

    def _plot_mds_rainbows(self, pool: mp.pool.Pool) -> dict:
        cdists_pointwise = {
            aa_codon: self.pointwise_analyzer._load_intermediate(
                f"{aa_codon}-dists-exp"
            )
            for aa_codon in ("aa", "codon")
        }

        scale = 1e3
        alpha = 0.5
        mds = manifold.MDS(
            n_components=2,
            metric=True,
            n_init=10,
            max_iter=2000,
            verbose=False,
            eps=1e-9,
            dissimilarity="precomputed",
            n_jobs=-1,
            random_state=42,
        )

        # Configure plotting a bit differently for AA and codon rainbow plots
        group_labels = {"aa": ACIDS, "codon": [aac[0] for aac in AA_CODONS]}
        point_labels = {"aa": list(ACIDS_1TO1AND3.values()), "codon": AA_CODONS}
        with_legend = {"aa": False, "codon": True}
        std_scales = {"aa": 1.0, "codon": 1.0}

        for ss_type, aa_codon in it.product(SS_TYPES, cdists_pointwise.keys()):
            LOGGER.info(f"Plotting {aa_codon} pointwise MDS rainbows for {ss_type}...")
            mu_d2 = np.real(cdists_pointwise[aa_codon][ss_type][0])
            std_d2 = np.imag(cdists_pointwise[aa_codon][ss_type][0])

            # Compute correction for distance estimate
            N = mu_d2.shape[0]
            S = np.zeros((N, 1), dtype=np.float32)
            D = np.zeros((N, N), dtype=np.float32)
            for i in range(N):
                S[i] = np.sqrt(0.25 * mu_d2[i, i])
                for j in range(N):
                    D[i, j] = np.sqrt(
                        mu_d2[i, j] - 0.5 * mu_d2[i, i] - 0.5 * mu_d2[j, j]
                    )

            D *= scale
            S *= scale
            X = mds.fit_transform(D)

            # Plot
            rainbow_data = np.hstack((X, S, S))
            fig_file = self.out_dir.joinpath(
                f"rainbow-mds-{aa_codon}", f"{ss_type}.pdf"
            )
            pp5.plot.rainbow(
                rainbow_data,
                group_labels=group_labels[aa_codon],
                point_labels=point_labels[aa_codon],
                all_groups=ACIDS_1TO1AND3,
                title=ss_type,
                alpha=alpha,
                err_scale=std_scales[aa_codon],
                error_ellipse=True,
                normalize=True,
                with_groups_legend=with_legend[aa_codon],
                outfile=fig_file,
            )
