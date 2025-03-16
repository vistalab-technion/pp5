import os
import logging
from typing import Optional
from pathlib import Path
from datetime import datetime
from multiprocessing.pool import Pool

import numpy as np
import pandas as pd

import pp5
from pp5.collect import DATASET_DIRNAME, ALL_STRUCTS_FILENAME
from pp5.collect.base import (
    COL_UNP_ID,
    COL_RESOLUTION,
    COL_REJECTED_BY,
    BLAST_SCORES_FILENAME,
    FILTERED_STRUCTS_FILENAME,
    REJECTED_STRUCTS_FILENAME,
    ParallelDataCollector,
    read_df_csv,
    write_df_csv,
    normalize_csv,
    pairwise_align_unp,
    collect_single_structure,
)
from pp5.external_dbs import pdb_api
from pp5.external_dbs.pdb import PDB_RCSB

_LOG = logging.getLogger(__name__)


class ProteinRecordCollector(ParallelDataCollector):
    DEFAULT_PREC_INIT_ARGS = dict()

    def __init__(
        self,
        pdb_source: str = PDB_RCSB,
        resolution: float = pp5.get_config(pp5.CONFIG_DEFAULT_RES),
        r_free: Optional[float] = pp5.get_config(pp5.CONFIG_DEFAULT_RFREE),
        expr_sys: Optional[str] = pp5.get_config(pp5.CONFIG_DEFAULT_EXPR_SYS),
        source_taxid: Optional[int] = pp5.get_config(pp5.CONFIG_DEFAULT_SOURCE_TAXID),
        query_max_chains: Optional[int] = pp5.get_config(
            pp5.CONFIG_DEFAULT_QUERY_MAX_CHAINS
        ),
        seq_similarity_thresh: float = pp5.get_config(
            pp5.CONFIG_DEFAULT_SEQ_SIMILARITY_THRESH
        ),
        deposition_min_date: Optional[str] = None,
        deposition_max_date: Optional[str] = None,
        prec_init_args=None,
        with_backbone: bool = False,
        with_contacts: bool = False,
        with_altlocs: bool = False,
        entity_single_chain: bool = False,
        out_dir: Path = pp5.out_subdir("prec-collected"),
        out_tag: Optional[str] = None,
        write_zip: bool = False,
        async_timeout: Optional[float] = 600,
        async_retry_delta: float = 1.0,
    ):
        """
        Collects ProteinRecords based on a PDB query results, and saves them
        in the local cache folder. Optionally also writes them to CSV.

        :param pdb_source: Source from which to obtain the pdb file.
        :param resolution: Resolution cutoff value in Angstroms.
        :param expr_sys: Expression system name.
        :param source_taxid: Taxonomy ID of source organism.
        :param query_max_chains: Limits collected structures only to those with no more than
        this number of chains.
        :param seq_similarity_thresh: PDB sequence similarity threshold. This is a
        fraction between 0 and 1.0 which represents the maximal percentage of
        similarity allowed between two collected structures. Use 1.0 to set no
        filter.
        :param deposition_min_date: A date in the format "YYYY-MM-DD" representing
        the minimum deposition date (inclusive).
        :param deposition_max_date: A date in the format "YYYY-MM-DD" representing
        the maximum deposition date (inclusive).
        :param out_dir: Output folder for collected metadata.
        :param out_tag: Extra tag to add to the output file names.
        :param prec_init_args: Arguments for initializing each ProteinRecord
        :param with_backbone: Whether to include a 'backbone' column which contain the
        backbone atom coordinates of each residue in the order N, CA, C, O.
        :param with_contacts: Whether to include tertiary contact features per residue.
        :param with_altlocs: Whether to include alternate locations (altlocs) in the
        output.
        :param entity_single_chain: Whether to collect only a single chain per entity.
        :param async_timeout: Total timeout for each async result. None means no limit.
        :param async_retry_delta: Number of seconds between each retry when
        waiting for an async result.
        """
        super().__init__(
            async_timeout=async_timeout,
            async_retry_delta=async_retry_delta,
            out_dir=out_dir,
            tag=out_tag,
            create_zip=write_zip,
            pdb_source=pdb_source,
        )
        if resolution is None:
            raise ValueError("Must specify resolution cutoff for collection")

        self.resolution = float(resolution)
        self.r_free = float(r_free) if r_free is not None else None
        self.expr_sys = str(expr_sys) if expr_sys else None
        self.source_taxid = (
            int(source_taxid) if (source_taxid not in (None, "")) else None
        )
        self.query_max_chains = int(query_max_chains) if query_max_chains else None
        self.deposition_min_date: Optional[datetime] = None
        self.deposition_max_date: Optional[datetime] = None

        ymd = "%Y-%m-%d"
        if deposition_min_date:
            self.deposition_min_date = datetime.strptime(deposition_min_date, ymd)
        if deposition_max_date:
            self.deposition_max_date = datetime.strptime(deposition_max_date, ymd)

        if not 0.0 < seq_similarity_thresh <= 1.0:
            raise ValueError("seq_similarity_thresh must be in (0, 1.0]")
        self.seq_similarity_thresh = seq_similarity_thresh

        queries = [pdb_api.PDBXRayResolutionQuery(resolution=self.resolution)]
        if self.r_free:
            queries.append(pdb_api.PDBRFreeQuery(rfree=self.r_free))
        if self.expr_sys:
            queries.append(pdb_api.PDBExpressionSystemQuery(expr_sys=self.expr_sys))
        if self.source_taxid:
            queries.append(
                pdb_api.PDBSourceTaxonomyIdQuery(taxonomy_id=self.source_taxid)
            )
        if self.query_max_chains:
            queries.append(
                pdb_api.PDBNumberOfChainsQuery(
                    n_chains=self.query_max_chains, comparison_operator="less_or_equal"
                )
            )
        if self.deposition_min_date or self.deposition_max_date:
            queries.append(
                pdb_api.PDBDepositionDateQuery(
                    min_date=self.deposition_min_date,
                    max_date=self.deposition_max_date,
                )
            )

        self.query = pdb_api.PDBCompositeQuery(
            *queries,
            logical_operator="and",
            return_type=pdb_api.PDBQuery.ReturnType.ENTITY,
            raise_on_error=False,
        )

        if prec_init_args:
            self.prec_init_args = prec_init_args
        else:
            self.prec_init_args = self.DEFAULT_PREC_INIT_ARGS

        self.with_backbone = with_backbone
        self.with_contacts = with_contacts
        self.with_altlocs = with_altlocs
        self.entity_single_chain = entity_single_chain

        # Unique output dir for this collection run
        self.prec_csv_out_dir = self.out_dir / DATASET_DIRNAME
        self.prec_csv_out_dir.mkdir(parents=True, exist_ok=True)

    def __repr__(self):
        return f"{self.__class__.__name__} query={self.query}"

    def _collection_functions(self):
        return {
            "Collect ProteinRecords": self._collect_precs,
            "Filter Collected": self._filter_collected,
            "Write dataset": self._write_dataset,
        }

    def _collect_precs(self, pool: Pool):
        meta = {}
        pdb_ids = self.query.execute()
        n_structs = len(pdb_ids)

        meta["query"] = str(self.query)
        meta["n_query_results"] = len(pdb_ids)
        _LOG.info(f"Got {n_structs} structures from PDB, collecting...")

        async_results = {}
        for i, pdb_id in enumerate(pdb_ids):
            args = (pdb_id, self.pdb_source, (i, n_structs))
            kwds = dict(
                csv_out_dir=self.prec_csv_out_dir,
                with_backbone=self.with_backbone,
                with_contacts=self.with_contacts,
                with_altlocs=self.with_altlocs,
                entity_single_chain=self.entity_single_chain,
                no_cache=True,
            )
            r = pool.apply_async(collect_single_structure, args, kwds)
            async_results[pdb_id] = r

        _, elapsed, pdb_id_metadata = self._handle_async_results(
            async_results, collect=True, flatten=True
        )

        # Create a dataframe containing metadata from the collected precs
        df_all = pd.DataFrame(pdb_id_metadata)
        n_collected = len(df_all)

        self._out_filepaths.append(
            write_df_csv(df_all, self.out_dir, ALL_STRUCTS_FILENAME)
        )

        meta["n_collected"] = n_collected
        _LOG.info(
            f"Done collecting: {n_collected}/{len(pdb_ids)} proteins collected "
            f"(elapsed={elapsed:.2f} seconds, "
            f"{len(pdb_ids) / elapsed:.1f} proteins/sec)."
        )
        return meta

    def _filter_collected(self, pool: Pool) -> dict:
        """
        Filters collected structures according to conditions on their metadata.
        """

        df_all: pd.DataFrame = read_df_csv(self.out_dir, ALL_STRUCTS_FILENAME)
        # A boolean series representing which structures to keep.
        filter_idx = pd.Series(data=[True] * len(df_all), index=df_all.index)
        rejected_counts = {"total": 0}
        rejected_idxs = {}

        def _update_rejected_counts(filter_name: str, idx: pd.Series):
            rejected_idx = ~idx
            n_rejected = rejected_idx.sum()
            rejected_counts["total"] += n_rejected
            if n_rejected:
                _LOG.info(
                    f"Filtered {n_rejected} structures with filter '{filter_name}'"
                )
            rejected_counts[filter_name] = n_rejected
            rejected_idxs[filter_name] = rejected_idx

        # Filter by metadata
        filter_idx_metadata = self._filter_metadata(pool, df_all)
        filter_idx &= filter_idx_metadata
        _update_rejected_counts("metadata", filter_idx_metadata)

        # Filter by sequence similarity
        filter_idx_redundant_unps = self._filter_redundant_unps(pool, df_all)
        filter_idx &= filter_idx_redundant_unps
        _update_rejected_counts("redundant_unps", filter_idx_redundant_unps)

        # Write the filtered structures
        df_filtered = df_all[filter_idx]
        self._out_filepaths.append(
            write_df_csv(df_filtered, self.out_dir, FILTERED_STRUCTS_FILENAME)
        )

        # Write the rejected structures and specify which filter rejected them
        df_rejected = df_all
        df_rejected[COL_REJECTED_BY] = ""
        for filter_name, rejected_idx in rejected_idxs.items():
            df_rejected.loc[rejected_idx, COL_REJECTED_BY] = filter_name
        df_rejected = df_rejected[~filter_idx]
        self._out_filepaths.append(
            write_df_csv(df_rejected, self.out_dir, REJECTED_STRUCTS_FILENAME)
        )

        return {
            "n_rejected": rejected_counts,
            "n_collected_filtered": len(df_filtered),
        }

    def _filter_metadata(self, pool: Pool, df_all: pd.DataFrame) -> pd.Series:
        # Even though we query by resolution, the metadata resolution is different
        # than what we can query on. Metadata shows resolution after refinement,
        # while the query is using data collection resolution.
        idx_filter = df_all[COL_RESOLUTION].astype(float) <= self.resolution
        return idx_filter

    def _filter_redundant_unps(self, pool: Pool, df_all: pd.DataFrame) -> pd.Series:
        if self.seq_similarity_thresh == 1.0:
            _LOG.info("Skipping sequence similarity filter...")
            return pd.Series(data=[True] * len(df_all), index=df_all.index)

        # Create a similarity matrix for pairs of structures (in terms of sequence)
        all_unp_ids = tuple(sorted(set(df_all[COL_UNP_ID])))
        n_unps = len(all_unp_ids)

        async_results = {}
        for i in range(n_unps):
            async_results[i] = pool.apply_async(
                pairwise_align_unp,
                kwds=dict(query_unp_id=all_unp_ids[i], target_unp_ids=all_unp_ids[i:]),
            )

        blast_matrix = np.full(
            shape=(n_unps, n_unps), fill_value=np.nan, dtype=np.float16
        )
        for i, scores in pp5.parallel.yield_async_results(async_results):
            # i is query idx, j is target idx
            for j, score in enumerate(scores, start=i):
                blast_matrix[i, j] = blast_matrix[j, i] = score

            if (i % 100) == 0 or (i == n_unps - 1):
                _LOG.info(f"Collected pairwise similarity scores ({i + 1}/{n_unps})")

        # Normalize each row and col by the self-similarity score of each structure
        # Result is that each entry S[i,j] contains the score between query i and target
        # j, where 1.0 is the maximal similarity, i.e. S[i,i] == 1.
        d = np.sqrt(np.diag(blast_matrix))
        blast_matrix /= d
        blast_matrix /= d.reshape((n_unps, 1))

        # Write the BLAST scores
        df_blast_scores = pd.DataFrame(
            data=blast_matrix, index=all_unp_ids, columns=all_unp_ids
        )
        self._out_filepaths.append(
            write_df_csv(
                df_blast_scores, self.out_dir, BLAST_SCORES_FILENAME, index=True
            )
        )

        # Filter out similar sequences
        selected = [0]
        unselected = set(range(1, n_unps))
        selected_min_similarities = [0.0]
        while len(selected) < n_unps:
            next_selected = None
            next_min_similarity_to_selected = np.inf

            # TODO: Vectorize this inner loop
            for j in unselected:
                # Find Max similarity from unselected j to selected i
                i = selected[np.argmax(blast_matrix[selected, j])]

                if blast_matrix[i, j] < next_min_similarity_to_selected:
                    # Min over the max similarities
                    next_selected = j
                    next_min_similarity_to_selected = blast_matrix[i, j]

            selected.append(next_selected)
            unselected.remove(next_selected)
            selected_min_similarities.append(next_min_similarity_to_selected)
            assert selected_min_similarities[-2] <= selected_min_similarities[-1]
        assert len(unselected) == 0

        # Filter based on the similarity threshold: keep only structures whose
        # similarity to each other is less than or equal to threshold
        idx_thresh = np.searchsorted(
            selected_min_similarities, self.seq_similarity_thresh, side="right"
        )
        selected_filtered = selected[:idx_thresh]
        filtered_unp_ids = [all_unp_ids[i] for i in selected_filtered]
        filtered_idx = df_all[COL_UNP_ID].isin(filtered_unp_ids)
        return filtered_idx

    def _write_dataset(self, pool: Pool) -> dict:
        all_csvs = tuple(self.prec_csv_out_dir.glob("*.csv"))
        n_pdb_ids = len(all_csvs)
        _LOG.info(f"Creating dataset file from {n_pdb_ids} precs...")

        # discover common columns
        _LOG.info(f"Normalizing columns...")
        common_columns_to_idx = {}
        for csv_path in all_csvs:
            with open(csv_path, "r", encoding="utf-8") as f:
                line = f.readline().strip()
            if not line:
                continue
            columns = line.split(",")
            for i, col in enumerate(columns):
                if col not in common_columns_to_idx:
                    common_columns_to_idx[col] = i + 1

        # give common columns a canonical order
        common_columns = tuple(
            col for col, i in sorted(common_columns_to_idx.items(), key=lambda x: x[1])
        )

        async_results = {}
        for i, csv_path in enumerate(all_csvs):
            kwds = dict(
                csv_path=csv_path,
                common_columns=common_columns,
                _seq=(i, len(all_csvs)),
            )
            r = pool.apply_async(normalize_csv, args=[], kwds=kwds)
            async_results[str(csv_path)] = r

        _, elapsed, n_rows_per_file = self._handle_async_results(
            async_results, collect=True, flatten=False
        )

        n_rows = sum(n_rows_per_file)
        dataset_size_mb = sum(os.path.getsize(p) for p in all_csvs) / 1024**2

        self._out_filepaths.extend(all_csvs)
        _LOG.info(f"Wrote {n_pdb_ids} files, ({n_rows=}, {dataset_size_mb:.2f}MB)")
        meta = {f"dataset_size_mb": f"{dataset_size_mb:.2f}", "n_entries": n_rows}
        return meta
