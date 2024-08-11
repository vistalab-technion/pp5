import os
import abc
import json
import time
import socket
import string
import logging
import zipfile
import warnings
import itertools
import multiprocessing as mp
from pprint import pformat
from typing import (
    Any,
    Set,
    Dict,
    List,
    Tuple,
    Union,
    Callable,
    Iterable,
    Optional,
    Sequence,
)
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from multiprocessing.pool import Pool, AsyncResult

import numpy as np
import pandas as pd
from Bio.Align import PairwiseAligner

import pp5
import pp5.parallel
from pp5.prec import ProteinRecord
from pp5.align import ProteinBLAST
from pp5.cache import ReprJSONEncoder
from pp5.utils import ProteinInitError, elapsed_seconds_to_dhms
from pp5.pgroup import ProteinGroup
from pp5.external_dbs import pdb, unp, pdb_api
from pp5.external_dbs.pdb import PDB_RCSB
from pp5.external_dbs.unp import unp_record

with warnings.catch_warnings():
    # Due to dask warning about pyarrow version - should update pyarrow and remove
    warnings.simplefilter(action="ignore", category=FutureWarning)
    import dask.dataframe as dd
    from dask.distributed import Client

LOGGER = logging.getLogger(__name__)

COL_UNP_ID = "unp_id"
COL_PDB_ID = "pdb_id"
COL_ENA_ID = "ena_id"
COL_RESOLUTION = "resolution"
COL_SEQ_LEN = "seq_len"
COL_SEQ_GAPS = "seq_gaps"
COL_DESCRIPTION = "description"
COL_DEPOSITION_DATE = "deposition_date"
COL_SRC_ORG = "src_org"
COL_HOST_ORG = "host_org"
COL_LIGANDS = "ligands"
COL_R_FREE = "r_free"
COL_R_WORK = "r_work"
COL_SPACE_GROUP = "space_group"
COL_CG_PH = "cg_ph"
COL_CG_TEMP = "cg_temp"
COL_PDB_SOURCE = "pdb_source"
COL_REJECTED_BY = "rejected_by"
COL_NUM_ALTLOCS = "num_altlocs"
COL_NUM_UNMODELLED = "num_unmodelled"

COLLECTION_METADATA_FILENAME = "meta.json"
ALL_STRUCTS_FILENAME = "meta-structs_all"
FILTERED_STRUCTS_FILENAME = "meta-structs_filtered"
REJECTED_STRUCTS_FILENAME = "meta-structs_rejected"
BLAST_SCORES_FILENAME = "meta-blast_scores"
DATASET_DIRNAME = "data-precs"


@dataclass(repr=False)
class CollectorStep:
    name: str
    elapsed: str
    result: str
    message: str

    def __repr__(self):
        return (
            f"{self.name}: completed in {self.elapsed} "
            f"result={self.result}"
            f'{f": {self.message}" if self.message else ""}'
        )


class ParallelDataCollector(abc.ABC):
    def __init__(
        self,
        id: str = None,
        out_dir: Path = None,
        tag: str = None,
        async_timeout: Optional[float] = None,
        async_retry_delta: float = 1.0,
        create_zip=True,
        pdb_source: str = PDB_RCSB,
    ):
        """
        :param id: Unique id of this collection. If None, will be generated
        from timestamp and hostname.
        :param out_dir: Output directory, if necessary. A subdirectory with
        a unique id will be created within foe this collection.
        :param tag: Tag (postfix) for unique id of output subdir.
        :param async_timeout: Total time to wait for each async result.
        None means no limit.
        :param async_retry_delta: Number of seconds between each retry when
        waiting for an async result.
        Number of tries will be async_timeout / async_retry_delta, so that the total
        wait time per result will be async_timeout.
        :param create_zip: Whether to create a zip file with all the result
        files.
        :param pdb_source: Source from which to obtain the pdb file.
        """
        hostname = socket.gethostname()
        if hostname:
            hostname = hostname.split(".")[0].strip()
        else:
            hostname = "localhost"

        # Timeout, if set, must be greater than retry delta
        assert async_timeout is None or (async_timeout > async_retry_delta)

        self.hostname = hostname
        self.out_tag = tag
        self.async_timeout = async_timeout
        self.async_retry_delta = async_retry_delta
        self.create_zip = create_zip
        self.pdb_source = pdb_source

        if id:
            self.id = id
        else:
            tag = f"-{self.out_tag}" if self.out_tag else ""
            timestamp = time.strftime(f"%Y%m%d_%H%M%S")
            self.id = time.strftime(f"{timestamp}-{hostname}{tag}")

        if out_dir is not None:
            out_dir = out_dir.joinpath(self.id)
            os.makedirs(str(out_dir), exist_ok=True)

        self.out_dir = out_dir

        self._collection_steps: List[CollectorStep] = []
        self._out_filepaths: List[Path] = []
        self._collection_meta = {"id": self.id}

    def collect(self) -> dict:
        """
        Performs all collection steps defined in this collector.
        :return: Collection metadata.
        """
        start_time = time.time()
        # Note: in python 3.7+ dict order is guaranteed to be insertion order
        collection_functions = self._collection_functions()
        collection_functions["Finalize"] = self._finalize_collection

        # Update metadata with configuration of this collector
        self._collection_meta.update(self._get_collection_config())

        for step_name, collect_fn in collection_functions.items():
            step_start_time = time.time()

            step_status, step_message = "RUNNING", None
            with pp5.parallel.global_pool() as pool:
                try:
                    step_meta = collect_fn(pool)
                    if step_meta:
                        self._collection_meta.update(step_meta)
                    step_status, step_message = "SUCCESS", None
                except Exception as e:
                    LOGGER.error(
                        f"Unexpected exception in top-level collect", exc_info=e
                    )
                    step_status, step_message = "FAIL", f"{e}"
                    break  # don't move on to the next step
                finally:
                    step_elapsed = time.time() - step_start_time
                    step_elapsed = elapsed_seconds_to_dhms(step_elapsed)
                    self._collection_steps.append(
                        CollectorStep(
                            step_name, step_elapsed, step_status, step_message
                        )
                    )

        end_time = time.time()
        time_str = elapsed_seconds_to_dhms(end_time - start_time)

        LOGGER.info(f"Completed collection for {self} in {time_str}")
        collection_meta_formatted = pformat(
            self._collection_meta,
            width=120,
            compact=True,
        )
        LOGGER.info(f"Collection metadata:\n" f"{collection_meta_formatted}")
        return self._collection_meta

    def _finalize_collection(self, pool: Pool):
        LOGGER.info(f"Finalizing collection for {self.id}...")
        if self.out_dir is None:
            return

        # Create a metadata file in the output dir based on the step results
        meta_filepath = self.out_dir.joinpath(COLLECTION_METADATA_FILENAME)
        meta = self._collection_meta
        meta["steps"] = [str(s) for s in self._collection_steps]
        with open(str(meta_filepath), "w", encoding="utf-8") as f:
            try:
                json.dump(meta, f, indent=2, cls=ReprJSONEncoder)
            except Exception as e:
                LOGGER.error(f"Failed to serialize metadata", exc_info=e)

        self._out_filepaths.append(meta_filepath)

        # Create a zip of the results
        if not self.create_zip:
            return

        zip_filename = Path(f"{self.id}.zip")
        zip_filepath = self.out_dir.joinpath(zip_filename)

        LOGGER.info(f"Compressing results into {zip_filename!s}...")
        with zipfile.ZipFile(
            zip_filepath, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
        ) as z:
            for out_filepath in self._out_filepaths:
                rel_out_filepath = out_filepath.relative_to(self.out_dir)
                arcpath = f"{zip_filename.stem}/{rel_out_filepath!s}"
                z.write(str(out_filepath), arcpath)

        zipsize_mb = os.path.getsize(str(zip_filepath)) / 1024 / 1024
        LOGGER.info(f"Wrote archive {zip_filepath} ({zipsize_mb:.2f}MB)")

    def _handle_async_results(
        self,
        async_results: Union[Dict[str, AsyncResult], List[AsyncResult]],
        collect: bool = False,
        flatten: bool = False,
        allow_none: bool = False,
        result_callback: Callable = None,
    ):
        """
        Handles a list of AsyncResult objects.

        :param async_results: List of async results, or a dict with the keys being
        identifiers.
        :param collect: Whether to add all obtained results to a list and
        return it.
        :param flatten: Whether to flatten results (useful i.e. if each
        result is a list or tuple).
        :param allow_none: Whether to allow None results.
        :param result_callback: Callable to invoke on each result.
        :return: Number of handled results, time elapsed in seconds, list of
        collected results (will be empty if collect is False).
        """
        count, start_time = 0, time.time()
        collected_results = []

        for res_id, res in pp5.parallel.yield_async_results(
            async_results,
            wait_time_sec=self.async_retry_delta,
            max_retries=int(self.async_timeout / self.async_retry_delta)
            if self.async_timeout is not None
            else None,
            re_raise=False,
        ):
            if result_callback is not None:
                result_callback(res)

            if not collect:
                continue

            if res is None and not allow_none:
                continue

            if flatten and isinstance(res, Iterable):
                collected_results.extend(res)
            else:
                collected_results.append(res)

        elapsed_time = time.time() - start_time
        return count, elapsed_time, collected_results

    @abc.abstractmethod
    def _collection_functions(
        self,
    ) -> Dict[str, Callable[[mp.pool.Pool], Optional[Dict]]]:
        """
        Defines the steps of the collection as a sequence of functions to
        call in order.
        Each collection function can return a dict with metadata about the
        collection. This metadata will be saved at the end of collection.
        :return: Dict mapping step name to a functions to call during collect.
        """
        return {}

    def _get_collection_config(self):
        cfg = {}
        for k, v in self.__dict__.items():
            # Ignore attributes marked with '_'
            if k.startswith("_"):
                continue

            # Convert paths to string for serialization
            if isinstance(v, (Path,)):
                cfg[k] = str(v)

            else:
                cfg[k] = v

        return cfg

    def __repr__(self):
        return f"{self.__class__.__name__} id={self.id}"


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
        LOGGER.info(f"Got {n_structs} structures from PDB, collecting...")

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
            r = pool.apply_async(_collect_single_structure, args, kwds)
            async_results[pdb_id] = r

        _, elapsed, pdb_id_metadata = self._handle_async_results(
            async_results, collect=True, flatten=True
        )

        # Create a dataframe containing metadata from the collected precs
        df_all = pd.DataFrame(pdb_id_metadata)
        n_collected = len(df_all)

        self._out_filepaths.append(
            _write_df_csv(df_all, self.out_dir, ALL_STRUCTS_FILENAME)
        )

        meta["n_collected"] = n_collected
        LOGGER.info(
            f"Done collecting: {n_collected}/{len(pdb_ids)} proteins collected "
            f"(elapsed={elapsed:.2f} seconds, "
            f"{len(pdb_ids) / elapsed:.1f} proteins/sec)."
        )
        return meta

    def _filter_collected(self, pool: Pool) -> dict:
        """
        Filters collected structures according to conditions on their metadata.
        """

        df_all: pd.DataFrame = _read_df_csv(self.out_dir, ALL_STRUCTS_FILENAME)
        # A boolean series representing which structures to keep.
        filter_idx = pd.Series(data=[True] * len(df_all), index=df_all.index)
        rejected_counts = {"total": 0}
        rejected_idxs = {}

        def _update_rejected_counts(filter_name: str, idx: pd.Series):
            rejected_idx = ~idx
            n_rejected = rejected_idx.sum()
            rejected_counts["total"] += n_rejected
            if n_rejected:
                LOGGER.info(
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
            _write_df_csv(df_filtered, self.out_dir, FILTERED_STRUCTS_FILENAME)
        )

        # Write the rejected structures and specify which filter rejected them
        df_rejected = df_all
        df_rejected[COL_REJECTED_BY] = ""
        for filter_name, rejected_idx in rejected_idxs.items():
            df_rejected.loc[rejected_idx, COL_REJECTED_BY] = filter_name
        df_rejected = df_rejected[~filter_idx]
        self._out_filepaths.append(
            _write_df_csv(df_rejected, self.out_dir, REJECTED_STRUCTS_FILENAME)
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
            LOGGER.info("Skipping sequence similarity filter...")
            return pd.Series(data=[True] * len(df_all), index=df_all.index)

        # Create a similarity matrix for pairs of structures (in terms of sequence)
        all_unp_ids = tuple(sorted(set(df_all[COL_UNP_ID])))
        n_unps = len(all_unp_ids)

        async_results = {}
        for i in range(n_unps):
            async_results[i] = pool.apply_async(
                _pairwise_align_unp,
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
                LOGGER.info(f"Collected pairwise similarity scores ({i + 1}/{n_unps})")

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
            _write_df_csv(
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
        LOGGER.info(f"Creating dataset file from {n_pdb_ids} precs...")

        # discover common columns
        LOGGER.info(f"Normalizing columns...")
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
            r = pool.apply_async(_normalize_csv, args=[], kwds=kwds)
            async_results[str(csv_path)] = r

        _, elapsed, n_rows_per_file = self._handle_async_results(
            async_results, collect=True, flatten=False
        )

        n_rows = sum(n_rows_per_file)
        dataset_size_mb = sum(os.path.getsize(p) for p in all_csvs) / 1024**2

        self._out_filepaths.extend(all_csvs)
        LOGGER.info(f"Wrote {n_pdb_ids} files, ({n_rows=}, {dataset_size_mb:.2f}MB)")
        meta = {f"dataset_size_mb": f"{dataset_size_mb:.2f}", "n_entries": n_rows}
        return meta


class ProteinGroupCollector(ParallelDataCollector):
    def __init__(
        self,
        resolution: float,
        pdb_source: str = PDB_RCSB,
        expr_sys: str = pp5.get_config("DEFAULT_EXPR_SYS"),
        source_taxid: int = pp5.get_config("DEFAULT_SOURCE_TAXID"),
        evalue_cutoff: float = 1.0,
        identity_cutoff: float = 30.0,
        b_max: float = 50.0,
        plddt_min: float = 70.0,
        sa_outlier_cutoff: float = 2.0,
        angle_aggregation: str = "circ",
        match_len: int = 2,
        context_len: int = 1,
        compare_contacts: bool = True,
        strict_codons: bool = True,
        out_dir=pp5.out_subdir("pgroup-collected"),
        pgroup_out_dir=pp5.out_subdir("pgroup"),
        write_pgroup_csvs=True,
        out_tag: str = None,
        ref_file: str = None,
        create_zip=True,
        async_timeout: Optional[float] = 3600,
        async_retry_delta: float = 1.0,
    ):
        """
        Collects ProteinGroup reference structures based on a PDB query
        results.

        :param resolution: Resolution cutoff value in Angstroms.
        :param pdb_source: Source from which to obtain the pdb file.
        :param expr_sys: Expression system name.
        :param source_taxid: Taxonomy ID of source organism.
        :param evalue_cutoff: Maximal expectation value allowed for BLAST
        matches when searching for proteins to include in pgroups.
        :param identity_cutoff: Minimal percent sequence identity
        allowed for BLAST matches when searching for proteins to include in
        pgroups.
        :param b_max: Maximal b-factor a residue can have
        (backbone-atom average) in order for it to be included in a match
        group. Only relevant if pdb_source is not af (alphafold).
        :param plddt_min: Minimal pLDDT value a residue can have in order for it to
        be included in a match. Only relevant if pdb_source is af (alphafold).
        :param sa_outlier_cutoff: RMS cutoff for determining outliers in
        structural alignment.
        :param angle_aggregation: Method for angle-aggregation of matching
        query residues of each reference residue. Options are
        'circ' - Circular mean;
        'frechet' - Frechet centroid;
        'max_res' - No aggregation, take angle of maximal resolution structure
        :param match_len: Number of residues to include in a match. Can be either 1
        or 2. If 2, the match dihedral angles will be the cross-bond angles (phi+1,
        psi+0) between the two residues.
        :param context_len: Number of stars required around an aligned AA
        pair to consider that pair for a match.
        :param compare_contacts: Whether to compare tertiary contacts contexts of
        potential matches.
        :param strict_codons: Whether to require that a codon assignment for each
        AA exists and is un-ambiguous.
        :param out_dir: Output directory for collection CSV files.
        :param pgroup_out_dir: Output directory for pgroup CSV files. Only
        relevant if write_pgroup_csvs is True.
        :param write_pgroup_csvs: Whether to write each pgroup's CSV files.
        Even if false, the collection files will still be writen.
        :param out_tag: Extra tag to add to the output file names.
        :param ref_file: Path of collector CSV file with references.
        Allows to skip the first and second collection steps (finding PDB
        IDs for the reference structures) and immediately collect
        ProteinGroups for the references in the file.
        :param async_timeout: Total timeout for each async result. None means no limit.
        :param async_retry_delta: Number of seconds between each retry when
        waiting for an async result.
        :param create_zip: Whether to create a zip file with all output files.
        """
        super().__init__(
            out_dir=out_dir,
            tag=out_tag,
            async_timeout=async_timeout,
            async_retry_delta=async_retry_delta,
            create_zip=create_zip,
            pdb_source=pdb_source,
        )

        self.resolution = float(resolution)
        self.expr_sys = expr_sys
        self.source_taxid = int(source_taxid) if source_taxid else None
        queries = [pdb_api.PDBXRayResolutionQuery(resolution=self.resolution)]
        if self.expr_sys:
            queries.append(pdb_api.PDBExpressionSystemQuery(expr_sys=self.expr_sys))
        if self.source_taxid:
            queries.append(
                pdb_api.PDBSourceTaxonomyIdQuery(taxonomy_id=self.source_taxid)
            )
        self.query = pdb_api.PDBCompositeQuery(
            *queries,
            logical_operator="and",
            return_type=pdb_api.PDBQuery.ReturnType.ENTITY,
            raise_on_error=False,
        )

        self.evalue_cutoff = evalue_cutoff
        self.identity_cutoff = identity_cutoff
        self.b_max = b_max
        self.plddt_min = plddt_min
        self.sa_outlier_cutoff = sa_outlier_cutoff
        self.angle_aggregation = angle_aggregation
        self.match_len = match_len
        self.context_len = context_len
        self.strict_codons = strict_codons
        self.compare_contacts = compare_contacts

        self.pgroup_out_dir = pgroup_out_dir
        self.write_pgroup_csvs = write_pgroup_csvs
        self.out_tag = out_tag

        self._df_all = None  # Metadata about all structures
        self._df_ref = None  # Metadata about collected reference structures
        self._df_pgroups = None  # Metadata for each pgroup
        self._df_pairwise = None  # Pairwise matches from all pgroups
        self._df_pointwise = None  # Pointwise matches from all pgroups

        if ref_file is None:
            self._all_file = None
            self._ref_file = None
        else:
            all_file = Path(str(ref_file).replace("ref", "all", 1))
            ref_file = Path(ref_file)
            if not all_file.is_file() or not ref_file.is_file():
                raise ValueError(
                    f"To skip the first two collection steps "
                    f"both collection files must exist:"
                    f"{all_file}, {ref_file}"
                )

            # Save path to skip first two collection steps
            self._all_file = all_file
            self._ref_file = ref_file

    def _collection_functions(self):
        return {
            "Collect precs": self._collect_all_structures,
            "Find references": self._collect_all_refs,
            "Collect pgroups": self._collect_all_pgroups,
        }

    def _collect_all_structures(self, pool: Pool):
        meta = {}

        if self._all_file:
            LOGGER.info(
                f"Skipping all-structure collection step: loading {self._all_file}"
            )
            read_csv_args = dict(comment="#", index_col=None, header=0)
            self._df_all = pd.read_csv(self._all_file, **read_csv_args)
            meta["init_from_all_file"] = str(self._all_file)
        else:
            # Execute PDB query to get a list of PDB IDs
            pdb_ids = self.query.execute()
            n_structs = len(pdb_ids)
            LOGGER.info(f"Got {n_structs} structure ids from PDB, collecting...")

            meta["query"] = str(self.query)
            meta["n_query_results"] = len(pdb_ids)

            async_results = {}
            for i, pdb_id in enumerate(pdb_ids):
                args = (pdb_id, self.pdb_source, (i, n_structs))
                r = pool.apply_async(_collect_single_structure, args=args)
                async_results[pdb_id] = r

            count, elapsed, pdb_id_data = self._handle_async_results(
                async_results,
                collect=True,
                flatten=True,
            )

            # Create a dataframe from the collected data
            self._df_all = pd.DataFrame(pdb_id_data)
            if len(self._df_all):
                self._df_all.sort_values(
                    by=["unp_id", "resolution"], inplace=True, ignore_index=True
                )

        # Even though we query by resolution, the metadata resolution is different
        # than what we can query on. Metadata shows resolution after refinement,
        # while the query is using data collection resolution.
        idx_filter = (
            self._df_all[COL_RESOLUTION].astype(float) <= self.resolution + 0.05
        )
        self._df_all = self._df_all[idx_filter]

        filepath = _write_df_csv(self._df_all, self.out_dir, "meta-structs_all")
        self._out_filepaths.append(filepath)

        meta["n_all_structures"] = len(self._df_all)
        return meta

    def _collect_all_refs(self, pool: Pool):
        meta = {}

        if self._ref_file:
            LOGGER.info(
                f"Skipping reference-structure collection step: "
                f"loading {self._ref_file}"
            )
            read_csv_args = dict(comment="#", index_col=None, header=0)
            self._df_ref = pd.read_csv(self._ref_file, **read_csv_args)
            meta["init_from_ref_file"] = str(self._all_file)
        else:
            # Find reference structure
            LOGGER.info(f"Finding reference structures...")
            groups = self._df_all.groupby("unp_id")

            async_results = {}
            for unp_id, df_group in groups:
                args = (unp_id, df_group)
                r = pool.apply_async(_collect_single_ref, args=args)
                async_results[unp_id] = r

            count, elapsed, group_datas = self._handle_async_results(
                async_results,
                collect=True,
            )
            group_datas = filter(None, group_datas)

            self._df_ref = pd.DataFrame(group_datas)
            if len(self._df_ref):
                self._df_ref.sort_values(
                    by=["group_size", "group_median_res"],
                    ascending=[False, True],
                    inplace=True,
                    ignore_index=True,
                )

        meta["n_ref_structures"] = len(self._df_ref)
        filepath = _write_df_csv(self._df_ref, self.out_dir, "meta-structs_ref")
        self._out_filepaths.append(filepath)
        return meta

    def _collect_all_pgroups(self, pool: Pool):
        meta = {}

        # Initialize a local BLAST DB.
        blast = ProteinBLAST(
            evalue_cutoff=self.evalue_cutoff,
            identity_cutoff=self.identity_cutoff,
            db_autoupdate_days=7,
        )

        LOGGER.info(f"Creating ProteinGroup for each reference...")
        ref_pdb_ids = self._df_ref["ref_pdb_id"].values
        async_results = {}
        all_pdb_ids = set(self._df_all["pdb_id"])
        for i, ref_pdb_id in enumerate(ref_pdb_ids):
            idx = (i, len(ref_pdb_ids))
            pgroup_out_dir = self.pgroup_out_dir if self.write_pgroup_csvs else None
            args = (
                ref_pdb_id,
                all_pdb_ids,
                self.pdb_source,
                blast,
                self.b_max,
                self.plddt_min,
                self.sa_outlier_cutoff,
                self.angle_aggregation,
                self.match_len,
                self.context_len,
                self.compare_contacts,
                self.strict_codons,
                pgroup_out_dir,
                self.out_tag,
                idx,
            )
            r = pool.apply_async(_collect_single_pgroup, args=args)
            async_results[ref_pdb_id] = r

        count, elapsed, collected_data = self._handle_async_results(
            async_results,
            collect=True,
            flatten=False,
        )

        # The pgroup_datas contains both metadata and also pairwise matches.
        # We need to write these things to different output files.
        pgroup_datas = []
        pairwise_dfs: List[pd.DataFrame] = []
        pointwise_dfs: List[pd.DataFrame] = []
        for pgroup_data in collected_data:
            if pgroup_data is None:
                continue

            # Save the pairwise and pointwise data from each pgroup.
            df_pairwise = pgroup_data.pop("pgroup_pairwise")
            pairwise_dfs.append(df_pairwise)
            df_pointwise = pgroup_data.pop("pgroup_pointwise")
            pointwise_dfs.append(df_pointwise)

            pgroup_datas.append(pgroup_data)

        # Create pgroup metadata dataframe
        self._df_pgroups = pd.DataFrame(pgroup_datas)
        if len(self._df_pgroups):
            self._df_pgroups.sort_values(
                by=["n_unp_ids", "n_total_matches"],
                ascending=False,
                inplace=True,
                ignore_index=True,
            )

        # Sum the counter columns into the collection step metadata
        meta["n_pgroups"] = len(self._df_pgroups)
        for c in [c for c in self._df_pgroups.columns if c.startswith("n_")]:
            meta[c] = int(self._df_pgroups[c].sum())  # converts from np.int64

        filepath = _write_df_csv(self._df_pgroups, self.out_dir, "meta-pgroups")
        self._out_filepaths.append(filepath)

        # Create the pairwise matches dataframe
        self._df_pairwise = pd.concat(pairwise_dfs, axis=0).reset_index()
        if len(self._df_pairwise):
            self._df_pairwise.sort_values(
                by=["ref_unp_id", "ref_idx", "type"], inplace=True, ignore_index=True
            )
        filepath = _write_df_csv(self._df_pairwise, self.out_dir, "data-pairwise")
        self._out_filepaths.append(filepath)

        # Create the pointwise matches dataframe
        self._df_pointwise = pd.concat(pointwise_dfs, axis=0).reset_index()
        if len(self._df_pointwise):
            self._df_pointwise.sort_values(
                by=["unp_id", "ref_idx"], inplace=True, ignore_index=True
            )
        filepath = _write_df_csv(self._df_pointwise, self.out_dir, "data-pointwise")
        self._out_filepaths.append(filepath)

        return meta


def _collect_single_structure(
    pdb_id: str,
    pdb_source: str,
    idx: tuple,
    csv_out_dir: Optional[Path] = None,
    csv_tag: str = None,
    with_backbone: bool = False,
    with_contacts: bool = False,
    with_altlocs: bool = False,
    entity_single_chain: bool = False,
    no_cache: bool = False,
) -> List[dict]:
    """
    Downloads a single PDB entry, and creates a prec for all its chains.

    :param pdb_id: The PDB id to download.
    :param pdb_source: Source from which to obtain the pdb file.
    :param idx: Index for logging; should be a tuple of (current, total).
    :param csv_out_dir: If provided, the prec of each chain will be written to CSV at
        this path.
    :param csv_tag: Tag to add to the output CSVs.
    :param with_backbone: Whether to write backbone atom locations to csv.
    :param with_contacts: Whether to run Arpeggio so that it's results are cached
    locally and also write contact features to csv. The dict contains kwargs for
    arpeggio.
    :param with_altlocs: Whether to include alternate locations in the prec.
    :param entity_single_chain: Whether to collect only a single chain from each
    unique entity. If True, the first chain will be collected from each entity.
    Otherwise, all chains will be collected.
    :param no_cache: If True, will not save precs to cache (only makes sense if
    csv_out_dir is not None)
    :return: A list of dicts, each containing metadata about one of the collected
        chains.
    """
    pdb_base_id, chain_id, entity_id = pdb.split_id_with_entity(pdb_id)

    pdb_dict = pdb.pdb_dict(pdb_id, pdb_source=pdb_source)
    meta = pdb.PDBMetadata.from_pdb(pdb_id, cache=True)
    chain_to_unp_ids = meta.chain_uniprot_ids

    # Determine all chains we need to collect from the PDB structure
    chains_to_collect: Sequence[str]
    if chain_id is not None:
        # If we got a single chain, use only that
        chains_to_collect = (chain_id,)

    elif entity_id is not None:
        # If we got an entity id, discover all corresponding chains
        chains_to_collect = tuple(
            chain_id
            for chain_id, chain_entity_id in meta.chain_entities.items()
            if entity_id == chain_entity_id
        )

        if entity_single_chain:
            chains_to_collect = (chains_to_collect[0],)
    else:
        # Otherwise, we have no entity id or chain id; we'll take chains from all
        # available entities.
        chains_to_collect = tuple(
            itertools.chain(
                *[
                    (chain_ids[0],) if entity_single_chain else chain_ids
                    for entity_id, chain_ids in meta.entity_chains.items()
                ]
            )
        )

    chain_data = []
    for chain_id in chains_to_collect:
        pdb_id_full = f"{pdb_base_id}:{chain_id}"
        entity_id = meta.chain_entities[chain_id]
        seq_len = len(meta.entity_sequence[entity_id])

        # Skip chains with no Uniprot ID
        if chain_id not in chain_to_unp_ids or not chain_to_unp_ids[chain_id]:
            LOGGER.warning(f"No Uniprot ID for {pdb_id_full}")
            continue

        # Skip chimeric chains
        if len(chain_to_unp_ids[chain_id]) > 1:
            LOGGER.warning(f"Discarding chimeric chain {pdb_id_full}")
            continue

        unp_id = chain_to_unp_ids[chain_id][0]

        # Create a ProteinRecord and save it so it's cached for when we
        # create the pgroups. Only collect structures for which we can
        # create a prec (e.g. they must have a DNA sequence).
        try:
            prec = ProteinRecord(
                pdb_id_full,
                pdb_source=pdb_source,
                pdb_dict=pdb_dict,
                with_altlocs=with_altlocs,
                with_backbone=with_backbone,
                with_contacts=with_contacts,
            )

            # Save into cache
            if not no_cache:
                prec.save()

            # Write CSV if requested
            if csv_out_dir is not None:
                prec.to_csv(csv_out_dir, tag=csv_tag)

        except Exception as e:
            LOGGER.warning(
                f"Failed to create ProteinRecord for {pdb_id} ({unp_id=}), "
                f"will not collect: {e}"
            )
            continue

        chain_data.append(
            {
                COL_UNP_ID: prec.unp_id,
                COL_PDB_ID: prec.pdb_id,
                COL_ENA_ID: prec.ena_id,
                COL_SEQ_LEN: seq_len,
                COL_SEQ_GAPS: str.join(";", [f"{s}-{e}" for (s, e) in prec.seq_gaps]),
                COL_NUM_ALTLOCS: prec.num_altlocs,
                **{
                    f"{COL_NUM_UNMODELLED}_{suffix}": n
                    for n, suffix in zip(
                        prec.num_unmodelled, ["nterm", "inter", "cterm"]
                    )
                },
                COL_PDB_SOURCE: pdb_source,
                **meta.as_dict(chain_id=chain_id, seq_to_str=True),
            }
        )

    msg = (
        f"Collected {len(chain_data)} chains from {pdb_id} "
        f"{chain_to_unp_ids} ({idx[0] + 1}/{idx[1]})"
    )
    LOGGER.log(level=logging.INFO if len(chain_data) else logging.WARNING, msg=msg)

    return chain_data


def _collect_single_ref(group_unp_id: str, df_group: pd.DataFrame) -> Optional[dict]:
    try:
        unp_rec = unp.unp_record(group_unp_id)
        unp_seq_len = len(unp_rec.sequence)
    except ValueError as e:
        pdb_ids = tuple(df_group["pdb_id"])
        LOGGER.error(
            f"Failed create Uniprot record for {group_unp_id=} {pdb_ids=}: {e}"
        )
        return None

    median_res = df_group["resolution"].median()
    group_size = len(df_group)
    df_group = df_group.sort_values(by=["resolution"])
    df_group["seq_ratio"] = df_group["seq_len"] / unp_seq_len

    # Keep only structures which have at least 90% of residues as
    # the Uniprot sequence, and not too many extras.
    filter_idx = (df_group["seq_ratio"] >= 0.9) & (df_group["seq_ratio"] <= 1.1)
    if filter_idx.sum() == 0:
        LOGGER.error(
            f"Failed to find reference structure for {group_unp_id=} {group_size=} "
            f"({df_group['seq_ratio'].min():.2f}, {df_group['seq_ratio'].max():.2f})"
        )
        return None

    df_group = df_group[filter_idx]

    ref_pdb_id = df_group.iloc[0]["pdb_id"]
    ref_res = df_group.iloc[0]["resolution"]
    ref_seq_ratio = df_group.iloc[0]["seq_ratio"]

    return dict(
        unp_id=group_unp_id,
        unp_name=unp_rec.entry_name,
        ref_pdb_id=ref_pdb_id,
        ref_res=ref_res,
        ref_seq_ratio=ref_seq_ratio,
        group_median_res=median_res,
        group_size=group_size,
    )


def _pairwise_align_unp(
    query_unp_id: str, target_unp_ids: Sequence[str]
) -> Sequence[float]:
    aligner = PairwiseAligner()
    query_seq = unp_record(query_unp_id).sequence

    scores = []
    for target_unp_id in target_unp_ids:
        target_seq = unp_record(target_unp_id).sequence
        alignment = aligner.align(query_seq, target_seq)
        scores.append(alignment.score)

    return scores


def _collect_single_pgroup(
    ref_pdb_id: str,
    all_pdb_ids: Set[str],
    pdb_source: str,
    blast: ProteinBLAST,
    b_max: float,
    plddt_min: float,
    sa_outlier_cutoff: float,
    angle_aggregation: str,
    match_len: int,
    context_len: int,
    compare_contacts: bool,
    strict_codons: bool,
    out_dir: Optional[Path],
    out_tag: str,
    idx: tuple,
) -> Optional[dict]:
    try:
        LOGGER.info(
            f"Creating ProteinGroup for {ref_pdb_id}, {b_max=}/{plddt_min=} "
            f"({idx[0] + 1}/{idx[1]})"
        )

        # Run BLAST to find query structures for the pgroup
        df_blast = blast.pdb(ref_pdb_id)

        # Only use query PDB ids that are part of the collected structure dataset.
        query_pdb_ids = sorted(set(df_blast.index) & all_pdb_ids)
        LOGGER.info(
            f"Got {len(df_blast)} BLAST hits for {ref_pdb_id}, of which "
            f"{len(query_pdb_ids)} query structures"
        )

        if not query_pdb_ids:
            LOGGER.info(f"No query structures for {ref_pdb_id}, skipping...")
            return None

        # Create a pgroup without an additional query, by specifying the
        # exact ids of the query structures.
        pgroup = ProteinGroup.from_query_ids(
            ref_pdb_id,
            pdb_source=pdb_source,
            query_pdb_ids=query_pdb_ids,
            b_max=b_max,
            plddt_min=plddt_min,
            sa_outlier_cutoff=sa_outlier_cutoff,
            angle_aggregation=angle_aggregation,
            match_len=match_len,
            context_len=context_len,
            compare_contacts=compare_contacts,
            strict_codons=strict_codons,
            parallel=False,
            prec_cache=True,
        )

        # Get the pairwise and pointwise matches from the pgroup
        pgroup_pairwise = pgroup.to_pairwise_dataframe()
        pgroup_pointwise = pgroup.to_pointwise_dataframe(
            with_ref_id=True, with_neighbors=True
        )

        # If necessary, also write the pgroup to CSV files
        if out_dir is not None:
            csv_filepaths = pgroup.to_csv(out_dir, tag=out_tag)

    except Exception as e:
        LOGGER.error(
            f"Failed to create ProteinGroup from "
            f"collected reference {ref_pdb_id}: {e}"
        )
        return None

    match_counts = {f"n_{k}": v for k, v in pgroup.match_counts.items()}
    return dict(
        ref_unp_id=pgroup.ref_prec.unp_id,
        ref_pdb_id=ref_pdb_id,
        n_unp_ids=pgroup.num_unique_proteins,
        n_pdb_ids=pgroup.num_query_structs,
        n_total_matches=pgroup.num_matches,
        **match_counts,
        pgroup_pairwise=pgroup_pairwise,
        pgroup_pointwise=pgroup_pointwise,
    )


def _load_prec_df_from_cache(
    pdb_id: str,
    pdb_source: str,
    with_backbone: bool = False,
    with_contacts: bool = False,
    with_altlocs: bool = False,
):
    try:
        prec = ProteinRecord.from_pdb(
            pdb_id,
            pdb_source=pdb_source,
            with_altlocs=with_altlocs,
            with_backbone=with_backbone,
            with_contacts=with_contacts,
            cache=True,
        )
        df = prec.to_dataframe()
        return df
    except ProteinInitError as e:
        LOGGER.error(f"Failed to create {pdb_id} from cache: {e}")
    except Exception as e:
        LOGGER.error(f"Unexpected error creating dataframe for {pdb_id}: {e}")
    return None


def _write_df_csv(df: pd.DataFrame, out_dir: Path, filename: str, index=False) -> Path:
    filename = f"{filename}.csv"
    filepath = out_dir.joinpath(filename)

    with open(str(filepath), mode="w", encoding="utf-8") as f:
        df.to_csv(f, header=True, index=index, float_format="%.2f")

    LOGGER.info(f"Wrote {filepath}")
    return filepath


def _read_df_csv(out_dir: Path, filename: str, usecols: list = None) -> pd.DataFrame:
    filename = f"{filename}.csv"
    filepath = out_dir.joinpath(filename)

    with open(str(filepath), mode="r", encoding="utf-8") as f:
        df = pd.read_csv(f, header=0, index_col=None, usecols=usecols)

    LOGGER.info(f"Loaded {filepath}")
    return df


def _normalize_csv(
    csv_path: Path, common_columns: Sequence[str], _seq: Optional[Tuple[int, int]]
) -> int:
    """
    Adds missing columns to a dataframe CSV file, and writes the result back to the
    same file.

    :param csv_path: The path to the file.
    :param common_columns: The common columns.
    :param _seq: Index and length of this file in the sequence of all files. Just for logging.
    :return: Number of rows in the file.
    """
    df = pd.read_csv(csv_path, header=0, index_col=None, encoding="utf-8")

    # add missing columns
    missing_cols = set(common_columns) - set(df.columns)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = df.assign(**{col: "" for col in missing_cols})

    # reorder
    df = df[[*common_columns]]

    # write back
    df.to_csv(csv_path, header=True, index=False)

    seq_str = f"({_seq[0]+1}/{_seq[1]})" if _seq else ""
    LOGGER.info(f"Normalized {csv_path!s} {seq_str}")
    return len(df)


class ZipDataset:
    """
    A class for loading a collected dataset from a zip file.
    """

    def __init__(
        self, dataset_zipfile_path: Union[Path, str], dataset_name: str = None
    ):
        """
        :param dataset_zipfile_path: The path to the zip file containing the dataset.
        :param dataset_name: Optional name for the dataset, in case the zipfile was
        renamed. This should be the name of the top-level directory inside the zipfile
        which contains the dataset.
        """
        self.zipfile_path = Path(dataset_zipfile_path)

        if not self.zipfile_path.is_file():
            raise FileNotFoundError(f"File not found: {self.zipfile_path}")

        if not dataset_name:
            dataset_name = self.zipfile_path.stem

        self.name: str = dataset_name
        self._collection_metadata_path: str = (
            f"{dataset_name}/{COLLECTION_METADATA_FILENAME}"
        )
        self._struct_metadata_path: str = f"{dataset_name}/{ALL_STRUCTS_FILENAME}.csv"
        self._prec_dir: str = f"{dataset_name}/{DATASET_DIRNAME}"

        self._collection_metadata: Dict[str, Any] = {}
        self._prec_paths: Dict[str, str] = {}  # pdb_id -> path in zip file
        with zipfile.ZipFile(self.zipfile_path, "r") as zip_file:
            # Load collection metadata
            with zip_file.open(self._collection_metadata_path, "r") as fileobj:
                self._collection_metadata.update(json.load(fileobj))

            # Load prec file names
            for file_path in zip_file.namelist():
                if file_path.startswith(self._prec_dir) and file_path.endswith(".csv"):
                    pdb_id = Path(file_path).stem.split("-")[0].replace("_", ":")
                    self._prec_paths[pdb_id] = file_path

    def _read_csv(self, file_path: Union[Path, str], **read_csv_kwargs):
        with zipfile.ZipFile(self.zipfile_path, "r") as zip_file:
            with zip_file.open(str(file_path), "r") as fileobj:
                return pd.read_csv(fileobj, **read_csv_kwargs)

    @property
    def collection_metadata(self) -> Dict[str, Any]:
        """
        :return: The collection metadata.
        """
        return self._collection_metadata.copy()

    @property
    def pdb_ids(self):
        """
        :return: The PDB IDs of all structures in the dataset.
        """
        return tuple(self._prec_paths.keys())

    def load_metadata(self) -> pd.DataFrame:
        """
        Load the metadata for all structures in the dataset.

        :return: The metadata as a DataFrame.
        """
        return self._read_csv(self._struct_metadata_path, header=0, index_col=None)

    def load_prec(self, pdb_id: str) -> pd.DataFrame:
        """
        Load the metadata for all structures in the dataset.

        :return: The metadata as a DataFrame.
        """
        if pdb_id not in self._prec_paths:
            raise ValueError(f"Structure {pdb_id} not found in dataset")
        return self._read_csv(self._prec_paths[pdb_id], header=0, index_col=None)
