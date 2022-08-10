import os
import abc
import json
import time
import socket
import string
import logging
import zipfile
import multiprocessing as mp
from pprint import pformat
from typing import Dict, List, Callable, Iterable, Optional, Sequence
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from multiprocessing.pool import AsyncResult

import numpy as np
import pandas as pd
import more_itertools
from Bio.Align import PairwiseAligner

import pp5
import pp5.parallel
from pp5.prec import ProteinRecord
from pp5.align import DEFAULT_ARPEGGIO_ARGS as ARPEGGIO_ARGS
from pp5.align import Arpeggio, ProteinBLAST
from pp5.utils import ReprJSONEncoder, ProteinInitError, elapsed_seconds_to_dhms
from pp5.pgroup import ProteinGroup
from pp5.external_dbs import pdb, unp, pdb_api
from pp5.external_dbs.unp import unp_record

LOGGER = logging.getLogger(__name__)

COL_UNP_ID = "unp_id"
COL_PDB_ID = "pdb_id"
COL_ENA_ID = "ena_id"
COL_RESOLUTION = "resolution"
COL_SEQ_LEN = "seq_len"
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
COL_REJECTED_BY = "rejected_by"


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
        async_timeout: float = None,
        create_zip=True,
    ):
        """
        :param id: Unique id of this collection. If None, will be generated
        from timestamp and hostname.
        :param out_dir: Output directory, if necessary. A subdirectory with
        a unique id will be created within foe this collection.
        :param tag: Tag (postfix) for unique id of output subdir.
        :param async_timeout: Timeout for async results.
        :param create_zip: Whether to create a zip file with all the result
        files.
        """
        hostname = socket.gethostname()
        if hostname:
            hostname = hostname.split(".")[0].strip()
        else:
            hostname = "localhost"

        self.hostname = hostname
        self.out_tag = tag
        self.async_timeout = async_timeout
        self.create_zip = create_zip

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
                        f"Unexpected exception in top-level " f"collect", exc_info=e
                    )
                    step_status, step_message = "FAIL", f"{e}"
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
            self._collection_meta, width=120, compact=True,
        )
        LOGGER.info(f"Collection metadata:\n" f"{collection_meta_formatted}")
        return self._collection_meta

    def _finalize_collection(self, pool):
        LOGGER.info(f"Finalizing collection for {self.id}...")
        if self.out_dir is None:
            return

        # Create a metadata file in the output dir based on the step results
        meta_filepath = self.out_dir.joinpath("meta.json")
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

        with zipfile.ZipFile(
            zip_filepath, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
        ) as z:
            for out_filepath in self._out_filepaths:
                LOGGER.info(f"Compressing {out_filepath}")
                arcpath = f"{zip_filename.stem}/{out_filepath.name}"
                z.write(str(out_filepath), arcpath)

        zipsize_mb = os.path.getsize(str(zip_filepath)) / 1024 / 1024
        LOGGER.info(f"Wrote archive {zip_filepath} ({zipsize_mb:.2f}MB)")

    def _handle_async_results(
        self,
        async_results: List[AsyncResult],
        collect=False,
        flatten=False,
        result_callback: Callable = None,
    ):
        """
        Handles a list of AsyncResult objects.
        :param async_results: List of objects.
        :param collect: Whether to add all obtained results to a list and
        return it.
        :param flatten: Whether to flatten results (useful i.e. if each
        result is a list or tuple).
        :param result_callback: Callable to invoke on each result.
        :return: Number of handled results, time elapsed in seconds, list of
        collected results (will be empty if collect is False).
        """
        count, start_time = 0, time.time()
        results = []
        for i, async_result in enumerate(async_results):
            try:
                res = async_result.get(self.async_timeout)
                count += 1

                if result_callback is not None:
                    result_callback(res)

                if not collect:
                    continue

                if flatten and isinstance(res, Iterable):
                    results.extend(res)
                else:
                    results.append(res)

            except mp.TimeoutError as e:
                LOGGER.error(
                    f"Timeout getting async result #{i}"
                    f"res={async_result}, skipping: {e}"
                )
            except ProteinInitError as e:
                LOGGER.error(f"Failed to create protein: {e}")
            except Exception as e:
                LOGGER.error("Unexpected error", exc_info=e)

        elapsed_time = time.time() - start_time
        return count, elapsed_time, results

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
    ALL_STRUCTS_FILENAME = "meta-structs_all"
    FILTERED_STRUCTS_FILENAME = "meta-structs_filtered"
    REJECTED_STRUCTS_FILENAME = "meta-structs_rejected"
    BLAST_SCORES_FILENAME = "meta-blast_scores"
    DATASET_FILENAME = "data-precs"

    def __init__(
        self,
        resolution: float = pp5.get_config("DEFAULT_RES"),
        r_free: Optional[float] = pp5.get_config("DEFAULT_RFREE"),
        expr_sys: Optional[str] = pp5.get_config("DEFAULT_EXPR_SYS"),
        source_taxid: Optional[int] = pp5.get_config("DEFAULT_SOURCE_TAXID"),
        seq_similarity_thresh: float = pp5.get_config("DEFAULT_SEQ_SIMILARITY_THRESH"),
        deposition_min_date: Optional[str] = None,
        deposition_max_date: Optional[str] = None,
        prec_init_args=None,
        with_backbone: bool = False,
        with_contacts: bool = False,
        out_dir: Path = pp5.out_subdir("prec-collected"),
        out_tag: Optional[str] = None,
        prec_out_dir: Path = pp5.out_subdir("prec"),
        write_csv=True,
        async_timeout=600,
    ):
        """
        Collects ProteinRecords based on a PDB query results, and saves them
        in the local cache folder. Optionally also writes them to CSV.
        :param resolution: Resolution cutoff value in Angstroms.
        :param expr_sys: Expression system name.
        :param source_taxid: Taxonomy ID of source organism.
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
        :param prec_out_dir: Output folder for prec CSV files.
        :param prec_init_args: Arguments for initializing each ProteinRecord
        :param with_backbone: Whether to include a 'backbone' column which contain the
        backbone atom coordinates of each residue in the order N, CA, C, O.
        :param with_contacts: Whether to include tertiary contact features per residue.
        :param write_csv: Whether to write the ProteinRecords to
            the prec_out_dir as CSV files.
        :param async_timeout: Timeout in seconds for each worker
            process result.
        """
        super().__init__(
            async_timeout=async_timeout, out_dir=out_dir, tag=out_tag, create_zip=False
        )
        if resolution is None:
            raise ValueError("Must specify resolution cutoff for collection")

        self.resolution = float(resolution)
        self.r_free = float(r_free) if r_free is not None else None
        self.expr_sys = str(expr_sys) if expr_sys else None
        self.source_taxid = (
            int(source_taxid) if (source_taxid not in (None, "")) else None
        )
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

        self.prec_out_dir = prec_out_dir
        self.write_csv = write_csv

    def __repr__(self):
        return f"{self.__class__.__name__} query={self.query}"

    def _collection_functions(self):
        return {
            "Collect ProteinRecords": self._collect_precs,
            "Filter Collected": self._filter_collected,
            "Write dataset": self._write_dataset,
        }

    def _collect_precs(self, pool: mp.Pool):
        meta = {}
        pdb_ids = self.query.execute()
        n_structs = len(pdb_ids)

        meta["query"] = str(self.query)
        meta["n_query_results"] = len(pdb_ids)
        LOGGER.info(f"Got {n_structs} structures from PDB, collecting...")

        async_results = []
        for i, pdb_id in enumerate(pdb_ids):
            args = (pdb_id, (i, n_structs))
            kwds = dict(
                csv_out_dir=self.prec_out_dir if self.write_csv else None,
                with_backbone=self.with_backbone,
                with_contacts=ARPEGGIO_ARGS if self.with_contacts else None,
            )
            r = pool.apply_async(_collect_single_structure, args, kwds)
            async_results.append(r)

        _, elapsed, pdb_id_data = self._handle_async_results(
            async_results, collect=True, flatten=True
        )

        # Create a dataframe from the collected data
        df_all = pd.DataFrame(pdb_id_data)
        n_collected = len(df_all)

        _write_df_csv(df_all, self.out_dir, self.ALL_STRUCTS_FILENAME)

        meta["n_collected"] = n_collected
        LOGGER.info(
            f"Done collecting: {n_collected}/{len(pdb_ids)} proteins collected "
            f"(elapsed={elapsed:.2f} seconds, "
            f"{len(pdb_ids) / elapsed:.1f} proteins/sec)."
        )
        return meta

    def _filter_collected(self, pool: mp.Pool) -> dict:
        """
        Filters collected structures according to conditions on their metadata.
        """

        df_all: pd.DataFrame = _read_df_csv(self.out_dir, self.ALL_STRUCTS_FILENAME)
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
        _write_df_csv(df_filtered, self.out_dir, self.FILTERED_STRUCTS_FILENAME)

        # Write the rejected structures and specify which filter rejected them
        df_rejected = df_all
        df_rejected[COL_REJECTED_BY] = ""
        for filter_name, rejected_idx in rejected_idxs.items():
            df_rejected.loc[rejected_idx, COL_REJECTED_BY] = filter_name
        df_rejected = df_rejected[~filter_idx]
        _write_df_csv(df_rejected, self.out_dir, self.REJECTED_STRUCTS_FILENAME)

        return {
            "n_rejected": rejected_counts,
            "n_collected_filtered": len(df_filtered),
        }

    def _filter_metadata(self, pool: mp.Pool, df_all: pd.DataFrame) -> pd.Series:
        # Even though we query by resolution, the metadata resolution is different
        # than what we can query on. Metadata shows resolution after refinement,
        # while the query is using data collection resolution.
        idx_filter = df_all[COL_RESOLUTION].astype(float) <= self.resolution
        return idx_filter

    def _filter_redundant_unps(self, pool: mp.Pool, df_all: pd.DataFrame) -> pd.Series:
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
        _write_df_csv(
            df_blast_scores, self.out_dir, self.BLAST_SCORES_FILENAME, index=True
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

    def _write_dataset(self, pool: mp.Pool) -> dict:
        df_pdb_ids: pd.DataFrame = _read_df_csv(
            self.out_dir, self.FILTERED_STRUCTS_FILENAME, usecols=["pdb_id"]
        )
        pdb_ids = df_pdb_ids["pdb_id"]
        LOGGER.info(f"Creating dataset file for {len(pdb_ids)} precs...")

        filepath = self.out_dir.joinpath(f"{self.DATASET_FILENAME}.csv")
        n_entries = 0

        # Parallelize creation of dataframes in chunks.
        chunk_size = 128
        for pdb_ids_chunk in more_itertools.chunked(pdb_ids, chunk_size):
            async_results = []
            for i, pdb_id in enumerate(pdb_ids_chunk):
                async_results.append(
                    pool.apply_async(
                        _load_prec_df_from_cache,
                        args=(pdb_id,),
                        kwds=dict(
                            with_backbone=self.with_backbone,
                            with_contacts=ARPEGGIO_ARGS if self.with_contacts else None,
                        ),
                    )
                )

            _, elapsed, pdb_id_dataframes = self._handle_async_results(
                async_results, collect=True, flatten=False,
            )

            # Writing the dataframes to a single file must be sequential
            with open(str(filepath), mode="a", encoding="utf-8") as f:
                for df in pdb_id_dataframes:
                    if df is None or df.empty:
                        continue
                    df.to_csv(
                        f, header=n_entries == 0, index=False, float_format="%.2f"
                    )
                    n_entries += len(df)

        dataset_size_mb = os.path.getsize(filepath) / 1024 / 1024
        LOGGER.info(f"Wrote {filepath} ({n_entries=}, {dataset_size_mb:.2f}MB)")
        meta = {f"dataset_size_mb": f"{dataset_size_mb:.2f}", "n_entries": n_entries}
        return meta


class ProteinGroupCollector(ParallelDataCollector):
    def __init__(
        self,
        resolution: float,
        expr_sys: str = pp5.get_config("DEFAULT_EXPR_SYS"),
        source_taxid: int = pp5.get_config("DEFAULT_SOURCE_TAXID"),
        evalue_cutoff: float = 1.0,
        identity_cutoff: float = 30.0,
        b_max: float = 30.0,
        match_len: int = 2,
        context_len: int = 1,
        compare_contacts: bool = True,
        out_dir=pp5.out_subdir("pgroup-collected"),
        pgroup_out_dir=pp5.out_subdir("pgroup"),
        write_pgroup_csvs=True,
        out_tag: str = None,
        ref_file: Path = None,
        async_timeout: float = 3600,
        create_zip=True,
    ):
        """
        Collects ProteinGroup reference structures based on a PDB query
        results.
        :param resolution: Resolution cutoff value in Angstroms.
        :param expr_sys: Expression system name.
        :param source_taxid: Taxonomy ID of source organism.
        :param evalue_cutoff: Maximal expectation value allowed for BLAST
        matches when searching for proteins to include in pgroups.
        :param identity_cutoff: Minimal percent sequence identity
        allowed for BLAST matches when searching for proteins to include in
        pgroups.
        :param b_max: Maximal b-factor a residue can have
        (backbone-atom average) in order for it to be included in a match
        group. None means no limit.
        :param match_len: Number of residues to include in a match. Can be either 1
        or 2. If 2, the match dihedral angles will be the cross-bond angles (phi+1,
        psi+0) between the two residues.
        :param context_len: Number of stars required around an aligned AA
        pair to consider that pair for a match.
        :param compare_contacts: Whether to compare tertiary contacts contexts of
        potential matches.
        :param out_dir: Output directory for collection CSV files.
        :param pgroup_out_dir: Output directory for pgroup CSV files. Only
            relevant if write_pgroup_csvs is True.
        :param write_pgroup_csvs: Whether to write each pgroup's CSV files.
            Even if false, the collection files will still be writen.
        :param out_tag: Extra tag to add to the output file names.
        :param  ref_file: Path of collector CSV file with references.
            Allows to skip the first and second collection steps (finding PDB
            IDs for the reference structures) and immediately collect
            ProteinGroups for the references in the file.
        :param async_timeout: Timeout in seconds for each worker
            process result, or None for no timeout.
        :param create_zip: Whether to create a zip file with all output files.
        """
        super().__init__(
            out_dir=out_dir,
            tag=out_tag,
            async_timeout=async_timeout,
            create_zip=create_zip,
        )

        self.resolution = float(resolution)
        self.expr_sys = expr_sys
        self.source_taxid = int(source_taxid) if source_taxid is not None else None
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
        self.match_len = match_len
        self.context_len = context_len
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

    def _collect_all_structures(self, pool: mp.Pool):
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

            async_results = []
            for i, pdb_id in enumerate(pdb_ids):
                args = (pdb_id, (i, n_structs))
                r = pool.apply_async(_collect_single_structure, args=args)
                async_results.append(r)

            count, elapsed, pdb_id_data = self._handle_async_results(
                async_results, collect=True, flatten=True,
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

    def _collect_all_refs(self, pool: mp.Pool):
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

            async_results = []
            for unp_id, df_group in groups:
                args = (unp_id, df_group)
                r = pool.apply_async(_collect_single_ref, args=args)
                async_results.append(r)

            count, elapsed, group_datas = self._handle_async_results(
                async_results, collect=True,
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

    def _collect_all_pgroups(self, pool: mp.Pool):
        meta = {}

        # Create a local BLAST DB containing all collected PDB IDs.
        all_pdb_ids = self._df_all["pdb_id"]
        alias_name = f"pgc-{self.out_tag}"
        blast_db = ProteinBLAST.create_db_subset_alias(
            all_pdb_ids, alias_name, db_autoupdate_days=7,
        )
        blast = ProteinBLAST(
            db_name=blast_db,
            evalue_cutoff=self.evalue_cutoff,
            identity_cutoff=self.identity_cutoff,
        )

        LOGGER.info(f"Creating ProteinGroup for each reference...")
        ref_pdb_ids = self._df_ref["ref_pdb_id"].values
        async_results = []
        for i, ref_pdb_id in enumerate(ref_pdb_ids):
            idx = (i, len(ref_pdb_ids))
            pgroup_out_dir = self.pgroup_out_dir if self.write_pgroup_csvs else None
            args = (
                ref_pdb_id,
                blast,
                self.b_max,
                self.match_len,
                self.context_len,
                self.compare_contacts,
                pgroup_out_dir,
                self.out_tag,
                idx,
            )
            r = pool.apply_async(_collect_single_pgroup, args=args)
            async_results.append(r)

        count, elapsed, collected_data = self._handle_async_results(
            async_results, collect=True, flatten=False,
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
    idx: tuple = (0, 0),
    csv_out_dir: Optional[Path] = None,
    csv_tag: str = None,
    with_backbone: bool = False,
    with_contacts: Optional[Dict] = None,
) -> List[dict]:
    """
    Downloads a single PDB entry, and creates a prec for all its chains.
    :param pdb_id: The PDB id to download.
    :param idx: Index for logging; should be a tuple of (current, total).
    :param csv_out_dir: If provided, the prec of each chain will be written to CSV at
        this path.
    :param csv_tag: Tag to add to the output CSVs.
    :param with_backbone: Whether to write backbone atom locations to csv.
    :param with_contacts: Whether to run Arpeggio so that it's results are cached
    locally and also write contact features to csv. The dict contains kwargs for
    arpeggio.
    :return: A list of dicts, each containing metadata about one of the collected
        chains.
    """
    pdb_id, chain_id, entity_id = pdb.split_id_with_entity(pdb_id)

    pdb_dict = pdb.pdb_dict(pdb_id)
    meta = pdb.PDBMetadata(pdb_id, struct_d=pdb_dict)
    pdb2unp = pdb.PDB2UNP.from_pdb(pdb_id, cache=True, struct_d=pdb_dict)

    # If we got an entity id instead of chain, discover the chain.
    if entity_id:
        entity_id = int(entity_id)
        chain_id = meta.get_chain(entity_id)

    if chain_id:
        # If we have a specific chain, use only that
        all_chains = (chain_id,)
    else:
        # Otherwise, we'll take all UNIQUE chains: only one chain from
        # each unique entity. This is important, since chains from the same
        # entity are identical, so they're redundant.
        all_chains = [meta.get_chain(e) for e in meta.entity_sequence]

    chain_data = []
    for chain_id in all_chains:
        pdb_id_full = f"{pdb_id}:{chain_id}"

        # Skip chains with no Uniprot ID
        if chain_id not in pdb2unp:
            LOGGER.warning(f"No Uniprot ID for {pdb_id_full}")
            continue

        # Skip chimeric chains
        if pdb2unp.is_chimeric(chain_id):
            LOGGER.warning(f"Discarding chimeric chain {pdb_id_full}")
            continue

        unp_id = pdb2unp.get_unp_id(chain_id)
        resolution = meta.resolution
        seq_len = len(meta.entity_sequence[meta.chain_entities[chain_id]])

        # Create a ProteinRecord and save it so it's cached for when we
        # create the pgroups. Only collect structures for which we can
        # create a prec (e.g. they must have a DNA sequence).
        try:
            nc = chain_id in string.digits
            prec = ProteinRecord(
                unp_id,
                pdb_id_full,
                pdb_dict=pdb_dict,
                strict_unp_xref=False,
                numeric_chain=nc,
            )

            # Save into cache
            prec.save()

            if with_contacts:
                arpeggio = Arpeggio(**with_contacts)
                _ = arpeggio.contact_df(pdb_id_full)

            # Write CSV if requested
            if csv_out_dir is not None:
                prec.to_csv(
                    csv_out_dir,
                    tag=csv_tag,
                    with_backbone=with_backbone,
                    with_contacts=with_contacts,
                )
        except Exception as e:
            LOGGER.warning(
                f"Failed to create ProteinRecord for "
                f"({unp_id}, {pdb_id}), will not collect: {e}"
            )
            continue

        chain_data.append(
            {
                COL_UNP_ID: prec.unp_id,
                COL_PDB_ID: prec.pdb_id,
                COL_ENA_ID: prec.ena_id,
                COL_RESOLUTION: resolution,
                COL_SEQ_LEN: seq_len,
                COL_DESCRIPTION: meta.description,
                COL_DEPOSITION_DATE: meta.deposition_date,
                COL_SRC_ORG: meta.src_org,
                COL_HOST_ORG: meta.host_org,
                COL_LIGANDS: meta.ligands,
                COL_R_FREE: meta.r_free,
                COL_R_WORK: meta.r_work,
                COL_SPACE_GROUP: meta.space_group,
                COL_CG_PH: meta.cg_ph,
                COL_CG_TEMP: meta.cg_temp,
            }
        )

    LOGGER.info(
        f"Collected {len(chain_data)} chains from {pdb_id} "
        f"{pdb2unp.get_chain_to_unp_ids()} ({idx[0] + 1}/{idx[1]})"
    )

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
    blast: ProteinBLAST,
    b_max: float,
    match_len: int,
    context_len: int,
    compare_contacts: bool,
    out_dir: Optional[Path],
    out_tag: str,
    idx: tuple,
) -> Optional[dict]:
    try:
        LOGGER.info(
            f"Creating ProteinGroup for {ref_pdb_id}, {b_max=} "
            f"({idx[0] + 1}/{idx[1]})"
        )

        # Run BLAST to find query structures for the pgroup
        df_blast = blast.pdb(ref_pdb_id)
        LOGGER.info(f"Got {len(df_blast)} BLAST hits for {ref_pdb_id}")

        # Create a pgroup without an additional query, by specifying the
        # exact ids of the query structures.
        pgroup = ProteinGroup.from_query_ids(
            ref_pdb_id,
            query_pdb_ids=df_blast.index,
            b_max=b_max,
            match_len=match_len,
            context_len=context_len,
            compare_contacts=compare_contacts,
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
    pdb_id: str, with_backbone: bool = False, with_contacts: Optional[Dict] = None,
):
    try:
        prec = ProteinRecord.from_pdb(pdb_id, cache=True)
        df = prec.to_dataframe(
            with_ids=True, with_backbone=with_backbone, with_contacts=with_contacts
        )
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
