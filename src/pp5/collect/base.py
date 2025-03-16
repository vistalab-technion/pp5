import os
import abc
import json
import time
import socket
import logging
import zipfile
import warnings
import itertools
import multiprocessing as mp
from pprint import pformat
from typing import Dict, List, Tuple, Union, Callable, Iterable, Optional, Sequence
from pathlib import Path
from dataclasses import dataclass
from multiprocessing.pool import Pool, AsyncResult

import pandas as pd
from Bio.Align import PairwiseAligner

import pp5
import pp5.parallel
from pp5.prec import ProteinRecord
from pp5.cache import ReprJSONEncoder
from pp5.utils import elapsed_seconds_to_dhms
from pp5.external_dbs import pdb
from pp5.external_dbs.pdb import PDB_RCSB
from pp5.external_dbs.unp import unp_record

_LOG = logging.getLogger(__name__)

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
                    _LOG.error(f"Unexpected exception in top-level collect", exc_info=e)
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

        _LOG.info(f"Completed collection for {self} in {time_str}")
        collection_meta_formatted = pformat(
            self._collection_meta,
            width=120,
            compact=True,
        )
        _LOG.info(f"Collection metadata:\n" f"{collection_meta_formatted}")
        return self._collection_meta

    def _finalize_collection(self, pool: Pool):
        _LOG.info(f"Finalizing collection for {self.id}...")
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
                _LOG.error(f"Failed to serialize metadata", exc_info=e)

        self._out_filepaths.append(meta_filepath)

        # Create a zip of the results
        if not self.create_zip:
            return

        zip_filename = Path(f"{self.id}.zip")
        zip_filepath = self.out_dir.joinpath(zip_filename)

        _LOG.info(f"Compressing results into {zip_filename!s}...")
        with zipfile.ZipFile(
            zip_filepath, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
        ) as z:
            for out_filepath in self._out_filepaths:
                rel_out_filepath = out_filepath.relative_to(self.out_dir)
                arcpath = f"{zip_filename.stem}/{rel_out_filepath!s}"
                z.write(str(out_filepath), arcpath)

        zipsize_mb = os.path.getsize(str(zip_filepath)) / 1024 / 1024
        _LOG.info(f"Wrote archive {zip_filepath} ({zipsize_mb:.2f}MB)")

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


def collect_single_structure(
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
            _LOG.warning(f"No Uniprot ID for {pdb_id_full}")
            continue

        # Skip chimeric chains
        if len(chain_to_unp_ids[chain_id]) > 1:
            _LOG.warning(f"Discarding chimeric chain {pdb_id_full}")
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
            _LOG.warning(
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
    _LOG.log(level=logging.INFO if len(chain_data) else logging.WARNING, msg=msg)

    return chain_data


def pairwise_align_unp(
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


def write_df_csv(df: pd.DataFrame, out_dir: Path, filename: str, index=False) -> Path:
    filename = f"{filename}.csv"
    filepath = out_dir.joinpath(filename)

    with open(str(filepath), mode="w", encoding="utf-8") as f:
        df.to_csv(f, header=True, index=index, float_format="%.2f")

    _LOG.info(f"Wrote {filepath}")
    return filepath


def read_df_csv(out_dir: Path, filename: str, usecols: list = None) -> pd.DataFrame:
    filename = f"{filename}.csv"
    filepath = out_dir.joinpath(filename)

    with open(str(filepath), mode="r", encoding="utf-8") as f:
        df = pd.read_csv(f, header=0, index_col=None, usecols=usecols)

    _LOG.info(f"Loaded {filepath}")
    return df


def normalize_csv(
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
    _LOG.info(f"Normalized {csv_path!s} {seq_str}")
    return len(df)
