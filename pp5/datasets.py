import abc
import logging
import multiprocessing as mp
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Callable, Any, Union

import pp5
from pp5.external_dbs import pdb
from pp5.external_dbs.pdb import PDBQuery
from pp5.protein import ProteinRecord, ProteinInitError
from pp5.align import structural_align

import pandas as pd

LOGGER = logging.getLogger(__name__)

BASE_WORKERS_DL_DIR = Path(tempfile.gettempdir()).joinpath('pp5_data')
os.makedirs(str(BASE_WORKERS_DL_DIR), exist_ok=True)


class ParallelDataCollector(abc.ABC):
    def __init__(self, max_processes=None, async_res_timeout_sec=30):
        self.max_processes = max_processes
        self.async_res_timeout_sec = async_res_timeout_sec

    @staticmethod
    def worker_process_init(base_workers_dl_dir, *args):
        # Provide each process with a unique download folder.
        pid = os.getpid()
        worker_dl_dir = base_workers_dl_dir.joinpath(f'{pid}')
        os.makedirs(worker_dl_dir, exist_ok=True)
        LOGGER.info(f'Worker process {pid} using dir {worker_dl_dir}...')
        # Override the default with the process-specific dir to have a
        # different download directory for each process.
        pp5.BASE_DOWNLOAD_DIR = worker_dl_dir

    @staticmethod
    def clean_worker_downloads(base_workers_dl_dir):
        n_copied = 0
        downloaded_files = base_workers_dl_dir.glob('**/*/*.*')
        for downloaded_file in downloaded_files:
            rel_path = downloaded_file.relative_to(base_workers_dl_dir)
            # Remove pid dir and append the relative path to the main data dir
            data_path = pp5.BASE_DATA_DIR.joinpath(*rel_path.parts[1:])
            if data_path.is_file():
                continue
            # The downloaded file is not in the data dir, so copy it.
            shutil.copy2(str(downloaded_file), str(data_path))
            n_copied += 1
        LOGGER.info(f'Copied {n_copied} downloaded files from '
                    f'{base_workers_dl_dir} into {pp5.BASE_DATA_DIR}')
        shutil.rmtree(base_workers_dl_dir)
        LOGGER.info(f'Deleted temp folder {base_workers_dl_dir}')

    def collect(self):
        mp_ctx = mp.get_context('spawn')
        try:
            with mp_ctx.Pool(processes=self.max_processes,
                             initializer=self.worker_process_init,
                             initargs=(BASE_WORKERS_DL_DIR,)) as pool:
                self._collect_with_pool(pool)
        finally:
            self.clean_worker_downloads(BASE_WORKERS_DL_DIR)

    @abc.abstractmethod
    def _collect_with_pool(self, pool: mp.Pool):
        pass


class ProteinRecordCollector(ParallelDataCollector):
    """
    Collects ProteinRecords based on a PDB query results, and invokes a
    custom callback on each of them.
    """
    DEFAULT_PREC_INIT_ARGS = dict(dihedral_est_name='erp')

    def __init__(self, query: PDBQuery, prec_init_args=None,
                 out_dir: Path = pp5.data_subdir('prec'),
                 prec_callback: Callable[
                     [ProteinRecord, Path], Any] = ProteinRecord.to_csv, **kw):
        """
        :param query: A PDBQuery to run. This query must return a list of
        PDB IDs, which can either bbe only structure id, can include chain
        ids or entity ids.
        :param out_dir: Output folder. Will be passed to callback.
        :param prec_init_args: Arguments for initializing each ProteinRecord
        :param prec_callback: A callback to invoke for each ProteinRecord.
        Will be invoked in the main process.
        Should be a function accepting a ProteinRecord and a path which is
        the output folder. The default callback will write the ProteinRecord to
        the output folder as a CSV file.
        :param kw: Extra args for ParallelDataCollector.
        """
        super().__init__(**kw)
        self.query = query
        if prec_init_args:
            self.prec_init_args = prec_init_args
        else:
            self.prec_init_args = self.DEFAULT_PREC_INIT_ARGS
        self.out_dir = out_dir
        self.prec_callback = prec_callback

    def _collect_with_pool(self, pool: mp.Pool):
        pdb_ids = self.query.execute()
        LOGGER.info(f"Got {len(pdb_ids)} structures from PDB")

        async_results = []
        for i, pdb_id in enumerate(pdb_ids):
            r = pool.apply_async(ProteinRecord.from_pdb, args=(pdb_id,),
                                 kwds=self.prec_init_args)
            async_results.append(r)

        start_time, counter, pps = time.time(), 0, 0
        for async_result in async_results:
            try:
                prec = async_result.get(self.async_res_timeout_sec)
                self.prec_callback(prec, self.out_dir)
            except TimeoutError as e:
                LOGGER.error("Timeout getting async result, skipping")
            except ProteinInitError as e:
                LOGGER.error(f"Failed to create protein: {e}")
            except Exception as e:
                LOGGER.error("Unexpected error", exc_info=e.__cause__)

            counter += 1
            pps = counter / (time.time() - start_time)

        LOGGER.info(f"Done: {counter}/{len(pdb_ids)} proteins collected "
                    f"(elapsed={time.time() - start_time:.2f} seconds, "
                    f"{pps:.1f} proteins/sec).")


if __name__ == '__main__':
    # Query PDB for structures
    query = pdb.PDBCompositeQuery(
        pdb.PDBExpressionSystemQuery('Escherichia Coli'),
        pdb.PDBResolutionQuery(max_res=0.8)
    )

    # collector = ProteinRecordCollector(
    #     query, prec_init_args={'dihedral_est_name': ''},
    #     out_dir=pp5.out_subdir('prec')
    # )

    for pdb_id in ['2wur:a', '5nl4:a', '5tnt:b', '1nkd:a', '1n19:a', '6mv4:h',
                   '6mv4:l', ]:
        # '1mwc:a', ' \ ''2wur:a', # '5jdt:a']:
        collector = ProteinGroupsCollector(pdb_id, blast_identity_cutoff=0)
        collector.collect()
