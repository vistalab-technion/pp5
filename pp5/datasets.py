import abc
import logging
import multiprocessing as mp
import time
from pathlib import Path
from typing import Callable, Any

import pp5
import pp5.parallel
from pp5.external_dbs import pdb
from pp5.external_dbs.pdb import PDBQuery
from pp5.protein import ProteinRecord, ProteinInitError

LOGGER = logging.getLogger(__name__)


class ParallelDataCollector(abc.ABC):
    def __init__(self, max_processes=pp5.parallel.MAX_PROCESSES,
                 async_res_timeout_sec=30):
        self.max_processes = max_processes
        self.async_res_timeout_sec = async_res_timeout_sec

    def collect(self):
        pool_name = self.__class__.__name__
        with pp5.parallel.pool(pool_name, self.max_processes) as pool:
            self._collect_with_pool(pool)

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

    ProteinRecordCollector(
        query, prec_init_args={'dihedral_est_name': ''},
        out_dir=pp5.out_subdir('prec')
    ).collect()
