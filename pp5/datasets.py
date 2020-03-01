import logging
import multiprocessing as mp
import multiprocessing.pool
import os
import shutil
import tempfile
import time
from pathlib import Path

import pp5
from pp5.external_dbs import pdb
from pp5.external_dbs.pdb import PDBQuery
from pp5.protein import ProteinRecord, ProteinInitError

LOGGER = logging.getLogger(__name__)


def worker_process_init(base_workers_dl_dir, *args):
    # Provide each process with a unique download folder.
    pid = os.getpid()
    worker_dl_dir = base_workers_dl_dir.joinpath(f'{pid}')
    os.makedirs(worker_dl_dir, exist_ok=True)
    LOGGER.info(f'Worker process {pid} using dir {worker_dl_dir}...')
    # Override the default with the process-specific dir to have a
    # different download directory for each process.
    pp5.BASE_DOWNLOAD_DIR = worker_dl_dir


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


def collect_data(query: PDBQuery):
    pdb_ids = query.execute()

    LOGGER.info(f"Got {len(pdb_ids)} structures from PDB")

    base_workers_dl_dir = Path(tempfile.gettempdir()).joinpath('pp5_data')
    os.makedirs(str(base_workers_dl_dir), exist_ok=True)

    async_results = []
    with mp.pool.Pool(processes=8,
                      initializer=worker_process_init,
                      initargs=(base_workers_dl_dir,)) as pool:
        for i, pdb_id in enumerate(pdb_ids):
            r = pool.apply_async(ProteinRecord.from_pdb, args=(pdb_id,),
                                 kwds=dict(dihedral_est_name='erp'))
            async_results.append(r)

        start_time, counter = time.time(), 0
        for async_result in async_results:
            try:
                protein_recs = async_result.get(30)
            except TimeoutError as e:
                LOGGER.error("Timeout getting async result, skipping")
            except ProteinInitError as e:
                LOGGER.error(f"Failed to create protein: {e}")
            except Exception as e:
                LOGGER.error("Unexpected error", exc_info=e.__cause__)

            counter += len(protein_recs)
            pps = counter / (time.time() - start_time)
            LOGGER.info(f'Collected {protein_recs} ({pps:.1f} proteins/sec)')

            # Write to file
            for prec in protein_recs:
                csv_path = prec.to_csv(pp5.data_subdir('proteins'))
                LOGGER.info(f'Wrote output CSV {csv_path}')

        clean_worker_downloads(base_workers_dl_dir)
        LOGGER.info(f"Done: {counter} proteins collected.")


if __name__ == '__main__':
    # Query PDB for structures
    query = pdb.PDBCompositeQuery(
        pdb.PDBExpressionSystemQuery('Escherichia Coli'),
        pdb.PDBResolutionQuery(max_res=1.0)
    )

    collect_data(query)
