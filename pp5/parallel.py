import atexit
import contextlib
import logging
import multiprocessing as mp
import multiprocessing.pool
import os
import shutil
import tempfile
from pathlib import Path
from typing import ContextManager

import pp5

LOGGER = logging.getLogger(__name__)

__GLOBAL_POOL = None
BASE_WORKERS_DL_DIR = Path(tempfile.gettempdir()).joinpath('pp5_data')
GLOBAL_WORKERS_DL_DIR = BASE_WORKERS_DL_DIR.joinpath(f'_global_{os.getpid()}')


def _worker_process_init(base_workers_dl_dir, *args):
    os.makedirs(str(base_workers_dl_dir), exist_ok=True)

    # Provide each process with a unique download folder.
    pid = os.getpid()
    worker_dl_dir = base_workers_dl_dir.joinpath(f'{pid}')
    os.makedirs(worker_dl_dir, exist_ok=True)
    LOGGER.info(f'Worker process {pid} using dir {worker_dl_dir}...')
    # Override the default with the process-specific dir to have a
    # different download directory for each process.
    pp5.BASE_DOWNLOAD_DIR = worker_dl_dir


def _clean_worker_downloads(base_workers_dl_dir):
    n_moved = 0
    downloaded_files = base_workers_dl_dir.glob('**/*/*.*')
    for downloaded_file in downloaded_files:
        rel_path = downloaded_file.relative_to(base_workers_dl_dir)
        # Remove pid dir and append the relative path to the main data dir
        data_path = pp5.BASE_DATA_DIR.joinpath(*rel_path.parts[1:])
        shutil.move(str(downloaded_file), str(data_path))
        n_moved += 1

    LOGGER.info(f'Moved {n_moved} data files from '
                f'{base_workers_dl_dir} into {pp5.BASE_DATA_DIR}')


def _remove_workers_dir(base_workers_dl_dir):
    try:
        shutil.rmtree(base_workers_dl_dir)
        LOGGER.info(f'Deleted temp folder {base_workers_dl_dir}')
    except Exception as e:
        LOGGER.warning(f'Failed to delete {base_workers_dl_dir}: {e}')


@contextlib.contextmanager
def global_pool() -> ContextManager[mp.pool.Pool]:
    global __GLOBAL_POOL
    base_workers_dl_dir = GLOBAL_WORKERS_DL_DIR

    n_processes = pp5.get_config('MAX_PROCESSES')
    if __GLOBAL_POOL is None:
        mp_ctx = mp.get_context('spawn')
        LOGGER.info(f'Starting global pool with {n_processes} processes')
        __GLOBAL_POOL = mp_ctx.Pool(processes=n_processes,
                                    initializer=_worker_process_init,
                                    initargs=(base_workers_dl_dir,))
    try:
        yield __GLOBAL_POOL
    finally:
        _clean_worker_downloads(base_workers_dl_dir)


@contextlib.contextmanager
def pool(name: str, processes=None) -> ContextManager[mp.pool.Pool]:
    base_workers_dl_dir = BASE_WORKERS_DL_DIR.joinpath(name)
    mp_ctx = mp.get_context('spawn')
    processes = processes if processes is not None else \
        pp5.get_config('MAX_PROCESSES')

    try:
        LOGGER.info(f'Starting pool {name} with {processes} processes')
        with mp_ctx.Pool(processes, initializer=_worker_process_init,
                         initargs=(base_workers_dl_dir,)) as p:
            yield p
    finally:
        _clean_worker_downloads(base_workers_dl_dir)
        _remove_workers_dir(base_workers_dl_dir)


def _cleanup():
    global __GLOBAL_POOL
    if __GLOBAL_POOL is not None:
        LOGGER.info(f'Closing global pool...')
        __GLOBAL_POOL.close()
        __GLOBAL_POOL.join()
        _remove_workers_dir(GLOBAL_WORKERS_DL_DIR)


atexit.register(_cleanup)
