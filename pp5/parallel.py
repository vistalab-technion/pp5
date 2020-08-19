import os
import atexit
import shutil
import signal
import logging
import tempfile
import contextlib
import multiprocessing as mp
import multiprocessing.pool
from typing import T, Any, Dict, List, Tuple, Union, Iterator, Generator, ContextManager
from pathlib import Path
from multiprocessing.pool import AsyncResult

import pp5

LOGGER = logging.getLogger(__name__)

__GLOBAL_POOL: mp.pool.Pool = None
BASE_WORKERS_DL_DIR = Path(tempfile.gettempdir()).joinpath("pp5_data")
GLOBAL_WORKERS_DL_DIR = BASE_WORKERS_DL_DIR.joinpath(f"_global_{os.getpid()}")


def _worker_process_init(base_workers_dl_dir, *args):
    # ignore SIGINT in workers, it should only be handled in the main process
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    os.makedirs(str(base_workers_dl_dir), exist_ok=True)

    # Provide each process with a unique download folder.
    pid = os.getpid()
    worker_dl_dir = base_workers_dl_dir.joinpath(f"{pid}")
    os.makedirs(worker_dl_dir, exist_ok=True)
    LOGGER.info(f"Worker process {pid} using dir {worker_dl_dir}...")
    # Override the default with the process-specific dir to have a
    # different download directory for each process.
    pp5.BASE_DOWNLOAD_DIR = worker_dl_dir


def _clean_worker_downloads(base_workers_dl_dir):
    n_moved = 0
    downloaded_files = base_workers_dl_dir.glob("**/*/*.*")
    for downloaded_file in downloaded_files:
        rel_path = downloaded_file.relative_to(base_workers_dl_dir)
        # Remove pid dir and append the relative path to the main data dir
        data_path = pp5.BASE_DATA_DIR.joinpath(*rel_path.parts[1:])
        shutil.move(str(downloaded_file), str(data_path))
        n_moved += 1

    LOGGER.info(
        f"Moved {n_moved} data files from "
        f"{base_workers_dl_dir} into {pp5.BASE_DATA_DIR}"
    )


def _remove_workers_dir(base_workers_dl_dir):
    try:
        shutil.rmtree(base_workers_dl_dir)
        LOGGER.info(f"Deleted temp folder {base_workers_dl_dir}")
    except Exception as e:
        LOGGER.warning(f"Failed to delete {base_workers_dl_dir}: {e}")


@contextlib.contextmanager
def global_pool() -> ContextManager[mp.pool.Pool]:
    global __GLOBAL_POOL
    base_workers_dl_dir = GLOBAL_WORKERS_DL_DIR

    n_processes = pp5.get_config("MAX_PROCESSES")
    if __GLOBAL_POOL is None:
        mp_ctx = mp.get_context("spawn")
        LOGGER.info(f"Starting global pool with {n_processes} processes")
        __GLOBAL_POOL = mp_ctx.Pool(
            processes=n_processes,
            initializer=_worker_process_init,
            initargs=(base_workers_dl_dir,),
        )
    try:
        yield __GLOBAL_POOL
    finally:
        _clean_worker_downloads(base_workers_dl_dir)


@contextlib.contextmanager
def pool(name: str, processes=None, context="spawn") -> ContextManager[mp.pool.Pool]:
    base_workers_dl_dir = BASE_WORKERS_DL_DIR.joinpath(name)
    mp_ctx = mp.get_context(context)
    processes = processes if processes is not None else pp5.get_config("MAX_PROCESSES")

    try:
        LOGGER.info(f"Starting pool {name} with {processes} processes")
        with mp_ctx.Pool(
            processes, initializer=_worker_process_init, initargs=(base_workers_dl_dir,)
        ) as p:
            yield p
    finally:
        _clean_worker_downloads(base_workers_dl_dir)
        _remove_workers_dir(base_workers_dl_dir)


def yield_async_results(
    async_results: Union[Dict[T, AsyncResult], List[AsyncResult]],
    wait_time_sec=0.1,
    max_retries=None,
    re_raise=False,
) -> Generator[Tuple[T, Any], None, None]:
    """

    Waits for async results to be ready, and yields them.
    This function waits for each result for a fixed time, and moves to the
    next result if it's not ready. Therefore, the order of yielded results is
    not guaranteed to be the same as the order of the AsyncResult objects.

    :param async_results: Either a dict mapping from some name to an
    AsyncResult to wait for, or a list of AsyncResults (in which case a name
    will be generated for each one based on it's index).
    :param wait_time_sec: Time to wait for each AsyncResult before moving to
    the next one if the current one is not ready.
    :param max_retries: Maximal number of times to wait for the same
    AsyncResult, before giving up on it. None means never give up. If
    max_retries is exceeded, an error will be logged.
    :param re_raise: Whether to re-raise an exception thrown in on of the
    tasks and stop handling. If False, exception will be logged instead and
    handling will continue.
    :return: A generator, where each element is a tuple. The first element
    in the tuple is the name of the result, and the second element is the
    actual result. In case the task raised an exception, the second element
    will be None.
    """

    if isinstance(async_results, (list, tuple)):
        async_results = {i: r for i, r in enumerate(async_results)}
    elif isinstance(async_results, dict):
        pass
    else:
        raise ValueError("async_results must be a dict or list of AsyncResult")

    if len(async_results) == 0:
        raise ValueError("No async results to wait for")

    # Map result to number of retries
    retry_counts = {res_name: 0 for res_name in async_results.keys()}

    while len(retry_counts) > 0:
        retry_counts_next = {}

        for res_name, retry_count in retry_counts.items():
            res: AsyncResult = async_results[res_name]

            res.wait(wait_time_sec)
            if not res.ready():
                retries = retry_counts[res_name] + 1
                if max_retries is not None and retries > max_retries:
                    LOGGER.error(f"*** MAX RETRIES FOR RESULT {res_name}")
                else:
                    retry_counts_next[res_name] = retries
                continue

            try:
                yield res_name, res.get()
            except Exception as e:
                if re_raise:
                    raise e
                LOGGER.error(
                    f"AsyncResult {res_name} raised {type(e)}: " f"{e}", exc_info=e
                )
                yield res_name, None

        retry_counts = retry_counts_next


def _cleanup():
    global __GLOBAL_POOL
    if __GLOBAL_POOL is not None:
        LOGGER.info(f"Closing global pool...")
        __GLOBAL_POOL.terminate()
        __GLOBAL_POOL.join()
        _clean_worker_downloads(GLOBAL_WORKERS_DL_DIR)
        _remove_workers_dir(GLOBAL_WORKERS_DL_DIR)


atexit.register(_cleanup)
