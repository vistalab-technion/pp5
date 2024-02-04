import os
import atexit
import shutil
import signal
import logging
import contextlib
import multiprocessing as mp
import multiprocessing.pool
from typing import (
    Any,
    Set,
    Dict,
    List,
    Tuple,
    Union,
    TypeVar,
    Generator,
    ContextManager,
)
from pathlib import Path
from collections import OrderedDict
from multiprocessing.pool import AsyncResult

import pp5
from pp5 import BASE_TEMP_DIR

LOGGER = logging.getLogger(__name__)

__GLOBAL_POOL: mp.pool.Pool = None
BASE_WORKERS_DL_DIR = BASE_TEMP_DIR.joinpath("workers")
GLOBAL_WORKERS_DL_DIR = BASE_WORKERS_DL_DIR.joinpath(f"_global_{os.getpid()}")


def _worker_process_init(base_workers_dl_dir: Path, pp5_config: dict, *args):
    # ignore SIGINT in workers, it should only be handled in the main process
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    os.makedirs(str(base_workers_dl_dir), exist_ok=True)

    # Provide each process with a unique download folder.
    pid = os.getpid()
    LOGGER.info(f"Initializing process {pid} ...")

    # Set global config
    for key, value in pp5_config.items():
        pp5.set_config(key, value)


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
            initargs=(base_workers_dl_dir, pp5.get_all_config()),
        )
    try:
        yield __GLOBAL_POOL
    finally:
        pass


@contextlib.contextmanager
def pool(name: str, processes=None, context="spawn") -> ContextManager[mp.pool.Pool]:
    base_workers_dl_dir = BASE_WORKERS_DL_DIR.joinpath(name)
    mp_ctx = mp.get_context(context)
    processes = processes if processes is not None else pp5.get_config("MAX_PROCESSES")

    try:
        LOGGER.info(f"Starting pool {name} with {processes} processes")
        with mp_ctx.Pool(
            processes,
            initializer=_worker_process_init,
            initargs=(base_workers_dl_dir, pp5.get_all_config()),
        ) as p:
            yield p
    finally:
        _remove_workers_dir(base_workers_dl_dir)


_T = TypeVar("_T")


def yield_async_results(
    async_results: Union[Dict[_T, AsyncResult], List[AsyncResult]],
    wait_time_sec=0.1,
    max_retries=None,
    re_raise=False,
) -> Generator[Tuple[_T, Any], None, None]:
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

    # Total number of times we waited for each result
    try_counts = {res_name: -1 for res_name in async_results.keys()}

    # Track which results were already processed
    success_results: Set[_T] = set()
    failed_results: Set[_T] = set()

    def _yield_result(_res_name: _T, _res: AsyncResult):
        try:
            success_results.add(_res_name)
            yield _res_name, _res.get()
        except Exception as e:
            if re_raise:
                raise e
            LOGGER.error(
                f"AsyncResult {_res_name} raised {type(e)}: " f"{e}", exc_info=e
            )
            failed_results.add(_res_name)
            yield _res_name, None

    while True:
        # Split by whether the result is ready, so we can get these without waiting.
        # Note that we maintain the order of the original dict, assuming that first
        # results started earlier and are more likely to be ready first.
        ready_results = {}
        not_ready_results = OrderedDict()  # OrderedDict supports popitem(last=False)
        for res_name, res in async_results.items():
            if res_name in success_results or res_name in failed_results:
                continue
            if res.ready():
                ready_results[res_name] = res
            else:
                not_ready_results[res_name] = res

        # Yield all the ready results
        for res_name, res in ready_results.items():
            yield from _yield_result(res_name, res)

        # Break if there's nothing left to do
        if not len(not_ready_results):
            LOGGER.info(f"Finished processing {len(async_results)} async results")
            break

        # Get next not-ready result
        res_name, res = not_ready_results.popitem(last=False)

        # Wait for it to become ready
        res.wait(wait_time_sec)

        # Result is ready: yield it (or log error if it raised an exception)
        if res.ready():
            yield from _yield_result(res_name, res)

        # Update try counter for this result
        try_counts[res_name] += 1
        if try_counts[res_name] > max_retries:
            failed_results.add(res_name)
            LOGGER.error(f"*** MAX RETRIES REACHED FOR {res_name}")

    # Sanity: Make sure we processed all AsyncResults
    assert len(success_results) + len(failed_results) == len(async_results)


def _cleanup():
    global __GLOBAL_POOL
    if __GLOBAL_POOL is not None:
        LOGGER.info(f"Closing global pool...")
        __GLOBAL_POOL.terminate()
        __GLOBAL_POOL.join()
        _remove_workers_dir(GLOBAL_WORKERS_DL_DIR)


atexit.register(_cleanup)
