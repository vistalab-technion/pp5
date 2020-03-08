import gzip
import logging
import os
import sys
from collections.abc import Mapping, Set, Sequence
from contextlib import contextmanager
from pathlib import Path

import requests
from requests import HTTPError

LOGGER = logging.getLogger(__name__)


def remote_dl(url: str, save_path: str, uncompress=False,
              skip_existing=False) -> Path:
    """
    Downloads contents of a remote file and saves it into a local file.
    :param url: The url to download from.
    :param save_path: Local file path to save to.
    :param uncompress: Whether to uncompress gzip files.
    :param skip_existing: Whether to skip download if a local file with
    the given path already exists.
    :return: A Path object for the downloaded file.
    """
    if skip_existing:
        if os.path.isfile(save_path) and os.path.getsize(save_path) > 0:
            LOGGER.debug(f"File {save_path} exists, skipping download...")
            return Path(save_path)

    req_headers = {'Accept-Encoding': 'gzip, identity'}
    with requests.get(url, stream=True, headers=req_headers) as response:
        response.raise_for_status()
        if 300 <= response.status_code < 400:
            raise HTTPError(f"Redirect {response.status_code} for url{url}",
                            response=response)

        if 'gzip' in response.headers.get('Content-Encoding', ''):
            uncompress = True

        save_dir = Path().joinpath(*Path(save_path).parts[:-1])
        os.makedirs(save_dir, exist_ok=True)

        with open(save_path, 'wb') as out_handle:
            try:
                if uncompress:
                    in_handle = gzip.GzipFile(fileobj=response.raw)
                else:
                    in_handle = response.raw
                out_handle.write(in_handle.read())
            finally:
                in_handle.close()

        size_bytes = os.path.getsize(save_path)
        LOGGER.info(f"Downloaded {save_path} ({size_bytes / 1024:.1f}kB)")
        return Path(save_path)


def deep_walk(obj, path=(), memo=None):
    str_types = (str, bytes)
    iteritems = lambda mapping: getattr(mapping, 'iteritems', mapping.items)()

    if memo is None:
        memo = set()
    iterator = None
    if isinstance(obj, Mapping):
        iterator = iteritems
    elif isinstance(obj, (Sequence, Set)) and not isinstance(obj, str_types):
        iterator = enumerate
    elif hasattr(obj, '__dict__'):
        iterator = lambda x: x.__dict__.items()

    if iterator:
        if id(obj) not in memo:
            memo.add(id(obj))
            for path_component, value in iterator(obj):
                for result in deep_walk(value, path + (path_component,), memo):
                    yield result
            memo.remove(id(obj))

    else:
        yield path, obj


@contextmanager
def out_redirected(stdout_stderr='stdout', to=os.devnull):
    '''
    Redirects stdout/stderr in a way that also affects C libraries called
    from python code.

    based on: https://stackoverflow.com/a/17954769/1230403
    but modified to support also stderr and to update loggers.
    '''
    assert stdout_stderr in {'stdout', 'stderr'}

    fd = getattr(sys, stdout_stderr).fileno()
    assert (stdout_stderr == 'stdout' and fd == 1) or \
           (stdout_stderr == 'stderr' and fd == 2)

    # Loggers which log to the stream being redirected must be modified
    logging_handlers = [h for h in logging.root.handlers
                        if isinstance(h, logging.StreamHandler)
                        and h.stream.fileno() == fd]

    def _redirect(to_stream):
        old_stream = getattr(sys, stdout_stderr)
        old_stream.flush()
        old_stream.close()

        os.dup2(to_stream.fileno(), fd)  # fd writes to 'to' file
        new_stream = os.fdopen(fd, 'w')
        setattr(sys, stdout_stderr, new_stream)  # Python writes to fd

        for lh in logging_handlers:
            try:
                lh.acquire()
                lh.stream = new_stream
            finally:
                lh.release()

    with os.fdopen(os.dup(fd), 'w') as old_stdout:
        with open(to, 'w') as file:
            _redirect(to_stream=file)
        try:
            yield  # allow code to be run with the redirected stdout
        finally:
            _redirect(to_stream=old_stdout)  # restore stdout.
            # buffering and flags such as CLOEXEC may be different
