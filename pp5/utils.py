import gzip
import importlib
import logging
import os
import sys
from collections.abc import Mapping, Set, Sequence
import contextlib
from io import UnsupportedOperation
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from IPython import get_ipython
from requests import HTTPError

import pp5

LOGGER = logging.getLogger(__name__)


def requests_retry(retries: int = None,
                   backoff_factor: float = 0.1,
                   status_forcelist: tuple = (413, 429, 500, 502, 503, 504),
                   session: requests.Session = None, ):
    """
    Creates a requests.Session configured to retry a request in case of
    failure.
    Based on:
    https://www.peterbe.com/plog/best-practice-with-retries-with-requests

    :param retries: Number of times to retry. Default is taken from pp5
    config.
    :param backoff_factor: Determines number of seconds to sleep between
    retry requests, using the following formula:
    {backoff factor} * (2 ^ ({number of total retries} - 1))
    :param status_forcelist: List of HTTP status codes to retry for
    :param session: Existing session object.
    :return: A session object.
    """
    if retries is None:
        retries = pp5.get_config('REQUEST_RETRIES')

    session = session or requests.Session()

    # Docs for Retry are here:
    # https://urllib3.readthedocs.io/en/latest/reference/urllib3.util.html
    retry = Retry(
        # Number of retries for various error types
        total=retries, read=retries, connect=retries, redirect=retries,
        # Retry on any HTTP verb, including POST
        method_whitelist=False,
        # List of status codes to retry for
        status_forcelist=status_forcelist,
        # See formula above
        backoff_factor=backoff_factor,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session


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
    with requests_retry().get(url, stream=True, headers=req_headers) as r:
        r.raise_for_status()
        if 300 <= r.status_code < 400:
            raise HTTPError(f"Redirect {r.status_code} for url{url}",
                            response=r)

        if 'gzip' in r.headers.get('Content-Encoding', ''):
            uncompress = True

        save_dir = Path().joinpath(*Path(save_path).parts[:-1])
        os.makedirs(save_dir, exist_ok=True)

        with open(save_path, 'wb') as out_handle:
            try:
                if uncompress:
                    in_handle = gzip.GzipFile(fileobj=r.raw)
                else:
                    in_handle = r.raw
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


def is_interactive():
    """
    :return: Whether python is running as an interactive shell.
    """
    # main = importlib.import_module('__main__')
    # return not hasattr(main, '__file__')
    ipy = get_ipython()
    return ipy is not None


@contextlib.contextmanager
def out_redirected(stdout_stderr='stdout', to=os.devnull,
                   standard_fds_only=True, no_interactive=True):
    """
    Redirects stdout/stderr in a way that also affects C libraries called
    from python code.

    Based on: https://stackoverflow.com/a/17954769/1230403
    but modified to support also stderr and to update loggers.
    See also:
    https://eli.thegreenplace.net/2015/redirecting-all-kinds-of-stdout-in-python/
    """
    assert stdout_stderr in {'stdout', 'stderr'}

    # In interactive python, don't redirect by changing file-descriptors
    # because this will crash the interactive shell (which also uses these
    # files). Use the regular contextlib context managers.
    if no_interactive and is_interactive():
        if stdout_stderr == 'stdout':
            redirect_fn = contextlib.redirect_stdout
        else:
            redirect_fn = contextlib.redirect_stderr

        with open(to, 'w') as file:
            with redirect_fn(file):
                yield
        return

    fd = getattr(sys, stdout_stderr).fileno()

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

    # By default we don't want to change any non-standard file descriptors,
    # so we only change 1 and 2
    if standard_fds_only and \
            not ((stdout_stderr == 'stdout' and fd == 1) or
                 (stdout_stderr == 'stderr' and fd == 2)):
        LOGGER.warning(f"None standard fd {stdout_stderr}={fd}")
        yield

    # In non-interactive python, we redirect by changing file descriptors.
    # The advantage is that this affect non-python code in this process,
    # such as compiled C libraries.
    else:
        with os.fdopen(os.dup(fd), 'w') as old_stdout:
            with open(to, 'w') as file:
                _redirect(to_stream=file)
            try:
                yield  # allow code to be run with the redirected stdout
            finally:
                _redirect(to_stream=old_stdout)  # restore stdout.
                # buffering and flags such as CLOEXEC may be different
