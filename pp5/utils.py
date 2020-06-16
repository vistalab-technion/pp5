import gzip
import importlib
import json
import logging
import os
import random
import sys
from collections.abc import Mapping, Set, Sequence
import contextlib
from datetime import datetime, timedelta
from io import UnsupportedOperation
from json import JSONEncoder
from pathlib import Path
from typing import Union, Callable, Any

import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from IPython import get_ipython
from requests import HTTPError

import pp5

LOGGER = logging.getLogger(__name__)


def requests_retry(retries: int = None,
                   backoff: float = 0.1,
                   status_forcelist: tuple = (413, 429, 500, 502, 503, 504),
                   session: requests.Session = None, ):
    """
    Creates a requests.Session configured to retry a request in case of
    failure.
    Based on:
    https://www.peterbe.com/plog/best-practice-with-retries-with-requests

    :param retries: Number of times to retry. Default is taken from pp5
    config.
    :param backoff: Determines number of seconds to sleep between
    retry requests, using the following formula:
    backoff * (2 ^ ({number of total retries} - 1))
    :param status_forcelist: List of HTTP status codes to retry for
    :param session: Existing session object.
    :return: A session object.
    """
    if retries is None:
        retries = pp5.get_config('REQUEST_RETRIES')

    session = session or requests.Session()

    # Randomize backoff a bit (20%)
    delta = random.uniform(-backoff * .2, backoff * .2)
    backoff += delta

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
        backoff_factor=backoff,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session


def remote_dl(url: str, save_path: str, uncompress=False,
              skip_existing=False, retries: int = None) -> Path:
    """
    Downloads contents of a remote file and saves it into a local file.
    :param url: The url to download from.
    :param save_path: Local file path to save to.
    :param uncompress: Whether to uncompress gzip files.
    :param skip_existing: Whether to skip download if a local file with
    the given path already exists.
    :param retries: Number of times to retry on failure. None means use
    default from config.
    :return: A Path object for the downloaded file.
    """
    if skip_existing:
        if os.path.isfile(save_path) and os.path.getsize(save_path) > 0:
            LOGGER.debug(f"File {save_path} exists, skipping download...")
            return Path(save_path)

    req_headers = {'Accept-Encoding': 'gzip, identity'}
    with requests_retry(retries=retries) \
            .get(url, stream=True, headers=req_headers) as r:
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


def elapsed_seconds_to_dhms(elapsed_sec: float):
    """
    Converts elapsed time in seconds to a string containing days, hours,
    minutes, seconds.
    :param elapsed_sec: Elapsed time to convert, in seconds.
    :return: A string.
    """
    dt = datetime(1, 1, 1) + timedelta(seconds=elapsed_sec)
    d, h, m, s = dt.day - 1, dt.hour, dt.minute, dt.second

    return f'{d:02d}+{h:02d}:{m:02d}:{s:02d}'


def sort_dict(d: dict, by_value=True, selector: Callable[[Any], Any] = None):
    """
    Sorts a dict by key or value.
    Assumes python 3.6+ since that's when dicts became ordered.
    :param d: A dict.
    :param by_value: Whether to sort by values (true) or by keys.
    :param selector: Function to apply to key or value (depending on
    by_value) to get the actual value for sorting. Useful in case e.g. the
    values in the given dict are complex objects from which the sort key
    needs to be extracted/computed. Also useful for reversing sort order,
    e.g. if selector=(lambda x: -x).
    :return: A new dict, which is the old one sorted.
    """
    i = 1 if by_value else 0
    selector = (lambda x: x) if selector is None else selector

    def sort_key(kv: tuple):
        return selector(kv[i])

    return {k: v for k, v in sorted(d.items(), key=sort_key)}


class JSONCacheableMixin(object):
    """
    Makes a class cacheable to JSON.
    """

    def __getstate__(self):
        return self.__dict__.copy()

    def __setstate__(self, state):
        self.__dict__.update(state)

    def to_cache(self,
                 cache_dir: Union[str, Path],
                 filename: Union[str, Path],
                 **json_kws) -> Path:
        """
        Write the object to a human-readable text file (json) which
        can also be loaded later using from_cache.
        :param cache_dir: Directory of cached files.
        :param filename: Cached file name (without directory).
        :return: The path of the written file.
        """
        filepath = pp5.get_resource_path(cache_dir, filename)
        os.makedirs(str(filepath.parent), exist_ok=True)

        with open(str(filepath), 'w', encoding='utf-8') as f:
            json.dump(self.__getstate__(), f, **json_kws)

        LOGGER.info(f'Wrote {self} to {filepath}')
        return filepath

    @classmethod
    def from_cache(cls,
                   cache_dir: Union[str, Path],
                   filename: Union[str, Path]):
        """
        Load the object from a cached file.
        :param cache_dir: Directory of cached file.
        :param filename: Cached file name (without directory).
        :return: The loaded object, or None if the file doesn't exist.
        """

        filepath = pp5.get_resource_path(cache_dir, filename)

        obj = None
        if filepath.is_file():
            try:
                with open(str(filepath), 'r', encoding='utf-8') as f:
                    state_dict = json.load(f)
                    obj = cls.__new__(cls)
                    obj.__setstate__(state_dict)
            except Exception as e:
                LOGGER.warning(
                    f'Failed to load cached {cls.__name__} {filepath} {e}')
        return obj


class ReprJSONEncoder(JSONEncoder):
    """
    A JSONEncoder that converts an object to it's representation string in
    case it's not serializable.
    """
    def default(self, o: Any) -> Any:
        try:
            return repr(o)
        except Exception as e:
            pass
        # Let the base class default method raise the TypeError
        return JSONEncoder.default(self, o)
