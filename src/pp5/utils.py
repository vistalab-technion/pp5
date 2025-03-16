import os
import sys
import gzip
import pickle
import random
import hashlib
import logging
import contextlib
from typing import Any, Union, Callable
from pathlib import Path
from datetime import datetime, timedelta
from collections.abc import Set, Mapping, Sequence

import yaml
import requests
from IPython import get_ipython
from urllib3 import Retry
from filelock import FileLock
from requests import HTTPError
from requests.adapters import HTTPAdapter

import pp5
from pp5 import TEMP_LOCKS_DIR

LOGGER = logging.getLogger(__name__)


def requests_retry(
    retries: int = None,
    backoff: float = 10,
    status_forcelist: tuple = (413, 429, 500, 502, 503, 504),
    session: requests.Session = None,
):
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
        retries = pp5.get_config("REQUEST_RETRIES")

    session = session or requests.Session()

    # Randomize backoff a bit (20%)
    delta = random.uniform(-backoff * 0.2, backoff * 0.2)
    backoff += delta

    # Docs for Retry are here:
    # https://urllib3.readthedocs.io/en/latest/reference/urllib3.util.html
    retry = Retry(
        # Number of retries for various error types
        total=retries,
        read=retries,
        connect=retries,
        redirect=retries,
        # List of status codes to retry for
        status_forcelist=status_forcelist,
        # See formula above
        backoff_factor=backoff,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


@contextlib.contextmanager
def filelock_context(
    path: Union[str, Path],
    lockfile_basedir: Path = TEMP_LOCKS_DIR,
    cleanup: bool = False,
):
    """
    :param path: The path to lock.
    :param lockfile_basedir: Base dir in which to create the lockfile.
    :param cleanup: Whether to delete the lockfile after the context exits.
    Can cause concurrency issues.
    :return:
    """
    lockfile_path_non_absolute = str(path).strip(os.sep) + ".lock"
    lockfile_path = lockfile_basedir / lockfile_path_non_absolute
    lockfile_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with FileLock(lockfile_path):
            yield
    finally:
        if cleanup:
            lockfile_path.unlink(missing_ok=True)


def remote_dl(
    url: str,
    save_path: Union[Path, str],
    uncompress=False,
    skip_existing=False,
    retries: int = None,
) -> Path:
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

    with filelock_context(save_path):
        if skip_existing:
            if os.path.isfile(save_path) and os.path.getsize(save_path) > 0:
                LOGGER.debug(f"File {save_path} exists, skipping download...")
                return Path(save_path)

        req_headers = {"Accept-Encoding": "gzip, identity"}
        with requests_retry(retries=retries).get(
            url, stream=True, headers=req_headers
        ) as r:
            r.raise_for_status()
            if 300 <= r.status_code < 400:
                raise HTTPError(f"Redirect {r.status_code} for url{url}", response=r)

            if "gzip" in r.headers.get("Content-Encoding", ""):
                uncompress = True

            save_dir = Path().joinpath(*Path(save_path).parts[:-1])
            os.makedirs(save_dir, exist_ok=True)

            with open(save_path, "wb") as out_handle:
                try:
                    if uncompress:
                        in_handle = gzip.GzipFile(fileobj=r.raw)
                    else:
                        in_handle = r.raw
                    out_handle.write(in_handle.read())
                finally:
                    in_handle.close()

            size_bytes = os.path.getsize(save_path)
            LOGGER.info(
                f"Downloaded {save_path} ({size_bytes / 1024:.1f}kB) from {url}"
            )
            return Path(save_path)


def deep_walk(obj, path=(), memo=None):
    str_types = (str, bytes)
    iteritems = lambda mapping: getattr(mapping, "iteritems", mapping.items)()

    if memo is None:
        memo = set()
    iterator = None
    if isinstance(obj, Mapping):
        iterator = iteritems
    elif isinstance(obj, (Sequence, Set)) and not isinstance(obj, str_types):
        iterator = enumerate
    elif hasattr(obj, "__dict__"):
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


def is_pytest():
    """
    :return: Whether we're currently running inside pytest
    """
    return "PYTEST_CURRENT_TEST" in os.environ


@contextlib.contextmanager
def out_redirected(
    stdout_stderr="stdout", to=os.devnull, standard_fds_only=True, no_interactive=True
):
    """
    Redirects stdout/stderr in a way that also affects C libraries called
    from python code.

    Based on: https://stackoverflow.com/a/17954769/1230403
    but modified to support also stderr and to update loggers.
    See also:
    https://eli.thegreenplace.net/2015/redirecting-all-kinds-of-stdout-in-python/
    """
    assert stdout_stderr in {"stdout", "stderr"}

    # In interactive python or in unit tests, don't redirect by changing
    # file-descriptors # because this will crash the interactive shell
    # (which also uses these files). In tests, this messed with pytest's capturing of
    # stdout.
    # Use the regular contextlib context managers.
    if no_interactive and (is_interactive() or is_pytest()):
        if stdout_stderr == "stdout":
            redirect_fn = contextlib.redirect_stdout
        else:
            redirect_fn = contextlib.redirect_stderr

        with open(to, "w") as file:
            with redirect_fn(file):
                yield
        return

    fd = getattr(sys, stdout_stderr).fileno()

    # Loggers which log to the stream being redirected must be modified
    logging_handlers = [
        h
        for h in logging.root.handlers
        if isinstance(h, logging.StreamHandler) and h.stream.fileno() == fd
    ]

    def _redirect(to_stream):
        old_stream = getattr(sys, stdout_stderr)
        old_stream.flush()
        old_stream.close()

        os.dup2(to_stream.fileno(), fd)  # fd writes to 'to' file
        new_stream = os.fdopen(fd, "w")
        setattr(sys, stdout_stderr, new_stream)  # Python writes to fd

        for lh in logging_handlers:
            try:
                lh.acquire()
                lh.stream = new_stream
            finally:
                lh.release()

    # By default we don't want to change any non-standard file descriptors,
    # so we only change 1 and 2
    if standard_fds_only and not (
        (stdout_stderr == "stdout" and fd == 1)
        or (stdout_stderr == "stderr" and fd == 2)
    ):
        LOGGER.warning(f"None standard fd {stdout_stderr}={fd}")
        yield

    # In non-interactive python, we redirect by changing file descriptors.
    # The advantage is that this affect non-python code in this process,
    # such as compiled C libraries.
    else:
        with os.fdopen(os.dup(fd), "w") as old_stdout:
            with open(to, "w") as file:
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

    return f"{d:02d}+{h:02d}:{m:02d}:{s:02d}"


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


def stable_hash(obj: Any, hash_len: int = 8) -> str:
    """
    Generates a stable hash for general python objects, as a hexadecimal string. Stable
    means that the exact-same input will produce exactly the same output, even across
    machines and processes. The provided object must be pickleable.

    :param obj: A python object. Must be pickle-able.
    :param hash_len: Desired length of hash string.
    :return: A string of the requested length comprised of hexadecimal digits,
        representing a number which is the hash value.
    """
    if hash_len < 2:
        raise ValueError(f"Invalid {hash_len=}, must be > 1")

    def _hash(bytelike: bytes) -> str:
        return hashlib.blake2b(bytelike, digest_size=hash_len // 2).hexdigest()

    obj_bytes: bytes = pickle.dumps(obj)

    return _hash(obj_bytes)


class ProteinInitError(ValueError):
    pass


class YamlDict(dict):
    """
    A dict subclass that writes itself to a yaml file on any change.

    Notes:
    1. This implementation assumes that all keys/values are serializable to yaml.
    2. This implementation is not efficient for large dictionaries,
       as it reads/writes the entire dictionary on every change.
    """

    def __init__(self, yaml_path: Union[str, Path], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.yaml_path = Path(yaml_path)
        self._load()

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        self._save()

    def __delitem__(self, key):
        super().__delitem__(key)
        self._save()

    def clear(self):
        super().clear()
        self._save()

    def pop(self, key, default=None):
        value = super().pop(key, default)
        self._save()
        return value

    def setdefault(self, key, default=None):
        value = super().setdefault(key, default)
        self._save()
        return value

    def update(self, m, /, **kwargs):
        super().update(m, **kwargs)
        self._save()

    def _load(self):
        if self.yaml_path.exists():
            with open(self.yaml_path, "r") as f:
                super().update(yaml.safe_load(f))

    def _save(self):
        self.yaml_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.yaml_path, "w") as f:
            yaml.dump(dict(self), f)
