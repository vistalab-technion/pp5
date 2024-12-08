import os
import json
import pickle
import logging
from abc import abstractmethod
from json import JSONEncoder
from typing import Any, Dict, Union, Callable, Optional, Sequence
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile
from dataclasses import dataclass

import pandas as pd

import pp5
from pp5.utils import sort_dict, stable_hash, filelock_context

CACHE_FORMAT_JSON = "json"
CACHE_FORMAT_PICKLE = "pkl"
CACHE_FORMATS = {CACHE_FORMAT_JSON, CACHE_FORMAT_PICKLE}


LOGGER = logging.getLogger(__name__)


@dataclass
class CacheSettings:
    """
    Settings for caching objects to file.
    """

    cache_dir: Path
    cache_format: str = CACHE_FORMAT_JSON
    cache_compression: bool = False

    def __post_init__(self):
        if self.cache_format not in CACHE_FORMATS:
            raise ValueError(f"Invalid {self.cache_format=}")

    def __str__(self):
        return f"{self.cache_format}{'-compressed' if self.cache_compression else ''}"


class Cacheable(object):
    """
    Makes a class cacheable to file.
    """

    # Subclasses may override this with the desired settings.
    _CACHE_SETTINGS = CacheSettings(cache_dir=pp5.data_subdir("cache"))

    def __getstate__(self):
        return self.__dict__.copy()

    def __setstate__(self, state):
        self.__dict__.update(state)

    @abstractmethod
    def cache_attribs(self) -> Dict[str, Any]:
        """
        :return: The attributes which determine the cache filename.
        """
        pass

    @classmethod
    def _cache_filename_prefix(cls, cache_attribs: Dict[str, Any]) -> str:
        """
        Generates the prefix of the cache filename.
        :param cache_attribs: Attributes which determine the cache filename.
        :return: The prefix of the cache filename.
        """
        return cls.__name__.lower()

    @classmethod
    def _cache_filename(cls, cache_attribs: Dict[str, Any]) -> str:
        """
        Generates the cache filename.
        :param cache_attribs: The attributes which determine the cache filename.
        :return: The cache filename.
        """
        return (
            f"{cls._cache_filename_prefix(cache_attribs=cache_attribs)}"
            "-"
            f"{stable_hash(sort_dict(cache_attribs,by_value=False))}.json"
        )

    def to_cache(
        self,
        cache_dir: Optional[Union[str, Path]] = None,
        filename: Optional[Union[str, Path]] = None,
        **json_kws,
    ) -> Path:
        """
        Write the object to a human-readable text file (json) which
        can also be loaded later using from_cache.
        :param cache_dir: Directory of cached files.
        :param filename: Cached file name (without directory).
        :return: The path of the written file.
        """
        if cache_dir is None:
            cache_dir = self._CACHE_SETTINGS.cache_dir
        if filename is None:
            filename = self._cache_filename(self.cache_attribs())

        filepath = pp5.get_resource_path(cache_dir, filename)
        os.makedirs(str(filepath.parent), exist_ok=True)

        with filelock_context(filepath):
            with open(str(filepath), "w", encoding="utf-8") as f:
                json.dump(self.__getstate__(), f, indent=2, **json_kws)

            if self._CACHE_SETTINGS.cache_compression:
                zip_filepath = filepath.with_suffix(".zip")
                with ZipFile(
                    zip_filepath, "w", compression=ZIP_DEFLATED, compresslevel=6
                ) as fzip:
                    fzip.write(str(filepath), arcname=filename)

                filepath.unlink()
                filepath = zip_filepath

        file_size = os.path.getsize(filepath)
        file_size_str = (
            f"{file_size / 1024:.1f}kB"
            if file_size < 1024 * 1024
            else f"{file_size / 1024 / 1024:.1f}MB"
        )
        LOGGER.info(f"Wrote cache file: {filepath} ({file_size_str})")
        return filepath

    @classmethod
    def from_cache(
        cls,
        cache_dir: Optional[Union[str, Path]] = None,
        cache_attribs: Optional[Dict[str, Any]] = None,
        filename: Optional[Union[str, Path]] = None,
    ):
        """
        Load the object from a cached file.
        :param cache_dir: Directory of cached file.
        :param cache_attribs: Attributes which determine the cache filename.
        :param filename: Cached filename (without directory). Won't be used if
        cache_attribs is given.
        :return: The loaded object, or None if the file doesn't exist.
        """
        if not (cache_attribs or filename):
            raise ValueError("cache_attribs or filename must be given")

        if cache_dir is None:
            cache_dir = cls._CACHE_SETTINGS.cache_dir

        if filename is None:
            filename = cls._cache_filename(cache_attribs)

        filepath = pp5.get_resource_path(cache_dir, filename)

        obj = None

        with filelock_context(filepath):
            zip_filepath = filepath.with_suffix(".zip")
            if cls._CACHE_SETTINGS.cache_compression and zip_filepath.is_file():
                with ZipFile(zip_filepath, "r") as fzip:
                    fzip.extractall(path=zip_filepath.parent)

            if filepath.is_file():
                try:
                    with open(str(filepath), "r", encoding="utf-8") as f:
                        state_dict = json.load(f)
                        obj = cls.__new__(cls)
                        obj.__setstate__(state_dict)
                except Exception as e:
                    LOGGER.warning(
                        f"Failed to load cached {cls.__name__} {filepath} {e}"
                    )
                finally:
                    if cls._CACHE_SETTINGS.cache_compression:
                        filepath.unlink()
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


def cached_call(
    target_fn: Callable[[Any], Any],
    cache_file_basename: str,
    target_fn_args: Optional[Dict[str, Any]] = None,
    hash_ignore_args: Optional[Sequence[str]] = None,
    cache_dir: Path = None,
    clear_cache: bool = False,
    cache_dump_fn: Callable[[Any, Path], None] = None,
    cache_load_fn: Callable[[Path], Any] = None,
    out_file_suffix: str = "pkl",
) -> Any:
    """
    Calls a function, caching the result to a file. If the file exists, the result is
    loaded from it instead of recomputing. If the file does not exist, the result is
    computed by calling the target function, and then saved to the file.

    :param target_fn: The function to call. Should return a value that can be pickled.
    :param cache_file_basename: The base name of the cache file, without suffix.
    The actual file name will be this base name plus a hash of the arguments and the
    specified suffix.
    :param target_fn_args: The arguments to pass to the target function.
    :param hash_ignore_args: A list of argument names to ignore when computing the hash.
    :param cache_dir: The directory in which to save the cache file.
    :param clear_cache: Whether to delete the cache file if it exists and force a recompute.
    :param cache_dump_fn: A function to use to save the cache file. If not provided,
    the result will be pickled.
    :param cache_load_fn: A function to use to load the cache file. If not provided,
    the file will be unpickled.
    :param out_file_suffix: The suffix to use for the cache file.
    :return: The result of the target function.
    """
    target_fn_args = target_fn_args or {}
    if (cache_load_fn is None) != (cache_dump_fn is None):
        raise ValueError(
            "Both or neither of cache_load_fn and cache_dump_fn must be provided."
        )

    out_file_path = None
    if cache_dir:
        sorted_args = {k: target_fn_args[k] for k in sorted(target_fn_args)}

        for arg_name in hash_ignore_args or []:
            sorted_args.pop(arg_name, None)

        args_hash = stable_hash(tuple(sorted_args.items()))
        out_file_path = (
            cache_dir / f"{cache_file_basename}-{args_hash}.{out_file_suffix}"
        )

    if clear_cache and out_file_path and out_file_path.is_file():
        LOGGER.info(f"Removing {out_file_path}")
        os.unlink(out_file_path)

    if out_file_path and out_file_path.is_file():
        LOGGER.info(f"Loading from {out_file_path}")

        if cache_load_fn:
            target_fn_result = cache_load_fn(out_file_path)
        else:
            with open(out_file_path, "rb") as f:
                target_fn_result = pickle.load(f)

    else:
        LOGGER.info(f"Computing, will save to {out_file_path}")

        target_fn_result = target_fn(**target_fn_args)

        if out_file_path:
            if cache_dump_fn:
                cache_dump_fn(target_fn_result, out_file_path)
            else:
                with open(out_file_path, "wb") as f:
                    pickle.dump(target_fn_result, f)

            LOGGER.info(f"Saved to {out_file_path}")

    return target_fn_result


def cached_call_csv(
    target_fn: Callable[[Any], Any],
    cache_file_basename: str,
    target_fn_args: Optional[Dict[str, Any]] = None,
    hash_ignore_args: Optional[Sequence[str]] = None,
    cache_dir: Path = None,
    clear_cache: bool = False,
    to_csv_kwargs: Optional[Dict[str, Any]] = None,
    read_csv_kwargs: Optional[Dict[str, Any]] = None,
) -> Any:
    """
    Calls a function which returns a pandas dataframe, and caches the result to a CSV
    file. If the file exists, the result is loaded from the file instead of recomputing.

    All parameters are the same as for `cached_call`, with the following new arguments:
    :param to_csv_kwargs: Keyword arguments to pass to the `DataFrame.to_csv` method
    when saving the result to a file.
    :param read_csv_kwargs: Keyword arguments to pass to the `pd.read_csv` function
    when loading the result from a file.
    :return: The result of the target function.
    """

    to_csv_kwargs = to_csv_kwargs or {}
    read_csv_kwargs = read_csv_kwargs or {}
    return cached_call(
        target_fn=target_fn,
        cache_file_basename=cache_file_basename,
        target_fn_args=target_fn_args,
        hash_ignore_args=hash_ignore_args,
        cache_dir=cache_dir,
        clear_cache=clear_cache,
        # Configure cached_call to save/load the result as a CSV file:
        out_file_suffix="csv",
        cache_dump_fn=lambda result, path: result.to_csv(path, **to_csv_kwargs),
        cache_load_fn=lambda path: pd.read_csv(path, **read_csv_kwargs),
    )
