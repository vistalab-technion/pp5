import os
import json
import logging
from abc import abstractmethod
from json import JSONEncoder
from typing import Any, Dict, Union, Optional
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile
from dataclasses import dataclass

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
