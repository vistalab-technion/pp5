import io
import abc
import json
import logging
import zipfile
import itertools as it
import contextlib
from typing import Any, Dict, Union, Callable
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from multiprocessing.context import SpawnContext

import pandas as pd
from pandas import DataFrame
from tqdm.auto import tqdm

from pp5.collect.base import (
    DATASET_DIRNAME,
    ALL_STRUCTS_FILENAME,
    COLLECTION_METADATA_FILENAME,
)

_LOG = logging.getLogger(__name__)


class CollectedDataset(abc.ABC):
    """
    A class for loading a collected dataset.
    """

    @property
    @abc.abstractmethod
    def name(self):
        """
        :return: The dataset name.
        """
        return

    @property
    @abc.abstractmethod
    def pdb_ids(self):
        """
        :return: The PDB IDs of all structures in the dataset.
        """
        pass

    @property
    @abc.abstractmethod
    def collection_metadata(self) -> Dict[str, Any]:
        """
        :return: The collection metadata.
        """
        pass

    @property
    @abc.abstractmethod
    def metadata_path(self) -> str:
        """
        :return: Path to the metadata file.
        """
        pass

    @property
    @abc.abstractmethod
    def prec_paths(self) -> Dict[str, str]:
        """
        :return: Dict from pdb_id to path to the prec file.
        """
        pass

    def load_metadata(self, **read_csv_kwargs) -> pd.DataFrame:
        """
        Load the metadata for all structures in the dataset.

        :return: The metadata as a DataFrame.
        """
        read_csv_kwargs = {
            **read_csv_kwargs,
            **dict(header=0, index_col=None),
        }
        return self._read_csv(self.metadata_path, **read_csv_kwargs)

    def load_prec(self, pdb_id: str, **read_csv_kwargs) -> pd.DataFrame:
        """
        Load the metadata for all structures in the dataset.

        :return: The metadata as a DataFrame.
        """
        if pdb_id not in self.prec_paths:
            raise ValueError(f"Structure {pdb_id} not found in dataset")

        read_csv_kwargs = {
            **read_csv_kwargs,
            **dict(header=0, index_col=None),
        }
        return self._read_csv(self.prec_paths[pdb_id], **read_csv_kwargs)

    def apply_parallel(
        self,
        apply_fn: Callable[[DataFrame], Any],
        workers: int = 1,
        chunksize: int = 100,
        limit: int = None,
    ) -> Dict[str, Any]:
        """
        Apply a function to each prec in the dataset in parallel, and collect results.

        :param apply_fn: The function to apply to each prec. Should accept a DataFrame
        representing the prec file as input. Can return any serializable type.
        :param workers: Number of worker processes to use.
        :param chunksize: The number of structures to process in each chunk. Chunks
        are sent to the workers processes.
        :param limit: Limit the number of structures to process. If not None,
        only the first `limit` structures will be processed.
        :return: A dictionary mapping pdb_id to the result of apply_fn.
        """
        pdb_id_to_result = {}
        pdb_ids = self.pdb_ids
        if limit:
            pdb_ids = pdb_ids[:limit]

        with ProcessPoolExecutor(
            max_workers=workers, mp_context=SpawnContext()
        ) as pool:
            map_results = pool.map(
                self._apply_fn_wrapper,
                pdb_ids,
                it.repeat(apply_fn, times=len(pdb_ids)),
                chunksize=chunksize,
            )
            with tqdm(total=len(pdb_ids), desc="applying") as pbar:
                for pdb_id, result in map_results:
                    pbar.set_postfix_str(pdb_id, refresh=False)
                    pbar.update()
                    pdb_id_to_result[pdb_id] = result

            return pdb_id_to_result

    @contextlib.contextmanager
    @abc.abstractmethod
    def _open(self, path: Union[Path, str], mode: str) -> io.IOBase:
        pass

    def _read_csv(self, file_path: Union[Path, str], **read_csv_kwargs):
        with self._open(file_path, "r") as fileobj:
            return pd.read_csv(fileobj, **read_csv_kwargs)

    def _apply_fn_wrapper(
        self, pdb_id: str, apply_fn: Callable[[DataFrame], Any]
    ) -> Any:
        """
        Wrapper for an apply function that loads the prec file for a given pdb_id and
        passes the resulting DataFrame to the apply function.
        This is needed so that the loading of the prec file can also be done in
        parallel.
        :param pdb_id: The pdb_id of the prec to load and process.
        :param apply_fn: The function to apply to the prec's dataframe.
        :return: A tuple (pdb_id, result of apply_fn).
        """
        df_prec = self.load_prec(pdb_id)
        try:
            return pdb_id, apply_fn(df_prec)
        except Exception as e:
            raise RuntimeError(f"Error processing {pdb_id}") from e


class FolderDataset(CollectedDataset):
    def __init__(self, dataset_dir_path: Union[Path, str]):
        """
        :param dataset_dir_path: The path to the folder file containing the dataset.
        The folder name is assumed to be the dataset name.
        """
        self.dataset_dir_path = Path(dataset_dir_path)

        if not self.dataset_dir_path.is_dir():
            raise FileNotFoundError(f"File not found: {self.dataset_dir_path}")

        self._name: str = self.dataset_dir_path.name
        self._collection_metadata_path: Path = (
            self.dataset_dir_path / COLLECTION_METADATA_FILENAME
        )
        if not self._collection_metadata_path.is_file():
            raise FileNotFoundError(f"File not found: {self._collection_metadata_path}")

        self._struct_metadata_path: Path = (
            self.dataset_dir_path / f"{ALL_STRUCTS_FILENAME}.csv"
        )
        if not self._struct_metadata_path.is_file():
            raise FileNotFoundError(f"Metadata not found: {self._struct_metadata_path}")

        self._prec_dir: Path = self.dataset_dir_path / DATASET_DIRNAME
        if not self._prec_dir.is_dir():
            raise FileNotFoundError(f"Prec dir not found: {self._prec_dir}")

        self._collection_metadata: Dict[str, Any] = {}
        self._prec_paths: Dict[str, str] = {}  # pdb_id -> path in prec dir

        # Load collection metadata
        with open(self._collection_metadata_path, "r") as fileobj:
            self._collection_metadata.update(json.load(fileobj))

        # Load prec file names
        for file_path in self._prec_dir.glob("*.csv"):
            pdb_id = Path(file_path).stem.split("-")[0].replace("_", ":")
            self._prec_paths[pdb_id] = str(file_path)

    @property
    def name(self):
        return self._name

    @property
    def metadata_path(self) -> str:
        return str(self._struct_metadata_path)

    @property
    def prec_paths(self) -> Dict[str, str]:
        return self._prec_paths.copy()

    @property
    def collection_metadata(self) -> Dict[str, Any]:
        return self._collection_metadata.copy()

    @property
    def pdb_ids(self):
        return tuple(self._prec_paths.keys())

    @contextlib.contextmanager
    def _open(self, path: Union[Path, str], mode: str) -> io.IOBase:
        with open(path, mode=mode) as fileobj:
            yield fileobj


class ZipDataset(CollectedDataset):
    """
    A class for loading a collected dataset from a zip file.
    """

    def __init__(
        self, dataset_zipfile_path: Union[Path, str], dataset_name: str = None
    ):
        """
        :param dataset_zipfile_path: The path to the zip file containing the dataset.
        :param dataset_name: Optional name for the dataset, in case the zipfile was
        renamed. This should be the name of the top-level directory inside the zipfile
        which contains the dataset.
        """
        self.zipfile_path = Path(dataset_zipfile_path)

        if not self.zipfile_path.is_file():
            raise FileNotFoundError(f"File not found: {self.zipfile_path}")

        if not dataset_name:
            dataset_name = self.zipfile_path.stem

        self._name: str = dataset_name
        self._collection_metadata_path: str = (
            f"{dataset_name}/{COLLECTION_METADATA_FILENAME}"
        )
        self._struct_metadata_path: str = f"{dataset_name}/{ALL_STRUCTS_FILENAME}.csv"
        self._prec_dir: str = f"{dataset_name}/{DATASET_DIRNAME}"

        self._collection_metadata: Dict[str, Any] = {}
        self._prec_paths: Dict[str, str] = {}  # pdb_id -> path in zip file
        with zipfile.ZipFile(self.zipfile_path, "r") as zip_file:
            # Load collection metadata
            with zip_file.open(self._collection_metadata_path, "r") as fileobj:
                self._collection_metadata.update(json.load(fileobj))

            # Load prec file names
            for file_path in zip_file.namelist():
                if file_path.startswith(self._prec_dir) and file_path.endswith(".csv"):
                    pdb_id = Path(file_path).stem.split("-")[0].replace("_", ":")
                    self._prec_paths[pdb_id] = file_path

    @property
    def name(self):
        return self._name

    @property
    def metadata_path(self) -> str:
        return self._struct_metadata_path

    @property
    def prec_paths(self) -> Dict[str, str]:
        return self._prec_paths.copy()

    @property
    def collection_metadata(self) -> Dict[str, Any]:
        return self._collection_metadata.copy()

    @property
    def pdb_ids(self):
        return tuple(self._prec_paths.keys())

    @contextlib.contextmanager
    def _open(self, path: Union[Path, str], mode: str) -> io.IOBase:
        with zipfile.ZipFile(self.zipfile_path, mode=mode) as zip_file:
            with zip_file.open(str(path), mode=mode) as fileobj:
                yield fileobj
