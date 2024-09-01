import io
import abc
import json
import logging
import zipfile
import contextlib
from typing import Any, Dict, Union
from pathlib import Path

import pandas as pd

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

    @contextlib.contextmanager
    @abc.abstractmethod
    def _open(self, path: Union[Path, str], mode: str) -> io.IOBase:
        pass

    def _read_csv(self, file_path: Union[Path, str], **read_csv_kwargs):
        with self._open(file_path, "r") as fileobj:
            return pd.read_csv(fileobj, **read_csv_kwargs)


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
