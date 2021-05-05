import os
import pickle
import shutil
import logging
from abc import ABC
from typing import Dict, Union
from pathlib import Path

from pp5.collect import ParallelDataCollector

LOGGER = logging.getLogger(__name__)


class ParallelAnalyzer(ParallelDataCollector, ABC):
    """
    Base class for analyzers.
    """

    def __init__(
        self,
        analysis_name,
        dataset_dir: Union[str, Path],
        input_file: Union[str, Path],
        out_dir: Union[str, Path] = None,
        out_tag: str = None,
        clear_intermediate=False,
    ):
        """

        :param analysis_name: Name of the analysis this instance is used for.
        :param dataset_dir: Path to directory with the dataset files.
        :param input_file: Name of input file.
        :param out_dir: Path to output directory. If None, will be set to
        <dataset_dir>/results. Another directory based on the analysis name and
        out tag will be created within.
        :param out_tag: Tag for output files.
        :param clear_intermediate: Whether to clear intermediate folder.
        """
        self.analysis_name = analysis_name
        self.dataset_dir = Path(dataset_dir)
        self.out_tag = out_tag

        if not self.dataset_dir.is_dir():
            raise ValueError(f"Dataset dir {self.dataset_dir} not found")

        self.input_file = self.dataset_dir.joinpath(input_file)
        if not self.input_file.is_file():
            raise ValueError(f"Dataset file {self.input_file} not found")

        tag = f"-{self.out_tag}" if self.out_tag else ""
        out_dir = out_dir or self.dataset_dir.joinpath("results")
        super().__init__(
            id=f"{self.analysis_name}{tag}",
            out_dir=out_dir,
            tag=out_tag,
            create_zip=False,
        )

        # Create clean directory for intermediate results between steps
        # Note that super() init set out_dir to include the id (analysis name).
        self.intermediate_dir = self.out_dir.joinpath("_intermediate_")
        if clear_intermediate and self.intermediate_dir.exists():
            shutil.rmtree(str(self.intermediate_dir))
        os.makedirs(str(self.intermediate_dir), exist_ok=True)

        # Create dict for storing paths of intermediate results
        self._intermediate_files: Dict[str, Path] = {}

    def _dump_intermediate(self, name: str, obj):
        # Update dict of intermediate files
        path = self.intermediate_dir.joinpath(f"{name}.pkl")
        self._intermediate_files[name] = path

        with open(str(path), "wb") as f:
            pickle.dump(obj, f, protocol=4)

        size_mbytes = os.path.getsize(path) / 1024 / 1024
        LOGGER.info(f"Wrote intermediate file {path} ({size_mbytes:.1f}MB)")
        return path

    def _load_intermediate(self, name, allow_old=True, raise_if_missing=False):
        path = self.intermediate_dir.joinpath(f"{name}.pkl")

        if name not in self._intermediate_files:
            # Intermediate files might exist from a previous run we wish to
            # resume
            if allow_old:
                LOGGER.warning(f"Loading old intermediate file {path}")
            else:
                return None

        if not path.is_file():
            msg = f"Can't find intermediate file {path}"
            if raise_if_missing:
                raise ValueError(msg)
            else:
                LOGGER.warning(msg + ", skipping...")
                return None

        self._intermediate_files[name] = path
        with open(str(path), "rb") as f:
            obj = pickle.load(f)

        LOGGER.info(f"Loaded intermediate file {path}")
        return obj

    def analyze(self):
        """
        Performs all analysis steps defined in this analyzer.
        :return: Analysis metadata.
        """
        return self.collect()
