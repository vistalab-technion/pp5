from __future__ import annotations

import os
import json
import logging
import zipfile
import subprocess
from time import time
from typing import Union, Optional
from pathlib import Path

import numpy as np
import pandas as pd

import pp5
from pp5.external_dbs import pdb
from pp5.external_dbs.pdb import PDB_RCSB

LOGGER = logging.getLogger(__name__)


DEFAULT_ARPEGGIO_ARGS = dict(
    interaction_cutoff=4.5, use_conda_env="arpeggio", cache=True
)


class Arpeggio(object):
    """
    A wrapper for running the arpeggio tool for contact annotation.

    https://github.com/PDBeurope/arpeggio
    """

    def __init__(
        self,
        out_dir: Union[Path, str] = pp5.out_subdir("arpeggio"),
        interaction_cutoff: float = 0.1,
        arpeggio_command: Optional[str] = None,
        use_conda_env: Optional[str] = None,
        cache: bool = False,
        pdb_source: str = PDB_RCSB,
    ):
        """
        :param out_dir: Output directory. JSON files will be written there with the
        names <pdb_id>.json
        :param interaction_cutoff: Cutoff (in angstroms) for detected interactions.
        :param arpeggio_command: Custom command name or path to the arpeggio executable.
        :param use_conda_env: Name of conda environment to use. This is useful,
        since arpeggio can be tricky to install with new versions of python.
        If this arg is provided, the arpeggio command will be run via `conda run`.
        The conda executable will be detected by from the `CONDA_EXE` env variable.
        :param cache: Whether to load arpeggio results from cache if available.
        :param pdb_source: Source from which to obtain the pdb file.
        """

        self.out_dir = Path(out_dir)
        self.interaction_cutoff = interaction_cutoff
        self.arpeggio_command = arpeggio_command or "arpeggio"
        self.cache = cache
        self.pdb_source = pdb_source

        if use_conda_env:
            # Use conda run to execute the arpeggio command in the specified conda env.
            conda_exe = os.getenv("CONDA_EXE", "conda")
            self.arpeggio_command = (
                f"{conda_exe} run --no-capture-output -n {use_conda_env} "
                f"{self.arpeggio_command}"
            )

    def contact_df(self, pdb_id: str, single_sided: bool = False) -> pd.DataFrame:
        """
        :param pdb_id: The PDB ID to run arpeggio against. Must include chain.
        :param single_sided: Whether to include only on side of each contact as in
        the original arpeggio output (True), or to duplicate each contact to both
        sides it touches (False).
        :return: A dataframe with the arpeggio contacts
        """
        pdb_base_id, pdb_chain_id = pdb.split_id(pdb_id)
        if not pdb_chain_id:
            raise ValueError("Must specify a chain")

        arpeggio_out_path = self._run_arpeggio(pdb_id)

        LOGGER.info(f"Parsing arpeggio output from {arpeggio_out_path!s}")
        if "zip" in arpeggio_out_path.suffix:
            with zipfile.ZipFile(arpeggio_out_path, "r") as zipf:
                with zipf.open(arpeggio_out_path.stem) as f:
                    out_json = json.load(f)
        else:  # json
            with open(arpeggio_out_path, "r") as f:
                out_json = json.load(f)

        # Convert nested json to dataframe and sort the columns
        df: pd.DataFrame = pd.json_normalize(out_json).sort_index(axis=1)

        if not single_sided:
            df1 = df

            # Obtain matching bgn.* and end.* columns
            bgn_cols = [c for c in df1.columns if c.startswith("bgn")]
            end_cols = [c.replace("bgn", "end") for c in bgn_cols]
            assert all(c in df1.columns for c in end_cols)

            # Obtain begin and end data
            df_bgn = df1[bgn_cols]
            df_end = df1[end_cols]

            # Create a copy df where bgn and end are swapped
            df2 = df1.copy()
            df2[bgn_cols] = df_end.values
            df2[end_cols] = df_bgn.values

            # Sanity check
            assert np.all(df1[bgn_cols].values == df2[end_cols].values)
            assert np.all(df1[end_cols].values == df2[bgn_cols].values)

            # Create double-sided dataframe
            df = pd.concat([df1, df2])

        # Sort and index by (Chain, Residue)
        index_cols = ["bgn.auth_asym_id", "bgn.auth_seq_id"]
        df.sort_values(by=index_cols, inplace=True)
        df.set_index(index_cols, inplace=True)

        return df

    @classmethod
    def can_execute(
        cls,
        arpeggio_command: Optional[str] = None,
        use_conda_env: Optional[str] = None,
        **kw,
    ) -> bool:
        """
        Checks whether arpeggio can be executed on the current machine.
        Arguments are the same as for init.
        :return: True if arpeggio can be executed successfully.
        """
        arpeggio = cls(arpeggio_command=arpeggio_command, use_conda_env=use_conda_env)

        try:
            exit_code = subprocess.Popen(
                args=[*arpeggio.arpeggio_command.split(), "--help"],
                encoding="utf-8",
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                text=True,
                shell=False,
            ).wait(timeout=10)
        except Exception:
            return False

        return exit_code == 0

    def _run_arpeggio(self, pdb_id: str) -> Path:
        """
        Helper to run the arpeggio command line.
        :return: Path of arpeggio output file.
        """

        pdb_base_id, pdb_chain_id = pdb.split_id(pdb_id)

        # Use cache if available
        cached_out_filename = (
            f"{pdb_base_id.upper()}_"
            f"{pdb_chain_id.upper()}-"
            f"i{self.interaction_cutoff:.1f}-"
            f"{self.pdb_source}.json.zip"
        )
        cached_out_path = self.out_dir.absolute() / cached_out_filename
        if self.cache and cached_out_path.is_file():
            LOGGER.info(f"Loading cached arpegio result from {cached_out_path!s}")
            return cached_out_path

        # Download structure cif file
        pdb_cif_path = pdb.pdb_download(pdb_id, pdb_source=self.pdb_source)

        # Construct the command-line for the arpeggio executable
        cline = [
            *self.arpeggio_command.split(),
            *f"-o {self.out_dir.absolute()!s}".split(),
            *f"-s /{pdb_chain_id}//".split(),
            *f"-i {self.interaction_cutoff:.2f}".split(),
            f"{pdb_cif_path!s}",
        ]

        LOGGER.info(f"Executing arpeggio command:\n{str.join(' ', cline)}")

        # Execute
        start_time = time()
        child_proc = subprocess.Popen(
            args=cline,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            shell=False,
        )

        stdout, stderr = child_proc.communicate()
        elapsed_time = time() - start_time

        LOGGER.info(
            f"Arpeggio run completed in {elapsed_time:.2f}s with code"
            f"={child_proc.returncode}"
        )
        if child_proc.returncode != 0:
            raise ValueError(
                f"Arpeggio returned code {child_proc.returncode}\n"
                f"{stdout=}\n\n{stderr=}"
            )
        LOGGER.debug(f"Arpeggio output\n{stdout=}\n\n{stderr=}")

        # Cache the result
        out_file_path = self.out_dir.absolute() / f"{pdb_cif_path.stem}.json"
        if self.cache:
            with zipfile.ZipFile(
                cached_out_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
            ) as out_zipfile:
                out_zipfile.write(out_file_path, arcname=cached_out_path.stem)
            out_file_path.unlink()
            out_file_path = cached_out_path

        return out_file_path
