from __future__ import annotations

import os
import json
import logging
import zipfile
import subprocess
from time import time
from typing import Union, Optional
from pathlib import Path
from functools import partial
from itertools import chain

import numpy as np
import pandas as pd

import pp5
from pp5.codons import ACIDS_3TO1, UNKNOWN_AA
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

    def contacts_df(self, pdb_id: str, single_sided: bool = False) -> pd.DataFrame:
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

        LOGGER.info(
            f"Generating contact features for {pdb_id} "
            f"(pdb_source={self.pdb_source}, "
            f"interaction_cutoff={self.interaction_cutoff})..."
        )

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

    def residue_contacts_df(self, pdb_id: str) -> pd.DataFrame:
        """
        Generates tertiary contact features per residue. Processes the raw arpeggio
        output by aggregating it at the residue level.

        :param pdb_id: The PDB ID to run arpeggio against. Must include chain.
        :return: A dataframe indexed by residue id and with columns corresponding to a
        summary of contacts per reisdue.
        """
        pdb_base_id, pdb_chain_id = pdb.split_id(pdb_id)

        # Invoke arpeggio to get the raw contact features.
        df_arp = self.contacts_df(pdb_id, single_sided=False)

        # Create a temp df to work with
        df = df_arp.copy().reset_index()

        # Convert 'contact' column to text
        df["contact"] = df["contact"].apply(lambda x: str.join(",", sorted(x)))

        # Ignore any water contacts
        idx_non_water = ~df["interacting_entities"].isin(
            ["SELECTION_WATER", "NON_SELECTION_WATER", "WATER_WATER"]
        )
        LOGGER.info(
            f"non-water proportion: "  #
            f"{sum(idx_non_water) / len(idx_non_water):.2f}"
        )

        # Ignore contacts which are of type 'proximal' only
        idx_non_proximal_only = df["contact"] != "proximal"
        LOGGER.info(
            f"non-proximal proportion: "
            f"{sum(idx_non_proximal_only) / len(idx_non_proximal_only):.2f}"
        )

        # Ignore contacts starting from another chain
        idx_non_other_chain = df["bgn.auth_asym_id"].str.lower() == pdb_chain_id.lower()
        LOGGER.info(
            f"start-in-chain proportion: "
            f"{sum(idx_non_other_chain) / len(idx_non_other_chain):.2f}"
        )

        # Find contacts ending on other chain
        idx_end_other_chain = df["end.auth_asym_id"].str.lower() != pdb_chain_id.lower()
        LOGGER.info(
            f"end-other-chain proportion: "
            f"{sum(idx_end_other_chain) / len(idx_end_other_chain):.2f}"
        )
        contact_any_ooc = df["end.auth_asym_id"].copy()
        contact_any_ooc[~idx_end_other_chain] = ""

        # Calculate sequence distance for each contact
        contact_sequence_dist = (df["end.auth_seq_id"] - df["bgn.auth_seq_id"]).abs()

        # If end is not on chain, set sdist to NaN to clarify that it's invalid
        contact_sequence_dist[idx_end_other_chain] = float("nan")

        # Find interactions with non-AAs (ligands)
        contact_non_aa = df["end.label_comp_id"].copy()
        idx_end_non_aa = ~contact_non_aa.isin(list(ACIDS_3TO1.keys()))
        contact_non_aa[~idx_end_non_aa] = ""
        LOGGER.info(
            f"end-non-aa proportion: "
            f"{sum(idx_end_non_aa) / len(idx_end_non_aa):.2f}"
        )

        # Filter only contacting and assign extra features
        df_filt = df[idx_non_water & idx_non_proximal_only & idx_non_other_chain]
        df_filt = df_filt.assign(
            # Note: this assign works because after filtering, the index remains intact
            contact_sdist=contact_sequence_dist,
            contact_any_ooc=contact_any_ooc,
            contact_non_aa=contact_non_aa,
        )
        df_filt = df_filt.drop("bgn.auth_asym_id", axis="columns")
        df_filt = df_filt.astype({"bgn.auth_seq_id": str})
        df_filt = df_filt.set_index(["bgn.auth_seq_id"])
        df_filt = df_filt.sort_values(by=["bgn.auth_seq_id"])
        df_filt = df_filt.rename_axis("res_id")

        # Aggregate contacts per AA
        def _agg_join(items):
            return str.join(",", [str(it) for it in items])

        def _agg_join_aas(items):
            return _agg_join(ACIDS_3TO1.get(aa, UNKNOWN_AA) for aa in items)

        def _agg_join_unique(items):
            return _agg_join(
                sorted(set(chain(*[str.split(it, ",") for it in items if it])))
            )

        def _join_aas_resids(row: pd.Series) -> str:
            return str.join(
                ",",
                map(
                    partial(str.join, ""),
                    zip(
                        str.split(row.contact_aas, ","),
                        str.split(row.contact_resids, ","),
                    ),
                ),
            )

        df_groups = df_filt.groupby(by=["res_id"]).aggregate(
            {
                # contacts count and type (unique)
                "contact": ["count", _agg_join_unique],
                # distances
                # note: min and max will ignore nans, and the lambda will count them
                "distance": [
                    "min",
                    "max",
                ],
                "contact_sdist": ["min", "max"],
                # OOC and non-AA contacts
                "contact_any_ooc": [_agg_join_unique],
                "contact_non_aa": [_agg_join_unique],
                # contact AAs and locations
                "end.label_comp_id": [_agg_join_aas],
                "end.auth_seq_id": [_agg_join],
            }
        )

        df_contacts = df_groups.set_axis(
            labels=[
                # count and type
                "contact_count",
                "contact_types",
                # distances
                "contact_dmin",
                "contact_dmax",
                "contact_smin",
                "contact_smax",
                # OOC and non-AA contacts
                "contact_ooc",
                "contact_non_aa",
                # contact AAs and locations
                "contact_aas",
                "contact_resids",
            ],
            axis="columns",
        )

        # Fix nans
        df_contacts["contact_count"].fillna(0, inplace=True)

        # Combine the contact AAs and residue ids columns together
        df_contacts["contact_aas"] = df_contacts.apply(func=_join_aas_resids, axis=1)
        df_contacts.drop("contact_resids", axis=1, inplace=True)

        df_contacts = (
            df_contacts.reset_index()
            .astype(
                {
                    "res_id": str,
                    "contact_count": pd.Int64Dtype(),
                    "contact_smin": pd.Int64Dtype(),
                    "contact_smax": pd.Int64Dtype(),
                }
            )
            .set_index("res_id")
            .sort_values(by=["res_id"])
        )

        return df_contacts

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