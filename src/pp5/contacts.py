from __future__ import annotations

import os
import json
import logging
import zipfile
import subprocess
from abc import ABC, abstractmethod
from time import time
from typing import Set, Dict, List, Tuple, Union, Optional, Sequence
from pathlib import Path
from functools import partial
from itertools import chain

import numpy as np
import pandas as pd
from Bio.PDB import NeighborSearch
from Bio.PDB.Atom import Atom
from Bio.PDB.Residue import Residue, DisorderedResidue

import pp5
from pp5.codons import ACIDS_3TO1, UNKNOWN_AA
from pp5.backbone import NO_ALTLOC, altloc_ctx_all, atom_altloc_ids
from pp5.external_dbs import pdb
from pp5.external_dbs.pdb import PDB_RCSB

LOGGER = logging.getLogger(__name__)


CONTACT_DEFAULT_RADIUS = 5.0  # Angstroms
CONTACT_METHOD_ARPEGGIO = "arp"
CONTACT_METHOD_NEIGHBOR = "ns"
CONTACT_METHODS = (CONTACT_METHOD_ARPEGGIO, CONTACT_METHOD_NEIGHBOR)

DEFAULT_ARPEGGIO_ARGS = dict(
    interaction_cutoff=4.5, use_conda_env="arpeggio", cache=True
)


def res_to_id(res: Residue) -> str:
    """
    Converts a biopython residue object to a string representing its ID.
    """
    return str.join("", map(str, res.get_id())).strip()


def format_residue_contact(
    tgt_chain_id: str,
    tgt_resname: str,
    tgt_seq_idx: int,
    tgt_altloc: str,
    contact_dist: float,
) -> str:
    """
    Formats a residue contact as a string.

    :param tgt_chain_id: The chain ID of the target residue.
    :param tgt_resname: The residue name of the target residue.
    :param tgt_seq_idx: The sequence index of the target residue.
    :param tgt_altloc: The altloc of the target residue.
    :param contact_dist: The distance to the contact residue.
    :return: A string representing the contact.
    Format is:
        chain:resname:seq_idx[-altloc]:contact_dist
    Where the [-altloc] part is only included if the altloc exists.
    """
    tgt_resname = ACIDS_3TO1.get(tgt_resname, tgt_resname)
    tgt_altloc = f"-{tgt_altloc}" if tgt_altloc else ""
    _contact_dist_str = f"{contact_dist:.2f}"
    return (
        f"{tgt_chain_id}"
        f":{tgt_resname}"
        f":{tgt_seq_idx}{tgt_altloc}"
        f":{_contact_dist_str}"
    )


class ContactsAssigner(ABC):
    """
    Calculates tertiary contacts for a given residue.
    """

    def __init__(
        self,
        pdb_id: str,
        pdb_source: str,
        contact_radius: float = CONTACT_DEFAULT_RADIUS,
        with_altlocs: bool = False,
    ):
        """
        :param pdb_id: The PDB ID to assign contacts for.
        :param pdb_source: The source from which to obtain the PDB file.
        :param contact_radius: The radius (in angstroms) to use for contact detection.
        :param with_altlocs: Whether to include altloc atoms in the contact detection.
        """
        self.pdb_id = pdb_id
        self.pdb_source = pdb_source
        self.contact_radius = contact_radius
        self.with_altlocs = with_altlocs

    @abstractmethod
    def assign(self, res: Residue) -> Dict[str, Optional[ResidueContacts]]:
        """
        Assigns contacts to a given residue.
        :param res: The residue to assign contacts for.
        :return: A dict mapping altloc ids to ResidueContacts objects.
        """
        pass


class ArpeggioContactsAssigner(ContactsAssigner):
    """
    Uses the arpeggio tool to assign contacts. Does not support altlocs.
    """

    def __init__(
        self,
        pdb_id: str,
        pdb_source: str,
        contact_radius: float = CONTACT_DEFAULT_RADIUS,
        **arpeggio_kwargs,
    ):
        super().__init__(
            pdb_id=pdb_id,
            pdb_source=pdb_source,
            contact_radius=contact_radius,
            with_altlocs=False,
        )

        self.arpeggio = Arpeggio(
            **{
                **DEFAULT_ARPEGGIO_ARGS,
                **arpeggio_kwargs,
                **dict(
                    interaction_cutoff=self.contact_radius,
                    pdb_source=self.pdb_source,
                ),
            }
        )

        self.contacts_df = self.arpeggio.residue_contacts_df(pdb_id=self.pdb_id)
        contacts_df_rows = self.contacts_df.reset_index().transpose().to_dict()
        self._contacts_from = {
            row["res_id"]: ResidueContacts(**row) for row in contacts_df_rows.values()
        }

    def assign(self, res: Residue) -> Dict[str, Optional[ResidueContacts]]:
        res_id = res_to_id(res)
        return {NO_ALTLOC: self._contacts_from.get(res_id, None)}


class NeighborSearchContactsAssigner(ContactsAssigner):
    """
    Uses the NeighborSearch algorithm to assign contacts.
    """

    def __init__(
        self,
        pdb_id: str,
        pdb_source: str,
        contact_radius: float = CONTACT_DEFAULT_RADIUS,
        with_altlocs: bool = False,
        pdb_dict: Optional[dict] = None,
    ):
        super().__init__(
            pdb_id=pdb_id,
            pdb_source=pdb_source,
            contact_radius=contact_radius,
            with_altlocs=with_altlocs,
        )

        self._pdb_struct = pdb.pdb_struct(
            self.pdb_id, pdb_source=self.pdb_source, struct_d=pdb_dict
        )

        # Get all atoms from within the structure, possibly including altloc atoms
        atoms = chain(
            *(
                a.disordered_get_list() if a.is_disordered() and with_altlocs else (a,)
                for a in self._pdb_struct.get_atoms()
            )
        )
        self._contacts_from = NeighborSearch(list(atoms))

    def assign(self, res: Residue) -> Dict[str, Optional[ResidueContacts]]:

        # In rare cases, a residue may be disordered and contain other residues.
        # This means there's a point mutation and both original and mutated residues
        # are present in the crystal. We ignore this and just use the selected residue.
        if isinstance(res, DisorderedResidue):
            res = res.disordered_get()

        # Get all atoms from within the residue, including side chain atoms
        all_atoms = tuple(res.get_atoms())

        altloc_ids: Sequence[str] = (NO_ALTLOC,)
        if self.with_altlocs:
            # We want the altlocs for all atoms, even if they are not common amongst
            # all other atoms in the residue.
            altloc_ids = atom_altloc_ids(
                *all_atoms, allow_disjoint=True, include_none=True
            )

        contacts = {NO_ALTLOC: None}

        # For each altloc, we want to move all the atoms to it (if it exists for a
        # particular atom) and then calculate the contacts from all the moved atoms.
        # We then join the contacts from all the atoms.
        for altloc_id in altloc_ids:
            with altloc_ctx_all(all_atoms, altloc_id) as all_atoms_alt:
                curr_altloc_contacts: Set[Tuple[Atom, Atom]] = set()

                # Loop over the atoms in the residue, after they've been moved to the
                # current altloc, and calculate contacts from each of them.
                for alt_atom in all_atoms_alt:
                    if alt_atom is None:
                        # this particular atom doesn't have this altloc
                        continue

                    # Sanity check: if using altlocs, the atoms we loop over should
                    # not be disordered atoms because we selected a specific altloc.
                    # is_disordered()==2 means it contains other atoms, 1 means
                    # it's a regular atom inside a disordered atom which is OK.
                    assert alt_atom.is_disordered() < 2 or altloc_id == NO_ALTLOC

                    # Search in a radius around the atom location for other atoms
                    new_contacts: Sequence[Atom] = self._contacts_from.search(
                        center=alt_atom.get_coord(),
                        radius=self.contact_radius,
                        level="A",
                    )

                    # Store each contact as (source_atom, target_atom)
                    curr_altloc_contacts.update((alt_atom, a) for a in new_contacts)

                # Convert the atom contacts to residue contacts
                contacts[altloc_id] = self._resolve_atom_contacts(
                    res, tuple(curr_altloc_contacts)
                )

        return contacts

    def _resolve_atom_contacts(
        self, src_res: Residue, atom_contacts: Sequence[Tuple[Atom, Atom]]
    ) -> ResidueContacts:
        """
        Resolves a list of atom contacts to a ResidueContacts object representing the
        residues (not atoms) which the current residue is in contact with.

        :param src_res: The current residue.
        :param atom_contacts: The list of atom contacts: (src, tgt) tuples.
        :return: A ResidueContacts.
        """

        src_hetflag, src_seq_idx, src_icode = src_res.get_id()

        contact_dists: List[float] = []
        sequence_dists: List[int] = []

        # TODO: Define a AtomContacts dataclass to store the contact ids and distances
        res_contacts_non_aa: Dict[tuple, List[float]] = {}
        res_contacts_ooc: Dict[tuple, List[float]] = {}
        res_contacts_aas: Dict[tuple, List[float]] = {}

        for src_atom, tgt_atom in atom_contacts:
            assert src_res == src_atom.get_parent()

            tgt_res: Residue = tgt_atom.get_parent()
            tgt_hetflag, tgt_seq_idx, tgt_icode = tgt_res.get_id()
            tgt_resname = tgt_res.get_resname()

            # Ignore any contacts that are atoms from the current residue
            if tgt_res == src_res:
                continue

            # Ignore any contacts with water
            if tgt_resname == "HOH":  # TODO: also check for hydrogen 1HJE
                continue

            src_chain = src_res.get_parent()
            tgt_chain = tgt_res.get_parent()
            tgt_chain_id = tgt_chain.get_id().strip()
            tgt_altloc = tgt_atom.get_altloc().strip() if self.with_altlocs else ""

            # Calculate contact distance in Angstroms
            contact_dist = src_atom - tgt_atom
            contact_dists.append(contact_dist)

            # Key uniquely identifying the contact target
            contact_tgt_key = (tgt_chain_id, tgt_resname, tgt_seq_idx, tgt_altloc)

            # Check if contact is a ligand (check hetero flag)
            if tgt_hetflag.startswith("H_"):
                res_contacts_non_aa.setdefault(contact_tgt_key, [])
                res_contacts_non_aa[contact_tgt_key].append(contact_dist)

            # Check if contact is out of chain
            elif src_chain != tgt_chain:
                res_contacts_ooc.setdefault(contact_tgt_key, [])
                res_contacts_ooc[contact_tgt_key].append(contact_dist)

            # Regular AA contact in current chain
            else:
                res_contacts_aas.setdefault(contact_tgt_key, [])
                res_contacts_aas[contact_tgt_key].append(contact_dist)

                # Calculate sequence distance (only in-chain)
                sequence_dists.append(abs(tgt_seq_idx - src_seq_idx))

        contact_count = (
            len(res_contacts_ooc) + len(res_contacts_non_aa) + len(res_contacts_aas)
        )

        contact_smin, contact_smax = -1, -1
        if sequence_dists:
            contact_smin, contact_smax = min(sequence_dists), max(sequence_dists)

        def _aggregate(_contacts: Dict[tuple, List[float]]) -> Dict[tuple, float]:
            """
            For each unique contact target key, aggregates the distances by taking
            the minimum.

            :param _contacts: The contacts to format: {tgt_key: [dist1, dist2, ...]}
            :return: The aggregated contacts: {tgt_key: min_dist}
            """
            _contacts_to_dist: Dict[tuple, float] = {}
            for _tgt_key, _dists in _contacts.items():
                _contacts_to_dist[_tgt_key] = np.min(_dists)
            return _contacts_to_dist

        def _format(_contacts: Dict[tuple, float]) -> Sequence[str]:
            """
            Formats contacts as a list of strings, by merging the contacts
            which have a unique target key.

            :param _contacts: The contacts to format: {tgt_key: dist}
            :return: A list of formatted contacts.
            """
            # Sort by distance
            _contacts = dict(sorted(_contacts.items(), key=lambda x: x[1]))
            _formatted_contacts = []
            for _tgt_key, _dist in _contacts.items():
                _tgt_chain_id, _tgt_resname, _tgt_seq_idx, _tgt_altloc = _tgt_key
                _formatted_contacts.append(
                    format_residue_contact(
                        _tgt_chain_id, _tgt_resname, _tgt_seq_idx, _tgt_altloc, _dist
                    )
                )
            return tuple(_formatted_contacts)

        return ResidueContacts(
            res_id=res_to_id(src_res),
            contact_count=contact_count,
            contact_types="proximal",  # use arpeggio name, but not meaningful here
            contact_smax=contact_smax,
            contact_ooc=_format(_aggregate(res_contacts_ooc)),
            contact_non_aa=_format(_aggregate(res_contacts_non_aa)),
            contact_aas=_format(_aggregate(res_contacts_aas)),
        )


class ResidueContacts(object):
    """
    Represents a single residue's tertiary contacts in a protein record.

    # TODO:
    #  This string-based ResidueContacts should be only for Arpeggio contacts
    #  (legacy), and a separate ResidueContacts class should be created based on
    #  a sequence of AtomContacts. Conversely, we can refactor the
    #  ArpeggioContactsAssigned to generate AtomContacts, and then we'd only need one
    #  type of ResidueContacts.
    """

    def __init__(
        self,
        res_id: Union[str, int],
        contact_count: int,
        contact_types: Union[Set[str], str],
        contact_smax: Union[int, float],
        contact_ooc: Union[Sequence[str], str],
        contact_non_aa: Union[Sequence[str], str],
        contact_aas: Union[Sequence[str], str],
        **kwargs_ignored,  # ignore any other args (passed in from Arpeggio)
    ):
        def _split(s: str):
            s_split = s.split(",")

            # In case of empty string input, output will be an empty set.
            if "" in s_split:
                s_split.remove("")

            return s_split

        if isinstance(contact_types, str):
            contact_types = sorted(set(_split(contact_types)))
        if isinstance(contact_ooc, str):
            contact_ooc = sorted(set(_split(contact_ooc)))
        if isinstance(contact_non_aa, str):
            contact_non_aa = sorted(set(_split(contact_non_aa)))
        if isinstance(contact_aas, str):
            contact_aas = sorted(set(_split(contact_aas)))

        self.res_id = str(res_id)
        self.contact_count = int(contact_count or 0)
        self.contact_types = tuple(contact_types)
        self.contact_smax = int(contact_smax or 0)
        self.contact_ooc = tuple(contact_ooc)
        self.contact_non_aa = tuple(contact_non_aa)
        self.contact_aas = tuple(contact_aas)

    def as_dict(self, key_postfix: str = "", join_lists: bool = True):
        def _join(s):
            return str.join(",", s) if join_lists else s

        d = dict(
            contact_count=self.contact_count,
            contact_types=_join(self.contact_types),
            contact_smax=self.contact_smax,
            contact_ooc=_join(self.contact_ooc),
            contact_non_aa=_join(self.contact_non_aa),
            contact_aas=_join(self.contact_aas),
        )

        if key_postfix:
            d = {f"{k}_{key_postfix}": v for k, v in d.items()}

        return d

    def __eq__(self, other):
        if not isinstance(other, ResidueContacts):
            return False
        return self.as_dict() == other.as_dict()

    def __hash__(self):
        return hash(tuple(self.as_dict().values()))


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
