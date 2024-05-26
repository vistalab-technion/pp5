from __future__ import annotations

import os
import json
import logging
import zipfile
import subprocess
from abc import ABC, abstractmethod
from time import time
from typing import Any, Set, Dict, List, Tuple, Union, Optional, Sequence
from pathlib import Path
from functools import partial
from itertools import chain

import attrs
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

CONTACT_TYPE_AAA = "AA"
CONTACT_TYPE_OOC = "OOC"
CONTACT_TYPE_LIG = "LIG"
CONTACT_TYPES = (CONTACT_TYPE_AAA, CONTACT_TYPE_OOC, CONTACT_TYPE_LIG)


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
    tgt_atom: str = None,
) -> str:
    """
    Formats a residue contact as a string.

    :param tgt_chain_id: The chain ID of the target residue.
    :param tgt_resname: The residue name of the target residue.
    :param tgt_seq_idx: The sequence index of the target residue.
    :param tgt_altloc: The altloc of the target residue.
    :param tgt_atom: The atom name of the target residue.
    :param contact_dist: The distance to the contact residue.
    :return: A string representing the contact.
    Format is:
        chain:resname:seq_idx[-altloc][@atom]:contact_dist
    Where the [-altloc] part is only included if the altloc exists, and the [@atom]
    is only included if the atom was specified.
    """
    tgt_resname = ACIDS_3TO1.get(tgt_resname, tgt_resname)
    tgt_altloc = f":{tgt_altloc}" if tgt_altloc else ""
    tgt_atom = f"@{tgt_atom}" if tgt_atom else ""
    _contact_dist_str = f"{contact_dist:.2f}"
    return (
        f"{tgt_chain_id}"
        f":{tgt_resname}"
        f":{tgt_seq_idx}{tgt_altloc}{tgt_atom}"
        f":{_contact_dist_str}"
    )


@attrs.define(repr=True, eq=True, hash=True)
class ResidueContactKey:
    """Unique key for a contact between two Residues (source or target)."""

    chain: str = attrs.field()
    resname: str = attrs.field()
    seq_idx: int = attrs.field()
    altloc: str = attrs.field()

    def __str__(self):
        return f"{self.chain}:{self.resname}:{self.seq_idx}:{self.altloc}"


@attrs.define(repr=True, eq=True, hash=True)
class AtomContactKey(ResidueContactKey):
    """Unique key for a contact between two atoms (source or target)."""

    atom: str = attrs.field()

    @classmethod
    def from_atom(cls, atom: Atom, with_altlocs: bool = True) -> AtomContactKey:
        """Creates an AtomContactKey from a biopython Atom object."""
        parent_res = atom.get_parent()
        parent_res_name = parent_res.get_resname().strip()
        return cls(
            chain=parent_res.get_parent().get_id().strip(),
            resname=ACIDS_3TO1.get(parent_res_name, parent_res_name),
            seq_idx=parent_res.get_id()[1],
            altloc=(atom.get_altloc().strip() or NO_ALTLOC)
            if with_altlocs
            else NO_ALTLOC,
            atom=atom.get_name().strip(),
        )

    def __str__(self):
        return f"{super().__str__()}@{self.atom}"


@attrs.define(repr=True, eq=True, hash=True)
class AtomContact:
    """Represents a contact between two atoms."""

    src_key: AtomContactKey = attrs.field()
    tgt_key: AtomContactKey = attrs.field()
    dist: float = attrs.field(hash=False)
    seq_dist: Optional[int] = attrs.field(hash=False)
    type: str = attrs.field(hash=False, validator=attrs.validators.in_(CONTACT_TYPES))

    @classmethod
    def from_atoms(cls, src: Atom, tgt: Atom, with_altlocs: bool = True) -> AtomContact:
        """Creates an AtomContact from two biopython Atom objects."""

        src_key = AtomContactKey.from_atom(src, with_altlocs=with_altlocs)
        tgt_key = AtomContactKey.from_atom(tgt, with_altlocs=with_altlocs)
        seq_dist = (
            abs(src_key.seq_idx - tgt_key.seq_idx)
            if src_key.chain == tgt_key.chain
            else None
        )

        # Assign type
        tgt_hetflag = tgt.get_parent().get_id()[0]
        if tgt_hetflag.startswith("H_"):
            contact_type = CONTACT_TYPE_LIG
        elif src_key.chain != tgt_key.chain:
            contact_type = CONTACT_TYPE_OOC
        else:
            contact_type = CONTACT_TYPE_AAA

        return cls(
            src_key=src_key,
            tgt_key=tgt_key,
            dist=src - tgt,
            seq_dist=seq_dist,
            type=contact_type,
        )

    def __str__(self):
        return f"{self.src_key!s} -> {self.tgt_key!s} ({self.type}): {self.dist:.2f}"

    def as_dict(self) -> Dict[str, Any]:
        return {
            **{f"src_{k}": v for k, v in attrs.asdict(self.src_key).items()},
            **{f"tgt_{k}": v for k, v in attrs.asdict(self.tgt_key).items()},
            **attrs.asdict(self, filter=attrs.filters.exclude("src_key", "tgt_key")),
        }


@attrs.define(repr=True, eq=True, hash=True)
class ResidueContact:
    """Represents a contact between two residues."""

    src_key: ResidueContactKey = attrs.field()
    tgt_key: ResidueContactKey = attrs.field()
    min_dist: float = attrs.field(hash=False)
    seq_dist: Optional[int] = attrs.field(hash=False)
    type: str = attrs.field(hash=False, validator=attrs.validators.in_(CONTACT_TYPES))

    def __str__(self):
        return (
            f"{self.src_key!s} -> {self.tgt_key!s} ({self.type}): {self.min_dist:.2f}A"
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
        with_atom_contacts: bool = False,
        pdb_dict: Optional[dict] = None,
    ):
        """
        :param pdb_id: The PDB ID to assign contacts for.
        :param pdb_source: The source from which to obtain the PDB file.
        :param contact_radius: The radius (in angstroms) to use for contact detection.
        :param with_altlocs: Whether to include altloc atoms in the contact detection.
        :param with_atom_contacts: Whether to include atom-level contacts in the output.
        :param pdb_dict: If provided, the PDB structure will be loaded from this dict.
        """
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
        self._with_atom_contacts = with_atom_contacts

    def assign(self, res: Residue) -> Dict[str, Optional[ResidueContacts]]:

        # In rare cases, a residue may be disordered and contain other residues.
        # This means there's a point mutation and both original and mutated residues
        # are present in the crystal. We ignore this and just use the selected residue.
        if isinstance(res, DisorderedResidue):
            res = res.disordered_get()

        # Get source residue info
        src_hetflag, src_seq_idx, src_icode = res.get_id()
        src_chain = res.get_parent().get_id().strip()
        src_resname = ACIDS_3TO1.get(res.get_resname(), res.get_resname())

        # Get all atoms from within the residue, including side chain atoms,
        # but ignore hydrogen atoms.
        all_atoms = tuple(a for a in res.get_atoms() if a.element != "H")

        altloc_ids: Sequence[str] = (NO_ALTLOC,)
        if self.with_altlocs:
            # We want the altlocs for all atoms, even if they are not common amongst
            # all other atoms in the residue.
            altloc_ids = atom_altloc_ids(
                *all_atoms, allow_disjoint=True, include_none=True
            )

        atom_contacts: List[AtomContact] = []
        altloc_to_residue_contacts: Dict[str, Optional[ResidueContacts]] = {
            NO_ALTLOC: None
        }

        # For each altloc, we want to move all the atoms to it (if it exists for a
        # particular atom) and then calculate the contacts from all the moved atoms.
        # We then join the contacts from all the atoms.
        for altloc_id in altloc_ids:

            # Create a contact key for this residue at the current altloc
            src_res_contact_key = ResidueContactKey(
                chain=src_chain,
                resname=src_resname,
                seq_idx=src_seq_idx,
                altloc=altloc_id,
            )

            with altloc_ctx_all(all_atoms, altloc_id) as all_atoms_alt:
                curr_altloc_contacts: Set[AtomContact] = set()

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

                    # Filter the contacts:
                    # - Ignore contacts with the same residue
                    # - Ignore contacts with water (H and O atoms in a HOH residue)
                    # - Ignore contacts with hydrogen atoms (H atoms in any residue)
                    new_contacts_filtered = []
                    for a in new_contacts:
                        if a.element == "H":
                            continue
                        parent_res: Residue = a.get_parent()
                        if parent_res == res:
                            continue
                        if parent_res.get_resname() == "HOH":
                            continue
                        new_contacts_filtered.append(a)

                    for a in new_contacts_filtered:
                        # Store each contact as (source_atom, target_atom)
                        atom_contact = AtomContact.from_atoms(
                            alt_atom, a, with_altlocs=self.with_altlocs
                        )
                        curr_altloc_contacts.add(atom_contact)
                        atom_contacts.append(atom_contact)

                # Convert the atom contacts to residue contacts
                residue_contacts = self._aggregate_atom_contacts(
                    src_res_contact_key, tuple(curr_altloc_contacts)
                )

                altloc_to_residue_contacts[altloc_id] = ResidueContacts.from_contacts(
                    res,
                    residue_contacts,
                    atom_contacts if self._with_atom_contacts else None,
                )

        return altloc_to_residue_contacts

    def _aggregate_atom_contacts(
        self, src_key: ResidueContactKey, atom_contacts: Sequence[AtomContact]
    ) -> Sequence[ResidueContact]:
        """
        Aggregates a list of atom contacts between a source residue and the atoms
        of other residues, into to a list of residue contacts.

        :param src_key: The key of the source residue.
        :param atom_contacts: The list of atom contacts.
        :return: The list of residue contacts, sorted by minimum distance.
        """

        # Group contacts by their target residue
        target_res_to_contacts: Dict[ResidueContactKey, List[AtomContact]] = {}
        for atom_contact in atom_contacts:
            tgt_atom_key = atom_contact.tgt_key
            tgt_res_key = ResidueContactKey(
                chain=tgt_atom_key.chain,
                resname=tgt_atom_key.resname,
                seq_idx=tgt_atom_key.seq_idx,
                altloc=tgt_atom_key.altloc,
            )
            target_res_to_contacts.setdefault(tgt_res_key, [])
            target_res_to_contacts[tgt_res_key].append(atom_contact)

        # Aggregate all contacts ending at the same target residue by taking the
        # minimum distance
        res_contacts: List[ResidueContact] = []
        for tgt_res_key, tgt_contacts in target_res_to_contacts.items():
            min_dist = min(c.dist for c in tgt_contacts)
            res_contact_type = tgt_contacts[0].type
            res_contact_seq_dist = tgt_contacts[0].seq_dist

            # Sanity: all contacts should have the same type and sequence distance
            assert all(c.type == res_contact_type for c in tgt_contacts)
            assert all(c.seq_dist == res_contact_seq_dist for c in tgt_contacts)

            res_contacts.append(
                ResidueContact(
                    src_key=src_key,
                    tgt_key=tgt_res_key,
                    min_dist=min_dist,
                    seq_dist=res_contact_seq_dist,
                    type=res_contact_type,
                )
            )

        # Sort by min distance
        return tuple(sorted(res_contacts, key=lambda c: (c.min_dist, str(c.tgt_key))))


class ResidueContacts(object):
    """
    Represents a single residue's tertiary contacts in a protein record.

    # TODO:
    #  This string-based ResidueContacts should be only for Arpeggio contacts
    #  (legacy), and a separate ResidueContacts class should be created based on
    #  a sequence of ResidueContacts. Conversely, we can refactor the
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
        atom_contacts: Optional[Sequence[AtomContact]] = None,
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
        self.atom_contacts = atom_contacts

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

    @classmethod
    def from_contacts(
        cls,
        src_res: Residue,
        contacts: Sequence[ResidueContact],
        atom_contacts: Sequence[AtomContact] = None,
    ) -> ResidueContacts:
        """
        Creates a ResidueContacts from a sequence of ResidueContact objects.

        :param src_res: The source residue.
        :param contacts: The sequence of ResidueContact objects.
        :param atom_contacts: Optional sequence of AtomContact objects. If provided,
        they will be included as-is in the ResidueContacts object.
        :return: A ResidueContacts object.
        """

        def _format(_contacts: Sequence[ResidueContact]) -> Sequence[str]:
            return tuple(
                format_residue_contact(
                    tgt_chain_id=_contact.tgt_key.chain,
                    tgt_resname=_contact.tgt_key.resname,
                    tgt_seq_idx=_contact.tgt_key.seq_idx,
                    tgt_altloc=_contact.tgt_key.altloc,
                    contact_dist=_contact.min_dist,
                )
                for _contact in _contacts
            )

        contact_smax = max(
            [
                c.seq_dist
                for c in contacts
                if (c.seq_dist is not None and c.type == CONTACT_TYPE_AAA)
            ]
        )
        contact_ooc = _format([c for c in contacts if c.type == CONTACT_TYPE_OOC])
        contact_non_aa = _format([c for c in contacts if c.type == CONTACT_TYPE_LIG])
        contact_aas = _format([c for c in contacts if c.type == CONTACT_TYPE_AAA])
        return cls(
            res_id=res_to_id(src_res),
            contact_count=len(contacts),
            contact_types="proximal",  # use arpeggio name, but not meaningful here
            contact_smax=contact_smax,
            contact_ooc=contact_ooc,
            contact_non_aa=contact_non_aa,
            contact_aas=contact_aas,
            atom_contacts=atom_contacts,
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
