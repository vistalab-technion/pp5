from __future__ import annotations

import os
import math
import pickle
import string
import logging
import warnings
from typing import (
    Any,
    Dict,
    List,
    Tuple,
    Union,
    Callable,
    Iterator,
    Optional,
    Sequence,
    ItemsView,
)
from pathlib import Path
from itertools import chain
from collections import OrderedDict

import numpy as np
import pandas as pd
from Bio.PDB import PPBuilder
from Bio.Seq import Seq
from Bio.Align import PairwiseAligner
from Bio.PDB.Chain import Chain
from Bio.SeqRecord import SeqRecord
from Bio.PDB.Residue import Residue
from Bio.PDB.Polypeptide import Polypeptide

import pp5
from pp5.align import BLOSUM80, pairwise_alignment_map
from pp5.utils import ProteinInitError, filelock_context
from pp5.codons import (
    ACIDS_1TO3,
    ACIDS_3TO1,
    UNKNOWN_AA,
    CODON_TABLE,
    STOP_CODONS,
    UNKNOWN_CODON,
    CODON_OPTS_SEP,
)
from pp5.backbone import (
    NO_ALTLOC,
    BACKBONE_ATOMS,
    BACKBONE_ATOM_CA,
    atom_altloc_ids,
    residue_altloc_sigmas,
    residue_altloc_ca_dists,
    residue_backbone_coords,
    residue_altloc_peptide_bond_lengths,
)
from pp5.contacts import (
    CONTACT_METHODS,
    CONTACT_DEFAULT_RADIUS,
    CONTACT_METHOD_ARPEGGIO,
    CONTACT_METHOD_NEIGHBOR,
    AtomContact,
    ResidueContacts,
    ContactsAssigner,
    ArpeggioContactsAssigner,
    NeighborSearchContactsAssigner,
    res_to_id,
)
from pp5.dihedral import (
    Dihedral,
    AtomLocationUncertainty,
    DihedralAngleCalculator,
    DihedralAnglesMonteCarloEstimator,
    DihedralAnglesUncertaintyEstimator,
)
from pp5.external_dbs import ena, pdb, unp
from pp5.external_dbs.pdb import PDB_AFLD, PDB_RCSB, PDBRecord, pdb_tagged_filepath
from pp5.external_dbs.unp import UNPRecord

with warnings.catch_warnings():
    warnings.simplefilter("ignore")

LOGGER = logging.getLogger(__name__)

DEFAULT_ANGLE_CALC = DihedralAngleCalculator()

DEFAULT_BFACTOR_CALC = AtomLocationUncertainty(
    backbone_only=True, unit_cell=None, isotropic=True, scale_as_bfactor=True
)

# Special insertion code to mark a residue that is unmodeled the structure.
ICODE_UNMODELED_RES = "U"


class ResidueRecord(object):
    """
    Represents a single residue in a protein record.
    """

    def __init__(
        self,
        res_seq_idx: int,
        unp_idx: int,
        rel_loc: float,
        name: str,
        codon_counts: Optional[Dict[str, int]],
        angles: Dihedral,
        bfactor: float,
        secondary: str,
        num_altlocs: int,
        backbone_coords: Optional[Dict[str, Optional[np.ndarray]]] = None,
        contacts: Optional[ResidueContacts] = None,
        res_icode: str = "",
        res_hflag: str = "",
    ):
        """
        :param res_seq_idx: index of this residue in the sequence.
        :param res_icode: insertion code, if present, which indicates some alteration
            or a missing residue in the structure if equal to ICODE_MISSING_RESIDUE.
        :param res_hflag: hetero flag, if present, which indicates a non-standard AA.
        :param unp_idx: index of this residue in the corresponding UNP record.
        :param rel_loc: relative location of this residue in the protein sequence,
            a number between 0 and 1.
        :param name: single-letter name of the residue or X for unknown.
        :param codon_counts: A dict mapping codons to the number of occurrences in
        DNA sequences associated with the Uniprot identifier of the containing protein.
        :param angles: A Dihedral object containing the dihedral angles.
        :param bfactor: Average b-factor along of the residue's backbone atoms.
        :param secondary: Single-letter secondary structure code.
        :param num_altlocs: Number of alternate conformations in the PDB entry of this residue.
        :param backbone_coords: A dict mapping atom names to their backbone coordinates.
        :param contacts: A ResidueContacts object containing the residue's tertiary contacts.
        """

        # # Get the best codon and calculate its 'score' based on how many
        # # other options there are
        best_codon, codon_score, codon_opts = None, 0, []
        if codon_counts is not None:
            best_codon = UNKNOWN_CODON
            max_count, total_count = 0, 0
            for codon, count in codon_counts.items():
                total_count += count
                if count > max_count and codon != UNKNOWN_CODON:
                    best_codon, max_count = codon, count
            codon_score = max_count / total_count if total_count else 0
            codon_opts = codon_counts.keys()

        self.res_seq_idx, self.res_icode, self.res_hflag = (
            res_seq_idx,
            res_icode,
            res_hflag,
        )
        self.name = name
        self.unp_idx, self.rel_loc = unp_idx, rel_loc
        self.codon, self.codon_score = best_codon, codon_score
        self.codon_opts = str.join(CODON_OPTS_SEP, codon_opts)
        self.codon_counts = codon_counts
        self.angles, self.bfactor, self.secondary = angles, bfactor, secondary
        self.num_altlocs = num_altlocs
        self.backbone_coords = backbone_coords or {}
        self.contacts = contacts

    @property
    def res_id(self) -> str:
        """
        :return: The residue id, including insertion code and hetero flag.
        Similar to biopython's id.
        """
        return f"{self.res_hflag}{self.res_seq_idx}{self.res_icode}"

    def as_dict(self, dihedral_args: Dict[str, Any] = None):
        """
        Creates a dict representation of the data in this residue. The angles object
        will we flattened out so its attributes will be placed directly in the
        resulting dict. The backbone angles will be converted to a nested list.
        :param dihedral_args: A dict of arguments to pass to Dihedral.as_dict.
        :return: A dict representation of this residue.
        """
        return dict(
            pdb_idx=self.res_seq_idx,
            unp_idx=self.unp_idx,
            res_name=self.name,
            res_icode=self.res_icode,
            res_hflag=self.res_hflag,
            rel_loc=self.rel_loc,
            **(
                dict(
                    codon=self.codon,
                    codon_score=self.codon_score,
                    codon_opts=self.codon_opts,
                )
                if self.codon is not None
                else {}
            ),
            secondary=self.secondary,
            **self.angles.as_dict(**(dihedral_args or {})),
            bfactor=self.bfactor,
            **{
                f"backbone_{atom_name}": str.join(
                    ",",
                    map(
                        str,
                        [
                            np.round(c, 4)
                            for c in (coords if coords is not None else [])
                        ],
                    ),
                )
                for atom_name, coords in self.backbone_coords.items()
            },
            **(self.contacts.as_dict() if self.contacts else {}),
            num_altlocs=self.num_altlocs,
        )

    def __repr__(self):
        return (
            f"{self.name} {self.res_id:<4s} [{self.codon}]"
            f"[{self.secondary}] {self.angles} b={self.bfactor:.2f}, "
            f"unp_idx={self.unp_idx}"
        )

    def __eq__(self, other):
        if self is other:
            return True
        if not isinstance(other, ResidueRecord):
            return False

        def _compare(a, b):
            eq = True
            if isinstance(a, (float, np.ndarray)):
                eq = np.allclose(a, b, equal_nan=True)

            elif isinstance(a, dict):
                for key, val in a.items():
                    # to handle dict that contains ndarrays
                    eq = _compare(val, b.get(key))
                    if not eq:
                        break
            else:
                eq = a == b

            return eq

        for k, v in self.__dict__.items():
            other_v = other.__dict__.get(k, math.inf)
            equal = _compare(v, other_v)
            if not equal:
                return False
        return True

    def __hash__(self):
        return hash(tuple(self.as_dict().items()))


class AltlocNameMap(dict):
    """
    Normalizes altloc ids to be in the range [A, Z] and always start from A.
    Maps from original altloc to new altloc id.
    """

    def __init__(self, map_altloc_ids: bool = True):
        """
        :param map_altloc_ids: Whether to map altloc ids.
        If False, then this class is a no-op: maps every altloc to itself.
        """
        self._new_altloc_ids = list(reversed(string.ascii_uppercase))
        self._should_map_altloc_ids = map_altloc_ids
        super().__init__({NO_ALTLOC: NO_ALTLOC})

    def add(self, orig_altloc_id: str) -> str:
        """
        Adds a new altloc id to the map.
        A new altloc id is chosen from the sequence A-Z and the original will be
        mapped to the new.
        :param orig_altloc_id: The original altloc id.
        :return: The new altloc id.
        """
        if orig_altloc_id not in self:
            super().__setitem__(
                orig_altloc_id,
                (
                    self._new_altloc_ids.pop()
                    if self._should_map_altloc_ids
                    else orig_altloc_id
                ),
            )
        return self[orig_altloc_id]

    def map_altloc(self, _altloc_id: str) -> str:
        """
        Maps e.g. X -> A
        """
        return self.get(_altloc_id, _altloc_id)

    def map_altloc_pair(self, altloc_pair_ids: str) -> str:
        """
        Maps pairs of altlocs, e.g.
        XY -> AB
        XY_norm -> AB_norm
        """
        altloc_pair_ids, *_suffix = altloc_pair_ids.split("_")
        assert len(altloc_pair_ids) == 2  # make sure it's really a pair
        altloc_pair_ids = str.join("", map(self.map_altloc, altloc_pair_ids))
        if _suffix:
            return str.join("_", [altloc_pair_ids, *_suffix])
        else:
            return altloc_pair_ids

    def __setitem__(self, key, value):
        raise KeyError("Can't manually set items in AltlocNameMap. Use add(key)")

    def __getitem__(self, altloc_id: str) -> str:
        if not altloc_id:
            raise KeyError("Empty altloc id")
        elif len(altloc_id) == 1:
            return self.map_altloc(altloc_id)
        else:
            return self.map_altloc_pair(altloc_id)


class AltlocResidueRecord(ResidueRecord):
    def __init__(
        self,
        res_seq_idx: int,
        res_icode: str,
        res_hflag: str,
        unp_idx: int,
        rel_loc: float,
        name: str,
        codon_counts: Optional[Dict[str, int]],
        secondary: str,
        altloc_ids: Dict[str, Sequence[str]],
        altloc_angles: Dict[str, Dihedral],
        altloc_bfactors: Dict[str, float],
        altloc_ca_dists: Dict[str, float],
        altloc_sigmas: Dict[str, Dict[str, float]],
        altloc_peptide_bond_lengths: Dict[str, float],
        altloc_contacts: Optional[Dict[str, ResidueContacts]] = None,
        backbone_coords: Optional[Dict[str, Optional[np.ndarray]]] = None,
    ):
        """
        Represents a residue with (potential) alternate conformations (altlocs).
        All params as for ResidueRecord, except:

        :param altloc_ids: A mapping from atom_name -> [altloc_id1, altloc_id2,...]
        representing the altloc ids of each atom in the residue.
        :param altloc_angles: A mapping from an altloc id to a Dihedral object
        containing the dihedral angles for that conformation.
        :param altloc_bfactors: A mapping from an altloc id to the average b-factor for
        that conformation.
        :param altloc_ca_dists: A mapping from an a pair of altloc ids (as a joined
        string, e.g. AB) to the CA-CA distance between them.
        :param altloc_sigmas: A mapping atom_name -> altloc_id -> sigma, where sigma is
        the standard deviation of the atom's location in the altloc conformation.
        :param altloc_peptide_bond_lengths: A mapping from an a pair of altloc ids
        to the peptide bond length between this residue and the next one with those
        altloc ids.
        :param altloc_contacts: A mapping from an altloc id to a ResidueContacts object.
        """
        num_altlocs = len(set(chain(*altloc_ids.values())))
        no_altloc_angle = altloc_angles.pop(NO_ALTLOC)
        no_altloc_bfactor = altloc_bfactors.pop(NO_ALTLOC)
        no_altloc_contacts = (altloc_contacts or {}).pop(NO_ALTLOC, None)
        self.altloc_ids = altloc_ids
        self.altloc_angles = altloc_angles
        self.altloc_bfactors = altloc_bfactors
        self.altloc_ca_dists = altloc_ca_dists or {}
        self.altloc_sigmas = altloc_sigmas or {}
        self.altloc_peptide_bond_lengths = altloc_peptide_bond_lengths or {}
        self.altloc_contacts = altloc_contacts or {}

        super().__init__(
            res_seq_idx=res_seq_idx,
            res_icode=res_icode,
            res_hflag=res_hflag,
            unp_idx=unp_idx,
            rel_loc=rel_loc,
            name=name,
            codon_counts=codon_counts,
            angles=no_altloc_angle,
            bfactor=no_altloc_bfactor,
            secondary=secondary,
            num_altlocs=num_altlocs,
            backbone_coords=backbone_coords,
            contacts=no_altloc_contacts,
        )

    @classmethod
    def from_residue(
        cls,
        r_curr: Residue,
        r_prev: Optional[Residue] = None,
        r_next: Optional[Residue] = None,
        unp_idx: Optional[int] = None,
        rel_loc: Optional[float] = None,
        codon_counts: Optional[Dict[str, int]] = None,
        secondary: Optional[str] = None,
        dihedral_est: DihedralAngleCalculator = DEFAULT_ANGLE_CALC,
        bfactor_est: AtomLocationUncertainty = DEFAULT_BFACTOR_CALC,
        with_backbone: bool = False,
        with_altlocs: bool = False,
        with_contacts: bool = False,
        contacts_assigner: Optional[ContactsAssigner] = None,
    ):
        # Parse residue id
        res_seq_idx: int
        res_hflag: str
        res_icode: str
        res_hflag, res_seq_idx, res_icode = r_curr.get_id()
        res_seq_idx = int(res_seq_idx)
        res_hflag = res_hflag.strip()
        res_icode = res_icode.strip()
        res_name: str = r_curr.get_resname()
        res_name = ACIDS_3TO1.get(res_name, res_name)  # keep name for non-standard AAs

        # mapping atom_name -> [altloc_id1, altloc_id2, ...]
        altloc_ids: Dict[str, Sequence[str]] = {
            atom_name: atom_altloc_ids(r_curr[atom_name])
            for atom_name in BACKBONE_ATOMS
            if atom_name in r_curr
        }

        altloc_angles: Dict[str, Dihedral] = dihedral_est.process_residues(
            r_curr, r_prev, r_next, with_altlocs=with_altlocs
        )
        altloc_bfactors: Dict[str, float] = bfactor_est.process_residue_altlocs(
            r_curr, with_altlocs=with_altlocs
        )

        # Distances
        altloc_ca_dists: Dict[str, float] = {}
        altloc_sigmas: Dict[str, Dict[str, float]] = {}
        altloc_peptide_bond_lengths: Dict[str, float] = {}
        if with_altlocs:
            altloc_ca_dists = residue_altloc_ca_dists(r_curr, normalize=True)
            altloc_sigmas = residue_altloc_sigmas(r_curr, atom_names=[BACKBONE_ATOM_CA])
            altloc_peptide_bond_lengths = residue_altloc_peptide_bond_lengths(
                r_curr, r_next, normalize=True
            )

        # Contacts
        altloc_contacts: Dict[str, ResidueContacts] = {}
        if with_contacts:
            if not contacts_assigner:
                raise ValueError(f"Must provide ContactsAssigner when {with_contacts=}")
            altloc_contacts = contacts_assigner.assign(r_curr)
            if not with_altlocs:  # make sure we don't have any altloc contacts here
                altloc_contacts = {NO_ALTLOC: altloc_contacts[NO_ALTLOC]}

        # Backbone
        altloc_backbone_coords: Dict[str, Optional[np.ndarray]] = {}
        if with_backbone:
            altloc_backbone_coords = residue_backbone_coords(
                r_curr, with_oxygen=True, with_altlocs=with_altlocs
            )

        return cls(
            res_seq_idx=res_seq_idx,
            res_icode=res_icode,
            res_hflag=res_hflag,
            unp_idx=unp_idx,
            rel_loc=rel_loc,
            name=res_name,
            codon_counts=codon_counts,
            secondary=secondary,
            altloc_ids=altloc_ids,
            altloc_angles=altloc_angles,
            altloc_bfactors=altloc_bfactors,
            altloc_ca_dists=altloc_ca_dists,
            altloc_sigmas=altloc_sigmas,
            altloc_peptide_bond_lengths=altloc_peptide_bond_lengths,
            altloc_contacts=altloc_contacts,
            backbone_coords=altloc_backbone_coords,
        )

    def as_dict(
        self, dihedral_args: Dict[str, Any] = None, map_altloc_ids: bool = True
    ):
        dihedral_args = dihedral_args or {}
        d = super().as_dict(dihedral_args)

        def _altloc_postfix(_altloc_id: str) -> str:
            return f"_{_altloc_id}"

        # Map altloc names in current AA
        altloc_map = AltlocNameMap(map_altloc_ids=map_altloc_ids)
        for _, orig_altloc_ids in self.altloc_ids.items():
            for orig_altloc_id in orig_altloc_ids:
                altloc_map.add(orig_altloc_id)

        # Map altloc names in next AA
        altloc_map_next = AltlocNameMap(map_altloc_ids=map_altloc_ids)
        for altloc_pair_ids, _ in self.altloc_peptide_bond_lengths.items():
            if altloc_pair_ids.startswith(NO_ALTLOC * 2):
                continue
            orig_altloc_pair_ids, *_suffix = altloc_pair_ids.split("_")
            for orig_altloc_id in orig_altloc_pair_ids:
                altloc_map_next.add(orig_altloc_id)

        for atom_name, altloc_ids in self.altloc_ids.items():
            # Write both the original altloc id and the new one for context,
            # if they're different.
            mapped_altloc_ids = [
                orig_altloc_id
                if altloc_map[orig_altloc_id] == orig_altloc_id
                else f"{altloc_map[orig_altloc_id]}({orig_altloc_id})"
                for orig_altloc_id in altloc_ids
            ]
            d[f"altlocs_{atom_name}"] = str.join(";", mapped_altloc_ids)

        for altloc_id, altloc_angles in self.altloc_angles.items():
            d.update(
                altloc_angles.as_dict(
                    **dihedral_args, postfix=_altloc_postfix(altloc_map[altloc_id])
                )
            )

        for altloc_id, altloc_bfactor in self.altloc_bfactors.items():
            d[f"bfactor{_altloc_postfix(altloc_map[altloc_id])}"] = altloc_bfactor

        for altloc_pair_ids, ca_dist in self.altloc_ca_dists.items():
            d[f"dist_CA{_altloc_postfix(altloc_map[altloc_pair_ids])}"] = ca_dist

        for atom_name, altloc_sigmas in self.altloc_sigmas.items():
            for altloc_id, sigma in altloc_sigmas.items():
                d[f"sigma_{atom_name}{_altloc_postfix(altloc_map[altloc_id])}"] = sigma

        for altloc_pair_ids, pb_len in self.altloc_peptide_bond_lengths.items():
            d[f"len_pb{_altloc_postfix(altloc_map_next[altloc_pair_ids])}"] = pb_len

        for altloc_id, contacts in self.altloc_contacts.items():
            d.update(contacts.as_dict(key_postfix=altloc_map[altloc_id]))

        return d


class ProteinRecord(object):
    """
    Represents a protein in our dataset. Includes:
    - Uniprot id defining the which protein this is.
    - PDB id of one structure representing this protein.
    - Amino acid sequence based on PDB structure,
    - Genetic codon sequence based on Uniprot cross-ref with ENA
      (and matching to AA sequence from PDB)
    - Dihedral angles at each AA.

    Protein AA sequence is defined by the specific PDB structure we're
    using, not the sequence from Uniprot. We want multiple different
    PDB structures with the same Uniprot id and possible slightly
    different AAs.
    """

    _SKIP_SERIALIZE = ["_unp_rec", "_pdb_rec", "_pdb_dict", "_pp"]

    @staticmethod
    def from_cache(
        pdb_id, pdb_source: str = PDB_RCSB, cache_dir: Union[str, Path] = None, tag=None
    ) -> Optional[ProteinRecord]:
        """
        Loads a cached ProteinRecord, if it exists.
        :param pdb_id: PDB ID with chain.
        :param pdb_source: Source from which to obtain the pdb file.
        :param cache_dir: Directory with cached files.
        :param tag: Optional extra tag on the filename.
        :return: Loaded ProteinRecord, or None if the cached prec does not
        exist.
        """
        # TODO: Prec should use Cacheable base class instead of this custom approach.

        if not isinstance(cache_dir, (str, Path)):
            cache_dir = pp5.PREC_DIR

        path = pdb_tagged_filepath(pdb_id, pdb_source, cache_dir, "prec", tag)
        filename = path.name
        path = pp5.get_resource_path(cache_dir, filename)
        prec = None
        with filelock_context(path):
            if path.is_file():
                try:
                    with open(str(path), "rb") as f:
                        prec = pickle.load(f)
                except Exception as e:
                    # If we can't unpickle, probably the code changed since
                    # saving this object. We'll just return None, so that a new
                    # prec will be created and stored.
                    LOGGER.warning(f"Failed to load cached ProteinRecord {path}")
            return prec

    @classmethod
    def from_pdb(
        cls,
        pdb_id: str,
        pdb_source: str = PDB_RCSB,
        pdb_dict=None,
        cache=False,
        cache_dir=pp5.PREC_DIR,
        strict_pdb_unp_xref=True,
        **kw_for_init,
    ) -> ProteinRecord:
        """
        Given a PDB id, finds the corresponding Uniprot id, and returns a
        ProteinRecord object for that protein.
        The PDB ID can be any valid format: either base ID only,
        e.g. '1ABC'; with a specified chain, e.g. '1ABC:D'; or with a
        specified entity, e.g. '1ABC:2'.
        :param pdb_id: The PDB id to query, with optional chain, e.g. '0ABC:D'.
        :param pdb_source: Source from which to obtain the pdb file.
        :param pdb_dict: Optional structure dict for the PDB record, in case it
        was already parsed.
        :param cache: Whether to load prec from cache if available.
        :param cache_dir: Where the cache dir is. ProteinRecords will be
        written to this folder after creation, unless it's None.
        :param strict_pdb_unp_xref: Whether to require that the given PDB ID
        maps uniquely to only one Uniprot ID.
        :param kw_for_init: Extra kwargs for the ProteinRecord initializer.
        :return: A ProteinRecord.
        """
        try:
            # Either chain or entity or none can be provided, but not both
            pdb_base_id, chain_id, entity_id = pdb.split_id_with_entity(pdb_id)
            if entity_id:
                entity_id = str(entity_id)

                meta = pdb.PDBMetadata.from_pdb(pdb_id, cache=cache)

                chain_id = None
                if entity_id in meta.entity_ids:
                    chain_id = meta.entity_chains[entity_id][0]

                if not chain_id:
                    # In rare cases the author chain is a number instead of a letter.
                    # We check for this, and if it's the case, we use the
                    # corresponding PDB chain instead. See e.g. 4N6V.
                    if entity_id in meta.auth_chain_ids:
                        # Chain is number, but use its string representation
                        chain_id = next(
                            iter(
                                c_id
                                for c_id, ac_id in (meta.chain_to_auth_chain.items())
                                if ac_id == entity_id
                            )
                        )
                    else:
                        raise ProteinInitError(
                            f"No matching chain found for entity "
                            f"{entity_id} in PDB structure {pdb_base_id}"
                        )

                pdb_id = f"{pdb_base_id}:{chain_id}"

            if cache and chain_id:
                prec = cls.from_cache(
                    pdb_id, cache_dir=cache_dir, pdb_source=pdb_source
                )
                if prec is not None:
                    return prec

            if not pdb_dict:
                pdb_dict = pdb.pdb_dict(
                    pdb_id, pdb_source=pdb_source, struct_d=pdb_dict
                )

            prec = cls(
                pdb_id,
                pdb_source=pdb_source,
                pdb_dict=pdb_dict,
                strict_pdb_unp_xref=strict_pdb_unp_xref,
                **kw_for_init,
            )
            if cache_dir:
                prec.save(out_dir=cache_dir)

            return prec
        except Exception as e:
            raise ProteinInitError(
                f"Failed to create protein record for pdb_id={pdb_id}: {e}"
            ) from e

    @classmethod
    def from_unp(
        cls,
        unp_id: str,
        cache=False,
        pdb_source: str = PDB_RCSB,
        cache_dir=pp5.PREC_DIR,
        xref_selector: Callable[[unp.UNPPDBXRef], Any] = None,
        **kw_for_init,
    ) -> ProteinRecord:
        """
        Creates a ProteinRecord from a Uniprot ID.
        The PDB structure with best resolution will be used.
        :param unp_id: The Uniprot id to query.
        :param xref_selector: Sort key for PDB cross refs. If None,
        resolution will be used.
        :param cache: Whether to load prec from cache if available.
        :param pdb_source: Source from which to obtain the pdb file.
        :param cache_dir: Where the cache dir is. ProteinRecords will be
        written to this folder after creation, unless it's None.
        :param kw_for_init: Extra args for the ProteinRecord initializer.
        :return: A ProteinRecord.
        """
        if not xref_selector:
            xref_selector = lambda xr: xr.resolution

        try:
            xrefs = unp.find_pdb_xrefs(unp_id)
            xrefs = sorted(xrefs, key=xref_selector)
            pdb_id = f"{xrefs[0].pdb_id}:{xrefs[0].chain_id}"

            if cache:
                prec = cls.from_cache(
                    pdb_id, cache_dir=cache_dir, pdb_source=pdb_source
                )
                if prec is not None:
                    return prec

            prec = cls(pdb_id, **kw_for_init)
            if cache_dir:
                prec.save(out_dir=cache_dir)

            return prec
        except Exception as e:
            raise ProteinInitError(
                f"Failed to create protein record for unp_id={unp_id}"
            ) from e

    def __init__(
        self,
        pdb_id: str,
        pdb_source: str = PDB_RCSB,
        pdb_dict: dict = None,
        dihedral_est_name: str = None,
        dihedral_est_args: dict = None,
        max_ena: int = None,
        with_altlocs: bool = True,
        with_backbone: bool = True,
        with_contacts: bool = True,
        with_atom_contacts: bool = False,
        with_codons: bool = True,
        strict_pdb_unp_xref: bool = True,
        contact_method: str = CONTACT_METHOD_NEIGHBOR,
        contact_radius: float = CONTACT_DEFAULT_RADIUS,
    ):
        """
        Don't call this directly. Use class methods from_pdb or from_unp instead.

        Initialize a protein record from PDB id.

        :param pdb_id: PDB id with chain (e.g. '1ABC:D') of the specific structure chain
        desired.
        :param pdb_source: Source from which to obtain the pdb file.
        :param dihedral_est_name: Method of dihedral angle estimation.
        Options are:
        None or empty to calculate angles without error estimation;
        'erp' for standard error propagation;
        'mc' for montecarlo error estimation.
        :param dihedral_est_args: Extra arguments for dihedral estimator.
        :param max_ena: Number of maximal ENA records (containing protein
        genetic data) to align to the PDB structure of this protein. None
        means no limit (all cross-refs from Uniprot will be aligned).
        :param with_altlocs: Whether to include alternate conformations in the
        protein record. If False, only the default conformation will be used.
        :param with_backbone: Whether to include backbone atoms in the protein record.
        :param with_contacts: Whether to calculate per-residue contacts.
        :param with_atom_contacts: If true, a separate output file will be created
        containing all atom level contacts.
        :param with_codons: Whether to assign codons to each residue.
        :param strict_pdb_unp_xref: Whether to require that the given PDB ID
        maps uniquely to only one Uniprot ID.
        :param contact_method: Method for calculating contacts.
        Options are: 'ns' for neighbor search; 'arp' for arpeggio.
        :param contact_radius: Radius for calculating contacts.
        """
        if not pdb_id:
            raise ProteinInitError("Must provide PDB ID")

        self.__setstate__({})

        # Parse the given PDB id and obtain metadata
        self.pdb_base_id, pdb_chain_id, ent_id = pdb.split_id_with_entity(pdb_id)

        LOGGER.info(f"{self.pdb_base_id}: Obtaining metadata...")
        self.pdb_meta = pdb.PDBMetadata.from_pdb(self.pdb_base_id, cache=True)

        if pdb_chain_id is None:
            if ent_id and len(self.pdb_meta.entity_chains.get(ent_id, [])) == 1:
                pdb_chain_id = self.pdb_meta.entity_chains[ent_id][0]
            elif len(self.pdb_meta.chain_ids) == 1:
                pdb_chain_id = next(iter(self.pdb_meta.chain_ids))
            else:
                raise ProteinInitError(
                    f"No chain specified in {pdb_id}, and multiple chains exist."
                )

        self.pdb_chain_id = pdb_chain_id
        self.pdb_id = f"{self.pdb_base_id}:{self.pdb_chain_id}"

        LOGGER.info(f"{self.pdb_id}: Constructing protein record...")

        # Obtain UniProt ID for the given PDB chain
        chain_unp_ids = self.pdb_meta.chain_uniprot_ids[self.pdb_chain_id]
        if not chain_unp_ids:
            raise ProteinInitError(f"No Uniprot ID found for chain {self.pdb_chain_id}")
        if len(chain_unp_ids) > 1:
            msg = f"Multiple UNP IDs for chain {self.pdb_chain_id}: {chain_unp_ids}"
            if strict_pdb_unp_xref:
                raise ProteinInitError(msg)
            else:
                LOGGER.warning(msg)

        self.unp_id = chain_unp_ids[0]
        rec_unp_id = self.unp_rec.accessions[0]
        if rec_unp_id != self.unp_id:
            LOGGER.warning(f"Replacing outdated UNP ID: {self.unp_id} -> {rec_unp_id}")
            self.unp_id = rec_unp_id

        if contact_method not in CONTACT_METHODS:
            raise ValueError(
                f"Unknown {contact_method=}, must be one of {CONTACT_METHODS}"
            )

        if with_altlocs and contact_method == CONTACT_METHOD_ARPEGGIO:
            raise ValueError(f"Altlocs not supported with {contact_method=}")

        self.with_altlocs = with_altlocs
        self.with_backbone = with_backbone
        self.with_contacts = with_contacts
        self.with_atom_contacts = with_atom_contacts
        self.contact_radius = contact_radius
        self.contact_method = contact_method

        self.pdb_source = pdb_source
        if pdb_dict:
            self._pdb_dict = pdb_dict

        if not self.pdb_meta.resolution and self.pdb_source != PDB_AFLD:
            raise ProteinInitError(f"Unknown resolution for {pdb_id}")

        self.pdb_entity_id = self.pdb_meta.chain_entities[self.pdb_chain_id]
        self.pdb_auth_chain_id = self.pdb_meta.chain_to_auth_chain[self.pdb_chain_id]

        chain_str = (
            self.pdb_chain_id
            if self.pdb_auth_chain_id == self.pdb_chain_id
            else f"{self.pdb_chain_id}({self.pdb_auth_chain_id})"
        )
        LOGGER.info(
            f"pdb_id={self.pdb_base_id}, chain={chain_str}, unp_id={self.unp_id}, "
            f"entity_id={self.pdb_entity_id}, "
            f"res={self.pdb_meta.resolution:.2f}Å, "
            f"desc={self.pdb_meta.entity_description[self.pdb_entity_id]}, "
            f"org={self.pdb_meta.entity_source_org[self.pdb_entity_id]} "
            f"({self.pdb_meta.entity_source_org_id[self.pdb_entity_id]}), "
            f"expr={self.pdb_meta.entity_host_org[self.pdb_entity_id]} "
            f"({self.pdb_meta.entity_host_org_id[self.pdb_entity_id]})"
        )

        # Make sure the structure is sane. See e.g. 1FFK.
        if not self.polypeptides:
            raise ProteinInitError(f"No parsable residues in {self.pdb_id}")

        # Get secondary-structure info using DSSP
        ss_dict, _ = pdb.pdb_to_secondary_structure(self.pdb_id, pdb_source=pdb_source)

        # Get estimators of dihedral angles and b-factor
        self.dihedral_est_name = dihedral_est_name
        dihedral_est, bfactor_est = self._get_dihedral_estimators(
            dihedral_est_name, dihedral_est_args
        )

        # Extract the residues from the PDB structure: these are the modelled
        # residues. We ignore water molecules.
        struct_chain: Chain = self.pdb_rec[0][self.pdb_auth_chain_id]
        modelled_residues: List[Residue] = [
            res for res in struct_chain.get_residues() if res.resname != "HOH"
        ]

        # Sequence of modelled residues from the PDB structure
        pdb_modelled_aa_seq: str = str.join(
            "",
            [ACIDS_3TO1.get(r.get_resname(), UNKNOWN_AA) for r in modelled_residues],
        )
        assert len(pdb_modelled_aa_seq) == len(modelled_residues)

        # The canonical AA sequence from the structure metadata
        pdb_meta_aa_seq = self.pdb_meta.entity_sequence[
            self.pdb_meta.chain_entities[self.pdb_chain_id]
        ]

        # Add un-modelled residues by aligning to the canonical PDB sequence.
        meta_to_struct_seq_alignment, meta_to_struct_idx = pairwise_alignment_map(
            pdb_meta_aa_seq, pdb_modelled_aa_seq
        )
        LOGGER.info(
            f"{self}: Canonical (target) to structure (query) sequence alignment:\n"
            f"{str(meta_to_struct_seq_alignment).strip()}"
        )
        matching_residues: List[Residue] = []  # residues both in modelled and in meta
        missing_residues: List[Residue] = []  # residues only in meta
        for curr_meta_seq_idx in range(len(pdb_meta_aa_seq)):
            if curr_meta_seq_idx in meta_to_struct_idx:
                # This residue is one of the modelled residues in the structure
                modelled_seq_idx = meta_to_struct_idx[curr_meta_seq_idx]
                curr_residue = modelled_residues[modelled_seq_idx]
            else:
                # This residue is not modelled (missing from the structure), need to add
                unmodelled_res_name_single = pdb_meta_aa_seq[curr_meta_seq_idx]
                unmodelled_res_name = ACIDS_1TO3.get(
                    unmodelled_res_name_single, UNKNOWN_AA
                )
                unmodelled_count = 0  # number of consecutive unmodelled residues

                # We need to determine the residue sequence index for the missing
                # residue. It needs to be consistent with the sequence index of the
                # modelled residues.
                if len(matching_residues) > 0:
                    # We have previous matching residues, so we can infer the index
                    # based on the last matching residue (LMR).
                    _, lmr_seq_idx, lmr_icode = matching_residues[-1].get_id()

                    # One or more unmodelled residues will be inserted after the
                    # previous matching residue using the same sequence index. We use
                    # the icode to disambiguate them. Each consecutive unmodelled
                    # residue will have an icode of the form U_i, where i is a counter.
                    unmodelled_seq_idx = lmr_seq_idx
                    if lmr_icode.strip().startswith(ICODE_UNMODELED_RES):
                        unmodelled_count = int(lmr_icode.split("_")[1]) + 1
                else:
                    # We have no matching residues yet. We'll insert the unmodelled
                    # residue(s) before the first matching residue (FMR), using a
                    # smaller sequence index.
                    fmr_meta_idx: int = next(iter(meta_to_struct_idx))
                    fmr_struct_idx: int = meta_to_struct_idx[fmr_meta_idx]
                    fmr_modelled: Residue = modelled_residues[fmr_struct_idx]
                    _, fmr_seq_idx, fmr_icode = fmr_modelled.get_id()
                    unmodelled_seq_idx = fmr_seq_idx - 1
                    # Sanity check: the first unmodelled residue should be inserted
                    # before a modelled one. We should only be in this 'else' once.
                    assert not fmr_icode.startswith(ICODE_UNMODELED_RES)

                unmodelled_res_icode = f"{ICODE_UNMODELED_RES}_{unmodelled_count:04d}"
                missing_residue_id = (" ", unmodelled_seq_idx, unmodelled_res_icode)
                curr_residue = Residue(missing_residue_id, unmodelled_res_name, 0)
                missing_residues.append(curr_residue)

            matching_residues.append(curr_residue)

        # Sanity check
        matching_aa_seq = str.join(
            "", [ACIDS_3TO1.get(r.get_resname(), UNKNOWN_AA) for r in matching_residues]
        )
        assert pdb_meta_aa_seq == matching_aa_seq

        # Detect any residues that are modelled but not in the canonical sequence,
        # these would usually be ligands and non-standard AAs.
        matching_modelled_residues = set(matching_residues) - set(missing_residues)
        extra_modelled_residues = set(modelled_residues) - matching_modelled_residues

        # Sort all residues by their sequence index, then icode. Include the extra
        # residues at the end.
        def _residue_sort_key(r: Residue) -> Tuple[int, str, str]:
            _het_flag, _seq_idx, _icode = r.get_id()
            # Ensure everything has the expected type
            _het_flag = str(_het_flag or "").strip()
            _icode = str(_icode or "").strip()
            _seq_idx = int(_seq_idx)
            return _seq_idx, _icode, _het_flag

        all_residues = sorted(
            [*matching_residues, *extra_modelled_residues], key=_residue_sort_key
        )
        n_residues = len(all_residues)

        # Obtain single-letter AA sequence with all residues (including unmodelled),
        # and with non-standard AAs or ligands represented by 'X'.
        all_aa_seq = str.join(
            "", [ACIDS_3TO1.get(r.get_resname(), UNKNOWN_AA) for r in all_residues]
        )

        # Find the best matching DNA for our AA sequence via pairwise alignment
        # between the PDB AA sequence and translated DNA sequences.
        self.ena_id = None
        self._dna_seq = None
        idx_to_codons = {}
        if with_codons:
            self.ena_id, self._dna_seq, idx_to_codons = self._find_dna_alignment(
                all_aa_seq, max_ena
            )

        # Calculate contacts if requested
        contacts_assigner: Optional[ContactsAssigner] = None
        if with_contacts:
            if contact_method == CONTACT_METHOD_ARPEGGIO:
                contacts_assigner = ArpeggioContactsAssigner(
                    pdb_id=self.pdb_id,
                    pdb_source=self.pdb_source,
                    contact_radius=self.contact_radius,
                )
            else:
                contacts_assigner = NeighborSearchContactsAssigner(
                    pdb_id=self.pdb_id,
                    pdb_source=self.pdb_source,
                    contact_radius=self.contact_radius,
                    with_altlocs=self.with_altlocs,
                    with_atom_contacts=with_atom_contacts,
                    pdb_dict=self._pdb_dict,
                )

        # Align PDB sequence to UNP
        unp_alignment, pdb_to_unp_idx = pairwise_alignment_map(
            all_aa_seq, self.unp_rec.sequence
        )
        LOGGER.info(f"{self}: PDB to UNP alignment score={unp_alignment.score}")

        # Create a ResidueRecord holding all data we need per residue
        residue_recs = []
        for i in range(n_residues):
            r_curr: Residue = all_residues[i]
            relative_location = (i + 1) / n_residues

            # Sanity check
            assert all_aa_seq[i] == ACIDS_3TO1.get(r_curr.get_resname(), UNKNOWN_AA)

            # Get the residues before and after this one
            r_prev: Optional[Residue] = all_residues[i - 1] if i > 0 else None
            r_next: Optional[Residue] = (
                all_residues[i + 1] if i < n_residues - 1 else None
            )

            # Get corresponding UNP index
            unp_idx: Optional[int] = pdb_to_unp_idx.get(i, None)

            # Secondary structure annotation
            secondary: str = ss_dict.get((self.pdb_chain_id, r_curr.get_id()), "-")

            # Codons options for residue
            codon_counts: Dict[str, int] = idx_to_codons.get(
                i, {} if with_codons else None
            )

            # Instantiate a ResidueRecord
            rr = AltlocResidueRecord.from_residue(
                r_curr,
                r_prev,
                r_next,
                unp_idx=unp_idx,
                rel_loc=relative_location,
                codon_counts=codon_counts,
                secondary=secondary,
                dihedral_est=dihedral_est,
                bfactor_est=bfactor_est,
                with_backbone=with_backbone,
                with_altlocs=with_altlocs,
                with_contacts=with_contacts,
                contacts_assigner=contacts_assigner,
            )
            residue_recs.append(rr)

        self._aa_seq = all_aa_seq
        self._residue_recs: Dict[str, ResidueRecord] = {
            rr.res_id: rr for rr in residue_recs
        }

    @property
    def unp_rec(self) -> UNPRecord:
        """
        :return: Uniprot record for this protein.
        """
        if not self._unp_rec:
            self._unp_rec = unp.unp_record(self.unp_id)
        return self._unp_rec

    @property
    def pdb_dict(self) -> dict:
        """
        :return: The PDB record for this protein as a raw dict parsed from
        an mmCIF file.
        """
        if not self._pdb_dict:
            self._pdb_dict = pdb.pdb_dict(self.pdb_id, pdb_source=self.pdb_source)
        return self._pdb_dict

    @property
    def pdb_rec(self) -> PDBRecord:
        """
        :return: PDB record for this protein. Note that this record may
        contain multiple chains and this prec only represents one of them
        (self.pdb_chain_id).
        """
        if not self._pdb_rec:
            self._pdb_rec = pdb.pdb_struct(
                self.pdb_id, pdb_source=self.pdb_source, struct_d=self.pdb_dict
            )
        return self._pdb_rec

    @property
    def dna_seq(self) -> Optional[SeqRecord]:
        """
        :return: DNA nucleotide sequence. This is the full DNA sequence which,
        after translation, best-matches to the PDB AA sequence.
        """
        if self.ena_id is None or self._dna_seq is None:
            return None
        return SeqRecord(Seq(self._dna_seq), self.ena_id, "", "")

    @property
    def aa_seq(self) -> str:
        """
        :return: Protein sequence as 1-letter AA names.
        Based on the residues found in the associated PDB structure, including those
        which are not modelled.
        Note that the sequence might contain the letter 'X' denoting a non-standard
        AA or a ligand.
        """
        return self._aa_seq

    @property
    def seq_gaps(self) -> Sequence[Tuple[str, str]]:
        """
        :return: A list of tuples (start, end) of residue ids corresponding to
        the beginning and end of gaps in the protein sequence. A gap is determined
        by one or more residues with a non-standard AA.
        """
        gaps = []
        res_iter = iter(self)
        for res in res_iter:
            curr_gap = []
            while res.name is UNKNOWN_AA:
                curr_gap.append(res.res_id)
                res = next(res_iter)
            if curr_gap:
                gaps.append((curr_gap[0], curr_gap[-1]))

        return tuple(gaps)

    @property
    def codons(self) -> Dict[str, str]:
        """
        :return: Protein sequence based on translating DNA sequence with
        standard codon table.
        """
        return {x.res_id: x.codon for x in self}

    @property
    def dihedral_angles(self) -> Dict[str, Dihedral]:
        return {x.res_id: x.angles for x in self}

    @property
    def contacts(self) -> Dict[str, ResidueContacts]:
        """
        :return: Mapping from residue id to its contact features.
        """
        if not self.with_contacts:
            raise ValueError("Contacts were not calculated for this protein record")
        return {res_id: res.contacts for res_id, res in self._residue_recs.items()}

    @property
    def polypeptides(self) -> List[Polypeptide]:
        """
        :return: List of Polypeptide objects corresponding to the PDB chain of
        this protein. If there is more than one, they are "sub chains"
        within the chain represented by this ProteinRecord.
        Even though we're working with one PDB chain, the results is a
        list of multiple Polypeptide objects because we split them at
        non-standard residues (HETATM atoms in PDB).
        https://proteopedia.org/wiki/index.php/HETATM
        """
        if not self._pp:
            # Use author chain id to get the polypeptides, as the author chain is
            # what's associated with the coordinates in the mmCIF file.
            chain = self.pdb_rec[0][self.pdb_auth_chain_id]
            pp_chains = PPBuilder().build_peptides(chain, aa_only=True)

            # Sort chain by sequence ID of first residue in the chain,
            # in case the chains are not returned in order.
            pp_chains = sorted(pp_chains, key=lambda ch: ch[0].get_id()[1])
            self._pp = pp_chains

        return self._pp

    @property
    def num_altlocs(self) -> int:
        """
        :return: Number of positions with alternate locations (altlocs) in this chain.
        """
        return sum(r.num_altlocs > 1 for r in self._residue_recs.values())

    @property
    def num_unmodelled(self) -> Tuple[int, int, int]:
        """
        Counts number of unmodelled residues in this chain.

        :return: A tuple of three integers:
        - Number of unmodelled residues before the first modelled residue (N-terminus).
        - Number of unmodelled residues between modelled residues.
        - Number of unmodelled residues after the last modelled residue (C-terminus).
        """
        count_pre, count_mid, count_post = 0, 0, 0
        modelled_idx = [
            i
            for i, res in enumerate(self)
            if not res.res_icode.startswith(ICODE_UNMODELED_RES)
            and res.name in ACIDS_1TO3
        ]

        if len(modelled_idx) > 1:
            first_modelled_idx, *_, last_modelled_idx = modelled_idx
            count_pre, count_mid, count_post = 0, 0, 0

            res: ResidueRecord
            for i, res in enumerate(self):
                if res.res_icode.startswith(ICODE_UNMODELED_RES):
                    if i < first_modelled_idx:
                        count_pre += 1
                    elif i > last_modelled_idx:
                        count_post += 1
                    else:
                        count_mid += 1

        return count_pre, count_mid, count_post

    def to_dataframe(self):
        """
        :return: A Pandas dataframe where each row is a ResidueRecord from
        this ProteinRecord.
        """
        dihedral_args = dict(
            degrees=True,
            skip_omega=False,
            with_std=(self.dihedral_est_name is not None),
        )
        df_data = []
        for res_id, res_rec in self.items():
            res_rec_dict = res_rec.as_dict(dihedral_args=dihedral_args)
            df_data.append(res_rec_dict)

        df_prec = pd.DataFrame(df_data)

        # Insert the Uniprot and PDB IDs as the first columns
        df_prec.insert(loc=0, column="unp_id", value=self.unp_id)
        df_prec.insert(loc=0, column="pdb_id", value=self.pdb_id)

        return df_prec

    def _atom_contacts_dataframe(self) -> Optional[pd.DataFrame]:
        if not self.with_atom_contacts:
            return None

        df_data = []
        for res_id, res_rec in self.items():
            res_atom_contacts: Sequence[AtomContact] = res_rec.contacts.atom_contacts
            df_data.extend(ac.as_dict() for ac in res_atom_contacts)

        return pd.DataFrame(df_data)

    def to_csv(self, out_dir=pp5.out_subdir("prec"), tag=None):
        """
        Writes the ProteinRecord as a CSV file, by writing the dataframe produced by
        self.to_dataframe() to CSV.

        Filename will be <PDB_ID>_<CHAIN_ID>_<TAG>.csv.
        Note that this is meant as a human-readable output format only,
        so a ProteinRecord cannot be re-created from this CSV.
        To save a ProteinRecord for later loading, use save().

        :param out_dir: Output dir.
        :param tag: Optional extra tag to add to filename.
        :return: The path to the written file.
        """
        os.makedirs(out_dir, exist_ok=True)
        filepath = pdb_tagged_filepath(
            self.pdb_id, self.pdb_source, out_dir, "csv", tag
        )
        df = self.to_dataframe()
        df.to_csv(
            filepath,
            na_rep="",
            header=True,
            index=False,
            encoding="utf-8",
            float_format="%.4f",
        )
        LOGGER.info(f"Wrote {self} to {filepath}")

        if self.with_atom_contacts:
            df_atom_contacts = self._atom_contacts_dataframe()
            filepath = pdb_tagged_filepath(
                self.pdb_id,
                self.pdb_source,
                out_dir,
                suffix="csv",
                tag=f"{tag}-atom-contacts" if tag else "atom-contacts",
            )
            df_atom_contacts.to_csv(
                filepath,
                na_rep="",
                header=True,
                index=False,
                encoding="utf-8",
                float_format="%.4f",
            )
            LOGGER.info(f"Wrote {self} atom contacts to {filepath}")

        return filepath

    def save(self, out_dir=pp5.data_subdir("prec"), tag=None):
        """
        Write the ProteinRecord to a binary file which can later to
        re-loaded into memory, recreating the ProteinRecord.
        :param out_dir: Output dir.
        :param tag: Optional extra tag to add to filename.
        :return: The path to the written file.
        """
        filepath = pdb_tagged_filepath(
            self.pdb_id, self.pdb_source, out_dir, "prec", tag
        )
        filepath = pp5.get_resource_path(filepath.parent, filepath.name)
        os.makedirs(filepath.parent, exist_ok=True)

        with filelock_context(filepath):
            with open(str(filepath), "wb") as f:
                pickle.dump(self, f, protocol=4)

        LOGGER.info(f"Wrote {self} to {filepath}")
        return filepath

    def _find_dna_alignment(
        self, pdb_aa_seq: str, max_ena: int
    ) -> Tuple[Optional[str], Optional[str], Dict[int, Dict[str, int]]]:
        """
        Aligns between this prec's AA sequence and all known DNA (from the
        ENA database) sequences of the corresponding Uniprot ID.
        :param pdb_aa_seq: AA sequence from PDB to align.
        :param max_ena: Maximal number of DNA sequences to consider.
        :return: A tuple:
            - The ENA id of the DNA sequence which best aligns to the provided AAs.
            - The DNA sequence which best aligns to the provided AAs.
            - SeqRecord of the DNA sequence which best aligns to the provided AAs.
            of codon counts. The second dict maps from a codon (e.g. 'CCT') to a
            count, representing the number of times this codon was found in the
            location of the corresponding AA index.
        """
        # Find cross-refs in ENA
        ena_molecule_types = ("mrna", "genomic_dna")
        ena_ids = unp.find_ena_xrefs(self.unp_rec, ena_molecule_types)

        # Map id to sequence by fetching from ENA API
        ena_seqs = []
        for i, ena_id in enumerate(ena_ids):
            try:
                ena_seqs.append(ena.ena_seq(ena_id))
            except IOError as e:
                LOGGER.warning(f"{self}: Invalid ENA id {ena_id}")
            if max_ena is not None and i > max_ena:
                LOGGER.warning(f"{self}: Over {max_ena} ENA ids, skipping")
                break

        aligner = PairwiseAligner(
            substitution_matrix=BLOSUM80, open_gap_score=-10, extend_gap_score=-0.5
        )
        alignments = []
        for seq in ena_seqs:
            # Handle case of DNA sequence with incomplete codons
            if len(seq) % 3 != 0:
                if seq[-3:].seq in STOP_CODONS:
                    seq = seq[-3 * (len(seq) // 3) :]
                else:
                    seq = seq[: 3 * (len(seq) // 3)]

            # Translate to AA sequence and align to the PDB sequence
            translated = seq.translate(stop_symbol="")
            alignment = aligner.align(pdb_aa_seq, translated.seq)
            alignments.append((seq, alignment))

        if len(alignments) == 0:
            LOGGER.warning(f"Can't find ENA id for {self.unp_id}")
            return None, None, {}

        # Sort alignments by negative score (we want the highest first)
        sorted_alignments = sorted(alignments, key=lambda x: -x[1].score)

        # Print best-matching alignment
        best_ena, best_alignments = sorted_alignments[0]
        best_alignment = best_alignments[0]
        LOGGER.info(f"{self}: ENA ID = {best_ena.id}")
        LOGGER.info(
            f"{self}: Translated DNA to PDB alignment "
            f"(norm_score="
            f"{best_alignments.score / len(pdb_aa_seq):.2f}, "
            f"num={len(best_alignments)})"
        )

        # Map each AA to a dict of (codon->count)
        idx_to_codons = {}
        for ena_seq, multi_alignment in sorted_alignments:
            alignment = multi_alignment[0]  # multiple equivalent alignments
            # Indices in the DNA AA seq which are aligned to the PDB AA seq
            aligned_idx_pdb_aa, aligned_idx_dna_aa = alignment.aligned
            dna_seq = str(ena_seq.seq)

            for i in range(len(aligned_idx_dna_aa)):
                # Indices of current matching segment of amino acids from the
                # PDB record and the translated DNA
                pdb_aa_start, pdb_aa_stop = aligned_idx_pdb_aa[i]
                dna_aa_start, dna_aa_stop = aligned_idx_dna_aa[i]

                for offset, k in enumerate(range(dna_aa_start, dna_aa_stop)):
                    k *= 3
                    codon = dna_seq[k : k + 3]
                    pdb_idx = offset + pdb_aa_start

                    # Skip "unknown" AAs - we put them there to represent
                    # non-standard AAs.
                    if pdb_aa_seq[pdb_idx] == UNKNOWN_AA:
                        continue

                    # List of codons at current index
                    codon_dict = idx_to_codons.get(pdb_idx, OrderedDict())

                    # Check if the codon actually matched the AA at the
                    # corresponding location. Sometimes there may be
                    # mismatches ('.') because the DNA alignment isn't perfect.
                    # In such a case we'll set the codon to None to specify we
                    # don't know what codon encoded the AA in the PDB sequence.
                    aa_name = CODON_TABLE.get(codon, None)
                    matches = aa_name == pdb_aa_seq[pdb_idx]
                    codon = codon if matches else UNKNOWN_CODON

                    # Map each codon to number of times seen
                    codon_dict[codon] = codon_dict.get(codon, 0) + 1
                    idx_to_codons[pdb_idx] = codon_dict

        return best_ena.id, str(best_ena.seq), idx_to_codons

    def _get_dihedral_estimators(self, est_name: str, est_args: dict):
        est_name = est_name.lower() if est_name else est_name
        est_args = {} if est_args is None else est_args

        if not est_name in {None, "", "erp", "mc"}:
            raise ProteinInitError(f"Unknown dihedral estimation method {est_name}")

        unit_cell = pdb.PDBUnitCell(self.pdb_id) if est_name else None
        args = dict(isotropic=False, n_samples=100, skip_omega=True)
        args.update(est_args)

        if est_name == "mc":
            d_est = DihedralAnglesMonteCarloEstimator(unit_cell, **args)
        elif est_name == "erp":
            d_est = DihedralAnglesUncertaintyEstimator(unit_cell, **args)
        else:
            if est_args:
                raise ProteinInitError(f"est_args not supported for {est_name=}")
            d_est = DEFAULT_ANGLE_CALC

        b_est = DEFAULT_BFACTOR_CALC
        return d_est, b_est

    def __iter__(self) -> Iterator[ResidueRecord]:
        return iter(self._residue_recs.values())

    def __getitem__(self, item: Union[str, int]) -> ResidueRecord:
        """
        :param item: A PDB residue id, either as an int e.g. 42 or a str e.g. 42A.
        :return: the corresponding residue record.
        """
        return self._residue_recs[str(item)]

    def __contains__(self, item: Union[str, int]) -> bool:
        return str(item) in self._residue_recs

    def items(self) -> ItemsView[str, ResidueRecord]:
        """
        :return: Entries of this prec as (residue id, residue record).
        """
        return self._residue_recs.items()

    def __repr__(self):
        return f"{self.pdb_id}"

    def __getstate__(self):
        # Prevent serializing Bio objects
        state = self.__dict__.copy()
        for attr in self._SKIP_SERIALIZE:
            del state[attr]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        for attr in self._SKIP_SERIALIZE:
            self.__setattr__(attr, None)

    def __len__(self):
        return len(self._residue_recs)

    def __eq__(self, other):
        if self is other:
            return True
        if not isinstance(other, ProteinRecord):
            return False
        if self.pdb_id != other.pdb_id:
            return False
        if self.unp_id != other.unp_id:
            return False
        if len(self) != len(other):
            return False
        return all(map(lambda x: x[0] == x[1], zip(self, other)))
