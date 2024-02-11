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
from Bio.SeqRecord import SeqRecord
from Bio.PDB.Residue import Residue
from Bio.PDB.Polypeptide import Polypeptide

import pp5
from pp5.align import BLOSUM80
from pp5.utils import ProteinInitError, filelock_context
from pp5.codons import (
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


class ResidueRecord(object):
    """
    Represents a single residue in a protein record.
    """

    def __init__(
        self,
        res_id: Union[str, int],
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
    ):
        """

        :param res_id: identifier of this residue in the sequence, usually an
            integer + insertion code if present, which indicates some alteration
            compared to the wild-type.
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

        self.res_id, self.name = str(res_id), name
        self.unp_idx, self.rel_loc = unp_idx, rel_loc
        self.codon, self.codon_score = best_codon, codon_score
        self.codon_opts = str.join(CODON_OPTS_SEP, codon_opts)
        self.codon_counts = codon_counts
        self.angles, self.bfactor, self.secondary = angles, bfactor, secondary
        self.num_altlocs = num_altlocs
        self.backbone_coords = backbone_coords or {}
        self.contacts = contacts

    def as_dict(self, dihedral_args: Dict[str, Any] = None):
        """
        Creates a dict representation of the data in this residue. The angles object
        will we flattened out so its attributes will be placed directly in the
        resulting dict. The backbone angles will be converted to a nested list.
        :param dihedral_args: A dict of arguments to pass to Dihedral.as_dict.
        :return: A dict representation of this residue.
        """
        return dict(
            res_id=self.res_id,
            name=self.name,
            unp_idx=self.unp_idx,
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
        for k, v in self.__dict__.items():
            other_v = other.__dict__.get(k, math.inf)
            if isinstance(v, (float, np.ndarray)):
                equal = np.allclose(v, other_v, equal_nan=True)
            else:
                equal = v == other_v
            if not equal:
                return False
        return True


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
        res_id: Union[str, int],
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
            res_id=res_id,
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
        res_id: str = res_to_id(r_curr)
        res_name: str = ACIDS_3TO1.get(r_curr.get_resname(), UNKNOWN_AA)

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
            res_id=res_id,
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
        strict_pdb_xref=True,
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
        :param strict_pdb_xref: Whether to require that the given PDB ID
        maps uniquely to only one Uniprot ID.
        :param kw_for_init: Extra kwargs for the ProteinRecord initializer.
        :return: A ProteinRecord.
        """
        try:
            # Either chain or entity or none can be provided, but not both
            pdb_base_id, chain_id, entity_id = pdb.split_id_with_entity(pdb_id)
            numeric_chain = False
            if entity_id:
                entity_id = int(entity_id)

                # Discover which chains belong to this entity
                pdb_dict = pdb.pdb_dict(
                    pdb_id, pdb_source=pdb_source, struct_d=pdb_dict
                )
                meta = pdb.PDBMetadata(pdb_id, pdb_source=pdb_source, struct_d=pdb_dict)
                chain_id = meta.get_chain(entity_id)

                if not chain_id:
                    # In rare cases the chain is a number instead of a letter,
                    # so there's no way to distinguish between entity id and
                    # chain except also trying to use our entity as a chain
                    # and finding the actual entity. See e.g. 4N6V.
                    if str(entity_id) in meta.chain_entities:
                        # Chain is number, but use its string representation
                        chain_id = str(entity_id)
                        numeric_chain = True
                    else:
                        raise ProteinInitError(
                            f"No matching chain found for entity "
                            f"{entity_id} in PDB structure {pdb_base_id}"
                        )

                pdb_id = f"{pdb_base_id}:{chain_id}"

            if cache and chain_id:
                prec = cls.from_cache(pdb_id, cache_dir=cache_dir)
                if prec is not None:
                    return prec

            if not pdb_dict:
                pdb_dict = pdb.pdb_dict(
                    pdb_id, pdb_source=pdb_source, struct_d=pdb_dict
                )

            unp_id = pdb.PDB2UNP.pdb_id_to_unp_id(
                pdb_id, strict=strict_pdb_xref, cache=cache
            )

            prec = cls(
                unp_id,
                pdb_id,
                pdb_source=pdb_source,
                pdb_dict=pdb_dict,
                numeric_chain=numeric_chain,
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
                prec = cls.from_cache(pdb_id, cache_dir=cache_dir)
                if prec is not None:
                    return prec

            prec = cls(unp_id, pdb_id, **kw_for_init)
            if cache_dir:
                prec.save(out_dir=cache_dir)

            return prec
        except Exception as e:
            raise ProteinInitError(
                f"Failed to create protein record for " f"unp_id={unp_id}"
            ) from e

    def __init__(
        self,
        unp_id: str,
        pdb_id: str,
        pdb_source: str = PDB_RCSB,
        pdb_dict: dict = None,
        dihedral_est_name: str = None,
        dihedral_est_args: dict = None,
        max_ena: int = None,
        strict_unp_xref: bool = True,
        numeric_chain: bool = False,
        with_altlocs: bool = True,
        with_backbone: bool = True,
        with_contacts: bool = True,
        with_codons: bool = True,
        contact_method: str = CONTACT_METHOD_NEIGHBOR,
        contact_radius: float = CONTACT_DEFAULT_RADIUS,
    ):
        """
        Initialize a protein record from both Uniprot and PDB ids.
        To initialize a protein from Uniprot id or PDB id only, use the
        class methods provided for this purpose.

        :param unp_id: Uniprot id which uniquely identifies the protein.
        :param pdb_id: PDB id with or without chain (e.g. '1ABC' or '1ABC:D')
        of the specific structure desired. Note that this structure must match
        the unp_id, i.e. it must exist in the cross-refs of the given unp_id.
        Otherwise an error will be raised (unless strict_unp_xref=False). If no
        chain is specified, a chain matching the unp_id will be used,
        if it exists.
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
        :param strict_unp_xref: Whether to require that there exist a PDB
        cross-ref for the given Uniprot ID.
        :param numeric_chain: Whether the given chain id (if any) is
        numeric. In rare cases PDB structures have numbers as chain ids.
        :param with_altlocs: Whether to include alternate conformations in the
        protein record. If False, only the default conformation will be used.
        :param with_backbone: Whether to include backbone atoms in the protein record.
        :param with_contacts: Whether to calculate per-residue contacts.
        :param with_codons: Whether to assign codons to each residue.
        :param contact_method: Method for calculating contacts.
        Options are: 'ns' for neighbor search; 'arp' for arpeggio.
        :param contact_radius: Radius for calculating contacts.
        """
        if not (unp_id and pdb_id):
            raise ProteinInitError("Must provide both Uniprot and PDB IDs")

        unp_id = unp_id.upper()
        LOGGER.info(f"{unp_id}: Initializing protein record...")
        self.__setstate__({})

        self.unp_id = unp_id
        rec_unp_id = self.unp_rec.accessions[0]
        if rec_unp_id != unp_id:
            LOGGER.warning(f"Replacing outdated UNP ID: {unp_id} -> {rec_unp_id}")
            self.unp_id = rec_unp_id

        if contact_method not in CONTACT_METHODS:
            raise ValueError(
                f"Unknown {contact_method=}, must be one of {CONTACT_METHODS}"
            )

        if with_altlocs and contact_method == CONTACT_METHOD_ARPEGGIO:
            raise ValueError(f"Altlocs not supported with {contact_method=}")

        self.strict_unp_xref = strict_unp_xref
        self.numeric_chain = numeric_chain
        self.with_altlocs = with_altlocs
        self.with_backbone = with_backbone
        self.with_contacts = with_contacts
        self.contact_radius = contact_radius
        self.contact_method = contact_method

        # First we must find a matching PDB structure and chain for the
        # Uniprot id. If a pdb_id is given we'll try to use that, depending
        # on whether there's a Uniprot xref for it and on strict_unp_xref.
        self.pdb_base_id, self.pdb_chain_id = self._find_pdb_xref(pdb_id)
        self.pdb_id = f"{self.pdb_base_id}:{self.pdb_chain_id}"
        self.pdb_source = pdb_source
        if pdb_dict:
            self._pdb_dict = pdb_dict

        self.pdb_meta = pdb.PDBMetadata(
            self.pdb_id, pdb_source=self.pdb_source, struct_d=self.pdb_dict
        )
        if not self.pdb_meta.resolution and self.pdb_source != PDB_AFLD:
            raise ProteinInitError(f"Unknown resolution for {pdb_id}")

        LOGGER.info(
            f"{self}: {self.pdb_meta.description}, "
            f"org={self.pdb_meta.src_org} ({self.pdb_meta.src_org_id}), "
            f"expr={self.pdb_meta.host_org} ({self.pdb_meta.host_org_id}), "
            f"res={self.pdb_meta.resolution:.2f}Å, "
            f"entity_id={self.pdb_meta.chain_entities[self.pdb_chain_id]}"
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

        # Extract the residues from the PDB structure
        pdb_aa_seq, residues = "", []
        for i, pp in enumerate(self.polypeptides):
            first_res: Residue = pp[0]
            _, curr_start_idx, _ = first_res.get_id()

            # More than one pp means there are gaps due to non-standard AAs
            if i > 0:
                # Calculate index gap between this polypeptide and previous
                prev_pp_last_res: Residue = self.polypeptides[i - 1][-1]
                _, prev_end_idx, _ = prev_pp_last_res.get_id()
                gap_len = curr_start_idx - prev_end_idx - 1

                # fill in the gaps
                pdb_aa_seq += UNKNOWN_AA * gap_len
                residues.extend(
                    [
                        Residue(("", j, ""), UNKNOWN_AA, i)
                        for j in range(prev_end_idx + 1, curr_start_idx)
                    ]
                )

            pdb_aa_seq += str(pp.get_sequence())
            residues.extend(pp)

        assert len(pdb_aa_seq) == len(residues)
        n_residues = len(pdb_aa_seq)

        # Find the alignment between the PDB AA sequence and the Uniprot AA sequence.
        pdb_to_unp_idx = self._find_unp_alignment(pdb_aa_seq, self.unp_rec.sequence)

        # Find the best matching DNA for our AA sequence via pairwise alignment
        # between the PDB AA sequence and translated DNA sequences.
        self.ena_id = None
        self._dna_seq = None
        idx_to_codons = {}
        if with_codons:
            self.ena_id, self._dna_seq, idx_to_codons = self._find_dna_alignment(
                pdb_aa_seq, max_ena
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
                    pdb_dict=self._pdb_dict,
                )

        # Create a ResidueRecord holding all data we need per residue
        residue_recs = []
        for i in range(n_residues):
            r_curr: Residue = residues[i]
            relative_location = (i + 1) / n_residues

            # Sanity check
            assert pdb_aa_seq[i] == ACIDS_3TO1.get(r_curr.get_resname(), UNKNOWN_AA)

            # Get the residues before and after this one
            r_prev: Optional[Residue] = residues[i - 1] if i > 0 else None
            r_next: Optional[Residue] = residues[i + 1] if i < n_residues - 1 else None

            # Alignment to UNP
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

        self._protein_seq = pdb_aa_seq
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
    def protein_seq(self) -> SeqRecord:
        """
        :return: Protein sequence as 1-letter AA names. Based on the
        residues found in the associated PDB structure.
        Note that the sequence might contain the letter 'X' denoting an
        unknown AA. This happens if the PDB entry contains non-standard AAs
        and we chose to ignore such AAs.
        """
        return SeqRecord(Seq(self._protein_seq), self.pdb_id, "", "")

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
            chain = self.pdb_rec[0][self.pdb_chain_id]
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

    def _find_unp_alignment(self, pdb_aa_seq: str, unp_aa_seq: str) -> Dict[int, int]:
        """
        Aligns between this prec's AA sequence (based on the PDB structure) and the
        Uniprot sequence.
        :param pdb_aa_seq: AA sequence from PDB to align.
        :param unp_aa_seq: AA sequence from UNP to align.
        :return: A dict mapping from an index in the PDB sequence to the
            corresponding index in the UNP sequence.
        """
        aligner = PairwiseAligner(
            substitution_matrix=BLOSUM80, open_gap_score=-10, extend_gap_score=-0.5
        )
        multi_alignments = aligner.align(pdb_aa_seq, unp_aa_seq)
        alignment = sorted(multi_alignments, key=lambda a: a.score)[-1]
        LOGGER.info(f"{self}: PDB to UNP sequence alignment score={alignment.score}")

        # Alignment contains two tuples each of length N (for N matching sub-sequences)
        # (
        #   ((t_start1, t_end1), (t_start2, t_end2), ..., (t_startN, t_endN)),
        #   ((q_start1, q_end1), (q_start2, q_end2), ..., (q_startN, q_endN))
        # )
        pdb_to_unp: List[Tuple[int, int]] = []
        pdb_subseqs, unp_subseqs = alignment.aligned
        assert len(pdb_subseqs) == len(unp_subseqs)
        for i in range(len(pdb_subseqs)):
            pdb_start, pdb_end = pdb_subseqs[i]
            unp_start, unp_end = unp_subseqs[i]
            assert pdb_end - pdb_start == unp_end - unp_start

            for j in range(pdb_end - pdb_start):
                if pdb_aa_seq[pdb_start + j] != unp_aa_seq[unp_start + j]:
                    # There are mismatches included in the match sequence (cases
                    # where a similar AA is not considered a complete mismatch).
                    # We are more strict: require exact match.
                    continue
                pdb_to_unp.append((pdb_start + j, unp_start + j))

        return dict(pdb_to_unp)

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
            f"num={len(best_alignments)})\n"
            f"{str(best_alignment).strip()}"
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

    def _find_pdb_xref(self, ref_pdb_id) -> Tuple[str, str]:
        ref_pdb_id, ref_chain_id, ent_id = pdb.split_id_with_entity(ref_pdb_id)
        if not ref_chain_id:
            if ent_id is not None and self.numeric_chain:
                # In rare cases the chain is a number and indistinguishable
                # from entity. Handle this case only if explicitly
                # requested.
                ref_chain_id = ent_id
            else:
                ref_chain_id = ""

        ref_pdb_id, ref_chain_id = ref_pdb_id.upper(), ref_chain_id.upper()

        xrefs = unp.find_pdb_xrefs(self.unp_rec, method="x-ray")

        # We'll sort the PDB entries according to multiple criteria based on
        # the resolution, number of chains and sequence length.
        def sort_key(xref: unp.UNPPDBXRef):
            id_cmp = xref.pdb_id.upper() != ref_pdb_id
            chain_cmp = xref.chain_id.upper() != ref_chain_id
            seq_len_diff = abs(xref.seq_len - self.unp_rec.sequence_length)
            # The sort key for PDB entries
            # First, if we have a matching id to the reference PDB id we take
            # it. Otherwise, we take the best match according to seq len and
            # resolution.
            return id_cmp, chain_cmp, seq_len_diff, xref.resolution

        xrefs = sorted(xrefs, key=sort_key)
        if not xrefs:
            msg = f"No PDB cross-refs for {self.unp_id}"
            if self.strict_unp_xref:
                raise ProteinInitError(msg)
            elif not ref_chain_id:
                raise ProteinInitError(f"{msg} and no chain provided in ref")
            else:
                LOGGER.warning(f"{msg}, using ref {ref_pdb_id}:{ref_chain_id}")
                return ref_pdb_id, ref_chain_id

        # Get best match according to sort key and return its id.
        xref = xrefs[0]
        LOGGER.info(f"{self.unp_id}: PDB XREF = {xref}")

        pdb_id = xref.pdb_id.upper()
        chain_id = xref.chain_id.upper()

        # Make sure we have a match with the Uniprot id. Id chain wasn't
        # specified, match only PDB ID, otherwise, both must match.
        if pdb_id != ref_pdb_id:
            msg = (
                f"Reference PDB ID {ref_pdb_id} not found as "
                f"cross-reference for protein {self.unp_id}"
            )
            if self.strict_unp_xref:
                raise ProteinInitError(msg)
            else:
                LOGGER.warning(msg)
                pdb_id = ref_pdb_id

        if ref_chain_id and chain_id != ref_chain_id:
            msg = (
                f"Reference chain {ref_chain_id} of PDB ID {ref_pdb_id} not"
                f"found as cross-reference for protein {self.unp_id}. "
                f"Did you mean chain {chain_id}?"
            )
            if self.strict_unp_xref:
                raise ProteinInitError(msg)
            else:
                LOGGER.warning(msg)
                chain_id = ref_chain_id

        return pdb_id.upper(), chain_id.upper()

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
        return f"({self.unp_id}, {self.pdb_id})"

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
