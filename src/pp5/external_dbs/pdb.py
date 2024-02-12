from __future__ import annotations

import re
import math
import logging
import warnings
from math import cos, sin
from math import degrees as deg
from math import radians as rad
from typing import Any, Set, Dict, List, Tuple, TypeVar, Callable, Optional, Sequence
from pathlib import Path
from datetime import datetime
from itertools import zip_longest
from collections import defaultdict

import numpy as np
from Bio import PDB as PDB
from Bio.PDB import Structure as PDBRecord
from Bio.PDB import MMCIF2Dict
from Bio.PDB.DSSP import dssp_dict_from_pdb_file
from Bio.PDB.Polypeptide import standard_aa_names
from Bio.PDB.PDBExceptions import PDBConstructionWarning, PDBConstructionException

from pp5 import PDB_DIR, get_resource_path
from pp5.utils import JSONCacheableMixin, remote_dl
from pp5.external_dbs import pdb_api

PDB_ID_PATTERN = re.compile(
    r"^(?P<id>[0-9][\w]{3})(?::(?:" r"(?P<chain>[a-z]{1,3})|(?P<entity>[0-9]{1,3})))?$",
    re.IGNORECASE | re.ASCII,
)

STANDARD_ACID_NAMES = set(standard_aa_names)

PDB_RCSB = "rc"
PDB_RCSB_DOWNLOAD_URL_TEMPLATE = r"https://files.rcsb.org/download/{pdb_id}.cif.gz"
PDB_REDO = "re"
PDB_REDO_DOWNLOAD_URL_TEMPLATE = "https://pdb-redo.eu/db/{pdb_id}/{pdb_id}_final.cif"
PDB_AFLD = "af"
PDB_AFLD_DOWNLOAD_URL_TEMPLATE = (
    "https://alphafold.ebi.ac.uk/files/AF-{unp_id}-F1-model_v4.cif"
)

PDB_DOWNLOAD_SOURCES: Dict[str, str] = {
    PDB_RCSB: PDB_RCSB_DOWNLOAD_URL_TEMPLATE,
    PDB_REDO: PDB_REDO_DOWNLOAD_URL_TEMPLATE,
    PDB_AFLD: PDB_AFLD_DOWNLOAD_URL_TEMPLATE,
}

PDB_MMCIF_ENTRY_ID = "_entry.id"
PDB_MMCIF_PDB_SOURCE = "_pdb_source"
ALPHAFOLD_ID_PREFIX = "AF"

LOGGER = logging.getLogger(__name__)


def split_id(pdb_id) -> Tuple[str, str]:
    """
    Splits and validates a full PDB id consisting of a base id and
    optionally also a chain, into a tuple with the base id and chain.
    Will raise an exception if the given id is not a valid PDB id.
    :param pdb_id: PDB id, either without a chain, e.g. '5JDT' or with a
    chain, e.g. '5JDT:A'.
    :return: A tuple (id, chain) where id is the base id and chain can be
    None. The returned strings will be upper-cased.
    """
    return split_id_with_entity(pdb_id)[0:2]


def split_id_with_entity(pdb_id) -> Tuple[str, str, Optional[str]]:
    """
    Splits and validates a full PDB id consisting of a base id and
    optionally also a chain OR entity id, into a tuple with the base id,
    chain and entity.
    Will raise an exception if the given id is not a valid PDB id.
    :param pdb_id: PDB id, either without a chain, e.g. '5JDT', with a
    chain, e.g. '5JDT:A' or with an entity id, e.g. '5JDT:1'.
    :return: A tuple (id, chain, entity) where id is the base id and chain
    or entity can be None (only one of them will not be None). The returned
    strings will be upper-cased.
    """
    pdb_id = pdb_id.upper()
    match = PDB_ID_PATTERN.match(pdb_id)
    if not match:
        raise ValueError(f"Invalid PDB id format: {pdb_id}")

    return match.group("id"), match.group("chain"), match.group("entity")


def pdb_download(pdb_id: str, pdb_dir=PDB_DIR, pdb_source: str = PDB_RCSB) -> Path:
    """
    Downloads a protein structure file from PDB.
    :param pdb_id: The id of the structure to download.
    :param pdb_dir: Directory to download PDB file to.
    :param pdb_source: Source from which to obtain the pdb file.
    """
    # Separate id and chain if chain was specified
    pdb_id, chain_id, entity_id = split_id_with_entity(pdb_id)

    if pdb_source not in PDB_DOWNLOAD_SOURCES:
        raise ValueError(
            f"Unknown {pdb_source=}, must be one of {tuple(PDB_DOWNLOAD_SOURCES)}"
        )

    download_url_template = PDB_DOWNLOAD_SOURCES[pdb_source]
    if "unp_id" in download_url_template:
        pdb_meta = PDBMetadata.from_pdb(pdb_id, cache=True)

        # The alphafold source requires downloading the data based on the uniprot id
        unp_ids = None
        if not chain_id:
            if not entity_id:
                raise ValueError(f"Chain or entity must be specified for {pdb_source=}")

            # Obtain uniprot ids from entity (entity -> chain -> unp ids)
            entity_chains: dict = pdb_meta.entity_uniprot_ids.get(entity_id, {})
            if not entity_chains:
                raise ValueError(f"Failed to obtain chain for {pdb_id}:{entity_id}")
            chain_id = [*entity_chains.keys()][0]  # arbitrary chain from the entity
            unp_ids = entity_chains[chain_id]

        filename = get_resource_path(
            pdb_dir, f"{pdb_id}_{chain_id}-{pdb_source}.cif".lower()
        )
        if filename.is_file():  # to prevent unnecessary API call if the file exists
            return filename

        # Get uniprot id for this chain (only if we didn't get them from entity)
        unp_ids = unp_ids or pdb_meta.chain_uniprot_ids.get(chain_id, [])
        if len(unp_ids) != 1:
            raise ValueError(
                f"Can't determine unique uniprot id for {pdb_id}:{chain_id}, "
                f"got {unp_ids=}"
            )
        unp_id = unp_ids[0]
        download_url = download_url_template.format(unp_id=unp_id)

    else:
        filename = get_resource_path(pdb_dir, f"{pdb_id}-{pdb_source}.cif".lower())
        download_url = download_url_template.format(pdb_id=pdb_id).lower()

    uncompress = download_url.endswith(("gz", "gzip", "zip"))
    return remote_dl(
        download_url, filename, uncompress=uncompress, skip_existing=True, retries=1
    )


def pdb_struct(
    pdb_id: str, pdb_dir=PDB_DIR, pdb_source: str = PDB_RCSB, struct_d=None
) -> PDBRecord:
    """
    Given a PDB structure id, returns an object representing the protein
    structure.
    :param pdb_id: The PDB id of the structure.
    :param pdb_dir: Directory to download PDB file to.
    :param pdb_source: Source from which to obtain the pdb file.
    :param struct_d: Optional dict containing the parsed structure as a
    dict. If provided, the file wont have to be re-parsed.
    :return: An biopython PDB Structure object.
    """
    pdb_base_id, chain_id = split_id(pdb_id)
    filename = pdb_download(pdb_id, pdb_dir=pdb_dir, pdb_source=pdb_source)

    # Parse the PDB file into a Structure object
    LOGGER.info(f"Parsing struct from PDB file {filename}...")
    parser = CustomMMCIFParser()
    return parser.get_structure(pdb_base_id, filename, mmcif_dict=struct_d)


def pdb_dict(
    pdb_id: str, pdb_dir=PDB_DIR, pdb_source: str = PDB_RCSB, struct_d=None
) -> dict:
    """
    Returns a dictionary containing all the contents of a PDB mmCIF file.
    :param pdb_id: The PDB id of the structure.
    :param pdb_dir: Directory to download PDB file to.
    :param pdb_source: Source from which to obtain the pdb file.
    :param struct_d: Optional dict of the structure if it was already loaded.
    This parameter exists to streamline other functions that use this one in
    case the file was already parsed.
    """
    pdb_base_id, chain_id = split_id(pdb_id)

    # No need to re-parse the file if we have a matching struct dict
    id_from_struct_d = struct_d[PDB_MMCIF_ENTRY_ID][0].upper() if struct_d else None
    source_from_struct_d = struct_d[PDB_MMCIF_PDB_SOURCE] if struct_d else None
    if (
        id_from_struct_d
        and id_from_struct_d == pdb_base_id
        and source_from_struct_d == pdb_source
    ):
        return struct_d

    filename = pdb_download(pdb_id, pdb_dir=pdb_dir, pdb_source=pdb_source)

    LOGGER.info(f"Parsing dict from PDB file {filename}...")
    struct_d = MMCIF2Dict.MMCIF2Dict(filename)

    # For alphafold structures, the id will be a uniprot id; add the pdb id.
    id_from_struct_d = struct_d[PDB_MMCIF_ENTRY_ID][0].upper()
    if id_from_struct_d.startswith(ALPHAFOLD_ID_PREFIX):
        struct_d[PDB_MMCIF_ENTRY_ID].insert(0, pdb_base_id)

    # Save the source from which this data was obtained
    struct_d[PDB_MMCIF_PDB_SOURCE] = pdb_source

    return struct_d


def pdb_to_secondary_structure(
    pdb_id: str, pdb_source: str = PDB_RCSB, pdb_dir=PDB_DIR
):
    """
    Uses DSSP to determine secondary structure for a PDB record.
    The DSSP codes for secondary structure used here are:
     H        Alpha helix (4-12)
     B        Isolated beta-bridge residue
     E        Strand
     G        3-10 helix
     I        Pi helix
     T        Turn
     S        Bend
     -        None
    :param pdb_id: The PDB id of the structure.
    :param pdb_source: Source from which to obtain the pdb file.
    :param pdb_dir: Directory to download PDB file to.
    :return: A tuple of
        ss_dict: maps from a residue id to the 1-char string denoting the
        type of secondary structure at that residue.
        keys: The residue ids.
    """
    path = pdb_download(pdb_id, pdb_dir, pdb_source)

    try:
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter("ignore")
            dssp_dict, keys = dssp_dict_from_pdb_file(
                str(path), DSSP="mkdssp", dssp_version="4.0.0"
            )
            if len(ws) > 0:
                for w in ws:
                    LOGGER.warning(f"Got DSSP warning for {pdb_id}: " f"{w.message}")
    except Exception as e:
        raise RuntimeError(f"Failed to get secondary structure for {pdb_id}")

    # dssp_dict maps a reisdue id to a tuple containing various things about
    # that residue. We take only the secondary structure info.
    ss_dict = {k: v[1] for k, v in dssp_dict.items()}

    return ss_dict, keys


_TC = TypeVar("_TC")


class PDBMetadata(JSONCacheableMixin):
    """
    Obtains and parses metadata from a PDB structure using PDB REST API.
    """

    def __init__(self, pdb_id: str):
        """
        :param pdb_id: The PDB ID of the structure. No chain.
        """

        self._pdb_id, _ = split_id(pdb_id)

        # Obtain structure-level metadata from the PDB API
        self._meta_struct: dict = pdb_api.execute_raw_data_query(self.pdb_id)
        self._meta_entities: Dict[str, dict] = {}
        self._meta_chains: Dict[str, dict] = {}
        entity_ids = self._meta_struct["rcsb_entry_container_identifiers"][
            "polymer_entity_ids"
        ]
        for entity_id in entity_ids:
            entity_id = str(entity_id)
            # Obtain entity-level metadata from the PDB API
            self._meta_entities[entity_id] = pdb_api.execute_raw_data_query(
                self.pdb_id, entity_id=entity_id
            )

            chain_ids = self._meta_entities[entity_id][
                "rcsb_polymer_entity_container_identifiers"
            ]["asym_ids"]
            for chain_id in chain_ids:
                # Obtain chain-level metadata from the PDB API
                self._meta_chains[chain_id] = pdb_api.execute_raw_data_query(
                    self.pdb_id, chain_id=chain_id
                )

    @staticmethod
    def _resolve(
        meta: dict, key: str, coerce_type: Callable[[Any], _TC]
    ) -> Optional[_TC]:
        for subkey in key.split("."):
            if isinstance(meta, (list, tuple)):
                subkey = int(subkey)
            elif not isinstance(meta, dict):
                raise ValueError(f"Can't resolve {key} in {meta}")
            elif subkey not in meta:
                return None

            meta = meta[subkey]

        if meta is not None:
            try:
                meta = coerce_type(meta)
            except ValueError:
                LOGGER.warning(f"Failed to coerce {meta}@{key} to {coerce_type}")

        return meta

    @property
    def pdb_id(self) -> str:
        return self._pdb_id

    @property
    def title(self) -> Optional[str]:
        return self._resolve(self._meta_struct, "struct.title", str)

    @property
    def description(self) -> Optional[str]:
        return self._resolve(self._meta_struct, "struct.pdbx_descriptor", str)

    @property
    def entity_description(self) -> Dict[str, Optional[str]]:
        return {
            entity_id: self._resolve(
                meta_entity, "rcsb_polymer_entity.pdbx_description", str
            )
            for entity_id, meta_entity in self._meta_entities.items()
        }

    @property
    def deposition_date(self) -> Optional[datetime]:
        return self._resolve(
            self._meta_struct,
            "pdbx_database_status.recvd_initial_deposition_date",
            datetime.fromisoformat,
        )

    @property
    def entity_source_org(self) -> Dict[str, Optional[str]]:
        return {
            entity_id: self._resolve(
                meta_entity, "rcsb_entity_source_organism.0.ncbi_scientific_name", str
            )
            or self._resolve(
                meta_entity, "entity_src_nat.0.pdbx_organism_scientific", str
            )
            or self._resolve(
                meta_entity, "entity_src_gen.0.pdbx_gene_src_scientific_name", str
            )
            for entity_id, meta_entity in self._meta_entities.items()
        }

    @property
    def entity_source_org_id(self) -> Dict[str, Optional[int]]:
        return {
            entity_id: self._resolve(
                meta_entity, "rcsb_entity_source_organism.0.ncbi_taxonomy_id", int
            )
            or self._resolve(meta_entity, "entity_src_nat.0.pdbx_ncbi_taxonomy_id", int)
            or self._resolve(
                meta_entity, "entity_src_gen.0.pdbx_gene_src_ncbi_taxonomy_id", int
            )
            for entity_id, meta_entity in self._meta_entities.items()
        }

    @property
    def entity_host_org(self) -> Dict[str, Optional[str]]:
        return {
            entity_id: self._resolve(
                meta_entity, "rcsb_entity_host_organism.0.ncbi_scientific_name", str
            )
            or self._resolve(
                meta_entity, "entity_src_gen.0.pdbx_host_org_scientific_name", str
            )
            for entity_id, meta_entity in self._meta_entities.items()
        }

    @property
    def entity_host_org_id(self) -> Dict[str, Optional[int]]:
        return {
            entity_id: self._resolve(
                meta_entity, "rcsb_entity_host_organism.0.ncbi_taxonomy_id", int
            )
            or self._resolve(
                meta_entity, "entity_src_gen.0.pdbx_host_org_ncbi_taxonomy_id", int
            )
            for entity_id, meta_entity in self._meta_entities.items()
        }

    @property
    def resolution(self) -> Optional[float]:
        return self._resolve(
            self._meta_struct, "rcsb_entry_info.diffrn_resolution_high.value", float
        ) or self._resolve(self._meta_struct, "reflns.0.d_resolution_high", float)

    @property
    def resolution_low(self) -> Optional[float]:
        return self._resolve(self._meta_struct, "reflns.0.d_resolution_low", float)

    @property
    def r_free(self) -> Optional[float]:
        return self._resolve(self._meta_struct, "refine.0.ls_rfactor_rfree", float)

    @property
    def r_work(self) -> Optional[float]:
        return self._resolve(self._meta_struct, "refine.0.ls_rfactor_rwork", float)

    @property
    def space_group(self) -> Optional[str]:
        return self._resolve(
            self._meta_struct, "symmetry.space_group_name_hm", str
        ) or self._resolve(self._meta_struct, "symmetry.space_group_name_H_M", str)

    @property
    def cg_ph(self) -> Optional[float]:
        return self._resolve(self._meta_struct, "exptl_crystal_grow.0.pH", float)

    @property
    def cg_temp(self) -> Optional[float]:
        return self._resolve(self._meta_struct, "exptl_crystal_grow.0.temp", float)

    @property
    def chain_ligands(self) -> Dict[str, Set[str]]:
        return {
            chain_id: set(
                [
                    ld.get("ligand_comp_id")
                    for ld in meta_chain.get("rcsb_ligand_neighbors", [])
                ]
            )
            for chain_id, meta_chain in self._meta_chains.items()
        }

    @property
    def ligands(self) -> str:
        return str.join(",", sorted(set.union(*self.chain_ligands.values())))

    @property
    def entity_ids(self) -> Sequence[str]:
        """
        :return: The entity ids which exist in the structure.
        """
        return tuple(self._meta_entities.keys())

    @property
    def chain_ids(self) -> Sequence[str]:
        """
        :return: The chain ids which exist in the structure.
        """
        return tuple(self._meta_chains.keys())

    @property
    def auth_chain_ids(self) -> Sequence[str]:
        """
        :return: The chain ids which exist in the structure.
        """
        return tuple(self.chain_to_auth_chain[chain_id] for chain_id in self.chain_ids)

    @property
    def entity_chains(self) -> Dict[str, Sequence[str]]:
        """
        :return: Mapping from entity id to a list of chains belonging to that entity.
        """
        return self._entity_chains(author=False)

    @property
    def entity_auth_chains(self) -> Dict[str, Sequence[str]]:
        """
        :return: Mapping from entity id to a list of chains belonging to that entity,
        using the original author's chain ids.
        """
        return self._entity_chains(author=True)

    def _entity_chains(self, author: bool = False) -> Dict[str, Sequence[str]]:
        """
        :param author: Whether to use author or canonical chain ids.
        :return: Mapping from entity id to a list of chains belonging to that entity.
        """
        asym_ids_key = "auth_asym_ids" if author else "asym_ids"
        key = f"rcsb_polymer_entity_container_identifiers.{asym_ids_key}"
        return {
            entity_id: self._resolve(meta_entity, key, tuple)
            for entity_id, meta_entity in self._meta_entities.items()
        }

    @property
    def chain_entities(self) -> Dict[str, str]:
        """
        :return: Mapping from chain id to its entity id.
        """
        chain_to_entity = {}
        for entity_id, chain_ids in self.entity_chains.items():
            chain_to_entity = {
                **chain_to_entity,
                **{chain_id: entity_id for chain_id in chain_ids},
            }
        return chain_to_entity

    @property
    def chain_to_auth_chain(self) -> Dict[str, str]:
        """
        :return: Mapping from PDB chain id to its author's chain id. If there are no
        different names for the author chains, the PDB chain names are mapped to
        themselves.
        """
        entity_auth_chains = self.entity_auth_chains
        chain_to_auth_chain = {}
        for entity_id, chain_ids in self.entity_chains.items():
            auth_chain_ids = entity_auth_chains[entity_id]
            chain_to_auth_chain = {
                **chain_to_auth_chain,
                **{
                    chain_id: auth_chain_id or chain_id
                    for chain_id, auth_chain_id in zip_longest(
                        chain_ids, auth_chain_ids
                    )
                },
            }
        return chain_to_auth_chain

    @property
    def entity_sequence(self) -> Dict[str, str]:
        return {
            entity_id: self._resolve(
                meta_entity, "entity_poly.pdbx_seq_one_letter_code_can", str
            )
            for entity_id, meta_entity in self._meta_entities.items()
        }

    @property
    def uniprot_ids(self) -> Sequence[str]:
        """
        :return: All Uniprot IDs associated with the PDB structure.
        """
        all_unp_ids = set()
        for chain_id, unp_ids in self.chain_uniprot_ids.items():
            all_unp_ids.update(unp_ids)
        return tuple(sorted(all_unp_ids))

    @property
    def chain_uniprot_ids(self) -> Dict[str, Sequence[str]]:
        """
        Retrieves all Uniprot IDs associated with a PDB structure chains.

        :return: a map: chain -> [unp1, unp2, ...]
        where unp1, unp2, ... are Uniprot IDs associated with the chain.
        """

        # entity -> chain -> unp -> [ (s1,e1), ... ]
        entity_map = self.entity_uniprot_id_alignments

        all_chain_map = {}
        for entity_id, chain_map in entity_map.items():
            for chain_id, unp_map in chain_map.items():
                # chain -> [unp1, unp2, ...]
                all_chain_map[chain_id] = tuple(unp_map.keys())

        return all_chain_map

    @property
    def chain_uniprot_id_alignments(
        self,
    ) -> Dict[str, Dict[str, List[Tuple[int, int]]]]:
        """
        Retrieves all Uniprot IDs associated with a PDB structure chains.

        :return: a map: chain -> unp -> [ (s1,e1), ... ]
        where (s1,e1) are alignment start,end indices between the UNP and PDB sequences.
        """
        # entity -> chain -> unp -> [ (s1,e1), ... ]
        entity_map = self.entity_uniprot_id_alignments

        all_chain_map = {}
        for entity_id, chain_map in entity_map.items():
            for chain_id, unp_map in chain_map.items():
                # chain -> unp -> [ (s1,e1), ... ]
                all_chain_map[chain_id] = unp_map

        return all_chain_map

    @property
    def entity_uniprot_ids(self) -> Dict[str, Dict[str, Sequence[str]]]:
        """
        Retrieves all Uniprot IDs associated with a PDB structure entities.

        :return: a map: entity -> chain ->[unp1, unp2, ...]
        where unp1, unp2, ... are Uniprot IDs associated with the entity.
        """
        # entity -> chain -> unp -> [ (s1,e1), ... ]
        entity_map = self.entity_uniprot_id_alignments

        new_entity_map = defaultdict(dict)
        for entity_id, chain_map in entity_map.items():
            for chain_id, unp_map in chain_map.items():
                # entity -> chain -> [unp1, unp2, ...]
                new_entity_map[entity_id][chain_id] = tuple(unp_map.keys())

        return dict(new_entity_map)

    @property
    def entity_uniprot_id_alignments(
        self,
    ) -> Dict[str, Dict[str, Dict[str, List[Tuple[int, int]]]]]:
        """
        Retrieves all Uniprot IDs associated with a PDB structure entities.

        :return: a map: entity -> chain -> unp -> [ (s1,e1), ...]
        where (s1,e1) are alignment start,end indices between the UNP and PDB sequences.
        """
        map_to_unp_ids = {}

        for entity_id, entity_meta in self._meta_entities.items():
            # Get list of chains and list of Uniprot IDs for this entity
            entity_containers = entity_meta["rcsb_polymer_entity_container_identifiers"]
            entity_unp_ids = entity_containers.get("uniprot_ids", [])

            unp_alignments: Dict[str, List[Tuple[int, int]]] = {
                unp_id: [] for unp_id in entity_unp_ids
            }
            for alignment_entry in entity_meta.get("rcsb_polymer_entity_align", []):
                if alignment_entry["reference_database_name"].lower() != "uniprot":
                    continue

                unp_id = alignment_entry["reference_database_accession"]
                if unp_id not in unp_alignments:
                    continue

                for alignment_region in alignment_entry["aligned_regions"]:
                    align_start = alignment_region["entity_beg_seq_id"]
                    align_end = align_start + alignment_region["length"] - 1
                    unp_alignments[unp_id].append((align_start, align_end))

            entity_chains = [
                # The same chain can be referred to by different labels,
                # the canonical PDB label and another label given by the
                # structure author.
                *entity_containers.get("asym_ids", []),
                *entity_containers.get("auth_asym_ids", []),
            ]

            map_to_unp_ids[entity_id] = {
                chain_id: unp_alignments for chain_id in entity_chains
            }

        return map_to_unp_ids

    def as_dict(self) -> Dict[str, Any]:
        return {
            k: getattr(self, k)
            for k, v in self.__class__.__dict__.items()
            if isinstance(v, property)
        }

    def __repr__(self):
        return str(self.as_dict())

    @classmethod
    def from_pdb(cls, pdb_id: str, cache=False) -> PDBMetadata:
        """
        Create a PDBMetadata object from a given PDB ID.
        :param pdb_id: The PDB ID to map for. Chain will be ignored if present.
        :param cache: Whether to load a cached mapping if available.
        :return: A PDBMetadata object.
        """
        pdb_base_id, _ = split_id(pdb_id)

        # TODO: Implement caching
        # if cache:
        #     pdb_meta = cls.from_cache(pdb_base_id)
        #     if pdb_meta is not None:
        #         return pdb_meta

        pdb_meta = cls(pdb_id)
        # pdb_meta.save()
        return pdb_meta

    @classmethod
    def pdb_id_to_unp_id(cls, pdb_id: str, strict=True, cache=False) -> str:
        """
        Given a PDB ID, returns a single Uniprot id for it.
        :param pdb_id: PDB ID, with optional chain.
        :param cache: Whether to use cached mapping.
        :param strict: Whether to raise an error (True) or just warn (False)
        if the PDB ID cannot be uniquely mapped to a single Uniprot ID.
        This can happen if: (1) Chain wasn't specified and there are
        different Uniprot IDs for different chains (e.g. 4HHB); (2) Chain was
        specified but there are multiple Uniprot IDs for the chain
        (chimeric entry, e.g. 3SG4:A).
        :return: A Uniprot ID.
        """
        pdb_base_id, chain_id = split_id(pdb_id)
        meta = cls.from_pdb(pdb_id, cache=cache)

        if not meta.uniprot_ids:
            raise ValueError(f"No Uniprot entries exist for {pdb_base_id}")

        if not chain_id:
            if len(meta.uniprot_ids) > 1:
                msg = f"Multiple Uniprot IDs for {pdb_base_id}, no chain specified."
                if strict:
                    raise ValueError(msg)
                LOGGER.warning(f"{msg} Returning first ID from the first chain.")

            for chain_id, unp_ids in meta.chain_uniprot_ids.items():
                return unp_ids[0]

        if chain_id not in meta.chain_uniprot_ids:
            raise ValueError(f"No Uniprot ID for chain {chain_id} of {pdb_base_id}")

        if len(meta.chain_uniprot_ids[chain_id]) > 1:
            msg = f"Multiple Uniprot IDs for {pdb_base_id} chain {chain_id} (chimeric)"
            if strict:
                raise ValueError(msg)
            LOGGER.warning(f"{msg} Returning the first Uniprot ID.")

        return meta.chain_uniprot_ids[chain_id][0]


class PDBUnitCell(object):
    """
    Represent a Unit Cell of a specific pdb structure.

    Trueblood, K. N. et al. Atomic displacement parameter nomenclature
    report of a subcommittee on atomic displacement parameter nomenclature.
    Acta Crystallogr. Sect. A Found. Crystallogr. 52, 770–781 (1996).

    Grosse-Kunstleve, R. W. & Adams, P. D. On the handling of atomic
    anisotropic displacement parameters. J. Appl. Crystallogr. 35, 477–480
    (2002).

    """

    def __init__(self, pdb_id, struct_d: dict = None):
        """
        :param struct_d: Optional dict containing the parsed structure.
        """
        d = pdb_dict(pdb_id, struct_d=struct_d)
        try:
            a, b = d["_cell.length_a"], d["_cell.length_b"]
            c = d["_cell.length_c"]
            alpha, beta = d["_cell.angle_alpha"], d["_cell.angle_beta"]
            gamma = d["_cell.angle_gamma"]
            a, b, c = float(a[0]), float(b[0]), float(c[0])
            alpha, beta = float(alpha[0]), float(beta[0])
            gamma = float(gamma[0])
        except KeyError:
            raise ValueError(f"Can't create UnitCell for {pdb_id}")

        self.pdb_id = pdb_id

        # a, b, c: Unit cell lengths in Angstroms
        # alpha, beta, gamma: Unit-cell angles in degrees
        self.a, self.b, self.c = a, b, c
        self.alpha, self.beta, self.gamma = alpha, beta, gamma

        cos_alpha, sin_alpha = cos(rad(self.alpha)), sin(rad(self.alpha))
        cos_beta, sin_beta = cos(rad(self.beta)), sin(rad(self.beta))
        cos_gamma, sin_gamma = cos(rad(self.gamma)), sin(rad(self.gamma))

        # Volume
        factor = math.sqrt(
            1
            - cos_alpha**2
            - cos_beta**2
            - cos_gamma**2
            + 2 * cos_alpha * cos_beta * cos_gamma
        )
        self.vol = self.a * self.b * self.c * factor

        # Reciprocal lengths
        self.a_r = self.b * self.c * sin_alpha / self.vol
        self.b_r = self.c * self.a * sin_beta / self.vol
        self.c_r = self.a * self.b * sin_gamma / self.vol

        # Reciprocal angles
        cos_alpha_r = (cos_beta * cos_gamma - cos_alpha) / (sin_beta * sin_gamma)
        cos_beta_r = (cos_gamma * cos_alpha - cos_beta) / (sin_gamma * sin_alpha)
        cos_gamma_r = (cos_alpha * cos_beta - cos_gamma) / (sin_alpha * sin_beta)
        self.alpha_r = deg(math.acos(cos_alpha_r))
        self.beta_r = deg(math.acos(cos_beta_r))
        self.gamma_r = deg(math.acos(cos_gamma_r))

        # Types of coordinate systems:
        # Fractional: no units
        # Cartesian: length
        # Direct lattice: length
        # Reciprocal lattice: 1/length

        # A: Transformation from fractional to Cartesian coordinates
        self.A = np.array(
            [
                [self.a, self.b * cos_gamma, self.c * cos_beta],
                [0, self.b * sin_gamma, -self.c * sin_beta * cos_alpha_r],
                [0, 0, 1 / self.c_r],
            ],
            dtype=np.float32,
        )

        # A^-1: Transformation from Cartesian to fractional coordinates
        self.Ainv = np.linalg.inv(self.A)

        # B: Transformation matrix from direct lattice coordinates to cartesian
        self.N = np.diag([self.a_r, self.b_r, self.c_r]).astype(np.float32)
        self.B = np.dot(self.A, self.N)

        # B^-1: Transformation matrix from Cartesian to direct lattice
        self.Binv = np.linalg.inv(self.B)

        # Fix precision issues
        [
            np.round(a, decimals=15, out=a)
            for a in (self.A, self.Ainv, self.B, self.Binv)
        ]

    def direct_lattice_to_cartesian(self, x: np.ndarray):
        assert 0 < x.ndim < 3
        if x.ndim == 1:
            return np.dot(self.B, x)
        elif x.ndim == 2:
            return np.dot(self.B, np.dot(x, self.B.T))

    def __repr__(self):
        abc = f"(a={self.a:.1f},b={self.b:.1f},c={self.c:.1f})"
        ang = f"(α={self.alpha:.1f},β={self.beta:.1f},γ={self.gamma:.1f})"
        return f"[{self.pdb_id}]{abc}{ang}"


class CustomMMCIFParser(PDB.MMCIFParser):
    """
    Override biopython's parser so that it accepts a structure dict,
    to prevent re-parsing in case it was already parsed.
    """

    def __init__(self, **kw):
        super().__init__(**kw)

    def get_structure(self, structure_id, filename, mmcif_dict=None):
        """Return the structure.

        Arguments:
         - structure_id - string, the id that will be used for the structure
         - filename - name of mmCIF file, OR an open text mode file handle

        """
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=PDBConstructionWarning)

            if not mmcif_dict:
                self._mmcif_dict = MMCIF2Dict.MMCIF2Dict(filename)
            else:
                self._mmcif_dict = mmcif_dict
                id_from_struct_d = self._mmcif_dict[PDB_MMCIF_ENTRY_ID][0]
                if not id_from_struct_d.lower() == structure_id.lower():
                    raise PDBConstructionException(
                        "PDB ID mismatch between provided struct dict and "
                        "desired structure id"
                    )

            self._build_structure(structure_id)
            self._structure_builder.set_header(self._get_header())

        return self._structure_builder.get_structure()


def pdb_tagged_filepath(
    pdb_id: str, pdb_source: str, out_dir: Path, suffix: str, tag: str = None
) -> Path:
    """
    Creates a file path for a PDB record, with an optional tag.
    Eg. pdb_id=1ABC:A, source=src, tag='foo' -> /path/to/out_dir/1ABC_A-src-foo.ext
    :param pdb_id: The PDB id, can include chain. Colon will be replaced with
    underscore.
    :param pdb_source: Source from which to obtain the pdb file.
    :param out_dir: Output directory.
    :param suffix: File suffix (extension).
    :param tag: Optional tag to add.
    :return: The path to the output file.
    """
    tag = f"-{tag}" if tag else ""
    filename = f'{pdb_id.replace(":", "_").upper()}-{pdb_source}{tag}'
    filepath = out_dir.joinpath(f"{filename}.{suffix}")
    return filepath
