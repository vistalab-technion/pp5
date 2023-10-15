from __future__ import annotations

import re
import math
import logging
import warnings
from math import cos, sin
from math import degrees as deg
from math import radians as rad
from typing import Any, Dict, List, Type, Tuple, Union, Optional, Sequence
from pathlib import Path
from collections import OrderedDict, defaultdict

import numpy as np
from Bio import PDB as PDB
from Bio.PDB import Structure as PDBRecord
from Bio.PDB import MMCIF2Dict
from Bio.PDB.DSSP import dssp_dict_from_pdb_file
from Bio.PDB.Polypeptide import standard_aa_names
from Bio.PDB.PDBExceptions import PDBConstructionWarning, PDBConstructionException

import pp5
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
        # The alphafold source requires downloading the data based on the uniprot id
        unp_ids = None
        if not chain_id:
            if not entity_id:
                raise ValueError(f"Chain or entity must be specified for {pdb_source=}")

            # Obtain uniprot ids from entity
            entity_chains: dict = PDB2UNP.query_all_uniprot_ids(
                pdb_id, map_from_entities=True
            ).get(entity_id, {})
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
        unp_ids = unp_ids or PDB2UNP.query_all_uniprot_ids(pdb_id).get(chain_id, [])
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
            dssp_dict, keys = dssp_dict_from_pdb_file(str(path), DSSP="mkdssp")
            if len(ws) > 0:
                for w in ws:
                    LOGGER.warning(f"Got DSSP warning for {pdb_id}: " f"{w.message}")
    except Exception as e:
        raise RuntimeError(f"Failed to get secondary structure for {pdb_id}")

    # dssp_dict maps a reisdue id to a tuple containing various things about
    # that residue. We take only the secondary structure info.
    ss_dict = {k: v[1] for k, v in dssp_dict.items()}

    return ss_dict, keys


class PDB2UNP(JSONCacheableMixin, object):
    """
    Maps PDB IDs (in each chain) to one or more Uniprot IDs which correspond
    to that chain, and their locations in the PDB sequence.
    """

    def __init__(self, pdb_id: str, pdb_source: str = PDB_RCSB, struct_d: dict = None):
        """
        Initialize a PDB to Uniprot mapping.
        :param pdb_id: PDB ID, without chain. Chain will be ignored if
        specified.
        :param pdb_source: Source from which to obtain the pdb file.
        :param struct_d: Optional parsed PDB file dict.
        """
        pdb_base_id, _ = split_id(pdb_id)
        struct_d = pdb_dict(pdb_id, pdb_source=pdb_source, struct_d=struct_d)

        # Get all chain Uniprot IDs by querying PDB. This gives us the most
        # up to date IDs, but it doesn't provide alignment info between the
        # PDB structure's sequence and the Uniprot xref sequence.
        # Map is chain -> [unp1, unp2, ...]
        chain_to_unp_ids = self.query_all_uniprot_ids(pdb_id)

        # Get Uniprot cross-refs from the PDB file. Map is
        # chain -> unp -> [ (s1,e1), (s2, e2), ... ]
        chain_to_unp_xrefs = self.parse_all_uniprot_xrefs(pdb_id, pdb_source, struct_d)

        # Make sure all Uniprot IDs we got from the PDB API exist in our
        # final dict and appear in the order we got them from the PDB API.
        # If not, we need to add them. For now we'll add
        # these missing Uniprot IDs with an empty list of ranges.
        # It's possible to query PDB for these ranges, see here:
        # https://www.rcsb.org/pdb/software/rest.do#dasfeatures
        for chain, unp_ids in chain_to_unp_ids.items():
            d = OrderedDict()
            curr_chain_xrefs = chain_to_unp_xrefs.setdefault(chain, d)
            for unp_id in reversed(unp_ids):
                curr_chain_xrefs.setdefault(unp_id, [])
                curr_chain_xrefs.move_to_end(unp_id, last=False)

        self.pdb_id = pdb_base_id
        self.chain_to_unp_xrefs = chain_to_unp_xrefs

    def get_unp_id(self, chain_id: str, strict=True) -> str:
        """
        :param chain_id: A chain in the PDB structure.
        :param strict: Whether to raise an error (True) or just warn (False)
        if the chain cannot be uniquely mapped to a single Uniprot ID.
        :return: the first unp id matching the given chain. Usually there's
        only one unless the entry is chimeric.
        """
        if not chain_id or chain_id.upper() not in self.chain_to_unp_xrefs:
            raise ValueError(f"No Uniprot ID for chain {chain_id} of" f" {self.pdb_id}")

        if self.is_chimeric(chain_id):
            msg = (
                f"{self.pdb_id} is chimeric at chain {chain_id}, "
                f"possible Uniprot IDs: "
                f"{self.get_all_chain_unp_ids(chain_id)}."
            )
            if strict:
                raise ValueError(msg)
            LOGGER.warning(f"{msg} Returning first ID.")

        for unp_id in self.chain_to_unp_xrefs[chain_id.upper()]:
            return unp_id

    def is_chimeric(self, chain_id: str) -> bool:
        """
        :param chain_id: A chain in the PDB structure.
        :return: Whether the sequence in the given chain is chimeric,
        i.e. is composed of regions from different proteins.
        """
        return len(self.chain_to_unp_xrefs[chain_id.upper()]) > 1

    def get_all_chain_unp_ids(self, chain_id) -> tuple:
        """
        :param chain_id: A chain in the PDB structure.
        :return: All unp ids matching the given chain.
        """
        return tuple(self.chain_to_unp_xrefs[chain_id.upper()].keys())

    def get_all_unp_ids(self) -> set:
        """
        :return: All Uniprot IDs for all chains in the PDB structure.
        """
        all_unp_ids = set()
        for chain in self.chain_to_unp_xrefs:
            all_unp_ids.update(self.get_all_chain_unp_ids(chain))
        return all_unp_ids

    def get_chain_to_unp_ids(self) -> Dict[str, Tuple[str]]:
        """
        :return: A mapping from chain it to a sequence of uniprot ids for
        that chain.
        """
        return {c: tuple(u.keys()) for c, u in self.chain_to_unp_xrefs.items()}

    def save(self, out_dir=pp5.PDB2UNP_DIR) -> Path:
        """
        Write the current mapping to a human-readable text file (json) which
        can also be loaded later using from_cache.
        :param out_dir: Output directory.
        :return: The path of the written file.
        """
        filename = f"{self.pdb_id}.json"
        return self.to_cache(out_dir, filename, indent=None)

    def __getitem__(self, chain_id: str):
        """
        :param chain_id: The chain.
        :return: Uniprot xrefs for a given chain
        """
        return self.chain_to_unp_xrefs[chain_id.upper()]

    def __contains__(self, chain_id: str):
        """
        :param chain_id: The chain.
        :return: Whether this mapping contains the given chain.
        """
        return chain_id.upper() in self.chain_to_unp_xrefs

    def __repr__(self):
        return f"PDB2UNP({self.pdb_id})={self.get_chain_to_unp_ids()}"

    @staticmethod
    def query_all_uniprot_ids(
        pdb_id: str, map_from_entities: bool = False
    ) -> Union[Dict[str, List[str]], Dict[str, Dict[str, List[str]]]]:
        """
        Retrieves all Uniprot IDs associated with a PDB structure by querying
        the PDB database. Returns a map from either chain id or entity id to the
        associated unp ids.

        :param pdb_id: The PDB ID to search for. Chain or entity will be ignored.
        :param map_from_entities: Whether to return a map from chain ids (False) or
        from entity ids (True).
        :return: if map_from_entities=False, mapping from each chain id in
        the given structure to a list of associated Uniprot IDs; otherwise a mapping
        from entity id to a mapping from chain to a list of uniprot ids.
        :raises pdb_api.PDBAPIException: If there's a problem obtaining the data.
        """
        map_to_unp_ids = {}

        # Make sure we have a base id
        pdb_id, _, _ = split_id_with_entity(pdb_id)

        # Get all data for the PDB structure
        entry_data = pdb_api.execute_raw_data_query(pdb_id)
        entry_containers = entry_data["rcsb_entry_container_identifiers"]

        # Find all polymer entities
        entity_ids = entry_containers.get("polymer_entity_ids", [])
        for entity_id in entity_ids:
            entity_id = str(entity_id)
            # Get all data about this entity
            entity_data = pdb_api.execute_raw_data_query(pdb_id, entity_id=entity_id)

            # Get list of chains and list of Uniprot IDs for this entity
            entity_containers = entity_data["rcsb_polymer_entity_container_identifiers"]
            entity_chains = entity_containers.get("asym_ids", [])
            entity_unp_ids = entity_containers.get("uniprot_ids", [])

            if map_from_entities:
                map_to_unp_ids[entity_id] = {
                    chain_id: entity_unp_ids for chain_id in entity_chains
                }
            else:
                # Save the mapping from each chain to these uniprot IDs.
                for chain_id in entity_chains:
                    map_to_unp_ids[chain_id] = entity_unp_ids

        return map_to_unp_ids

    @staticmethod
    def parse_all_uniprot_xrefs(
        pdb_id: str, pdb_source: str = PDB_RCSB, struct_d: dict = None
    ) -> Dict[str, Dict[str, List[tuple]]]:
        """
        Parses the Uniprot cross references and sequence ranges from a PDB
        file of a given PDB ID.
        :param pdb_id: The PDB ID.
        :param pdb_source: Source from which to obtain the pdb file.
        :param struct_d: Optional parsed PDB file.
        :return: Uniprot cross-refs from the PDB file. Mapping is
        chain -> unp -> [ (s1,e1), (s2, e2), ... ]
        """

        pdb_base_id, _ = split_id(pdb_id)
        struct_d = pdb_dict(pdb_id, pdb_source=pdb_source, struct_d=struct_d)

        # Go over referenced DBs and take all uniprot IDs
        unp_ids = set()
        if "_struct_ref.db_name" in struct_d:
            for i, db_name in enumerate(struct_d["_struct_ref.db_name"]):
                if db_name.lower() == "unp":
                    unp_ids.add(struct_d["_struct_ref.pdbx_db_accession"][i])

        # Get Uniprot cross-refs from the PDB file. Map is
        # chain -> unp -> [ (s1,e1), (s2, e2), ... ]
        chain_to_unp_xrefs: Dict[str, Dict[str, List[tuple]]] = {}

        if not unp_ids or "_struct_ref_seq.pdbx_db_accession" not in struct_d:
            return chain_to_unp_xrefs

        for i, curr_id in enumerate(struct_d["_struct_ref_seq.pdbx_db_accession"]):
            curr_id = curr_id.upper()

            # In case the xref DB id is not from Uniprot
            if curr_id not in unp_ids:
                continue

            if "_struct_ref_seq.pdbx_strand_id" not in struct_d:
                continue

            curr_chain = struct_d["_struct_ref_seq.pdbx_strand_id"][i].upper()

            d = OrderedDict()
            curr_chain_xrefs = chain_to_unp_xrefs.setdefault(curr_chain, d)
            curr_chain_unp_ranges = curr_chain_xrefs.setdefault(curr_id, [])

            if (
                "_struct_ref_seq.seq_align_beg" not in struct_d
                or "_struct_ref_seq.seq_align_end" not in struct_d
            ):
                continue

            ref_start = int(struct_d["_struct_ref_seq.seq_align_beg"][i])
            ref_end = int(struct_d["_struct_ref_seq.seq_align_end"][i])
            curr_chain_unp_ranges.append((ref_start, ref_end))

        return chain_to_unp_xrefs

    @classmethod
    def pdb_id_to_unp_id(
        cls,
        pdb_id: str,
        pdb_source: str = PDB_RCSB,
        strict=True,
        cache=False,
        struct_d: dict = None,
    ) -> str:
        """
        Given a PDB ID, returns a single Uniprot id for it.
        :param pdb_id: PDB ID, with optional chain. If provided chain will
        be used.
        :param pdb_source: Source from which to obtain the pdb file.
        :param cache: Whether to use cached mapping.
        :param strict: Whether to raise an error (True) or just warn (False)
        if the PDB ID cannot be uniquely mapped to a single Uniprot ID.
        This can happen if: (1) Chain wasn't specified and there are
        different Uniprot IDs for different chains (e.g. 4HHB); (2) Chain was
        specified but there are multiple Uniprot IDs for the chain
        (chimeric entry, e.g. 3SG4:A).
        :param struct_d: Optional parsed PDB file.
        :return: A Uniprot ID.
        """
        pdb_base_id, chain_id = split_id(pdb_id)
        pdb2unp = cls.from_pdb(pdb_id, pdb_source, cache=cache, struct_d=struct_d)

        all_unp_ids = pdb2unp.get_all_unp_ids()
        if not all_unp_ids:
            raise ValueError(f"No Uniprot entries exist for {pdb_base_id}")

        if not chain_id:
            if len(all_unp_ids) > 1:
                msg = (
                    f"Multiple Uniprot IDs exists for {pdb_base_id}, and no "
                    f"chain specified."
                )
                if strict:
                    raise ValueError(msg)
                LOGGER.warning(
                    f"{msg} Returning the first Uniprot ID " f"from the first chain."
                )

            for chain_id, unp_ids in pdb2unp.get_chain_to_unp_ids().items():
                return unp_ids[0]

        return pdb2unp.get_unp_id(chain_id, strict=strict)

    @classmethod
    def from_pdb(
        cls, pdb_id: str, pdb_source: str = PDB_RCSB, cache=False, struct_d: dict = None
    ) -> PDB2UNP:
        """
        Create a PDB2UNP mapping from a given PDB ID.
        :param pdb_id: The PDB ID to map for. Chain will be ignored if present.
        :param pdb_source: Source from which to obtain the pdb file.
        :param cache: Whether to load a cached mapping if available.
        :param struct_d: Optional parsed PDB file.
        :return: A PDB2UNP mapping object.
        """
        pdb_base_id, _ = split_id(pdb_id)

        if cache:
            pdb2unp = cls.from_cache(pdb_base_id)
            if pdb2unp is not None:
                return pdb2unp

        pdb2unp = cls(pdb_id, pdb_source=pdb_source, struct_d=struct_d)
        pdb2unp.save()
        return pdb2unp

    @classmethod
    def from_cache(
        cls, pdb_id, cache_dir: Union[str, Path] = pp5.PDB2UNP_DIR
    ) -> Optional[PDB2UNP]:
        pdb_id, _ = split_id(pdb_id)
        filename = f"{pdb_id}.json"
        return super(PDB2UNP, cls).from_cache(cache_dir, filename)


class PDBMetadata(object):
    """
    Extracts metadata from a PDB structure.
    Helpful metadata fields:
    https://www.rcsb.org/pdb/results/reportField.do
    """

    def __init__(self, pdb_id: str, pdb_source: str = PDB_RCSB, struct_d=None):
        """
        :param pdb_id: The PDB ID of the structure.
        :param struct_d: Optional dict which will be used if given, instead of
        parsing the PDB file.
        :param pdb_source: Source from which to obtain the pdb file.
        """
        pdb_base_id, chain_id = split_id(pdb_id)
        struct_d = pdb_dict(pdb_id, pdb_source=pdb_source, struct_d=struct_d)

        # For alphafold structures, default to zero resolution instead of NaN.
        default_res = 0.0 if pdb_source == PDB_AFLD else None

        def _meta(key: str, convert_to: Type = str, default=None):
            val = struct_d.get(key, None)
            if not val:
                return default
            if isinstance(val, list):
                val = val[0]
            if not val or val == "?":
                return default
            try:
                return convert_to(val)
            except ValueError:
                return default

        title = _meta("_struct.title")
        description = _meta("_entity.pdbx_description")
        deposition_date = _meta("_pdbx_database_status.recvd_initial_deposition_date")

        src_org = _meta("_entity_src_nat.pdbx_organism_scientific")
        if not src_org:
            src_org = _meta("_entity_src_gen.pdbx_gene_src_scientific_name")

        src_org_id = _meta("_entity_src_nat.pdbx_ncbi_taxonomy_id", int)
        if not src_org_id:
            src_org_id = _meta("_entity_src_gen.pdbx_gene_src_ncbi_taxonomy_id", int)

        host_org = _meta("_entity_src_gen.pdbx_host_org_scientific_name")
        host_org_id = _meta("_entity_src_gen.pdbx_host_org_ncbi_taxonomy_id", int)
        resolution = _meta("_refine.ls_d_res_high", float, default=default_res)
        resolution_low = _meta("_refine.ls_d_res_low", float, default=default_res)
        r_free = _meta("_refine.ls_R_factor_R_free", float)
        r_work = _meta("_refine.ls_R_factor_R_work", float)
        space_group = _meta("_symmetry.space_group_name_H-M")

        # Find ligands
        ligands = set()
        for i, chemical_type in enumerate(struct_d["_chem_comp.id"]):
            if chemical_type.lower() == "hoh":
                continue
            if chemical_type not in STANDARD_ACID_NAMES:
                ligands.add(chemical_type)
        ligands = str.join(",", ligands)

        # Crystal growth details
        cg_ph = _meta("_exptl_crystal_grow.pH", float)
        cg_temp = _meta("_exptl_crystal_grow.temp", float)

        # Map each chain to entity id, and entity to 1-letter sequence.
        chain_entities, entity_seq = {}, {}
        for i, entity_id in enumerate(struct_d["_entity_poly.entity_id"]):
            if not struct_d["_entity_poly.type"][i].startswith("polypeptide"):
                continue

            entity_id = int(entity_id)
            chains_str = struct_d["_entity_poly.pdbx_strand_id"][i]
            for chain in chains_str.split(","):
                chain_entities[chain] = entity_id

            seq_str: str = struct_d["_entity_poly.pdbx_seq_one_letter_code_can"][i]
            seq_str = seq_str.replace("\n", "")
            entity_seq[entity_id] = seq_str

        self.pdb_id: str = pdb_base_id
        self.pdb_source: str = pdb_source
        self.title: str = title
        self.description: str = description
        self.deposition_date: str = deposition_date
        self.src_org: str = src_org
        self.src_org_id: int = src_org_id
        self.host_org: str = host_org
        self.host_org_id: int = host_org_id
        self.resolution: float = resolution
        self.resolution_low: float = resolution_low
        self.r_free: float = r_free
        self.r_work: float = r_work
        self.space_group: str = space_group
        self.ligands: str = ligands
        self.cg_ph: float = cg_ph  # crystal growth pH
        self.cg_temp: float = cg_temp  # crystal growth temperature
        # mapping from chain_id to  entity_id
        self.chain_entities: Dict[str, int] = chain_entities
        # mapping from entity_id to sequence
        self.entity_sequence: Dict[int, str] = entity_seq

    def get_chain(self, entity_id: int) -> Optional[str]:
        """
        :param entity_id: An ID of one of the entities in this structure.
        :return: One of the chains from teh structure belonging to this entity id,
        or None if this is not a valid entity if for the given structure.
        """
        chains = [c for c, e in self.chain_entities.items() if e == entity_id]
        if not chains:
            return None
        return sorted(chains)[0]

    def as_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}

    @property
    def entity_chains(self) -> Dict[int, Sequence[str]]:
        """
        :return: Mapping from entity id to a list of chains belonging to that entity.
        """
        entity_chains = defaultdict(list)
        for chain, entity in self.chain_entities.items():
            entity_chains[entity].append(chain)
        return dict(entity_chains)

    def __repr__(self):
        return str(self.as_dict())


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
