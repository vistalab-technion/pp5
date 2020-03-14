from pathlib import Path
from typing import Set, List, Iterable, NamedTuple, Union
from urllib.parse import urlsplit

import requests
import Bio.SwissProt
from Bio.SwissProt import Record as UNPRecord
from requests import HTTPError

from pp5 import UNP_DIR, get_resource_path
from pp5.utils import remote_dl

import logging

UNP_URL_TEMPLATE = r"https://www.uniprot.org/uniprot/{}.txt"
UNP_REPLACE_TEMPLATE = r"https://www.uniprot.org/uniprot/?query=replaces:{}" \
                       r"&format=list"
LOGGER = logging.getLogger(__name__)


def replacement_ids(unp_id: str):
    """
    Sometimes a uniprot ID is not valid and there are replacement ids for it.
    This method retrieves them.
    :param unp_id: The id to find a replacement for.
    :return: A list of replacement ids.
    """
    replaces_url = UNP_REPLACE_TEMPLATE.format(unp_id)
    replaces = requests.get(replaces_url)
    replaces.raise_for_status()
    ids = replaces.text.split()
    if not ids:
        raise ValueError(f"UNP id {unp_id} has no replacements")
    return ids


def unp_download(unp_id: str, unp_dir=UNP_DIR) -> Path:
    url = UNP_URL_TEMPLATE.format(unp_id)
    filename = get_resource_path(unp_dir, f'{unp_id}.txt')

    try:
        return remote_dl(url, filename, skip_existing=True)
    except HTTPError as e:
        if e.response.status_code == 300:
            # We got a kind of redirect to a different Uniprot id.
            # We need to query Uniprot for replacement a replacement ID and
            # download that instead.
            new_unp_id = replacement_ids(unp_id)[0]
            LOGGER.warning(f"UNP id {unp_id} replaced by {new_unp_id}")
            return unp_download(new_unp_id, unp_dir)
        else:
            # Other download error, we can't handle this here
            raise e from None


def unp_record(unp_id: str, unp_dir=UNP_DIR) -> UNPRecord:
    """
    Create a Record object holding the information about a protein based on
    its Uniprot id.
    :param unp_id: The Uniprot id.
    :param unp_dir: Directory to download Uniprot file to.
    :return: A biopython Record object.
    """
    filename = unp_download(unp_id, unp_dir)

    try:
        with open(str(filename), 'r') as local_handle:
            return Bio.SwissProt.read(local_handle)
    except ValueError as e:
        raise ValueError(f'Failed to read Uniprot record {unp_id} from file '
                         f'{filename}')


def as_record(unp_id_or_rec: Union[UNPRecord, str]):
    """
    Convert either id or record to a record.
    :param unp_id_or_rec: ID as string or a record object.
    :return: Corresponding record.
    """
    if isinstance(unp_id_or_rec, UNPRecord):
        return unp_id_or_rec
    else:
        return unp_record(unp_id_or_rec)


def find_ena_xrefs(unp: Union[UNPRecord, str], molecule_types: Iterable[str]) \
        -> List[str]:
    """
    Find EMBL ENA cross-references to specific molecule types in a Uniprot
    record.
    :param unp: A Uniprot record or id.
    :param molecule_types: Which types of molecules are allowed for the
    returned references. For example, 'mrna' or 'genomic_dna'.
    :return: A list of ENA ids which can be used to retrieve ENA records.
    """

    unp_rec = as_record(unp)
    ena_ids = []
    cross_refs = unp_rec.cross_references

    if isinstance(molecule_types, str):
        molecule_types = (molecule_types,)
    molecule_types = {t.lower() for t in molecule_types}

    embl_refs = (x for x in cross_refs if x[0].lower() == 'embl')
    for dbname, id1, id2, comment, molecule_type in embl_refs:
        molecule_type = molecule_type.lower()
        if molecule_type in molecule_types and id2 and len(id2) > 3:
            ena_ids.append(id2)

    return ena_ids


class UNPPDBXRef(NamedTuple):
    """
    Represents a PDB cross-ref within a Uniprot record
    """
    pdb_id: str
    chain_id: str
    seq_len: int
    method: str
    resolution: float

    def __repr__(self):
        return f'{self.pdb_id}:{self.chain_id} (res={self.resolution:.2f}Å, ' \
               f'len={self.seq_len})'


def find_pdb_xrefs(unp: Union[UNPRecord, str], method='x-ray') \
        -> List[UNPPDBXRef]:
    """
    Find PDB cross-references with a specific methods type in a Uniprot
    record.
    :param unp: A Uniprot record or id.
    :param method: Currently only 'x-ray' is supported.
    :return:
    """
    unp_rec = as_record(unp)
    cross_refs = unp_rec.cross_references

    # PDB cross refs are ('PDB', id, method, resolution, chains)
    # E.g: ('PDB', '5EWX', 'X-ray', '2.60 A', 'A/B=1-35, A/B=38-164')
    pdb_xrefs = (x for x in cross_refs if x[0].lower() == 'pdb')
    pdb_xrefs = (x for x in pdb_xrefs if x[2].lower() == 'x-ray')

    def split_xref_chains(xref_chains: str):
        # Example xref_chains format
        # A/B/C=1-100,A/B/C=110-121,X/Y/Z=122-200
        # Returns a dict from chain name to it's length in residues
        res = {}
        for chain_str in xref_chains.split(','):
            chain_names, chain_seqs = chain_str.split('=')
            seq_start, seq_end = chain_seqs.split('-')
            for chain_name in chain_names.split('/'):
                chain_name = chain_name.strip()
                res.setdefault(chain_name, 0)
                res[chain_name] += int(seq_end) - int(seq_start)
        return res

    res = []
    for _, pdb_id, method, resolution, chains_str in pdb_xrefs:
        resolution = float(resolution.split()[0])
        chains = split_xref_chains(chains_str)
        for chain, seq_len in chains.items():
            xref = UNPPDBXRef(pdb_id, chain, seq_len, method, resolution)
            res.append(xref)

    return res
