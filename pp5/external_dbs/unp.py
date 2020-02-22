from pathlib import Path
from typing import Set, List, Iterable
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
            new_unp_id = replacement_ids(unp_id)[0]
            LOGGER.warning(f"UNP id {unp_id} replaced by {new_unp_id}")
            return unp_download(new_unp_id, unp_dir)


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


def find_ena_xrefs(unp_rec: UNPRecord, molecule_types: Iterable[str]) \
        -> List[str]:
    """
    Find EMBL ENA cross-references to specific molecule types in a Uniprot
    record.
    :param unp_rec: The Uniprot record.
    :param molecule_types: Which types of molecules are allowed for the
    returned references. For example, 'mrna' or 'genomic_dna'.
    :return: A list of ENA ids which can be used to retrieve ENA records.
    """

    ena_ids = []
    cross_refs = unp_rec.cross_references
    molecule_types = {t.lower() for t in molecule_types}

    embl_refs = (x for x in cross_refs if x[0].lower() == 'embl')
    for dbname, id1, id2, comment, molecule_type in embl_refs:
        molecule_type = molecule_type.lower()
        if molecule_type in molecule_types and id2 and len(id2) > 3:
            ena_ids.append(id2)

    return ena_ids
