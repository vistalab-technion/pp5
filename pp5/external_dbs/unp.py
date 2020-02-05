from pathlib import Path
from urllib.parse import urlsplit

import requests
from Bio import SwissProt as SwissProt
from requests import HTTPError

from pp5 import UNP_DIR
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
    filename = unp_dir.joinpath(f'{unp_id}.txt')

    try:
        return remote_dl(url, filename, skip_existing=True)
    except HTTPError as e:
        if e.response.status_code == 300:
            new_unp_id = replacement_ids(unp_id)[0]
            LOGGER.warning(f"UNP id {unp_id} replaced by {new_unp_id}")
            return unp_download(new_unp_id, unp_dir)


def unp_record(unp_id: str, unp_dir=UNP_DIR) -> SwissProt.Record:
    """
    Create a Record object holding the information about a protein based on
    its Uniprot id.
    :param unp_id: The Uniprot id.
    :param unp_dir: Directory to download Uniprot file to.
    :return: A biopython Record object.
    """
    filename = unp_download(unp_id, unp_dir)

    with open(filename, 'r') as local_handle:
        return SwissProt.read(local_handle)
