from Bio import SwissProt as SwissProt

from pp5 import UNP_DIR
from pp5.utils import remote_dl

import logging

UNP_URL_TEMPLATE = r"https://www.uniprot.org/uniprot/{}.txt"
LOGGER = logging.getLogger(__name__)


def unp_record(unp_id: str, unp_dir=UNP_DIR) -> SwissProt.Record:
    """
    Create a Record object holding the information about a protein based on
    its Uniprot id.
    :param unp_id: The Uniprot id.
    :param unp_dir: Directory to download Uniprot file to.
    :return: A biopython Record object.
    """
    url = UNP_URL_TEMPLATE.format(unp_id)
    filename = unp_dir.joinpath(f'{unp_id}.txt')
    remote_dl(url, filename, skip_existing=True)

    with open(filename, 'r') as local_handle:
        return SwissProt.read(local_handle)
