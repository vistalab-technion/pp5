from pathlib import Path

from Bio import SeqIO as SeqIO
from Bio.SeqRecord import SeqRecord
from requests import RequestException

from pp5 import ENA_DIR
from pp5.utils import remote_dl

import logging

ENA_URL_TEMPLATE = r"https://www.ebi.ac.uk/ena/browser/api/fasta/{}"
LOGGER = logging.getLogger(__name__)


def ena_download(ena_id: str, ena_dir=ENA_DIR) -> Path:
    url = ENA_URL_TEMPLATE.format(ena_id)
    filename = ena_dir.joinpath(f'{ena_id}.fa')
    return remote_dl(url, filename, skip_existing=True)


def ena_seq(ena_id: str, ena_dir=ENA_DIR) -> SeqRecord:
    """
    Given an ENI (European Nucleotide Archive) id, returns the corresponding
    nucleotide sequence.
    :param ena_id: id of data to fetch.
    :param ena_dir: Directory to download ENA file to.
    :return: A biopython Sequence object.
    """
    filename = ena_download(ena_id, ena_dir)

    with open(filename, 'r') as local_handle:
        return SeqIO.read(local_handle, 'fasta')
