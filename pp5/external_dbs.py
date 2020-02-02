"""
This module contains helper functions to work with external databases such
as PDB, UniProt and ENA.
"""

import os
import io
import gzip
import logging
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List
from urllib.request import urlopen


PDB_URL_TEMPLATE = r"https://files.rcsb.org/download/{}.cif.gz"
UNP_URL_TEMPLATE = r"https://www.uniprot.org/uniprot/{}.txt"
ENA_URL_TEMPLATE = r"https://www.ebi.ac.uk/ena/browser/api/fasta/{}"
LOGGER = logging.getLogger(__name__)


def remote_dl(url: str, save_path: str, uncompress=False, skip_existing=False):
    """
    Downloads contents of a remote file and saves it into a local file.
    :param url: The url to download from.
    :param save_path: Local file path to save to.
    :param uncompress: Whether to uncompress gzip files.
    :param skip_existing: Whether to skip download if a local file with
    the given path already exists.
    :return: A Path object for the downloaded file.
    """
    if os.path.isfile(save_path) and skip_existing:
        LOGGER.info(f"Local file {save_path} exists, skipping download...")
        return Path(save_path)

    with urlopen(url) as remote_handle, open(save_path, 'wb') as out_handle:
        LOGGER.info(f"Downloading {url}...")
        try:
            if uncompress:
                in_handle = gzip.GzipFile(fileobj=remote_handle)
            else:
                in_handle = remote_handle
            out_handle.write(in_handle.read())
        finally:
            in_handle.close()

    return Path(save_path)


