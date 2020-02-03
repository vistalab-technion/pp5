"""
This module contains helper functions to work with external databases such
as PDB, Uniprot and ENA.
"""

import gzip
import logging
import os
from pathlib import Path
from typing import List
from urllib.request import urlopen

import Bio.PDB as pdb
import Bio.PDB.MMCIF2Dict
import Bio.PDB.Structure
import Bio.SeqIO as seqio
import Bio.SwissProt as sprot
from Bio.SeqRecord import SeqRecord

from pp5 import PDB_DIR, UNP_DIR, ENA_DIR

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
        try:
            if uncompress:
                in_handle = gzip.GzipFile(fileobj=remote_handle)
            else:
                in_handle = remote_handle
            out_handle.write(in_handle.read())
        finally:
            in_handle.close()

    size_bytes = os.path.getsize(save_path)
    LOGGER.info(f"Downloaded {save_path} ({size_bytes / 1024:.1f}kB)")
    return Path(save_path)


def pdb_download(pdb_id: str, pdb_dir=PDB_DIR):
    """
    Downloads a protein structure file from PDB.
    :param pdb_id: The id of the structure to download.
    :param pdb_dir: Directory to download PDB file to.
    """
    pdb_id = pdb_id.lower()
    filename = pdb_dir.joinpath(f'{pdb_id}.cif')
    url = PDB_URL_TEMPLATE.format(pdb_id)
    return remote_dl(url, filename, uncompress=True, skip_existing=True)


def pdb_struct(pdb_id: str, pdb_dir=PDB_DIR) -> pdb.Structure:
    """
    Given a PDB structure id, returns an object representing the protein
    structure.
    :param pdb_id: The PDB id of the structure.
    :param pdb_dir: Directory to download PDB file to.
    :return: An biopython Structure object.
    """
    filename = pdb_download(pdb_id, pdb_dir=pdb_dir)

    # Parse the PDB file into a Structure object
    LOGGER.info(f"Loading PDB file {filename}...")
    parser = pdb.MMCIFParser(QUIET=True)
    return parser.get_structure(pdb_id, filename)


def pdbid_to_unpids(pdb_id: str, pdb_dir=PDB_DIR) -> List[str]:
    """
    Extracts Uniprot protein ids from a PDB protein structure.
    :param pdb_id: The PDB id of the structure.
    :param pdb_dir: Directory to download PDB file to.
    :return: A list of Uniprot ids.
    """
    filename = pdb_download(pdb_id, pdb_dir=pdb_dir)
    pdb_dict = pdb.MMCIF2Dict.MMCIF2Dict(filename)

    # Go over referenced DBs and take first accession id belonging to Uniprot
    unp_ids = []
    for i, db_name in enumerate(pdb_dict['_struct_ref.db_name']):
        if db_name.lower() == 'unp':
            unp_ids.append(pdb_dict['_struct_ref.pdbx_db_accession'][i])

    return unp_ids


def unp_record(unp_id: str, unp_dir=UNP_DIR) -> sprot.Record:
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
        return sprot.read(local_handle)


def ena_seq(ena_id: str, ena_dir=ENA_DIR) -> SeqRecord:
    """
    Given an ENI (European Nucleotide Archive) id, returns the corresponding
    nucleotide sequence.
    :param ena_id: id of data to fetch.
    :param ena_dir: Directory to download ENA file to.
    :return: A biopython Sequence object.
    """
    url = ENA_URL_TEMPLATE.format(ena_id)
    filename = ena_dir.joinpath(f'{ena_id}.fa')
    remote_dl(url, filename, skip_existing=True)

    with open(filename, 'r') as local_handle:
        return seqio.read(local_handle, 'fasta')
