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

import Bio.PDB as PDB
import Bio.PDB.MMCIF2Dict
import Bio.PDB.Structure
import Bio.SeqIO as SeqIO
import Bio.SwissProt as SwissProt
import requests
from Bio.SeqRecord import SeqRecord
import yattag

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


def pdb_struct(pdb_id: str, pdb_dir=PDB_DIR) -> PDB.Structure:
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
    parser = PDB.MMCIFParser(QUIET=True)
    return parser.get_structure(pdb_id, filename)


def pdbid_to_unpids(pdb_id: str, pdb_dir=PDB_DIR) -> List[str]:
    """
    Extracts Uniprot protein ids from a PDB protein structure.
    :param pdb_id: The PDB id of the structure.
    :param pdb_dir: Directory to download PDB file to.
    :return: A list of Uniprot ids.
    """
    filename = pdb_download(pdb_id, pdb_dir=pdb_dir)
    pdb_dict = PDB.MMCIF2Dict.MMCIF2Dict(filename)

    # Go over referenced DBs and take first accession id belonging to Uniprot
    unp_ids = []
    for i, db_name in enumerate(pdb_dict['_struct_ref.db_name']):
        if db_name.lower() == 'unp':
            unp_ids.append(pdb_dict['_struct_ref.pdbx_db_accession'][i])

    return unp_ids


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
        return SeqIO.read(local_handle, 'fasta')


class PDBQuery:
    """
    Represents a query that can be sent to the PDB search API to obtain PDB
    ids matching some criteria.
    To implement new search criteria, simply derive from this class.

    See documentation here:
    https://www.rcsb.org/pages/webservices/rest-search#search
    """
    TAG_QUERY = 'orgPdbQuery'
    TAG_QUERY_TYPE = 'queryType'
    TAG_DESCRIPTION = 'description'

    def to_xml(self):
        raise NotImplementedError("Override this")

    def to_xml_pretty(self):
        return yattag.indent(self.to_xml())

    def execute(self):
        """
        Executes the query on PDB.
        :return: A list of PDB IDs for proteins matching the query.
        """
        header = {'Content-Type': 'application/x-www-form-urlencoded'}
        search_url = 'https://www.rcsb.org/pdb/rest/search'
        query = self.to_xml()

        pdb_ids = []
        try:
            response = requests.post(search_url, data=query, headers=header)
            response.raise_for_status()
            pdb_ids = response.text.split()
        except requests.exceptions.RequestException as e:
            LOGGER.error('Failed to query PDB', exc_info=e)

        return pdb_ids


class PDBCompositeQuery(PDBQuery):
    """
    A composite query is composed of multiple regular PDBQueries.
    It creates a query that represents "query1 AND query2 AND ... queryN".
    """
    TAG_COMPOSITE = 'orgPdbCompositeQuery'
    TAG_REFINEMENT = 'queryRefinement'
    TAG_REFINEMENT_LEVEL = 'queryRefinementLevel'
    TAG_CONJ_TYPE = 'conjunctionType'

    def __init__(self, *queries: PDBQuery):
        super().__init__()
        self.queries = queries

    def to_xml(self):
        doc, tag, text, line = yattag.Doc().ttl()

        with tag(self.TAG_COMPOSITE, version="1.0"):
            for i, query in enumerate(self.queries):
                with tag(self.TAG_REFINEMENT):
                    line(self.TAG_REFINEMENT_LEVEL, i)

                    if i > 0:
                        line(self.TAG_CONJ_TYPE, 'and')

                    # Insert XML from regular query as-is
                    doc.asis(query.to_xml())

        return doc.getvalue()


class PDBResolutionQuery(PDBQuery):
    RES_QUERY_TYPE = 'org.pdb.query.simple.ResolutionQuery'
    TAG_RES_COMP = 'refine.ls_d_res_high.comparator'
    TAG_RES_MIN = 'refine.ls_d_res_high.min'
    TAG_RES_MAX = 'refine.ls_d_res_high.max'

    def __init__(self, min_res=0., max_res=2.):
        super().__init__()
        self.min_res = min_res
        self.max_res = max_res
        self.description = f'Resolution between {min_res} and {max_res}'

    def to_xml(self):
        doc, tag, text, line = yattag.Doc().ttl()

        with tag(self.TAG_QUERY):
            line(self.TAG_QUERY_TYPE, self.RES_QUERY_TYPE)
            line(self.TAG_DESCRIPTION, self.description)
            line(self.TAG_RES_COMP, 'between')
            line(self.TAG_RES_MIN, self.min_res)
            line(self.TAG_RES_MAX, self.max_res)

        return doc.getvalue()


class PDBExpressionSystemQuery(PDBQuery):
    COMP_TYPES = {'contains', 'equals', 'startswith', 'endswith',
                  '!contains', '!startswith', '!endswith'}

    EXPR_SYS_QUERY_TYPE = 'org.pdb.query.simple.ExpressionOrganismQuery'
    TAG_COMP = 'entity_src_gen.pdbx_host_org_scientific_name.comparator'
    TAG_NAME = 'entity_src_gen.pdbx_host_org_scientific_name.value'

    def __init__(self, expr_sys: str, comp_type: str = 'contains'):
        super().__init__()
        self.expr_sys = expr_sys
        if comp_type not in self.COMP_TYPES:
            raise ValueError(f"Unknown comparison type {comp_type}, must be "
                             f"one of {self.COMP_TYPES}.")
        self.comp_type = comp_type
        self.description = f'Expression system {self.comp_type}' \
                           f' {self.expr_sys}'

    def to_xml(self):
        doc, tag, text, line = yattag.Doc().ttl()

        with tag(self.TAG_QUERY):
            line(self.TAG_QUERY_TYPE, self.EXPR_SYS_QUERY_TYPE)
            line(self.TAG_DESCRIPTION, self.description)
            line(self.TAG_COMP, self.comp_type)
            line(self.TAG_NAME, self.expr_sys)

        return doc.getvalue()


if __name__ == '__main__':
    query = PDBCompositeQuery(
        PDBResolutionQuery(max_res=0.7),
        PDBExpressionSystemQuery(expr_sys='Escherichia Coli',
                                 comp_type='contains'))
    pdb_ids = query.execute()
    print(f'{pdb_ids}')
    print(f'Got {len(pdb_ids)} ids')
