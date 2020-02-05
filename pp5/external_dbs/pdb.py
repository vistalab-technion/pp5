import abc
import logging
from typing import List

import requests
import yattag
from Bio import PDB as PDB
from Bio.PDB import Structure, MMCIF2Dict

from pp5 import PDB_DIR
from pp5.utils import remote_dl

PDB_SEARCH_URL = 'https://www.rcsb.org/pdb/rest/search'
PDB_DOWNLOAD_URL_TEMPLATE = r"https://files.rcsb.org/download/{}.cif.gz"
LOGGER = logging.getLogger(__name__)


def pdb_download(pdb_id: str, pdb_dir=PDB_DIR):
    """
    Downloads a protein structure file from PDB.
    :param pdb_id: The id of the structure to download.
    :param pdb_dir: Directory to download PDB file to.
    """
    pdb_id = pdb_id.lower()
    filename = pdb_dir.joinpath(f'{pdb_id}.cif')
    url = PDB_DOWNLOAD_URL_TEMPLATE.format(pdb_id)
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
    unp_ids = set()
    for i, db_name in enumerate(pdb_dict['_struct_ref.db_name']):
        if db_name.lower() == 'unp':
            unp_ids.add(pdb_dict['_struct_ref.pdbx_db_accession'][i])

    return list(unp_ids)


class PDBQuery(abc.ABC):
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

    @abc.abstractmethod
    def to_xml(self):
        raise NotImplementedError("Override this")

    @abc.abstractmethod
    def description(self):
        raise NotImplementedError("Override this")

    def to_xml_pretty(self):
        return yattag.indent(self.to_xml())

    def execute(self):
        """
        Executes the query on PDB.
        :return: A list of PDB IDs for proteins matching the query.
        """
        header = {'Content-Type': 'application/x-www-form-urlencoded'}
        query = self.to_xml()

        pdb_ids = []
        try:
            LOGGER.info(f'Executing PDB query: {self.description()}')
            response = requests.post(PDB_SEARCH_URL, query, headers=header)
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

    def description(self):
        descriptions = [f'({q.description()})' for q in self.queries]
        return str.join(' AND ', descriptions)

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

    def description(self):
        return f'Resolution between {self.min_res} and {self.max_res}'

    def to_xml(self):
        doc, tag, text, line = yattag.Doc().ttl()

        with tag(self.TAG_QUERY):
            line(self.TAG_QUERY_TYPE, self.RES_QUERY_TYPE)
            line(self.TAG_DESCRIPTION, self.description())
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

    def description(self):
        return f'Expression system {self.comp_type} {self.expr_sys}'

    def to_xml(self):
        doc, tag, text, line = yattag.Doc().ttl()

        with tag(self.TAG_QUERY):
            line(self.TAG_QUERY_TYPE, self.EXPR_SYS_QUERY_TYPE)
            line(self.TAG_DESCRIPTION, self.description())
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
