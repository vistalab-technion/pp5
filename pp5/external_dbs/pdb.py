from __future__ import annotations

import re
import abc
import math
import warnings
from math import cos, sin, radians as rad, degrees as deg
import logging
from pathlib import Path
from typing import NamedTuple, Type

import requests
import yattag
import numpy as np
from Bio import PDB as PDB
from Bio.PDB import Structure as PDBRecord, MMCIF2Dict
from Bio.PDB.DSSP import dssp_dict_from_pdb_file
from Bio.PDB.PDBExceptions import PDBConstructionWarning, \
    PDBConstructionException

from pp5 import PDB_DIR, get_resource_path
from pp5.utils import remote_dl

PDB_ID_PATTERN = re.compile(r'^(?P<id>[0-9][\w]{3})(?::(?P<chain>[a-z]))?$',
                            re.IGNORECASE | re.ASCII)

PDB_SEARCH_URL = 'https://www.rcsb.org/pdb/rest/search'
PDB_DOWNLOAD_URL_TEMPLATE = r"https://files.rcsb.org/download/{}.cif.gz"
LOGGER = logging.getLogger(__name__)


def split_id(pdb_id):
    """
    Splits and validates a full PDB id consisting of a base id and
    optionally also a chain, into a tuple with the base id and chain.
    Will raise an exception if the given id is not a valid PDB id.
    :param pdb_id: PDB id, either without a chain, e.g. '5JDT' or with a
    chain, e.g. '5JDT:A'.
    :return: A tuple (id, chain) where id is the base id and chain can be
    None. The returned strings will be upper-cased.
    """
    pdb_id = pdb_id.upper()
    match = PDB_ID_PATTERN.match(pdb_id)
    if not match:
        raise ValueError(f"Invalid PDB id format: {pdb_id}")

    return match.group('id'), match.group('chain')


def pdb_download(pdb_id: str, pdb_dir=PDB_DIR) -> Path:
    """
    Downloads a protein structure file from PDB.
    :param pdb_id: The id of the structure to download.
    :param pdb_dir: Directory to download PDB file to.
    """
    # Separate id and chain if chain was specified
    pdb_id, chain_id = split_id(pdb_id)

    pdb_id = pdb_id.lower()
    filename = get_resource_path(pdb_dir, f'{pdb_id}.cif')
    url = PDB_DOWNLOAD_URL_TEMPLATE.format(pdb_id)
    return remote_dl(url, filename, uncompress=True, skip_existing=True)


def pdb_struct(pdb_id: str, pdb_dir=PDB_DIR, struct_d=None) -> PDBRecord:
    """
    Given a PDB structure id, returns an object representing the protein
    structure.
    :param pdb_id: The PDB id of the structure.
    :param pdb_dir: Directory to download PDB file to.
    :param struct_d: Optional dict containing the parsed structure as a
    dict. If provided, the file wont have to be re-parsed.
    :return: An biopython PDB Structure object.
    """
    pdb_id, chain_id = split_id(pdb_id)
    filename = pdb_download(pdb_id, pdb_dir=pdb_dir)

    # Parse the PDB file into a Structure object
    LOGGER.info(f"Loading PDB file {filename}...")
    parser = CustomMMCIFParser()
    return parser.get_structure(pdb_id, filename, mmcif_dict=struct_d)


def pdb_dict(pdb_id: str, pdb_dir=PDB_DIR, struct_d=None) -> dict:
    """
    Returns a dictionary containing all the contents of a PDB mmCIF file.
    :param pdb_id: The PDB id of the structure.
    :param pdb_dir: Directory to download PDB file to.
    :param struct_d: Optional dict of the structure if it was already loaded.
    This parameter exists to streamline other functions that use this one in
    case the file was already parsed.
    """
    pdb_id, chain_id = split_id(pdb_id)
    # No need to re-parse the file if we have a matching struct dict
    if struct_d and struct_d['_entry.id'][0].upper() == pdb_id:
        return struct_d

    filename = pdb_download(pdb_id, pdb_dir=pdb_dir)
    return MMCIF2Dict.MMCIF2Dict(filename)


def pdbid_to_unpid(pdb_id: str, pdb_dir=PDB_DIR, struct_d=None) -> str:
    """
    Extracts a Uniprot protein id from a PDB protein structure.
    :param pdb_id: The PDB id of the structure, optionally including chain.
    If no chain is provided, and there are multiple chains with different
    Uniprot IDs, the first Uniprot ID will be returned.
    :param pdb_dir: Directory to download PDB file to.
    :param struct_d: Optional dict containing the parsed structure.
    :return: A Uniprot id.
    """
    pdb_id, chain_id = split_id(pdb_id)
    struct_d = pdb_dict(pdb_id, pdb_dir, struct_d=struct_d)

    # Go over referenced DBs and take all uniprot IDs
    unp_ids = set()
    for i, db_name in enumerate(struct_d['_struct_ref.db_name']):
        if db_name.lower() == 'unp':
            unp_ids.add(struct_d['_struct_ref.pdbx_db_accession'][i])

    if not unp_ids:
        raise ValueError(f'No Uniprot cross-references found in {pdb_id}')

    if not chain_id:
        if len(unp_ids) > 1:
            LOGGER.warning(f"Multiple Uniprot IDs exists for {pdb_id}, and no "
                           f"chain specified. Returning first Uniprot ID.")
        return next(iter(unp_ids))

    # Loop over extra information about each cross-reference.
    # The chains are contained here. Find the Uniprot ID for the requested
    # chain.
    for i, curr_id in enumerate(struct_d['_struct_ref_seq.pdbx_db_accession']):
        if curr_id not in unp_ids:
            continue

        curr_chain = struct_d['_struct_ref_seq.pdbx_strand_id'][i]
        if curr_chain.lower() == chain_id.lower():
            return curr_id

    raise ValueError(f"Can't find Uniprot ID for chain {chain_id} of {pdb_id}")


def pdb_to_secondary_structure(pdb_id: str, pdb_dir=PDB_DIR):
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
     -       None
    :param pdb_id: The PDB id of the structure.
    :param pdb_dir: Directory to download PDB file to.
    :return: A tuple of
        ss_dict: maps from a residue id to the 1-char string denoting the
        type of secondary structure at that residue.
        keys: The residue ids.
    """
    path = pdb_download(pdb_id, pdb_dir)

    try:
        dssp_dict, keys = dssp_dict_from_pdb_file(str(path), DSSP='mkdssp')
    except Exception as e:
        raise RuntimeError(f"Failed to get secondary structure for {pdb_id}")

    # dssp_dict maps a reisdue id to a tuple containing various things about
    # that residue. We take only the secondary structure info.
    ss_dict = {k: v[1] for k, v in dssp_dict.items()}

    return ss_dict, keys


def pdb_to_unit_cell(pdb_id: str, pdb_dir=PDB_DIR, struct_d=None) \
        -> PDBUnitCell:
    """
    :return: a UnitCell object given a PDB id.
    """
    d = pdb_dict(pdb_id, pdb_dir, struct_d=struct_d)
    try:
        a, b, c = d['_cell.length_a'], d['_cell.length_b'], d['_cell.length_c']
        alpha, beta = d['_cell.angle_alpha'], d['_cell.angle_beta']
        gamma = d['_cell.angle_gamma']

        a, b, c = float(a[0]), float(b[0]), float(c[0])
        alpha, beta, gamma = float(alpha[0]), float(beta[0]), float(gamma[0])
    except KeyError:
        raise ValueError(f"Can't create UnitCell for {pdb_id}")

    return PDBUnitCell(pdb_id, a, b, c, alpha, beta, gamma)


def pdb_metadata(pdb_id: str, pdb_dir=PDB_DIR, struct_d=None) -> PDBMetadata:
    """
    Extracts metadata from a PDB structure.
    Helpful metadata fields:
    https://www.rcsb.org/pdb/results/reportField.do

    :param pdb_id: The PDB ID of the structure.
    :param pdb_dir: Where to store PDB files.
    :param struct_d: Optional dict which will be used if given, instead of
    parsing the PDB file.
    :return: A PDBMetadata object.
    """
    pdb_id, chain_id = split_id(pdb_id)
    struct_d = pdb_dict(pdb_id, pdb_dir, struct_d)

    def meta(key: str, convert_to: Type = str):
        val = struct_d.get(key, None)
        if not val:
            return None
        if isinstance(val, list):
            val = val[0]
        if not val or val == '?':
            return None
        return convert_to(val)

    title = meta('_struct.title')

    src_org = meta('_entity_src_nat.pdbx_organism_scientific')
    if not src_org:
        src_org = meta('_entity_src_gen.pdbx_gene_src_scientific_name')

    src_org_id = meta('_entity_src_nat.pdbx_ncbi_taxonomy_id', int)
    if not src_org_id:
        src_org_id = meta('_entity_src_gen.pdbx_gene_src_ncbi_taxonomy_id',
                          int)

    host_org = meta('_entity_src_gen.pdbx_host_org_scientific_name')
    host_org_id = meta('_entity_src_gen.pdbx_host_org_ncbi_taxonomy_id', int)
    resolution = meta('_refine.ls_d_res_high', float)
    resolution_low = meta('_refine.ls_d_res_low', float)

    return PDBMetadata(pdb_id, title, src_org, src_org_id,
                       host_org, host_org_id, resolution, resolution_low)


class PDBMetadata(NamedTuple):
    pdb_id: str
    title: str
    src_org: str
    src_org_id: int
    host_org: str
    host_org_id: int
    resolution: float
    resolution_low: float


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

    def __init__(self, pdb_id, a, b, c, alpha, beta, gamma):
        """
        a, b, c: Unit cell lengths in Angstroms
        alpha, beta, gamma: Unit-cell angles in degrees
        """
        self.pdb_id = pdb_id
        self.a, self.b, self.c = a, b, c
        self.alpha, self.beta, self.gamma = alpha, beta, gamma

        cos_alpha, sin_alpha = cos(rad(self.alpha)), sin(rad(self.alpha))
        cos_beta, sin_beta = cos(rad(self.beta)), sin(rad(self.beta))
        cos_gamma, sin_gamma = cos(rad(self.gamma)), sin(rad(self.gamma))

        # Volume
        factor = math.sqrt(1 - cos_alpha ** 2 - cos_beta ** 2 - cos_gamma ** 2
                           + 2 * cos_alpha * cos_beta * cos_gamma)
        self.vol = self.a * self.b * self.c * factor

        # Reciprocal lengths
        self.a_r = self.b * self.c * sin_alpha / self.vol
        self.b_r = self.c * self.a * sin_beta / self.vol
        self.c_r = self.a * self.b * sin_gamma / self.vol

        # Reciprocal angles
        cos_alpha_r = (cos_beta * cos_gamma - cos_alpha) / (sin_beta *
                                                            sin_gamma)
        cos_beta_r = (cos_gamma * cos_alpha - cos_beta) / (sin_gamma *
                                                           sin_alpha)
        cos_gamma_r = (cos_alpha * cos_beta - cos_gamma) / (sin_alpha *
                                                            sin_beta)
        self.alpha_r = deg(math.acos(cos_alpha_r))
        self.beta_r = deg(math.acos(cos_beta_r))
        self.gamma_r = deg(math.acos(cos_gamma_r))

        # Types of coordinate systems:
        # Fractional: no units
        # Cartesian: length
        # Direct lattice: length
        # Reciprocal lattice: 1/length

        # A: Transformation from fractional to Cartesian coordinates
        self.A = np.array([[self.a, self.b * cos_gamma, self.c * cos_beta],
                           [0, self.b * sin_gamma,
                            -self.c * sin_beta * cos_alpha_r],
                           [0, 0, 1 / self.c_r]], dtype=np.float32)

        # A^-1: Transformation from Cartesian to fractional coordinates
        self.Ainv = np.linalg.inv(self.A)

        # B: Transformation matrix from direct lattice coordinates to cartesian
        self.N = np.diag([self.a_r, self.b_r, self.c_r]).astype(np.float32)
        self.B = np.dot(self.A, self.N)

        # B^-1: Transformation matrix from Cartesian to direct lattice
        self.Binv = np.linalg.inv(self.B)

        # Fix precision issues
        [np.round(a, decimals=15, out=a) for a in (self.A, self.Ainv,
                                                   self.B, self.Binv)]

    def direct_lattice_to_cartesian(self, x: np.ndarray):
        assert 0 < x.ndim < 3
        if x.ndim == 1:
            return np.dot(self.B, x)
        elif x.ndim == 2:
            return np.dot(self.B, np.dot(x, self.B.T))

    def __repr__(self):
        abc = f'(a={self.a:.1f},b={self.b:.1f},c={self.c:.1f})'
        ang = f'(α={self.alpha:.1f},β={self.beta:.1f},γ={self.gamma:.1f})'
        return f'[{self.pdb_id}]{abc}{ang}'


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

    def __repr__(self):
        return self.description()


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
    """
    Query PDB for structures within a range of X-ray resolutions.
    """
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
    """
    Query PDB for structures with a specified expression system.
    """
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


class PDBSequenceQuery(PDBQuery):
    """
    Query PDB for structures
    """
    TOOL_TYPES = {'blast', 'psiblast'}

    SEQUENCE_QUERY_TYPE = 'org.pdb.query.simple.SequenceQuery'
    TAG_STRUCTURE_ID = 'structureId'
    TAG_CHAIN_ID = 'chainId'
    TAG_SEQUENCE = 'sequence'
    TAG_ECUTOFF = 'eValueCutoff'
    TAG_MASK_LOW_COMP = 'maskLowComplexity'
    TAG_IDENTITY_CUTOFF = 'sequenceIdentityCutoff'

    def __init__(self, pdb_id: str = None, sequence: str = None,
                 search_tool: str = 'blast', e_cutoff=10.0,
                 mask_low_complexity=True, identity_cutoff=0.):
        both = pdb_id and sequence
        neither = not pdb_id and not sequence
        if both or neither:
            raise ValueError('Must provide either pdb_id sequence')

        self.pdb_id = pdb_id
        self.sequence = sequence
        self.search_tool = search_tool
        self.e_cutoff = e_cutoff
        self.mask_low_complexity = 'yes' if mask_low_complexity else 'no'
        self.identity_cutoff = identity_cutoff

        if pdb_id:
            self.pdb_id, self.chain_id = split_id(pdb_id)
            if not self.chain_id:
                raise ValueError('Must provide chain info for query')

    def description(self):
        if self.pdb_id:
            seq_str = f'Structure:Chain = {self.pdb_id}:{self.chain_id}'
        else:
            n = 10
            groups = [self.sequence[i:i + n]
                      for i in range(0, len(self.sequence), n)]
            seq_str = str.join(" ", groups)

        return f'Sequence Search ({seq_str}, ' \
               f'Expectation Value = {self.e_cutoff:.1f}, ' \
               f'Sequence Identity = {self.identity_cutoff * 100:.0f}, ' \
               f'Search Tool = {self.search_tool}, ' \
               f'Mask Low Complexity={self.mask_low_complexity})'

    def to_xml(self):
        doc, tag, text, line = yattag.Doc().ttl()

        with tag(self.TAG_QUERY):
            line(self.TAG_QUERY_TYPE, self.SEQUENCE_QUERY_TYPE)
            line(self.TAG_DESCRIPTION, self.description())
            line(self.TAG_ECUTOFF, self.e_cutoff)
            line(self.TAG_MASK_LOW_COMP, self.mask_low_complexity)
            line(self.TAG_IDENTITY_CUTOFF, self.identity_cutoff)
            if self.pdb_id:
                line(self.TAG_STRUCTURE_ID, self.pdb_id)
                line(self.TAG_CHAIN_ID, self.chain_id)
            else:
                line(self.TAG_SEQUENCE, self.sequence)

        return doc.getvalue()


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
                id_from_struct_d = self._mmcif_dict['_entry.id'][0]
                if not id_from_struct_d.lower() == structure_id.lower():
                    raise PDBConstructionException(
                        "PDB ID mismatch between provided struct dict and "
                        "desired structure id")

            self._build_structure(structure_id)
            self._structure_builder.set_header(self._get_header())

        return self._structure_builder.get_structure()


if __name__ == '__main__':
    uc = pdb_to_unit_cell('1b0y')
    j = 3
