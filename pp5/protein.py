from __future__ import annotations

import logging
import math
import os
import warnings
import enum
import pickle
from pathlib import Path
from typing import List, Tuple, Dict, Iterator, Callable, Any, Union, \
    Iterable, Optional, Set
from collections import OrderedDict, Counter
import itertools as it

import pandas as pd
from Bio.Align import PairwiseAligner
from Bio.Data import CodonTable
from Bio.PDB import PPBuilder
from Bio.PDB.Polypeptide import Polypeptide
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from pytest import approx

import pp5
from pp5.dihedral import Dihedral, DihedralAnglesEstimator, \
    DihedralAnglesUncertaintyEstimator, DihedralAnglesMonteCarloEstimator, \
    AtomLocationUncertainty
from pp5.external_dbs import pdb, unp, ena
from pp5.external_dbs.pdb import PDBRecord, PDBQuery
from pp5.external_dbs.unp import UNPRecord
from pp5.align import StructuralAlignment
from pp5.align import PYMOL_SA_GAP_SYMBOLS as PSA_GAP
from pp5.parallel import global_pool

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from Bio.Align import substitution_matrices

BLOSUM62 = substitution_matrices.load("BLOSUM62")
BLOSUM80 = substitution_matrices.load("BLOSUM80")
CODON_TABLE = CodonTable.standard_dna_table.forward_table
START_CODONS = CodonTable.standard_dna_table.start_codons
STOP_CODONS = CodonTable.standard_dna_table.stop_codons
UNKNOWN_AA = 'X'
UNKNOWN_CODON = '---'

LOGGER = logging.getLogger(__name__)


def _tagged_filepath(pdb_id: str, out_dir: Path, suffix: str, tag: str):
    tag = f'-{tag}' if tag else ''
    filename = f'{pdb_id.replace(":", "_").upper()}{tag}'
    filepath = out_dir.joinpath(f'{filename}.{suffix}')
    return filepath


class ProteinInitError(ValueError):
    pass


class ResidueRecord(object):
    """
    Represents a singe residue in a protein record.
    - Sequence id: (identifier of this residue in the sequence, usually an
    integer + insertion code if present, which indicates some alteration
    compared to the wild-type)
    - Name: (single letter AA code or X for unknown)
    - Codon: three-letter nucleotide sequence
    - Codon score: Confidence measure for the codon match
    - Codon opts: All possible codons found in DNA sequences of the protein
    - phi/psi/omega: dihedral angles in degrees
    - bfactor: average b-factor along of the residue's backbone atoms
    - secondary: single-letter secondary structure code
    """

    def __init__(self, res_id: Union[str, int], name: str,
                 codon: str, codon_score: float,
                 codon_opts: Union[Iterable[str], str],
                 angles: Dihedral, bfactor: float, secondary: str):
        self.res_id, self.name = str(res_id).strip(), name
        self.codon, self.codon_score = codon, codon_score
        if isinstance(codon_opts, str):
            self.codon_opts = codon_opts
        else:
            self.codon_opts = str.join('/', codon_opts)
        self.angles, self.bfactor, self.secondary = angles, bfactor, secondary

    def as_dict(self, skip_omega=False):
        d = self.__dict__.copy()
        # Replace angles object with the angles themselves
        d.pop('angles')
        a = self.angles
        d.update(a.as_dict(degrees=True, with_std=True, skip_omega=skip_omega))
        return d

    def __repr__(self):
        return f'{self.name}{self.res_id:<4s} [{self.codon}]' \
               f'[{self.secondary}] {self.angles} b={self.bfactor:.2f}'

    def __eq__(self, other):
        if self is other:
            return True
        if not isinstance(other, ResidueRecord):
            return False
        for k, v in self.__dict__.items():
            other_v = other.__dict__.get(k, math.inf)
            if isinstance(v, float):
                equal = v == approx(other_v, nan_ok=True)
            else:
                equal = v == other_v
            if not equal:
                return False
        return True


class ResidueMatchType(enum.IntEnum):
    REFERENCE = enum.auto()
    VARIANT = enum.auto()
    SAME = enum.auto()
    SILENT = enum.auto()
    MUTATION = enum.auto()
    ALTERATION = enum.auto()


class ResidueMatch(ResidueRecord):
    """
    Represents a residue match between a reference structure and a query
    structure.
    """

    @classmethod
    def from_residue(cls, query_res: ResidueRecord,
                     query_idx: int,
                     match_type: ResidueMatchType,
                     ang_dist: float, context: int):
        return cls(query_idx, match_type, ang_dist, context,
                   **query_res.__dict__)

    def __init__(self, query_idx: int, match_type: ResidueMatchType,
                 ang_dist: float, context: int, **res_rec_args):
        self.idx = query_idx
        self.type = match_type
        self.ang_dist = ang_dist
        self.context = context
        super().__init__(**res_rec_args)

    def __repr__(self):
        return f'[{self.idx}] {self.type.name}, ' \
               f'diff={self.ang_dist:.2f}, context={self.context}'


class ResidueMatchGroup(object):
    """
    Represents a group of residue matches which share a common Uniprot ID
    and codon.
    """

    def __init__(self, unp_id: str, codon: str, pdb_ids: Tuple[str],
                 group_idxs: Tuple[int], group_res_ids: Tuple[str],
                 group_contexts: Tuple[int], group_angles: Tuple[Dihedral],
                 match_type: ResidueMatchType,
                 name: str, codon_opts: set, secondary: str,
                 avg_phipsi: Dihedral, ang_dist: float):
        assert len(pdb_ids) == len(group_idxs) == len(group_res_ids) == \
               len(group_contexts) == len(group_angles) > 0

        self.unp_id = unp_id
        self.codon = codon
        self.group_size = len(pdb_ids)
        self.name = name
        self.codon_opts = codon_opts
        self.secondary = secondary
        self.match_type = match_type
        self.avg_phipsi = avg_phipsi
        self.norm_factor = math.sqrt(
            avg_phipsi.phi_std_deg ** 2 + avg_phipsi.psi_std_deg ** 2
        )
        self.ang_dist = ang_dist
        self.pdb_ids = pdb_ids
        self.idxs = group_idxs
        self.res_ids = group_res_ids
        self.contexts = group_contexts
        self.angles = tuple((a.phi_deg, a.psi_deg) for a in group_angles)

    def __repr__(self):
        return f'[{self.unp_id}, {self.codon}] {self.match_type.name} ' \
               f'{self.avg_phipsi}, n={self.group_size},' \
               f'ang_dist={self.ang_dist:.2f}'

    @property
    def codon_opts_str(self): return self._join(self.codon_opts, '/')

    @property
    def pdb_ids_str(self): return self._join(self.pdb_ids)

    @property
    def idxs_str(self): return self._join(self.idxs)

    @property
    def res_ids_str(self): return self._join(self.res_ids)

    @property
    def contexts_str(self): return self._join(self.contexts)

    @property
    def angles_str(self):
        return self._join(f'{phi:.3f},{psi:.3f}' for phi, psi, in self.angles)

    @staticmethod
    def _join(seq, d=';'): return str.join(d, map(str, seq))


class ProteinRecord(object):
    """
    Represents a protein in our dataset. Includes:
    - Uniprot id defining the which protein this is.
    - PDB id of one structure representing this protein.
    - Amino acid sequence based on PDB structure,
    - Genetic codon sequence based on Uniprot cross-ref with ENA
      (and matching to AA sequence from PDB)
    - Dihedral angles at each AA.

    Protein AA sequence is defined by the specific PDB structure we're
    using, not the sequence from Uniprot. We want multiple different
    PDB structures with the same Uniprot id and possible slightly
    different AAs.
    """
    _SKIP_SERIALIZE = ['_unp_rec', '_pdb_rec', '_pdb_dict', '_pp']

    @staticmethod
    def from_cache(pdb_id, cache_dir: Union[str, Path] = None, tag=None) \
            -> Optional[ProteinRecord]:
        """
        Loads a cached ProteinRecord, if it exists.
        :param pdb_id: PDB ID with chain.
        :param cache_dir: Directory with cached files.
        :param tag: Optional extra tag on the filename.
        :return: Loaded ProteinRecord, or None if the cached prec does not
        exist.
        """
        if not isinstance(cache_dir, (str, Path)):
            cache_dir = pp5.PREC_DIR

        path = _tagged_filepath(pdb_id, cache_dir, 'prec', tag)
        filename = path.name
        path = pp5.get_resource_path(cache_dir, filename)
        prec = None
        if path.is_file():
            try:
                with open(str(path), 'rb') as f:
                    prec = pickle.load(f)
            except Exception as e:
                # If we can't unpickle, probably the code changed since
                # saving this object. We'll just return None, so that a new
                # prec will be created and stored.
                LOGGER.warning(f'Failed to load cached ProteinRecord {path}')
        return prec

    @classmethod
    def from_pdb(cls, pdb_id: str, pdb_dict=None,
                 cache=False, cache_dir=pp5.PREC_DIR,
                 strict_pdb_xref=True, **kw_for_init) -> ProteinRecord:
        """
        Given a PDB id, finds the corresponding Uniprot id, and returns a
        ProteinRecord object for that protein.
        The PDB ID can be any valid format: either base ID only,
        e.g. '1ABC'; with a specified chain, e.g. '1ABC:D'; or with a
        specified entity, e.g. '1ABC:2'.
        :param pdb_id: The PDB id to query, with optional chain, e.g. '0ABC:D'.
        :param pdb_dict: Optional structure dict for the PDB record, in case it
        was already parsed.
        :param cache: Whether to load prec from cache if available.
        :param cache_dir: Where the cache dir is. ProteinRecords will be
        written to this folder after creation, unless it's None.
        :param strict_pdb_xref: Whether to require that the given PDB ID
        maps uniquely to only one Uniprot ID.
        :param kw_for_init: Extra kwargs for the ProteinRecord initializer.
        :return: A ProteinRecord.
        """
        try:
            # Either chain or entity or none can be provided, but not both
            pdb_id, chain_id, entity_id = pdb.split_id_with_entity(pdb_id)
            numeric_chain = False
            if entity_id:
                entity_id = int(entity_id)

                # Discover which chains belong to this entity
                pdb_dict = pdb.pdb_dict(pdb_id, struct_d=pdb_dict)
                meta = pdb.PDBMetadata(pdb_id, struct_d=pdb_dict)
                chain_id = meta.get_chain(entity_id)

                if not chain_id:
                    # In rare cases the chain is a number instead of a letter,
                    # so there's no way to distinguish between entity id and
                    # chain except also trying to use our entity as a chain
                    # and finding the actual entity. See e.g. 4N6V.
                    if str(entity_id) in meta.chain_entities:
                        # Chain is number, but use it's string representation
                        chain_id = str(entity_id)
                        numeric_chain = True
                    else:
                        raise ProteinInitError(
                            f'No matching chain found for entity '
                            f'{entity_id} in PDB structure {pdb_id}')

                pdb_id = f'{pdb_id}:{chain_id}'

            elif chain_id:
                pdb_id = f'{pdb_id}:{chain_id}'

            if cache and chain_id:
                prec = cls.from_cache(pdb_id, cache_dir)
                if prec is not None:
                    return prec

            if not pdb_dict:
                pdb_dict = pdb.pdb_dict(pdb_id, struct_d=pdb_dict)

            unp_id = pdb.PDB2UNP.pdb_id_to_unp_id(
                pdb_id, strict=strict_pdb_xref, cache=True, struct_d=pdb_dict
            )

            prec = cls(unp_id, pdb_id, pdb_dict=pdb_dict,
                       numeric_chain=numeric_chain, **kw_for_init)
            if cache_dir:
                prec.save(out_dir=cache_dir)

            return prec
        except Exception as e:
            raise ProteinInitError(f"Failed to create protein record for "
                                   f"pdb_id={pdb_id}: {e}") from e

    @classmethod
    def from_unp(cls, unp_id: str, cache=False, cache_dir=pp5.PREC_DIR,
                 xref_selector: Callable[[unp.UNPPDBXRef], Any] = None,
                 **kw_for_init) -> ProteinRecord:
        """
        Creates a ProteinRecord from a Uniprot ID.
        The PDB structure with best resolution will be used.
        :param unp_id: The Uniprot id to query.
        :param xref_selector: Sort key for PDB cross refs. If None,
        resolution will be used.
        :param cache: Whether to load prec from cache if available.
        :param cache_dir: Where the cache dir is. ProteinRecords will be
        written to this folder after creation, unless it's None.
        :param kw_for_init: Extra args for the ProteinRecord initializer.
        :return: A ProteinRecord.
        """
        if not xref_selector:
            xref_selector = lambda xr: xr.resolution

        try:
            xrefs = unp.find_pdb_xrefs(unp_id)
            xrefs = sorted(xrefs, key=xref_selector)
            pdb_id = f'{xrefs[0].pdb_id}:{xrefs[0].chain_id}'

            if cache:
                prec = cls.from_cache(pdb_id, cache_dir=cache_dir)
                if prec is not None:
                    return prec

            prec = cls(unp_id, pdb_id, **kw_for_init)
            if cache_dir:
                prec.save(out_dir=cache_dir)

            return prec
        except Exception as e:
            raise ProteinInitError(f"Failed to create protein record for "
                                   f"unp_id={unp_id}") from e

    def __init__(self, unp_id: str, pdb_id: str, pdb_dict: dict = None,
                 dihedral_est_name='erp', dihedral_est_args: dict = None,
                 max_ena=None, strict_unp_xref=True, numeric_chain=False):
        """
        Initialize a protein record from both Uniprot and PDB ids.
        To initialize a protein from Uniprot id or PDB id only, use the
        class methods provided for this purpose.

        :param unp_id: Uniprot id which uniquely identifies the protein.
        :param pdb_id: PDB id with or without chain (e.g. '1ABC' or '1ABC:D')
        of the specific structure desired. Note that this structure must match
        the unp_id, i.e. it must exist in the cross-refs of the given unp_id.
        Otherwise an error will be raised (unless strict_unp_xref=False). If no
        chain is specified, a chain matching the unp_id will be used,
        if it exists.
        :param dihedral_est_name: Method of dihedral angle estimation.
        Options are:
        None or empty to calculate angles without error estimation;
        'erp' for standard error propagation;
        'mc' for montecarlo error estimation.
        :param dihedral_est_args: Extra arguments for dihedral estimator.
        :param max_ena: Number of maximal ENA records (containing protein
        genetic data) to align to the PDB structure of this protein. None
        means no limit (all cross-refs from Uniprot will be aligned).
        :param strict_unp_xref: Whether to require that there exist a PDB
        cross-ref for the given Uniprot ID.
        :param numeric_chain: Whether the given chain id (if any) is
        numeric. In rare cases PDB structures have numbers as chain ids.
        """
        if not (unp_id and pdb_id):
            raise ProteinInitError("Must provide both Uniprot and PDB IDs")

        unp_id = unp_id.upper()
        LOGGER.info(f'{unp_id}: Initializing protein record...')
        self.__setstate__({})

        self.unp_id = unp_id

        # First we must find a matching PDB structure and chain for the
        # Uniprot id. If a pdb_id is given we'll try to use that, depending
        # on whether there's a Uniprot xref for it and on strict_unp_xref.
        self.strict_unp_xref = strict_unp_xref
        self.numeric_chain = numeric_chain
        self.pdb_base_id, self.pdb_chain_id = self._find_pdb_xref(pdb_id)
        self.pdb_id = f'{self.pdb_base_id}:{self.pdb_chain_id}'
        if pdb_dict:
            self._pdb_dict = pdb_dict

        self.pdb_meta = pdb.PDBMetadata(self.pdb_id, struct_d=self.pdb_dict)
        if not self.pdb_meta.resolution:
            raise ProteinInitError(f'Unknown resolution for {pdb_id}')

        LOGGER.info(
            f'{self}: {self.pdb_meta.description}, '
            f'org={self.pdb_meta.src_org} ({self.pdb_meta.src_org_id}), '
            f'expr={self.pdb_meta.host_org} ({self.pdb_meta.host_org_id}), '
            f'res={self.pdb_meta.resolution:.2f}Å, '
            f'entity_id={self.pdb_meta.chain_entities[self.pdb_chain_id]}'
        )

        # Make sure the structure is sane. See e.g. 1FFK.
        if not self.polypeptides:
            raise ProteinInitError(f"No parsable residues in {self.pdb_id}")

        # Get secondary-structure info using DSSP
        ss_dict, _ = pdb.pdb_to_secondary_structure(self.pdb_id)

        # Get estimators of dihedral angles and b-factor
        dihedral_est, bfactor_est = self._get_dihedral_estimators(
            dihedral_est_name, dihedral_est_args
        )

        # Extract the PDB AA sequence, dihedral angles and b-factors
        # from the PDB structure.
        # Even though we're working with one PDB chain, the results is a
        # list of multiple Polypeptide objects because we split them at
        # non-standard residues (HETATM atoms in PDB).
        pdb_aa_seq, aa_ids, angles, bfactors, sstructs = '', [], [], [], []
        for i, pp in enumerate(self.polypeptides):
            curr_start_idx = pp[0].get_id()[1]
            curr_end_idx = pp[-1].get_id()[1]

            # More than one pp means there are gaps due to non-standard AAs
            if i > 0:
                # Calculate index gap between this polypeptide and previous
                prev_end_idx = self.polypeptides[i - 1][-1].get_id()[1]
                gap_len = curr_start_idx - prev_end_idx - 1

                # fill in the gaps
                pdb_aa_seq += UNKNOWN_AA * gap_len
                aa_ids.extend(range(prev_end_idx + 1, curr_start_idx))
                angles.extend([Dihedral.empty()] * gap_len)
                bfactors.extend([math.nan] * gap_len)
                sstructs.extend(['-'] * gap_len)

            pdb_aa_seq += str(pp.get_sequence())
            aa_ids.extend([str.join("", map(str, res.get_id())) for res in pp])
            angles.extend(dihedral_est.estimate(pp))
            bfactors.extend(bfactor_est.mean_uncertainty(pp, True))
            res_ids = ((self.pdb_chain_id, res.get_id()) for res in pp)
            sss = (ss_dict.get(res_id, '-') for res_id in res_ids)
            sstructs.extend(sss)

        # Find the best matching DNA for our AA sequence via pairwise alignment
        # between the PDB AA sequence and translated DNA sequences.
        dna_seq_record, idx_to_codons = self._find_dna_alignment(
            pdb_aa_seq, max_ena
        )
        dna_seq = str(dna_seq_record.seq)
        self.ena_id = dna_seq_record.id

        # Create a ResidueRecord holding all data we need per residue
        residue_recs = []
        for i in range(len(pdb_aa_seq)):
            # Get best codon and calculate it's 'score' based on how many
            # other options there are
            codon_counts = idx_to_codons.get(i, {})
            best_codon, max_count, total_count = UNKNOWN_CODON, 0, 0
            for codon, count in codon_counts.items():
                total_count += count
                if count > max_count and codon != UNKNOWN_CODON:
                    best_codon, max_count = codon, count
            codon_score = max_count / total_count if total_count else 0
            codon_opts = codon_counts.keys()

            rr = ResidueRecord(
                res_id=aa_ids[i], name=pdb_aa_seq[i],
                codon=best_codon, codon_score=codon_score,
                codon_opts=codon_opts, angles=angles[i],
                bfactor=bfactors[i], secondary=sstructs[i],
            )
            residue_recs.append(rr)

        self._protein_seq = pdb_aa_seq
        self._dna_seq = dna_seq
        self._residue_recs = tuple(residue_recs)

    @property
    def unp_rec(self) -> UNPRecord:
        """
        :return: Uniprot record for this protein.
        """
        if not self._unp_rec:
            self._unp_rec = unp.unp_record(self.unp_id)
        return self._unp_rec

    @property
    def pdb_dict(self) -> dict:
        """
        :return: The PDB record for this protein as a raw dict parsed from
        an mmCIF file.
        """
        if not self._pdb_dict:
            self._pdb_dict = pdb.pdb_dict(self.pdb_id)
        return self._pdb_dict

    @property
    def pdb_rec(self) -> PDBRecord:
        """
        :return: PDB record for this protein. Note that this record may
        contain multiple chains and this protein only represents one of them
        (self.pdb_chain_id).
        """
        if not self._pdb_rec:
            self._pdb_rec = pdb.pdb_struct(self.pdb_id, struct_d=self.pdb_dict)
        return self._pdb_rec

    @property
    def dna_seq(self) -> SeqRecord:
        """
        :return: DNA nucleotide sequence. This is the full DNA sequence which,
        after translation, best-matches to the PDB AA sequence.
        """
        return SeqRecord(Seq(self._dna_seq), self.ena_id, '', '')

    @property
    def protein_seq(self) -> SeqRecord:
        """
        :return: Protein sequence as 1-letter AA names. Based on the
        residues found in the associated PDB structure.
        Note that the sequence might contain the letter 'X' denoting an
        unknown AA. This happens if the PDB entry contains non-standard AAs
        and we chose to ignore such AAs.
        """
        return SeqRecord(Seq(self._protein_seq), self.pdb_id, '', '')

    @property
    def codons(self) -> List[str]:
        """
        :return: Protein sequence based on translating DNA sequence with
        standard codon table.
        """
        return [x.codon for x in self]

    @property
    def dihedral_angles(self) -> List[Dihedral]:
        return [x.angles for x in self]

    @property
    def polypeptides(self) -> List[Polypeptide]:
        """
        :return: List of Polypeptide objects corresponding to the PDB chain of
        this protein. If there is more than one, they are "sub chains"
        within the chain represented by this ProteinRecord.
        Even though we're working with one PDB chain, the results is a
        list of multiple Polypeptide objects because we split them at
        non-standard residues (HETATM atoms in PDB).
        https://proteopedia.org/wiki/index.php/HETATM
        """
        if not self._pp:
            chain = self.pdb_rec[0][self.pdb_chain_id]
            pp_chains = PPBuilder().build_peptides(chain, aa_only=True)

            # Sort chain by sequence ID of first residue in the chain,
            # in case the chains are not returned in order.
            pp_chains = sorted(pp_chains, key=lambda ch: ch[0].get_id()[1])
            self._pp = pp_chains

        return self._pp

    def to_dataframe(self):
        """
        :return: A Pandas dataframe where each row is a ResidueRecord from
        this ProteinRecord.
        """
        # use the iterator of this class to get the residue recs
        data = [res_rec.as_dict(skip_omega=True) for res_rec in self]
        return pd.DataFrame(data)

    def to_csv(self, out_dir=pp5.out_subdir('prec'), tag=None):
        """
        Writes the ProteinRecord as a CSV file.
        Filename will be <PDB_ID>_<CHAIN_ID>_<TAG>.csv.
        Note that this is meant as a human-readable output format only,
        so a ProteinRecord cannot be re-created from this CSV.
        To save a ProteinRecord for later loading, use save().
        :param out_dir: Output dir.
        :param tag: Optional extra tag to add to filename.
        :return: The path to the written file.
        """
        filepath = _tagged_filepath(self.pdb_id, out_dir, 'csv', tag)
        df = self.to_dataframe()
        df.to_csv(filepath, na_rep='nan', header=True, index=False,
                  encoding='utf-8', float_format='%.3f')

        LOGGER.info(f'Wrote {self} to {filepath}')
        return filepath

    def save(self, out_dir=pp5.data_subdir('prec'), tag=None):
        """
        Write the ProteinRecord to a binary file which can later to
        re-loaded into memory, recreating the ProteinRecord.
        :param out_dir: Output dir.
        :param tag: Optional extra tag to add to filename.
        :return: The path to the written file.
        """
        filepath = _tagged_filepath(self.pdb_id, out_dir, 'prec', tag)
        filepath = pp5.get_resource_path(filepath.parent, filepath.name)
        os.makedirs(filepath.parent, exist_ok=True)

        with open(str(filepath), 'wb') as f:
            pickle.dump(self, f, protocol=4)

        LOGGER.info(f'Wrote {self} to {filepath}')
        return filepath

    def _find_dna_alignment(self, pdb_aa_seq: str, max_ena: int) \
            -> Tuple[SeqRecord, Dict[int, str]]:
        # Find cross-refs in ENA
        ena_molecule_types = ('mrna', 'genomic_dna')
        ena_ids = unp.find_ena_xrefs(self.unp_rec, ena_molecule_types)

        # Map id to sequence by fetching from ENA API
        ena_seqs = []
        for i, ena_id in enumerate(ena_ids):
            try:
                ena_seqs.append(ena.ena_seq(ena_id))
            except IOError as e:
                LOGGER.warning(f"{self}: Invalid ENA id {ena_id}")
            if max_ena is not None and i > max_ena:
                LOGGER.warning(f"{self}: Over {max_ena} ENA ids, "
                               f"skipping")
                break

        aligner = PairwiseAligner(substitution_matrix=BLOSUM80,
                                  open_gap_score=-10, extend_gap_score=-0.5)
        alignments = []
        for seq in ena_seqs:
            # Handle case of DNA sequence with incomplete codons
            if len(seq) % 3 != 0:
                if seq[-3:].seq in STOP_CODONS:
                    seq = seq[-3 * (len(seq) // 3):]
                else:
                    seq = seq[:3 * (len(seq) // 3)]

            # Translate to AA sequence and align to the PDB sequence
            translated = seq.translate(stop_symbol='')
            alignment = aligner.align(pdb_aa_seq, translated.seq)
            alignments.append((seq, alignment))

        if len(alignments) == 0:
            raise ProteinInitError(f"Can't find ENA id for {self.unp_id}")

        # Sort alignments by negative score (we want the highest first)
        sorted_alignments = sorted(alignments, key=lambda x: -x[1].score)

        # Print best-matching alignment
        best_ena, best_alignments = sorted_alignments[0]
        best_alignment = best_alignments[0]
        LOGGER.info(f'{self}: ENA ID = {best_ena.id}')
        LOGGER.info(f'{self}: Translated DNA to PDB alignment '
                    f'(norm_score='
                    f'{best_alignments.score / len(pdb_aa_seq):.2f}, '
                    f'num={len(best_alignments)})\n'
                    f'{str(best_alignment).strip()}')

        # Map each AA to a dict of (codon->count)
        idx_to_codons = {}
        for ena_seq, multi_alignment in sorted_alignments:
            alignment = multi_alignment[0]  # multiple equivalent alignments
            # Indices in the DNA AA seq which are aligned to the PDB AA seq
            aligned_idx_pdb_aa, aligned_idx_dna_aa = alignment.aligned
            dna_seq = str(ena_seq.seq)

            for i in range(len(aligned_idx_dna_aa)):
                # Indices of current matching segment of amino acids from the
                # PDB record and the translated DNA
                pdb_aa_start, pdb_aa_stop = aligned_idx_pdb_aa[i]
                dna_aa_start, dna_aa_stop = aligned_idx_dna_aa[i]

                for offset, k in enumerate(range(dna_aa_start, dna_aa_stop)):
                    k *= 3
                    codon = dna_seq[k:k + 3]
                    pdb_idx = offset + pdb_aa_start

                    # Skip "unknown" AAs - we put them there to represent
                    # non-standard AAs.
                    if pdb_aa_seq[pdb_idx] == UNKNOWN_AA:
                        continue

                    # List of codons at current index
                    codon_dict = idx_to_codons.get(pdb_idx, OrderedDict())

                    # Check if the codon actually matched the AA at the
                    # corresponding location. Sometimes there may be
                    # mismatches ('.') because the DNA alignment isn't perfect.
                    # In such a case we'll set the codon to None to specify we
                    # don't know what codon encoded the AA in the PDB sequence.
                    aa_name = CODON_TABLE.get(codon, None)
                    matches = aa_name == pdb_aa_seq[pdb_idx]
                    codon = codon if matches else UNKNOWN_CODON

                    # Map each codon to number of times seen
                    codon_dict[codon] = codon_dict.get(codon, 0) + 1
                    idx_to_codons[pdb_idx] = codon_dict

        return best_ena, idx_to_codons

    def _find_pdb_xref(self, ref_pdb_id) -> Tuple[str, str]:
        ref_pdb_id, ref_chain_id, ent_id = pdb.split_id_with_entity(ref_pdb_id)
        if not ref_chain_id:
            if ent_id is not None and self.numeric_chain:
                # In rare cases the chain is a number and indistinguishable
                # from entity. Handle this case only if explicitly
                # requested.
                ref_chain_id = ent_id
            else:
                ref_chain_id = ''

        ref_pdb_id, ref_chain_id = ref_pdb_id.upper(), ref_chain_id.upper()

        xrefs = unp.find_pdb_xrefs(self.unp_rec, method='x-ray')

        # We'll sort the PDB entries according to multiple criteria based on
        # the resolution, number of chains and sequence length.
        def sort_key(xref: unp.UNPPDBXRef):
            id_cmp = xref.pdb_id.upper() != ref_pdb_id
            chain_cmp = xref.chain_id.upper() != ref_chain_id
            seq_len_diff = abs(xref.seq_len - self.unp_rec.sequence_length)
            # The sort key for PDB entries
            # First, if we have a matching id to the reference PDB id we take
            # it. Otherwise, we take the best match according to seq len and
            # resolution.
            return id_cmp, chain_cmp, seq_len_diff, xref.resolution

        xrefs = sorted(xrefs, key=sort_key)
        if not xrefs:
            msg = f"No PDB cross-refs for {self.unp_id}"
            if self.strict_unp_xref:
                raise ProteinInitError(msg)
            elif not ref_chain_id:
                raise ProteinInitError(f"{msg} and no chain provided in ref")
            else:
                LOGGER.warning(f'{msg}, using ref {ref_pdb_id}:{ref_chain_id}')
                return ref_pdb_id, ref_chain_id

        # Get best match according to sort key and return its id.
        xref = xrefs[0]
        LOGGER.info(f'{self.unp_id}: PDB XREF = {xref}')

        pdb_id = xref.pdb_id.upper()
        chain_id = xref.chain_id.upper()

        # Make sure we have a match with the Uniprot id. Id chain wasn't
        # specified, match only PDB ID, otherwise, both must match.
        if pdb_id != ref_pdb_id:
            msg = f"Reference PDB ID {ref_pdb_id} not found as " \
                  f"cross-reference for protein {self.unp_id}"
            if self.strict_unp_xref:
                raise ProteinInitError(msg)
            else:
                LOGGER.warning(msg)
                pdb_id = ref_pdb_id

        if ref_chain_id and chain_id != ref_chain_id:
            msg = f"Reference chain {ref_chain_id} of PDB ID {ref_pdb_id} not" \
                  f"found as cross-reference for protein {self.unp_id}. " \
                  f"Did you mean chain {chain_id}?"
            if self.strict_unp_xref:
                raise ProteinInitError(msg)
            else:
                LOGGER.warning(msg)
                chain_id = ref_chain_id

        return pdb_id.upper(), chain_id.upper()

    def _get_dihedral_estimators(self, est_name: str, est_args: dict):
        est_name = est_name.lower() if est_name else est_name
        est_args = {} if est_args is None else est_args

        if not est_name in {None, '', 'erp', 'mc'}:
            raise ProteinInitError(
                f'Unknown dihedral estimation method {est_name}')

        unit_cell = pdb.PDBUnitCell(self.pdb_id, struct_d=self.pdb_dict)
        args = dict(isotropic=False, n_samples=100, skip_omega=True)
        args.update(est_args)

        if est_name == 'mc':
            d_est = DihedralAnglesMonteCarloEstimator(unit_cell, **args)
        elif est_name == 'erp':
            d_est = DihedralAnglesUncertaintyEstimator(unit_cell, **args)
        else:
            d_est = DihedralAnglesEstimator(**args)

        b_est = AtomLocationUncertainty(backbone_only=True, unit_cell=None,
                                        isotropic=True)
        return d_est, b_est

    def __iter__(self) -> Iterator[ResidueRecord]:
        return iter(self._residue_recs)

    def __getitem__(self, item: int) -> ResidueRecord:
        return self._residue_recs[item]

    def __repr__(self):
        return f'({self.unp_id}, {self.pdb_id})'

    def __getstate__(self):
        # Prevent serializing Bio objects
        state = self.__dict__.copy()
        for attr in self._SKIP_SERIALIZE:
            del state[attr]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        for attr in self._SKIP_SERIALIZE:
            self.__setattr__(attr, None)

    def __len__(self):
        return len(self._residue_recs)

    def __eq__(self, other):
        if self is other:
            return True
        if not isinstance(other, ProteinRecord):
            return False
        if self.pdb_id != other.pdb_id:
            return False
        if self.unp_id != other.unp_id:
            return False
        if len(self) != len(other):
            return False
        return all(map(lambda x: x[0] == x[1], zip(self, other)))


class ProteinGroup(object):
    """
    A ProteinGroup represents a group of protein structures which are
    similar in terms of sequence and structure, but may belong to different
    proteins (Uniprot IDs).

    A group is defined by a reference structure.
    All proteins in the group are aligned to the reference and different
    residue pairs are created. The pairs have different types based on the
    properties of the alignment (variant/same/silent/mutation/alteration).

    This class allows creation of Pairwise and Pointwise datasets for the
    structures in the group.
    """
    DEFAULT_EXPR_SYS = pp5.get_config('DEFAULT_EXPR_SYS')
    DEFAULT_RES = pp5.get_config('DEFAULT_RES')

    @classmethod
    def from_pdb_ref(cls, ref_pdb_id: str,
                     expr_sys_query: Union[str, PDBQuery] = DEFAULT_EXPR_SYS,
                     resolution_query: Union[float, PDBQuery] = DEFAULT_RES,
                     blast_e_cutoff: float = 1.,
                     blast_identity_cutoff: float = 30.,
                     **kw_for_init) -> ProteinGroup:
        """
        Creates a ProteinGroup given a reference PDB ID.
        Performs a query combining expression system, resolution and
        sequence matching (BLAST) to find PDB IDs of the group.
        Then initializes a ProteinGroup with the PDB IDs obtained from the
        query.

        :param ref_pdb_id: PDB ID of reference structure. Should include chain.
        :param expr_sys_query: Expression system query object or a a string
        containing the organism name.
        :param resolution_query: Resolution query or a number specifying the
        maximal resolution value.
        :param blast_e_cutoff: Expectation value cutoff parameter for BLAST.
        :param blast_identity_cutoff: Identity cutoff parameter for BLAST.
        :param kw_for_init: Keyword args for ProteinGroup.__init__()
        :return: A ProteinGroup for the given reference id.
        """

        ref_pdb_id = ref_pdb_id.upper()
        ref_pdb_base_id, ref_chain = pdb.split_id(ref_pdb_id)
        if not ref_chain:
            raise ValueError('Must provide chain for reference')

        if isinstance(expr_sys_query, str):
            expr_sys_query = pdb.PDBExpressionSystemQuery(expr_sys_query)

        if isinstance(resolution_query, (int, float)):
            resolution_query = pdb.PDBResolutionQuery(max_res=resolution_query)

        sequence_query = pdb.PDBSequenceQuery(
            pdb_id=ref_pdb_id, e_cutoff=blast_e_cutoff,
            identity_cutoff=blast_identity_cutoff
        )

        composite_query = pdb.PDBCompositeQuery(
            expr_sys_query, resolution_query, sequence_query
        )

        pdb_entities = set(composite_query.execute())
        LOGGER.info(f'Initializing ProteinGroup for {ref_pdb_id} with '
                    f'{len(pdb_entities)} query results...')

        pgroup = cls.from_query_ids(ref_pdb_id, pdb_entities, **kw_for_init)
        LOGGER.info(f'{pgroup}: '
                    f'#unp_ids={pgroup.num_unique_proteins} '
                    f'#structures={pgroup.num_query_structs} '
                    f'#matches={pgroup.num_matches}')

        return pgroup

    @classmethod
    def from_query_ids(cls, ref_pdb_id: str, query_pdb_ids: Iterable[str],
                       **kw_for_init) -> ProteinGroup:
        return cls(ref_pdb_id, query_pdb_ids, **kw_for_init)

    @classmethod
    def from_structures_csv(cls, ref_pdb_id: str,
                            indir=pp5.out_subdir('pgroup'), tag=None,
                            **kw_for_init) -> ProteinGroup:
        """
        Creates a ProteinGroup based on the structures specified in the CSV
        group file. This file is the one generated by to_csv with
        type='struct'.
        :param ref_pdb_id: Reference PDB id.
        :param indir: Folder to search for the group file in.
        :param tag: The tag given to the file, if any.
        :param kw_for_init: Extra args for the ProteinGroup __init__()
        :return: A ProteinGroup
        """
        tag = f'structs-{tag}' if tag else 'structs'
        filepath = _tagged_filepath(ref_pdb_id, indir, 'csv', tag)
        if not filepath.is_file():
            raise ProteinInitError(f'ProteinGroup structure CSV not found:'
                                   f'{filepath}')
        df = pd.read_csv(filepath, header=0, index_col=0)
        query_pdb_ids = list(df['pdb_id'])
        return cls.from_query_ids(ref_pdb_id, query_pdb_ids, **kw_for_init)

    def __init__(self, ref_pdb_id: str, query_pdb_ids: Iterable[str],
                 context_len: int = 1, prec_cache=False,
                 sa_outlier_cutoff: float = 2.,
                 sa_max_all_atom_rmsd: float = 2.,
                 sa_min_aligned_residues: int = 50,
                 b_max: float = math.inf,
                 angle_aggregation='circ',
                 strict_pdb_xref=True, strict_unp_xref=False,
                 parallel=True):
        """
        Creates a ProteinGroup based on a reference PDB ID, and a sequence of
        query PDB IDs. Structural alignment will be performed, and some
        query structures may be rejected.
        :param ref_pdb_id: Reference structure PDB ID.
        :param query_pdb_ids: List of PDB IDs of query structures.
        :param context_len: Number of stars required around an aligmed AA
        pair to consider that pair for a match.
        :param prec_cache:  Whether to load ProteinRecords from cache if
        available.
        :param sa_outlier_cutoff: RMS cutoff for determining outliers in
        structural alignment.
        :param sa_max_all_atom_rmsd: Maximal allowed average RMSD
        after structural alignment to include a structure in a group.
        :param sa_min_aligned_residues: Minimal number of aligned residues (stars)
        required to include a structure in a group.
        :param b_max: Maximal b-factor a residue can have
        (backbone-atom average) in order for it to be included in a match
        group.
        :param angle_aggregation: Method for angle-aggregation of matching
        query residues of each reference residue. Options are
        'circ' - Circular mean;
        'frechet' - Frechet centroid;
        'max_res' - No aggregation, take angle of maximal resolution structure
        :param strict_pdb_xref: Whether to require that the given PDB ID
        maps uniquely to only one Uniprot ID.
        :param strict_unp_xref: Whether to require that there exist a PDB
        cross-ref for the given Uniprot ID.
        :param parallel: Whether to process query structures in parallel using
        the global worker process pool.
        """
        self.ref_pdb_id = ref_pdb_id.upper()
        self.ref_pdb_base_id, self.ref_pdb_chain = pdb.split_id(ref_pdb_id)
        if not self.ref_pdb_chain:
            raise ProteinInitError('ProteinGroup reference structure must '
                                   'specify the chain id.')

        ref_pdb_dict = pdb.pdb_dict(self.ref_pdb_base_id)
        ref_pdb_meta = pdb.PDBMetadata(self.ref_pdb_base_id,
                                       struct_d=ref_pdb_dict)
        if self.ref_pdb_chain not in ref_pdb_meta.chain_entities:
            raise ProteinInitError(f'Unknown PDB entity for {self.ref_pdb_id}')

        self.ref_pdb_entity = ref_pdb_meta.chain_entities[self.ref_pdb_chain]

        if context_len < 1:
            raise ProteinInitError("Context size must be > 1")

        self.context_size = context_len
        self.sa_outlier_cutoff = sa_outlier_cutoff
        self.sa_max_all_atom_rmsd = sa_max_all_atom_rmsd
        self.sa_min_aligned_residues = sa_min_aligned_residues
        self.b_max = b_max
        self.prec_cache = prec_cache
        self.strict_pdb_xref = strict_pdb_xref
        self.strict_unp_xref = strict_unp_xref

        angle_aggregation_methods = {
            'circ': self._aggregate_fn_circ,
            'frechet': self._aggregate_fn_frechet,
            'max_res': self._aggregate_fn_best_res
        }
        if angle_aggregation not in angle_aggregation_methods:
            raise ProteinInitError(
                f'Unknown aggregation method: {angle_aggregation}, '
                f'must be one of {tuple(angle_aggregation_methods.keys())}'
            )
        self.angle_aggregation = angle_aggregation
        aggregation_fn = angle_aggregation_methods[angle_aggregation]

        # Make sure that the query IDs are valid
        query_pdb_ids = list(query_pdb_ids)  # to allow multiple iteration
        if not query_pdb_ids:
            raise ProteinInitError('No query PDB IDs provided')
        try:
            split_ids = [pdb.split_id_with_entity(query_id)
                         for query_id in query_pdb_ids]

            # Make sure all query ids have either chain or entity
            if not all(map(lambda x: x[1] or x[2], split_ids)):
                raise ValueError('Must specify chain or entity for all '
                                 'structures')

            # If there's no entry for the reference, add it
            if not any(map(lambda x: x[0] == self.ref_pdb_base_id, split_ids)):
                query_pdb_ids.insert(0, self.ref_pdb_id)

            ref_with_entity = f'{self.ref_pdb_base_id}:{self.ref_pdb_entity}'

            def is_ref(q_pdb_id: str):
                q_pdb_id = q_pdb_id.upper()
                # Check if query is identical to teh reference structure. We
                # need to check both with chain and entity.
                return q_pdb_id == ref_with_entity or \
                       q_pdb_id == self.ref_pdb_id

            def sort_key(q_pdb_id: str):
                return not is_ref(q_pdb_id), q_pdb_id

            # Sort query ids so that the reference is first, then by id
            query_pdb_ids = sorted(query_pdb_ids, key=sort_key)

            # Check whether reference structure is in the query list
            if is_ref(query_pdb_ids[0]):
                # In case it's specified as an entity, replace it with the
                # chain id, because we use this PDB IDs with chain later for
                # comparison to the reference structure.
                query_pdb_ids[0] = self.ref_pdb_id
            else:
                # If the reference is not included, add it.
                query_pdb_ids.insert(0, self.ref_pdb_id)

        except ValueError as e:
            # can also be raised by split, if format invalid
            raise ProteinInitError(str(e)) from None

        self.ref_prec = ProteinRecord.from_pdb(
            self.ref_pdb_id, cache=self.prec_cache,
            strict_pdb_xref=self.strict_pdb_xref,
            strict_unp_xref=self.strict_unp_xref,
            pdb_dict=ref_pdb_dict,
        )

        # Align all query structure residues to the reference structure
        # Process different query structures in parallel
        align_fn = self._align_query_residues_to_ref
        if parallel:
            with global_pool() as pool:
                q_aligned = pool.map(align_fn, query_pdb_ids)
        else:
            q_aligned = list(map(align_fn, query_pdb_ids))

        self.ref_matches: Dict[int, Dict[str, ResidueMatch]] = OrderedDict()
        self.ref_groups: Dict[int, List[ResidueMatchGroup]] = OrderedDict()
        self.query_pdb_ids = []
        self.query_pdb_to_prec: Dict[str, ProteinRecord] = OrderedDict()
        self.query_pdb_to_sa: Dict[str, StructuralAlignment] = OrderedDict()

        # Go over aligned queries. Save the matching residues and group them.
        for qa in q_aligned:
            if not qa:
                continue

            # qa contains a dict of matches for one query structure
            q_prec, q_alignment, q_matches_dict = qa

            for ref_res_idx, res_matches in q_matches_dict.items():
                self.ref_matches.setdefault(ref_res_idx, OrderedDict())
                self.ref_matches[ref_res_idx].update(res_matches)

            self.query_pdb_ids.append(q_prec.pdb_id)
            self.query_pdb_to_prec[q_prec.pdb_id] = q_prec
            self.query_pdb_to_sa[q_prec.pdb_id] = q_alignment

        LOGGER.info(f'{self}: Grouping matches with angle_aggregation='
                    f'{self.angle_aggregation}')
        self.ref_groups = self._group_matches(aggregation_fn)

    @property
    def num_query_structs(self) -> int:
        """
        :return: Number of query structures in this pgroup.
        """
        return len(self.query_pdb_to_prec)

    @property
    def num_matches(self) -> int:
        """
        :return: Total number of matches in this pgroup.
        """
        return sum(len(res_m) for res_m in self.ref_matches.values())

    @property
    def num_unique_proteins(self):
        """
        :return: Total number of unique proteins in this pgroup (unique
        Uniprot IDs).
        """
        unp_ids = set()
        for ref_idx, match_groups in self.ref_groups.items():
            for match_group in match_groups:
                unp_ids.add(match_group.unp_id)
        return len(unp_ids)

    @property
    def match_counts(self) -> Dict[str, int]:
        """
        :return: Number of matches per type.
        """
        counts = {t.name: 0 for t in ResidueMatchType}
        for ref_idx, matches in self.ref_matches.items():
            for match in matches.values():
                counts[match.type.name] += 1
        return counts

    @property
    def unp_to_pdb(self) -> Dict[str, Set[str]]:
        """
        :return: A mapping from Uniprot ID to a the set of structures
        which represent it within this pgroup.
        """

        # Make sure to include reference...
        res = {
            self.ref_prec.unp_id: {self.ref_pdb_id}
        }

        for q_pdb_id, q_prec in self.query_pdb_to_prec.items():
            q_unp_id = q_prec.unp_id
            pdb_id_set: set = res.setdefault(q_unp_id, set())
            pdb_id_set.add(q_pdb_id)

        return res

    def to_pointwise_dataframe(self, with_ref_id=False, with_neighbors=False):
        """
        :param with_ref_id: Whether to include a column with the reference
        structure Uniprot id. This is generally redundant as it
        will have the same value for all rows. However may be useful when
        creating multiple pointwise dataframes from different pgroups.
        :param with_neighbors: Whether to include the neighbors of each
        location (one before and one after) in each row. This will also add
        redundancy to the output, since this can be calculated from the
        regular version.
        :return: A dataframe containing the aligned residue VARIANT match
        groups at each reference index.
        This is very similar to the 'groups' dataframe, but with only
        VARIANT groups, and optionally with a comparison to the previous and
        next aligned residues.
        """
        df_index = []
        df_data = []

        # Get the VARIANT match group at each reference index
        variant_groups: Dict[int, ResidueMatchGroup] = {
            i: g[0] for i, g in self.ref_groups.items()
        }

        # Number of reference structure residues
        n_ref = len(self.ref_prec)

        # Create consecutive indices of residues.
        if with_neighbors:
            # Currently we use a default context of 1 residue before and after.
            # We can change this index later on if more/less context is needed.
            idx = zip(range(0, n_ref - 2), range(1, n_ref - 1),
                      range(2, n_ref))
        else:
            idx = zip(range(n_ref))  # produces [(0,), (1,) ...]

        if with_ref_id:
            curr_index_base = (self.ref_prec.unp_id,)
            df_index_names = ['unp_id', 'ref_idx', 'names']
        else:
            curr_index_base = tuple()
            df_index_names = ['ref_idx', 'names']

        for iii in idx:
            # Get VARIANTS at these indices
            # Make sure we have all consecutive residues for these indices
            vgroups = [variant_groups.get(i) for i in iii]
            if not all(vgroups):
                continue

            # We index the resulting dataframe by (unp_id, location, AA names)
            ref_idx = iii[0] + len(iii) // 2  # use central index
            names = str.join('', [v.name for v in vgroups])
            curr_index = curr_index_base + (ref_idx, names)

            # Get match group data from consecutive variant groups
            curr_data = {}
            for j, vgroup in enumerate(vgroups):
                # Relative idx is e.g. -1/+0/+1, relative to ref_idx
                rel_idx = f'{j - len(iii) // 2:+0d}' if len(iii) > 1 else ''
                angles = {
                    f'{k}{rel_idx}': v for k, v in
                    vgroup.avg_phipsi.as_dict(
                        degrees=True, skip_omega=True, with_std=True
                    ).items()
                }
                curr_data.update({
                    f'codon{rel_idx}': vgroup.codon,
                    f'codon_opts{rel_idx}': str.join('/', vgroup.codon_opts),
                    f'secondary{rel_idx}': str.join('/', vgroup.secondary),
                    f'group_size{rel_idx}': vgroup.group_size,
                    **angles
                })

            df_index.append(curr_index)
            df_data.append(curr_data)

        df_index = pd.MultiIndex.from_tuples(df_index, names=df_index_names)
        df = pd.DataFrame(data=df_data, index=df_index)
        return df

    def to_pairwise_dataframe(self, with_ref_id=False):
        """
        :param with_ref_id: Whether to include a column with the reference
        structure ids (pdb and unp). This is generally redundant as it will
        have the same value for all rows. However may be useful when
        creating multiple pairwise dataframes from different pgroups.
        :return: A dataframe containing the aligned residue match groups (i.e.,
        matches grouped by Uniprot ID and codon), but compared to the VARIANT
        group at that reference index.
        This is very similar to the 'groups' dataframe, but here each line
        is a *pair* of ResidueMatchGroups at an index: the first is the
        VARIANT group, and the second is another type of group (e.g. SILENT).

        A crucial difference between this and the groups dataframe is that
        this will not include residues where only a VARIANT group exists.
        """
        df_index = []
        df_data = []

        ref_data_base = {}
        if with_ref_id:
            ref_data_base['ref_pdb_id'] = self.ref_pdb_id
            ref_data_base['ref_unp_id'] = self.ref_prec.unp_id

        for ref_idx, grouped_matches in self.ref_groups.items():
            grouped_matches: List[ResidueMatchGroup]

            # Separate the VARIANT group from the others as the reference group
            ref_group: Optional[ResidueMatchGroup] = None
            other_groups: List[ResidueMatchGroup] = []
            for group in grouped_matches:
                if group.match_type == ResidueMatchType.VARIANT:
                    ref_group = group
                else:
                    other_groups.append(group)
            assert ref_group is not None

            # Get angles of variant group
            ref_avg_phipsi = ref_group.avg_phipsi.as_dict(
                degrees=True, skip_omega=True, with_std=True
            )
            ref_avg_phipsi = {
                f'ref_{k}': v
                for k, v in ref_avg_phipsi.items()
            }

            ref_data = ref_data_base.copy()
            ref_data.update({
                'ref_codon': ref_group.codon,
                'ref_name': ref_group.name,
                'ref_codon_opts': ref_group.codon_opts_str,
                'ref_secondary': ref_group.secondary,
                'ref_group_size': ref_group.group_size,
                'ref_norm_factor': ref_group.norm_factor,
                **ref_avg_phipsi,
                'ref_pdb_ids': ref_group.pdb_ids_str,
                'ref_idxs': ref_group.idxs_str,
                'ref_res_ids': ref_group.res_ids_str,
                'ref_contexts': ref_group.contexts_str,
                'ref_angles': ref_group.angles_str,
            })
            ref_data.update(ref_avg_phipsi)

            for match_group in other_groups:
                data = ref_data.copy()
                match_group_avg_phipsi = match_group.avg_phipsi.as_dict(
                    degrees=True, skip_omega=True, with_std=True
                )

                data.update({
                    'type': match_group.match_type.name,
                    'unp_id': match_group.unp_id,
                    'codon': match_group.codon,
                    'name': match_group.name,
                    'codon_opts': match_group.codon_opts_str,
                    'secondary': match_group.secondary,
                    'group_size': match_group.group_size,
                    'ang_dist': match_group.ang_dist,
                    'norm_factor': match_group.norm_factor,
                    **match_group_avg_phipsi,
                    'pdb_ids': match_group.pdb_ids_str,
                    'idxs': match_group.idxs_str,
                    'res_ids': match_group.res_ids_str,
                    'contexts': match_group.contexts_str,
                    'angles': match_group.angles_str,
                })

                df_data.append(data)
                df_index.append(ref_idx)

        df_index = pd.Index(data=df_index, name='ref_idx')
        df = pd.DataFrame(data=df_data, index=df_index)
        return df

    def to_groups_dataframe(self) -> pd.DataFrame:
        """
        :return: A DataFrame containing the aligned residue matches, grouped
        by Uniprot ID and codon for each residue in the reference structure.
        For each reference residues index, there will be one or more
        lines containing the information from the corresponding
        ResidueMatchGroups.
        """
        df_index = []
        df_data = []

        for ref_idx, grouped_matches in self.ref_groups.items():
            grouped_matches: List[ResidueMatchGroup]

            for match_group in grouped_matches:
                idx = (ref_idx, match_group.unp_id, match_group.codon)
                data = {
                    'type': match_group.match_type.name,
                    'name': match_group.name,
                    'codon_opts': match_group.codon_opts_str,
                    'secondary': match_group.secondary,
                    'group_size': match_group.group_size,
                    'ang_dist': match_group.ang_dist,
                    'pdb_ids': match_group.pdb_ids_str,
                    'idxs': match_group.idxs_str,
                    'res_ids': match_group.res_ids_str,
                    'contexts': match_group.contexts_str,
                }
                data.update(match_group.avg_phipsi.as_dict(
                    degrees=True, skip_omega=True, with_std=True)
                )
                df_index.append(idx)
                df_data.append(data)

        df_index_names = ['ref_idx', 'unp_id', 'codon']
        df_index = pd.MultiIndex.from_tuples(df_index, names=df_index_names)
        df = pd.DataFrame(data=df_data, index=df_index)
        return df

    def to_residue_dataframe(self) -> pd.DataFrame:
        """
        :return: A DataFrame containing the aligned residues matches for
        each residue in the reference structure.
        """
        df_index = []
        df_data = []

        for ref_idx, matches in self.ref_matches.items():
            for query_pdb_id, match in matches.items():
                q_prec = self.query_pdb_to_prec[query_pdb_id]

                data = {'unp_id': q_prec.unp_id}
                data.update(match.as_dict(skip_omega=True))
                data['type'] = match.type.name  # change from number to name

                df_data.append(data)
                df_index.append((ref_idx, query_pdb_id))

        df_index = pd.MultiIndex.from_tuples(df_index,
                                             names=['ref_idx', 'query_pdb_id'])
        df = pd.DataFrame(data=df_data, index=df_index)
        return df

    def to_struct_dataframe(self) -> pd.DataFrame:
        """
        :return: A DataFrame containing metadata about each of the
        structures in the ProteinGroup.
        """
        data = []
        for q_pdb_id in self.query_pdb_ids:
            q_prec = self.query_pdb_to_prec[q_pdb_id]
            q_alignment = self.query_pdb_to_sa[q_pdb_id]
            data.append({
                'unp_id': q_prec.unp_id, 'pdb_id': q_prec.pdb_id,
                'resolution': q_prec.pdb_meta.resolution,
                'struct_rmse': q_alignment.rmse,
                'n_stars': q_alignment.n_stars,
                'seq_len': len(q_alignment.ungapped_seq_2),  # seq2 is query
                'description': q_prec.pdb_meta.description,
                'src_org': q_prec.pdb_meta.src_org,
                'src_org_id': q_prec.pdb_meta.src_org_id,
                'host_org': q_prec.pdb_meta.host_org,
                'host_org_id': q_prec.pdb_meta.host_org_id,
                'ligands': q_prec.pdb_meta.ligands,
                'space_group': q_prec.pdb_meta.space_group,
                'r_free': q_prec.pdb_meta.r_free,
                'r_work': q_prec.pdb_meta.r_work,
                'cg_ph': q_prec.pdb_meta.cg_ph,
                'cg_temp': q_prec.pdb_meta.cg_temp,
            })

        df = pd.DataFrame(data)
        df['ref_group'] = df['unp_id'] == self.ref_prec.unp_id
        df = df.astype({'src_org_id': "Int32", 'host_org_id': "Int32"})
        df.sort_values(by=['ref_group', 'unp_id', 'struct_rmse'],
                       ascending=[False, True, True], inplace=True,
                       ignore_index=True)
        return df

    def to_csv(self, out_dir=pp5.out_subdir('pgroup'), types=('all',),
               tag=None) -> Dict[str, Path]:
        """
        Writes one or more CSV files describing this protein group.
        :param out_dir: Output directory.
        :param types: What CSV types to write. Can be either 'all',
        or a list containing a combination of:
        'structs' - metadata about each structure in the pgroup
        'residues' - per-residue alignment between ref structure and queries
        'groups' - per-residue alignment data grouped by unp_id and codon
        'pairwise' - like groups but each group is compared to VARIANT group
        :param tag: Optional tag to add to the output filenames.
        :return: A dict from type to the path of the file written.
        Keys can be one of the types specified above.
        """

        df_funcs = {
            'structs': self.to_struct_dataframe,
            'residues': self.to_residue_dataframe,
            'groups': self.to_groups_dataframe,
            'pairwise': self.to_pairwise_dataframe
        }

        os.makedirs(str(out_dir), exist_ok=True)
        if isinstance(types, str):
            types = [types]

        if not types or types[0] == 'all':
            types = df_funcs.keys()

        filepaths = {}
        for csv_type in types:
            df_func = df_funcs.get(csv_type)
            if not df_func:
                LOGGER.warning(f'{self}: Unknown ProteinGroup CSV type'
                               f' {csv_type}')
                continue

            df = df_func()
            tt = f'{csv_type}-{tag}' if tag else csv_type
            filepath = _tagged_filepath(self.ref_pdb_id, out_dir, 'csv', tt)

            df.to_csv(filepath, na_rep='', header=True, index=True,
                      encoding='utf-8', float_format='%.3f')

            filepaths[csv_type] = filepath

        LOGGER.info(f'Wrote {self} to {list(map(str, filepaths.values()))}')
        return filepaths

    def _align_query_residues_to_ref(self, q_pdb_id: str):
        try:
            return self._align_query_residues_to_ref_inner(q_pdb_id)
        except Exception as e:
            LOGGER.error(f"Failed to align pgroup reference "
                         f"{self.ref_pdb_id} to query {q_pdb_id}: "
                         f"{e.__class__.__name__}: {e}", exc_info=e)
            return None

    def _align_query_residues_to_ref_inner(
            self, q_pdb_id: str
    ) -> Optional[Tuple[ProteinRecord, StructuralAlignment,
                        Dict[int, Dict[str, ResidueMatch]]]]:
        try:
            q_prec = ProteinRecord.from_pdb(
                q_pdb_id, cache=self.prec_cache,
                strict_pdb_xref=self.strict_pdb_xref,
                strict_unp_xref=self.strict_unp_xref,
            )
        except ProteinInitError as e:
            LOGGER.error(f'{self}: Failed to create prec for query structure:'
                         f' {e}')
            return None

        alignment = self._struct_align_filter(q_prec)
        if alignment is None:
            return None

        r_seq_pymol = alignment.aligned_seq_1
        q_seq_pymol = alignment.aligned_seq_2
        stars = alignment.aligned_stars
        n = len(stars)

        # Map from index in the structural alignment to the index within each
        # sequence we got from pymol
        r_idx, q_idx = -1, -1
        stars_to_pymol_idx = {}
        for i in range(n):
            if r_seq_pymol[i] not in PSA_GAP:
                r_idx += 1
            if q_seq_pymol[i] not in PSA_GAP:
                q_idx += 1
            stars_to_pymol_idx[i] = (r_idx, q_idx)

        # Map from pymol sequence index to the index in the precs
        # Need to do another pairwise alignment for this
        # Align the ref and query seqs from our prec and pymol
        r_pymol_to_prec = self._align_pymol_to_prec(self.ref_prec, r_seq_pymol)
        q_pymol_to_prec = self._align_pymol_to_prec(q_prec, q_seq_pymol)

        # Context size is the number of stars required on EACH SIDE of a match
        ctx = self.context_size
        stars_ctx = '*' * self.context_size

        matches = OrderedDict()
        for i in range(ctx, n - ctx):
            # Check that context around i has only stars
            point = stars[i]
            pre, post = stars[i - ctx:i], stars[i + 1:i + 1 + ctx]
            if pre != stars_ctx or post != stars_ctx:
                continue

            # We allow the AA at the match point to differ (mutation and
            # alteration), but if it's the same AA we require a star there
            # (structural alignment as well as sequence alignment)
            if r_seq_pymol[i] == q_seq_pymol[i] and point != '*':
                continue

            # We allow them to differ, but both must be an aligned AA,
            # not a gap symbol
            if r_seq_pymol[i] in PSA_GAP or q_seq_pymol[i] in PSA_GAP:
                continue

            # Now we need to convert i into the index in the prec of each
            # structure
            r_idx_pymol, q_idx_pymol = stars_to_pymol_idx[i]
            r_idx_prec = r_pymol_to_prec.get(r_idx_pymol)
            q_idx_prec = q_pymol_to_prec.get(q_idx_pymol)

            # The pymol seq index might be inside a gap in the alignment
            # between pymol and prec seq. Skip these gaps.
            if r_idx_prec is None or q_idx_prec is None:
                continue

            # Get the matching residues, and make sure they are usable:
            # We require known AA, codon, and bfactor within maximum value.
            r_res, q_res = self.ref_prec[r_idx_prec], q_prec[q_idx_prec]
            if r_res.name == UNKNOWN_AA or q_res.name == UNKNOWN_AA:
                continue
            if r_res.codon == UNKNOWN_CODON or q_res.codon == UNKNOWN_CODON:
                continue
            if r_res.bfactor > self.b_max or q_res.bfactor > self.b_max:
                continue

            # Make sure we got from i to the correct residues in the precs
            assert r_res.name == r_seq_pymol[i], i
            assert q_res.name == q_seq_pymol[i], i

            # Compute type of match
            pdb_match = self.ref_prec.pdb_id == q_prec.pdb_id
            unp_match = self.ref_prec.unp_id == q_prec.unp_id
            aa_match = r_res.name == q_res.name
            codon_match = r_res.codon == q_res.codon
            match_type = self._match_type(pdb_match, unp_match, aa_match,
                                          codon_match)
            if match_type is None:
                continue

            # Calculate angle distance between match and reference
            ang_dist = Dihedral.flat_torus_distance(r_res.angles, q_res.angles,
                                                    degrees=True)

            # Calculate full context length
            context_len = 0
            for d in range(1, min(i, n - 1 - i)):
                if stars[i - d] == stars[i + d] == '*':
                    context_len += 1
                else:
                    break

            # Save match object
            match = ResidueMatch.from_residue(
                q_res, q_idx_prec, match_type, ang_dist, context_len
            )
            res_matches = matches.setdefault(r_idx_prec, OrderedDict())
            res_matches[q_prec.pdb_id] = match

        return q_prec, alignment, matches

    def _struct_align_filter(self, q_prec: ProteinRecord) \
            -> Optional[StructuralAlignment]:
        """
        Performs structural alignment between the query and the reference
        structure. Rejects query structures which do not conform to the
        requires structural alignment parameters.
        :param q_pdb_id: Query PDB ID.
        :return: Alignment object, or None if query was rejected.
        """
        q_pdb_id = q_prec.pdb_id
        try:
            sa = StructuralAlignment.from_pdb(
                self.ref_pdb_id, q_pdb_id, cache=True,
                outlier_rejection_cutoff=self.sa_outlier_cutoff
            )
        except Exception as e:
            LOGGER.warning(f'{self}: Rejecting {q_pdb_id} due to failed '
                           f'structural alignment: {e.__class__.__name__} {e}')
            return None

        if sa.rmse > self.sa_max_all_atom_rmsd:
            LOGGER.info(
                f'{self}: Rejecting {q_pdb_id} due to insufficient structural '
                f'similarity, RMSE={sa.rmse:.3f}')
            return None

        if sa.n_stars < self.sa_min_aligned_residues:
            LOGGER.info(f'{self}: Rejecting {q_pdb_id} due to insufficient '
                        f'aligned residues, n_stars={sa.n_stars}')
            return None

        return sa

    @staticmethod
    def _align_pymol_to_prec(prec: ProteinRecord, pymol_seq: str) \
            -> Dict[int, int]:
        sa_r_seq = StructuralAlignment.ungap(pymol_seq)

        # Align prec and pymol sequences
        aligner = PairwiseAligner(substitution_matrix=BLOSUM80,
                                  open_gap_score=-10, extend_gap_score=-0.5)
        rr_alignments = aligner.align(prec.protein_seq.seq, sa_r_seq)

        # Take the alignment with shortest path (fewer gap openings)
        rr_alignments = sorted(rr_alignments, key=lambda a: len(a.path))
        rr_alignment = rr_alignments[0]
        rr_idx_prec, rr_idx_pymol = rr_alignment.aligned
        assert len(rr_idx_prec) == len(rr_idx_pymol)

        # Map pymol index to prec index
        pymol_to_prec = {}
        for j in range(len(rr_idx_prec)):
            prec_start, prec_end = rr_idx_prec[j]
            pymol_start, pymol_end = rr_idx_pymol[j]
            pymol_to_prec.update(zip(
                range(pymol_start, pymol_end + 1),
                range(prec_start, prec_end + 1)
            ))

        return pymol_to_prec

    @staticmethod
    def _match_type(pdb_match: bool, unp_match: bool, aa_match: bool,
                    codon_match: bool) -> Optional[ResidueMatchType]:
        if pdb_match:
            match_type = ResidueMatchType.REFERENCE
        elif unp_match:
            if aa_match:
                if codon_match:
                    match_type = ResidueMatchType.VARIANT
                else:
                    return None  # This is not a match!
            else:
                match_type = ResidueMatchType.ALTERATION
        else:
            if aa_match:
                if codon_match:
                    match_type = ResidueMatchType.SAME
                else:
                    match_type = ResidueMatchType.SILENT
            else:
                match_type = ResidueMatchType.MUTATION
        return match_type

    def _group_matches(self, aggregate_fn: Callable[[Dict], Dihedral]) \
            -> Dict[int, List[ResidueMatchGroup]]:
        """
        Groups the matches of each reference residue into subgroups by their
        unp_id and codon, and computes an aggregate angle statistic based on
        the structures in each subgroup.
        :param aggregate_fn: Function for angle aggregation.
        Should accept a dict from pdb_id to residue match and return an angle
        pair.
        :return: Mapping from residue id in the reference to a list of
        subgroups.
        """

        grouped_matches: Dict[int, List[ResidueMatchGroup]] = OrderedDict()

        for ref_res_idx, matches in self.ref_matches.items():
            ref_res = self.ref_prec[ref_res_idx]

            # Group matches of queries to current residue by unp_id and codon
            match_groups: Dict[Tuple[str, str], Dict[str, ResidueMatch]] = {}
            ref_group_idx = None
            for q_pdb_id, q_match in matches.items():
                q_prec = self.query_pdb_to_prec[q_pdb_id]
                unp_id = q_prec.unp_id
                codon = q_match.codon

                # Reference group will include the REFERENCE residue and zero
                # or more VARIANT residues. We want to calculate the angle
                # difference between all other groups and this group.
                if q_match.type == ResidueMatchType.REFERENCE:
                    ref_group_idx = (unp_id, codon)

                match_group_idx = (unp_id, codon)
                match_group = match_groups.setdefault(match_group_idx, {})
                match_group[q_pdb_id] = q_match

            # Compute reference group aggregate angles
            assert ref_group_idx is not None
            ref_group_avg_phipsi = aggregate_fn(match_groups[ref_group_idx])

            # Compute aggregate statistics in each group
            for match_group_idx, match_group in match_groups.items():
                (unp_id, codon) = match_group_idx
                match_group: Dict[str, ResidueMatch]

                # Save information about the structures in this group
                group_pdb_ids = tuple(match_group.keys())
                vs = ((m.idx, m.res_id, m.context, m.angles)
                      for m in match_group.values())
                idxs, res_ids, contexts, angles = [tuple(z) for z in zip(*vs)]

                # Calculate angle difference w.r.t. ref group
                if match_group_idx == ref_group_idx:
                    group_avg_phipsi = ref_group_avg_phipsi
                else:
                    group_avg_phipsi = aggregate_fn(match_group)
                ang_dist = Dihedral.flat_torus_distance(
                    ref_group_avg_phipsi, group_avg_phipsi, degrees=True)

                # Collect sets of features from the matches in the group
                vs = ((
                    m.type, m.name, m.secondary, m.codon_opts,
                ) for m in match_group.values())
                types, names, secondaries, opts = [set(z) for z in zip(*vs)]

                # Make sure all members of group have the same match type,
                # except the VARIANT group which should have one REFERENCE.
                # Also, AA name should be the same since codon is the same.
                if ResidueMatchType.REFERENCE in types:
                    types.remove(ResidueMatchType.REFERENCE)
                    types.add(ResidueMatchType.VARIANT)
                group_type = types.pop()
                group_aa_name = names.pop()
                assert len(types) == 0, types
                assert len(names) == 0, names

                # Get alternative possible codon options. Remove the
                # group codon to prevent redundancy.  Note that
                # UNKNOWN_CODON is a possibility, we leave it in.
                group_codon_opts = set(it.chain(*[o.split('/') for o in opts]))
                group_codon_opts.remove(codon)

                # Assign a SS to a group based on majority-vote. Since
                # we're dealing with different structures of the same protein,
                # they should have the same SS.
                group_secondary, _ = Counter(secondaries).most_common(1)[0]

                # Store the aggregate info
                ref_res_groups = grouped_matches.setdefault(ref_res_idx, [])
                ref_res_groups.append(ResidueMatchGroup(
                    unp_id, codon,
                    group_pdb_ids, idxs, res_ids, contexts, angles,
                    group_type, group_aa_name, group_codon_opts,
                    group_secondary, group_avg_phipsi, ang_dist
                ))

        return grouped_matches

    def _aggregate_fn_best_res(self, match_group: Dict[str, ResidueMatch]) \
            -> Dihedral:
        """
        Aggregator which selects the angles from the best resolution
        structure in a match group.
        :param match_group: Match group dict, keys are query pdb_ids.
        :return: The dihedral angles frmo the best-resolution structure.
        """

        def sort_key(q_pdb_id):
            return self.query_pdb_to_prec[q_pdb_id].pdb_meta.resolution

        q_pdb_id_best_res = sorted(match_group, key=sort_key)[0]
        return match_group[q_pdb_id_best_res].angles

    @staticmethod
    def _aggregate_fn_frechet(
            match_group: Dict[str, ResidueMatch]) -> Dihedral:
        """
        Aggregator which computes the Frechet mean of the dihedral angles in
        the group.
        :param match_group: Match group dict, keys are query pdb_ids.
        :return: The Frechet mean of the dihedral angles in
        the group.
        """
        return Dihedral.frechet_centroid(
            *[m.angles for m in match_group.values()]
        )

    @staticmethod
    def _aggregate_fn_circ(match_group: Dict[str, ResidueMatch]) -> Dihedral:
        """
        Aggregator which computes the circular mean of the dihedral angles in
        the group.
        :param match_group: Match group dict, keys are query pdb_ids.
        :return: The mean of the dihedral angles in the group.
        """
        return Dihedral.circular_centroid(
            *[m.angles for m in match_group.values()]
        )

    def __getitem__(self, ref_idx: int):
        """
        :param ref_idx: A residue index in the reference structure.
        :return: The match groups matching at that reference.
        """
        return self.ref_groups.get(ref_idx)

    def __repr__(self):
        return f'{self.__class__.__name__} {self.ref_pdb_id}'
