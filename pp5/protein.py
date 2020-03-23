from __future__ import annotations

import logging
import math
import os
import warnings
import enum
import pickle
from pathlib import Path
from typing import List, Tuple, Dict, Iterator, Callable, Any, Union, Iterable, \
    Generator, Optional
from collections import OrderedDict

import pandas as pd
from Bio.Align import PairwiseAligner, MultipleSeqAlignment
from Bio.Data import CodonTable
from Bio.PDB import PPBuilder
from Bio.PDB.Polypeptide import Polypeptide
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from pytest import approx

import pp5
from pp5.dihedral import Dihedral, DihedralAnglesEstimator, \
    DihedralAnglesUncertaintyEstimator, DihedralAnglesMonteCarloEstimator, \
    BFactorEstimator
from pp5.external_dbs import pdb, unp, ena
from pp5.external_dbs.pdb import PDBRecord
from pp5.external_dbs.unp import UNPRecord
from pp5.align import structural_align

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

    def __getstate__(self):
        d = self.__dict__.copy()
        d['angles'] = self.angles.__dict__
        return d

    def __setstate__(self, state):
        angles = Dihedral.empty()
        angles.__dict__.update(state['angles'])
        state['angles'] = angles
        self.__dict__.update(state)

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
    def from_residue(cls, res: ResidueRecord,
                     match_type: ResidueMatch.Type, diff_deg: float):
        return cls(match_type, diff_deg, **res.__dict__)

    def __init__(self, match_type: ResidueMatchType, diff_deg: float,
                 **res_rec_args):
        super().__init__(**res_rec_args)
        self.type = match_type
        self.ang_dist = diff_deg

    def __repr__(self):
        return f'{self.type.name}, diff={self.ang_dist:.2f}'


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
    def from_cache(pdb_id, cache_dir: Union[bool, str, Path] = None,
                   tag=None) -> Optional[ProteinRecord]:
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

        path = ProteinRecord._tagged_filepath(pdb_id, cache_dir, 'prec', tag)
        filename = path.name
        path = pp5.get_resource_path(cache_dir, filename)
        prec = None
        if path.is_file():
            with open(str(path), 'rb') as f:
                prec = pickle.load(f)
                LOGGER.info(f'Loaded cached ProteinRecord: {path}')
        return prec

    @classmethod
    def from_pdb(cls, pdb_id: str, pdb_dict: dict = None, cache=False,
                 cache_dir=pp5.PREC_DIR, **kwargs) -> ProteinRecord:
        """
        Given a PDB id (and optionally a chain), finds the
        corresponding Uniprot id, and returns a ProteinRecord object for
        that protein.
        :param pdb_id: The PDB id to query, with optional chain, e.g. '0ABC:D'.
        :param pdb_dict: Optional structure dict for the PDB record, in case it
        was already parsed.
        :param cache: Whether to load prec from cache if available.
        :param cache_dir: Where the cache dir is. ProteinRecords will be
        written to this folder after creation, unless it's None.
        :param kwargs: Extra args for the ProteinRecord initializer.
        :return: A ProteinRecord.
        """
        try:
            if cache:
                prec = cls.from_cache(pdb_id, cache_dir)
                if prec is not None:
                    return prec

            pdb_dict = pdb.pdb_dict(pdb_id, struct_d=pdb_dict)
            unp_id = pdb.pdbid_to_unpid(pdb_id, struct_d=pdb_dict)

            prec = cls(unp_id, pdb_id, pdb_dict=pdb_dict, **kwargs)
            if cache_dir:
                prec.save(out_dir=cache_dir)

            return prec
        except Exception as e:
            raise ProteinInitError(f"Failed to created protein record for "
                                   f"pdb_id={pdb_id}: {e}") from e

    @classmethod
    def from_pdb_entity(cls, pdb_id: str, entity_id: Union[int, str],
                        cache=False, cache_dir=pp5.PREC_DIR, **kw) \
            -> ProteinRecord:
        """
        Creates a ProteinRecord based on a PDB ID and an entity ID (not
        chain) within that structure.
        Entities are the distinct chemical components of structures in the
        PDB. Unlike chains, entities do not include duplicate copies. In
        other words, each entity in a structure is different from every
        other entity in the structure.
        One of the chains belonging to the desired entity will be selected.
        :param pdb_id: PDB ID, should not contain chain.
        :param entity_id: ID of the desired entity (a number).
        :param cache: Whether to load prec from cache if available.
        :param cache_dir: Where the cache dir is. ProteinRecords will be
        written to this folder after creation, unless it's None.
        :return: A ProteinRecord.
        """

        # Make sure PDB ID has correct format and ignore chain if provided
        pdb_id, _ = pdb.split_id(pdb_id)
        entity_id = int(entity_id)

        # Discover which chains belong to this entity
        struct_d = pdb.pdb_dict(pdb_id)
        meta = pdb.pdb_metadata(pdb_id, struct_d=struct_d)
        chain = meta.get_chain(entity_id)
        if not chain:
            raise ProteinInitError(f'No matching chain found for entity '
                                   f'{entity_id} in PDB structure {pdb_id}')

        pdb_id = f'{pdb_id}:{chain}'
        return cls.from_pdb(pdb_id, pdb_dict=struct_d, cache=cache,
                            cache_dir=cache_dir, **kw)

    @classmethod
    def from_unp(cls, unp_id: str, cache=False, cache_dir=pp5.PREC_DIR,
                 xref_selector: Callable[[unp.UNPPDBXRef], Any] = None,
                 **kwargs) -> ProteinRecord:
        """
        Creates a ProteinRecord from a Uniprot ID.
        The PDB structure with best resolution will be used.
        :param unp_id: The Uniprot id to query.
        :param xref_selector: Sort key for PDB cross refs. If None,
        resolution will be used.
        :param cache: Whether to load prec from cache if available.
        :param cache_dir: Wheere the cache dir is. ProteinRecords will be
        written to this folder after creation, unless it's None.
        :param kwargs: Extra args for the ProteinRecord initializer.
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

            prec = cls(unp_id, pdb_id, **kwargs)
            if cache_dir:
                prec.save(out_dir=cache_dir)

            return prec
        except Exception as e:
            raise ProteinInitError(f"Failed to created protein record for "
                                   f"unp_id={unp_id}") from e

    def __init__(self, unp_id: str, pdb_id, pdb_dict: dict = None,
                 dihedral_est_name='erp', dihedral_est_args={}, **kw):
        """
        Initialize a protein record from both Uniprot and PDB ids.
        To initialize a protein from Uniprot id or PDB id only, use the
        class methods provided for this purpose.

        :param unp_id: Uniprot id which uniquely identifies the protein.
        :param pdb_id: PDB id with or without chain (e.g. '1ABC' or '1ABC:D')
        of the specific structure desired. Note that this structure must match
        the unp_id, i.e. it must exist in the cross-refs of the given unp_id.
        Otherwise an error will be raised. If no chain is specified, a chain
        matching the unp_id will be used, if it exists.
        :param dihedral_est_name: Method of dihedral angle estimation. None
        or empty to calculate angles without error estimation; 'erp' for
        standard error propagation; 'mc' for montecarlo error estimation.
        :param dihedral_est_args: Extra arguments for dihedral estimator.
        """
        if not (unp_id and pdb_id):
            raise ProteinInitError("Must provide both Uniprot and PDB IDs")

        LOGGER.info(f'{unp_id}: Initializing protein record...')
        self.__setstate__({})

        # First we must find a matching PDB structure and chain for the
        # Uniprot id. If a proposed_pdb_id is given we'll try to use that.
        self.unp_id = unp_id
        self.pdb_base_id, self.pdb_chain_id = self._find_pdb_xref(pdb_id)
        self.pdb_id = f'{self.pdb_base_id}:{self.pdb_chain_id}'
        if pdb_dict:
            self._pdb_dict = pdb_dict

        self.pdb_meta = pdb.pdb_metadata(self.pdb_id, struct_d=self.pdb_dict)
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
            dihedral_est_name, **dihedral_est_args
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
            bfactors.extend(bfactor_est.average_bfactors(pp))
            res_ids = ((self.pdb_chain_id, res.get_id()) for res in pp)
            sss = (ss_dict.get(res_id, '-') for res_id in res_ids)
            sstructs.extend(sss)

        # Find the best matching DNA for our AA sequence via pairwise alignment
        # between the PDB AA sequence and translated DNA sequences.
        dna_seq_record, idx_to_codons = self._find_dna_alignment(pdb_aa_seq)
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
    def dna_seq(self) -> Seq:
        """
        :return: DNA nucleotide sequence. This is the full DNA sequence which,
        after translatoin, best-matches to the PDB AA sequence.
        """
        return Seq(self._dna_seq)

    @property
    def protein_seq(self) -> Seq:
        """
        :return: Protein sequence as 1-letter AA names. Based on the
        residues found in the associated PDB structure.
        Note that the sequence might contain the letter 'X' denoting an
        unknown AA. This happens if the PDB entry contains non-standard AAs
        and we chose to ignore such AAs.
        """
        return Seq(self._protein_seq)

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
            self._pp = pp_chains
        return self._pp

    def to_dataframe(self):
        """
        :return: A Pandas dataframe where each row is a ResidueRecord from
        this ProteinRecord.
        """
        # use the iterator of this class to get the residue recs
        data = []
        for rec in self:
            rec_dict = rec.__dict__.copy()
            del rec_dict['angles']
            angles_dict = rec.angles.as_dict(degrees=True, with_std=True)
            rec_dict.update(angles_dict)
            data.append(rec_dict)
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
        filepath = self._tagged_filepath(self.pdb_id, out_dir, 'csv', tag)
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
        filepath = self._tagged_filepath(self.pdb_id, out_dir, 'prec', tag)
        filepath = pp5.get_resource_path(filepath.parent, filepath.name)
        os.makedirs(filepath.parent, exist_ok=True)

        with open(str(filepath), 'wb') as f:
            pickle.dump(self, f, protocol=4)

        LOGGER.info(f'Wrote {self} to {filepath}')
        return filepath

    @staticmethod
    def _tagged_filepath(pdb_id: str, out_dir: Path, suffix: str, tag: str):
        tag = f'_{tag}' if tag else ''
        filename = f'{pdb_id.replace(":", "_").upper()}{tag}'
        filepath = out_dir.joinpath(f'{filename}.{suffix}')
        return filepath

    def _find_dna_alignment(self, pdb_aa_seq: str) \
            -> Tuple[SeqRecord, Dict[int, str]]:

        # Find cross-refs in ENA
        ena_molecule_types = ('mrna', 'genomic_dna')
        ena_ids = unp.find_ena_xrefs(self.unp_rec, ena_molecule_types)
        if len(ena_ids) == 0:
            raise ProteinInitError(f"Can't find ENA id for {self.unp_id}")

        # Map id to sequence by fetching from ENA API
        ena_seqs = []
        max_enas = 50
        for i, ena_id in enumerate(ena_ids):
            try:
                ena_seqs.append(ena.ena_seq(ena_id))
            except IOError as e:
                LOGGER.warning(f"{self}: Invalid ENA id {ena_id}")
            if i > max_enas:
                LOGGER.warning(f"{self}: Over {max_enas} ENA ids, "
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
        ref_pdb_id, ref_chain_id = pdb.split_id(ref_pdb_id)
        if not ref_chain_id:
            ref_chain_id = ''

        ref_pdb_id, ref_chain_id = ref_pdb_id.lower(), ref_chain_id.lower()

        xrefs = unp.find_pdb_xrefs(self.unp_rec, method='x-ray')

        # We'll sort the PDB entries according to multiple criteria based on
        # the resolution, number of chains and sequence length.
        def sort_key(xref: unp.UNPPDBXRef):
            id_cmp = xref.pdb_id.lower() != ref_pdb_id
            chain_cmp = xref.chain_id.lower() != ref_chain_id
            seq_len_diff = abs(xref.seq_len - self.unp_rec.sequence_length)
            # The sort key for PDB entries
            # First, if we have a matching id to the reference PDB id we take
            # it. Otherwise, we take the best match according to seq len and
            # resolution.
            return id_cmp, chain_cmp, seq_len_diff, xref.resolution

        xrefs = sorted(xrefs, key=sort_key)
        if not xrefs:
            raise ProteinInitError(f"No PDB cross-refs for {self.unp_id}")

        # Get best match according to sort key and return its id.
        xref = xrefs[0]
        LOGGER.info(f'{self.unp_id}: PDB XREF = {xref}')

        pdb_id = xref.pdb_id.lower()
        chain_id = xref.chain_id.lower()

        # Make sure we have a match with the Uniprot id. Id chain wasn't
        # specified, match only PDB ID, otherwise, both must match.
        if pdb_id != ref_pdb_id:
            raise ProteinInitError(
                f"Reference PDB ID {ref_pdb_id} not found as "
                f"cross-reference for protein {self.unp_id}")
        if ref_chain_id and chain_id != ref_chain_id:
            raise ProteinInitError(
                f"Reference chain {ref_chain_id} of PDB ID {ref_pdb_id} not"
                f"found as cross-reference for protein {self.unp_id}. "
                f"Did you mean chain {chain_id}?")

        return pdb_id.upper(), chain_id.upper()

    def _get_dihedral_estimators(self, est_name, **est_args):
        est_name = est_name.lower() if est_name else est_name
        if not est_name in {None, '', 'erp', 'mc'}:
            raise ProteinInitError(
                f'Unknown dihedral estimation method {est_name}')

        unit_cell = pdb.pdb_to_unit_cell(self.pdb_id, struct_d=self.pdb_dict)
        args = dict(isotropic=False, n_samples=100, skip_omega=True)
        args.update(est_args)

        if est_name == 'mc':
            d_est = DihedralAnglesMonteCarloEstimator(unit_cell, **args)
        elif est_name == 'erp':
            d_est = DihedralAnglesUncertaintyEstimator(unit_cell, **args)
        else:
            d_est = DihedralAnglesEstimator(**args)

        b_est = BFactorEstimator(backbone_only=True, unit_cell=None,
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
    similar in terms of sequence and structure.

    A group is defined by a reference structure.
    All proteins in the group are aligned to the reference and different
    residue pairs are created. The pairs have different types based on the
    properties of the alignment (variant/same/silent/mutation/alteration).

    TODO
    """

    @classmethod
    def from_ref(cls, pdb_id: str):
        # Use collector?
        pass

    @classmethod
    def from_pdb(cls, ref_pdb_id: str, query_pdb_ids: Iterable[str], **kw):
        return cls(ref_pdb_id, query_pdb_ids, **kw)

    @classmethod
    def from_collector_csv(cls, ref_pdb_id: str, filepath: str, **kw):
        # From a collector csv
        # This class will generate a different CSV which specifies how
        # everything is aligned to the reference.
        df = pd.read_csv(filepath, header=0, index_col=0)
        query_pdb_ids = list(df['pdb_id'])
        return cls.from_pdb(ref_pdb_id, query_pdb_ids, **kw)

    def __init__(self, ref_pdb_id: str, query_pdb_ids: Iterable[str],
                 context_len: int = 1, max_all_atom_rmsd: float = 2.,
                 min_aligned_residues: int = 50, **kw):

        self.ref_pdb_id = ref_pdb_id.upper()
        self.ref_pdb_base_id, self.ref_pdb_chain = pdb.split_id(ref_pdb_id)
        if not self.ref_pdb_chain:
            raise ProteinInitError('ProteinGroup reference structure must '
                                   'specify the chain id.')

        # Make sure that the query IDs are valid
        if not query_pdb_ids:
            raise ProteinInitError('No query PDB IDs provided')
        try:
            split_ids = [pdb.split_id(query_id) for query_id in query_pdb_ids]
            if not all(map(lambda x: x[1], split_ids)):
                raise ValueError('Must specify chain for all structures')
        except ValueError as e:
            # can also be raised by split, if format invalid
            raise ProteinInitError(str(e)) from None

        if context_len < 1:
            raise ProteinInitError("Context size must be > 1")

        self.context_size = context_len
        self.max_all_atom_rmsd = max_all_atom_rmsd
        self.min_aligned_residues = min_aligned_residues

        self.ref_prec = ProteinRecord.from_pdb(self.ref_pdb_id, cache=True)

        # TODO: replace with parallel maps
        q_aligned = map(self._align_query_residues_to_ref, query_pdb_ids)

        # Find residues matches with the reference
        self.ref_matches: Dict[int, Dict[str, ResidueMatch]] = OrderedDict()
        self.query_pdb_to_prec = {}
        for qa in q_aligned:
            if not qa:
                continue
            q_prec, q_matches_dict = qa
            for res_idx, res_matches in q_matches_dict.items():
                if res_idx not in self.ref_matches:
                    self.ref_matches[res_idx] = OrderedDict()
                self.ref_matches[res_idx].update(res_matches)
            self.query_pdb_to_prec[q_prec.pdb_id] = q_prec

        # TODO: Compute aggregates

    def to_dataframe(self):
        df_index = []
        df_data = []

        for ref_idx, matches in self.ref_matches.items():
            for query_pdb_id, match in matches.items():
                q_prec = self.query_pdb_to_prec[query_pdb_id]
                data = OrderedDict(match.__dict__.copy())

                del data['angles']
                data.update(match.angles.as_dict(
                    degrees=True, skip_omega=True, with_std=True))

                data['type'] = match.type.name
                data.move_to_end('type', last=False)
                data['unp_id'] = q_prec.unp_id
                data.move_to_end('unp_id', last=False)
                data['resolution'] = q_prec.pdb_meta.resolution
                data.move_to_end('resolution', last=False)

                df_data.append(data)
                df_index.append((ref_idx, query_pdb_id))

        df_index = pd.MultiIndex.from_tuples(df_index,
                                             names=['ref_idx', 'query_pdb_id'])
        df = pd.DataFrame(data=df_data, index=df_index)
        return df

    def to_csv(self, out_dir=pp5.out_subdir('pgroup'), tag=None):
        df = self.to_dataframe()
        tag = f'_{tag}' if tag else ''
        filename = f'{self.ref_pdb_base_id}_{self.ref_pdb_chain}{tag}'
        filepath = out_dir.joinpath(f'{filename}.csv')
        df.to_csv(filepath, na_rep='nan', header=True, index=True,
                  encoding='utf-8', float_format='%.3f')
        LOGGER.info(f'Wrote {self} to {filepath}')
        return filepath

    def _align_query_residues_to_ref(self, q_pdb_id: str) -> Optional[
        Tuple[ProteinRecord, Dict[int, Dict[str, ResidueMatch]]]
    ]:
        try:
            q_prec = ProteinRecord.from_pdb(q_pdb_id, cache=True)
        except ProteinInitError as e:
            LOGGER.error(f'Failed to create prec for query structure: {e}')
            return None

        msa = self._struct_align_filter(q_prec.pdb_id)
        if msa is None:
            return None

        stars = msa.column_annotations['clustal_consensus']
        n = msa.get_alignment_length()
        assert n == len(stars)

        r_seq_pymol = msa[0].seq
        q_seq_pymol = msa[1].seq

        # Map from index in the structural alignment to the index within each
        # sequence we got from pymol
        r_idx, q_idx = -1, -1
        stars_to_pymol_idx = {}
        for i in range(n):
            if r_seq_pymol[i] is not '-':
                r_idx += 1
            if q_seq_pymol[i] is not '-':
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
            if r_seq_pymol[i] == '-' or q_seq_pymol[i] == '-':
                continue

            # Now we need to convert i into the index in the prec of each
            # structure
            r_idx_pymol, q_idx_pymol = stars_to_pymol_idx[i]
            r_idx_prec = r_pymol_to_prec[r_idx_pymol]
            q_idx_prec = q_pymol_to_prec[q_idx_pymol]

            # Get the matching residues
            r_res, q_res = self.ref_prec[r_idx_prec], q_prec[q_idx_prec]

            # Compute type of match
            pdb_match = self.ref_prec.pdb_id == q_prec.pdb_id
            unp_match = self.ref_prec.unp_id == q_prec.unp_id
            aa_match = r_res.name == q_res.name
            codon_match = r_res.codon == q_res.codon
            match_type = self._match_type(pdb_match, unp_match, aa_match,
                                          codon_match)
            if match_type is None:
                continue
            ang_dist = self._angle_distance(r_res, q_res)

            # Save match object
            match = ResidueMatch.from_residue(q_res, match_type, ang_dist)
            res_matches = matches.setdefault(r_idx_prec, OrderedDict())
            res_matches[q_prec.pdb_id] = match

        return q_prec, matches

    def _struct_align_filter(self, q_pdb_id: str) \
            -> Optional[MultipleSeqAlignment]:
        """
        Performs structural alignment between the query and the reference
        structure. Rejects query structures which do not conform to the
        requires structural alignment parameters.
        :param q_pdb_id: Query PDB ID.
        :return: Alignment object, or None if query was rejected.
        """
        rmse, n_stars, msa = structural_align(self.ref_pdb_id, q_pdb_id)

        if rmse is None or rmse > self.max_all_atom_rmsd:
            LOGGER.info(
                f'Rejecting {q_pdb_id} due to insufficient structural '
                f'similarity, RMSE={rmse}')
            return None

        if n_stars < self.min_aligned_residues:
            LOGGER.info(f'Rejecting {q_pdb_id} due to insufficient aligned '
                        f'residues, n_stars={n_stars}')
            return None

        return msa

    @staticmethod
    def _align_pymol_to_prec(prec: ProteinRecord, pymol_seq: Seq) \
            -> Dict[int, int]:

        # Align the ref seq from our prec and pymol
        aligner = PairwiseAligner(substitution_matrix=BLOSUM80,
                                  open_gap_score=-10, extend_gap_score=-0.5)
        sa_r_seq = str(pymol_seq).replace('-', '').replace('?', '')
        rr_alignment = aligner.align(prec.protein_seq, sa_r_seq)[0]
        rr_idx_prec, rr_idx_pymol = rr_alignment.aligned
        assert len(rr_idx_prec) == len(rr_idx_pymol)

        # Map pymol index to prec index
        pymol_to_prec = {}
        for j in range(len(rr_idx_prec)):
            prec_start, prec_end = rr_idx_prec[j]
            pymol_start, pymol_end = rr_idx_pymol[j]
            pymol_to_prec.update(zip(
                range(pymol_start, pymol_end), range(prec_start, prec_end)
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

    @staticmethod
    def _angle_distance(r_res: ResidueRecord, q_res: ResidueRecord) -> float:
        dphi = Dihedral.wraparound_diff(r_res.angles.phi, q_res.angles.phi)
        dpsi = Dihedral.wraparound_diff(r_res.angles.psi, q_res.angles.psi)
        dist = math.sqrt(dphi ** 2 + dpsi ** 2)
        return math.degrees(dist)

    def __repr__(self):
        return f'{self.ref_pdb_id}, size={len(self.query_pdb_to_prec)}'
