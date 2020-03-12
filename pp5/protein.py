from __future__ import annotations

import logging
import math
import warnings
from typing import List, Tuple, Dict, Iterator, Callable, Any, Union
from collections import OrderedDict

import pandas as pd
from Bio.Align import PairwiseAligner
from Bio.Data import CodonTable
from Bio.PDB import PPBuilder
from Bio.PDB.Polypeptide import Polypeptide
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

import pp5
from pp5.dihedral import Dihedral, DihedralAnglesEstimator, \
    DihedralAnglesUncertaintyEstimator, DihedralAnglesMonteCarloEstimator, \
    BFactorEstimator
from pp5.external_dbs import pdb, unp, ena
from pp5.external_dbs.pdb import PDBRecord
from pp5.external_dbs.unp import UNPRecord

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

    def __init__(self, res_id: Union[str, int], name: str, codon_counts: dict,
                 angles: Dihedral, bfactor: float, secondary: str):
        self.res_id, self.name = str(res_id).strip(), name
        self.angles, self.bfactor, self.secondary = angles, bfactor, secondary

        codon_counts = {} if not codon_counts else codon_counts
        best_codon, max_count, total_count = UNKNOWN_CODON, 0, 0
        for codon, count in codon_counts.items():
            total_count += count
            if count > max_count and codon != UNKNOWN_CODON:
                best_codon, max_count = codon, count
        self.codon = best_codon
        self.codon_score = max_count / total_count if total_count else 0
        self.codon_opts = '/'.join(codon_counts.keys())

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

    @classmethod
    def from_pdb(cls, pdb_id: str, **kwargs) -> ProteinRecord:
        """
        Given a PDB id (and optionally a chain), finds the
        corresponding Uniprot id, and returns a ProteinRecord object for
        that protein.
        :param pdb_id: The PDB id to query, with optional chain, e.g. '0ABC:D'.
        :param kwargs: Extra args for the ProteinRecord initializer.
        :return: A ProteinRecord.
        """
        try:
            pdb_dict = pdb.pdb_dict(pdb_id)
            unp_id = pdb.pdbid_to_unpid(pdb_id, struct_d=pdb_dict)
            return cls(unp_id, pdb_id, pdb_dict=pdb_dict, **kwargs)
        except Exception as e:
            raise ProteinInitError(f"Failed to created protein record for "
                                   f"pdb_id={pdb_id}: {e}") from e

    @classmethod
    def from_unp(cls, unp_id: str,
                 xref_selector: Callable[[unp.UNPPDBXRef], Any] = None,
                 **kwargs) -> ProteinRecord:
        """
        Creates a ProteinRecord from a Uniprot ID.
        The PDB structure with best resolution will be used.
        :param unp_id: The Uniprot id to query.
        :param xref_selector: Sort key for PDB cross refs. If None,
        resolution will be used.
        :param kwargs: Extra args for the ProteinRecord initializer.
        :return: A ProteinRecord.
        """
        if not xref_selector:
            xref_selector = lambda xr: xr.resolution

        try:
            xrefs = unp.find_pdb_xrefs(unp_id)
            xrefs = sorted(xrefs, key=xref_selector)
            pdb_id = f'{xrefs[0].pdb_id}:{xrefs[0].chain_id}'
            return cls(unp_id, pdb_id, **kwargs)
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

        LOGGER.info(f'{self}: '
                    f'res={self.pdb_meta.resolution:.2f}Å, '
                    f'org={self.pdb_meta.src_org} ({self.pdb_meta.src_org_id}) '
                    f'expr={self.pdb_meta.host_org} ({self.pdb_meta.host_org_id})')

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

        residue_recs = []
        for i in range(len(pdb_aa_seq)):
            rr = ResidueRecord(
                res_id=aa_ids[i], name=pdb_aa_seq[i],
                codon_counts=idx_to_codons.get(i, {}),
                angles=angles[i],
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
            rec_dict = rec.__getstate__()
            del rec_dict['angles']
            angles_dict = dict(phi=rec.angles.phi_deg,
                               psi=rec.angles.psi_deg,
                               omega=rec.angles.omega_deg,
                               phi_std=rec.angles.phi_std_deg,
                               psi_std=rec.angles.psi_std_deg,
                               omega_std=rec.angles.omega_std_deg, )
            rec_dict.update(angles_dict)
            data.append(rec_dict)
        return pd.DataFrame(data)

    def to_csv(self, out_dir=pp5.out_subdir('precs'), tag=None):
        df = self.to_dataframe()
        tag = f'_{tag}' if tag else ''
        filename = f'{self.pdb_base_id}_{self.pdb_chain_id}{tag}'
        filepath = out_dir.joinpath(f'{filename}.csv')
        df.to_csv(filepath, na_rep='nan', header=True, index=False,
                  encoding='utf-8', float_format='%.3f')
        LOGGER.info(f'Wrote {self} to {filepath}')
        return filepath

    def __iter__(self) -> Iterator[ResidueRecord]:
        return iter(self._residue_recs)

    def __getitem__(self, item: int):
        return self._residue_recs[item]

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
                    f'{best_alignment}')

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
                f"found as cross-reference for protein {self.unp_id}."
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
