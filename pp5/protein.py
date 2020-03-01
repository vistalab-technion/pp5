from __future__ import annotations

import logging
import math
import warnings
from typing import List, Tuple, NamedTuple, Dict, Iterator

import pandas as pd
from Bio.Align import PairwiseAligner
from Bio.Data import CodonTable
from Bio.PDB import PPBuilder
from Bio.PDB.Polypeptide import Polypeptide
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from requests import RequestException

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

LOGGER = logging.getLogger(__name__)


class ProteinInitError(ValueError):
    pass


class ResidueRecord(object):
    """
    Represents a singe residue in a protein record.
    - Sequence id: (number of this residue in the sequence)
    - Name: (single letter AA code or X for unknown)
    - Codon: three-letter nucleotide sequence
    - phi/psi/omega: dihedral angles in degrees
    - bfactor: average b-factor along of the residue's backbone atoms
    - secondary: single-letter secondary structure code
    """

    def __init__(self, seq_id: int, name: str, codon: str,
                 angles: Dihedral, bfactor: float, secondary: str):
        self.seq_id, self.name, self.codon = seq_id, name, codon
        self.angles, self.bfactor, self.secondary = angles, bfactor, secondary

    def __getstate__(self):
        d = self.__dict__.copy()
        d['angles'] = self.angles.__dict__
        return d

    def __setstate__(self, state):
        angles = Dihedral.empty()
        angles.__dict__.update(state['angles'])
        state['angles'] = angles
        self.__init__(**state)

    def __repr__(self):
        return f'#{self.seq_id:03d}: {self.name} [{self.codon}] ' \
               f'{self.angles} b={self.bfactor:.2f}, {self.secondary}'


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
    _SKIP_SERIALIZE = ['_unp_rec', '_pdb_rec', '_pp']

    def __init__(self, unp_id: str, proposed_pdb_id=None,
                 dihedral_est_name=None, dihedral_est_args={}):
        """
        Initialize a protein record.
        :param unp_id: Uniprot id which uniquely identifies the protein.
        :param proposed_pdb_id: Optional PDB id in case a *specific*
        structure is desired. If not provided, OR if provided but it does
        not exist as a cross-reference in the Uniprot record, the PDB id of
        the matching structure with best X-ray resolution and best-matching
        sequence length will be selected.
        :param dihedral_est_name: Method of dihedral angle estimation. None
        or empty to calculate angles without error estimation; 'erp' for
        standard error propagation; 'mc' for montecarlo error estimation.
        :param dihedral_est_args: Extra arguments for dihedral estimator.
        """
        LOGGER.info(f'{unp_id}: Initializing protein record...')
        self.__setstate__({})

        # First we must find a matching PDB structure and chain for the
        # Uniprot id. If a proposed_pdb_id is given we'll try to use that.
        self.unp_id = unp_id
        self.pdb_id, self.pdb_chain_id = self._find_pdb_xref(proposed_pdb_id)

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
        pdb_aa_seq, aa_idxs, angles, bfactors, sstructs = '', [], [], [], []
        for i, pp in enumerate(self.polypeptides):
            curr_start_idx = pp[0].get_id()[1]
            curr_end_idx = pp[-1].get_id()[1]

            # More than one pp means there are gaps due to non-standard AAs
            if i > 0:
                # Calculate index gap between this polypeptide and previous
                prev_end_idx = self.polypeptides[i - 1][-1].get_id()[1]
                gap_len = curr_start_idx - prev_end_idx - 1

                # fill in the gaps
                pdb_aa_seq += 'X' * gap_len
                aa_idxs.extend(range(prev_end_idx + 1, curr_start_idx))
                angles.extend([Dihedral.empty()] * gap_len)
                bfactors.extend([math.nan] * gap_len)
                sstructs.extend(['-'] * gap_len)

            pdb_aa_seq += str(pp.get_sequence())
            aa_idxs.extend(range(curr_start_idx, curr_end_idx + 1))
            angles.extend(dihedral_est.estimate(pp))
            bfactors.extend(bfactor_est.average_bfactors(pp))
            res_ids = ((self.pdb_chain_id, res.get_id()) for res in pp)
            sss = (ss_dict.get(res_id, '-') for res_id in res_ids)
            sstructs.extend(sss)

        # Find the best matching DNA for our AA sequence via pairwise alignment
        # between the PDB AA sequence and translated DNA sequences.
        ena_ids = unp.find_ena_xrefs(
            self.unp_rec, molecule_types=('mrna', 'genomic_dna')
        )
        dna_seq_record, idx_to_codon = self._find_dna_alignment(
            ena_ids, pdb_aa_seq
        )
        dna_seq = str(dna_seq_record.seq)
        self.ena_id = dna_seq_record.id

        residue_recs = []
        for i in range(len(pdb_aa_seq)):
            rr = ResidueRecord(
                seq_id=aa_idxs[i], name=pdb_aa_seq[i],
                codon=idx_to_codon.get(i, None),
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
    def pdb_rec(self) -> PDBRecord:
        """
        :return: PDB record for this protein. Note that this record may
        contain multiple chains and this protein only represents one of them
        (self.pdb_chain_id).
        """
        if not self._pdb_rec:
            self._pdb_rec = pdb.pdb_struct(self.pdb_id)
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

    def to_csv(self, out_dir=pp5.data_subdir('precs'), tag=None):
        df = self.to_dataframe()
        tag = f'_{tag}' if tag else ''
        filename = f'{self.pdb_id.upper()}_{self.pdb_chain_id.upper()}{tag}'
        filepath = out_dir.joinpath(f'{filename}.csv')
        df.to_csv(filepath, na_rep='nan', header=True, index=False,
                  encoding='utf-8', )
        return filepath

    def __iter__(self) -> Iterator[ResidueRecord]:
        return iter(self._residue_recs)

    def __getitem__(self, item: int):
        return self._residue_recs[item]

    def _find_dna_alignment(self, ena_ids, pdb_aa_seq: str) \
            -> Tuple[SeqRecord, Dict[int, str]]:

        # Map id to sequence by fetching from ENA API
        ena_seqs = []
        max_enas = 50
        for i, ena_id in enumerate(ena_ids):
            try:
                ena_seqs.append(ena.ena_seq(ena_id))
            except RequestException as e:
                LOGGER.warning(f"{self}: Invalid ENA id {ena_id}")
            if i > max_enas:
                LOGGER.warning(f"{self}: Over {max_enas} ENA ids, "
                               f"skipping")
                break

        if len(ena_ids) == 0:
            raise ProteinInitError(f"Can't find ENA id for {self.unp_id}")

        aligner = PairwiseAligner(substitution_matrix=BLOSUM80,
                                  open_gap_score=-10, extend_gap_score=-0.5)
        alignments = []
        for seq in ena_seqs:
            translated = seq.translate(stop_symbol='')
            alignment = aligner.align(pdb_aa_seq, translated.seq)
            alignments.append(alignment)

        # Sort alignments by score
        # best_ena, best_alignments = max(
        #     zip(ena_seqs, alignments), key=lambda x: x[1].score
        # )
        alignments = sorted(
            zip(ena_seqs, alignments), key=lambda x: x[1].score
        )
        best_ena, best_alignments = alignments[0]
        best_alignment = best_alignments[0]

        LOGGER.info(f'{self}: ENA ID = {best_ena.id}')
        LOGGER.info(f'{self}: Translated DNA to PDB alignment '
                    f'(norm_score='
                    f'{best_alignments.score / len(pdb_aa_seq):.2f}, '
                    f'num={len(best_alignments)})\n'
                    f'{best_alignment}')

        # idx_to_codon = self._match_codons()

        # Indices in the DNA AA seq which are aligned to the PDB AA seq
        aligned_idx_pdb_aa, aligned_idx_dna_aa = best_alignment.aligned
        dna_seq = str(best_ena.seq)

        idx_to_codon = {}
        for i in range(len(aligned_idx_dna_aa)):
            # Indices of current matching segment of amino acids from the
            # PDB record and the translated DNA
            pdb_aa_start, pdb_aa_stop = aligned_idx_pdb_aa[i]
            dna_aa_start, dna_aa_stop = aligned_idx_dna_aa[i]

            for pdb_offset, k in enumerate(range(dna_aa_start, dna_aa_stop)):
                k *= 3
                codon = dna_seq[k:k + 3]
                pdb_idx = pdb_offset + pdb_aa_start

                # Skip "unknown" AAs - we put them there to represent
                # non-standard AAs.
                if pdb_aa_seq[pdb_idx] == 'X':
                    continue

                # Check if the codon actually matched the AA at the
                # corresponding location. Sometimes there may be
                # mismatches ('.') because the DNA alignment isn't perfect.
                # In such a case we'll set the codon to None to specify we
                # don't know what codon encoded the AA in the PDB sequence.
                matches = CODON_TABLE.get(codon, None) == pdb_aa_seq[pdb_idx]
                idx_to_codon[pdb_idx] = codon if matches else None

        return best_ena, idx_to_codon

    def _find_pdb_xref(self, proposed_pdb_id=None) -> Tuple[str, str]:
        cross_refs = self.unp_rec.cross_references
        proposed_pdb_id = '' if not proposed_pdb_id else proposed_pdb_id

        # PDB cross refs are ('PDB', id, method, resolution, chains)
        # E.g: ('PDB', '5EWX', 'X-ray', '2.60 A', 'A/B=1-35, A/B=38-164')
        pdb_xrefs = (x for x in cross_refs if x[0].lower() == 'pdb')
        pdb_xrefs = (x for x in pdb_xrefs if x[2].lower() == 'x-ray')

        # We'll sort the PDB entries according to multiple criteria based on
        # the resolution, number of chains and sequence length.
        def sort_key(xref):
            resolution = float(xref[3].split()[0])
            chains_groups = xref[4].split(',')
            chains = set()
            seq_len_diff = 0
            for chain_str in chains_groups:
                chain_names, chain_seqs = chain_str.split('=')
                chains.update(chain_names.split('/'))
                seq_start, seq_end = chain_seqs.split('-')
                seq_len = int(seq_end) - int(seq_start)
                seq_len_diff += abs(self.unp_rec.sequence_length - seq_len)

            id_cmp = xref[1].lower() != proposed_pdb_id.lower()
            n_groups = len(chains_groups)
            n_chains = len(chains)

            # The sort key for PDB entries
            # First, if we have a matching id to the proposed PDB id we take
            # it. Otherwise, we take the best match according to seq len and
            # resolution.
            return id_cmp, seq_len_diff, resolution, n_groups, n_chains

        pdb_xrefs = sorted(pdb_xrefs, key=sort_key)
        if not pdb_xrefs:
            raise ProteinInitError(f"No PDB cross-refs for {self.unp_id}")

        # Get best match according to sort key and return its id.
        xref = pdb_xrefs[0]
        LOGGER.info(f'{self.unp_id}: PDB ID = {xref[1]}|{xref[3]}|{xref[4]}')

        # We just need one of the chain IDs in the cross-reference, so we'll
        # take the first one.
        pdb_id = xref[1]
        chains_str = xref[4]
        chain_id = chains_str[0]
        if proposed_pdb_id and pdb_id != proposed_pdb_id:
            LOGGER.warning(f"Proposed PDB id {proposed_pdb_id} not found as "
                           f"cross-reference for protein {self.unp_id}, "
                           f"using {pdb_id} instead.")

        return pdb_id, chain_id

    def _get_dihedral_estimators(self, est_name, **est_args):
        est_name = est_name.lower() if est_name else est_name
        if not est_name in {None, '', 'erp', 'mc'}:
            raise ProteinInitError(
                f'Unknown dihedral estimation method {est_name}')

        unit_cell = pdb.pdb_to_unit_cell(self.pdb_id)
        args = dict(isotropic=False, n_samples=100, skip_omega=True)
        args.update(est_args)

        if est_name == 'mc':
            d_est = DihedralAnglesMonteCarloEstimator(unit_cell, **args)
        elif est_name == 'erp':
            d_est = DihedralAnglesUncertaintyEstimator(unit_cell, **est_args)
        else:
            d_est = DihedralAnglesEstimator(**est_args)

        b_est = BFactorEstimator(backbone_only=True, unit_cell=None,
                                 isotropic=True)
        return d_est, b_est

    def __repr__(self):
        return f'({self.unp_id}, {self.pdb_id}:{self.pdb_chain_id})'

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

    @classmethod
    def from_pdb(cls, pdb_id: str, **kwargs) -> List[ProteinRecord]:
        """
        Given a PDB id, finds all the proteins it contains (usually one) in
        terms of unique Uniprot ids, and returns a sequence of ProteinRecord
        objects for each.
        :param pdb_id: The PDB id to query.
        :param kwargs: Extra args for the ProteinRecord initializer.
        :return: A sequence of ProteinRecord (lazily generated).
        """
        # pdb id -> mlutiple uniprot ids -> multiple ProteinRecords
        unp_ids = pdb.pdbid_to_unpids(pdb_id)
        if not unp_ids:
            raise ProteinInitError(f"Can't find Uniprot cross-reference for "
                                   f"pdb_id={pdb_id}")

        protein_recs = [cls(unp_id, pdb_id, **kwargs) for unp_id in unp_ids]
        return protein_recs


if __name__ == '__main__':
    pass
    prec = ProteinRecord.from_pdb('2WUR')[0]
    prec.to_csv()
