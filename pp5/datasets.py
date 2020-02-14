from __future__ import annotations

import warnings

import multiprocessing as mp
import multiprocessing.pool
import logging
import time
import Bio
from typing import Iterable, List, Tuple, NamedTuple, Dict, Iterator
from Bio.PDB import PPBuilder
from Bio.PDB.Polypeptide import Polypeptide, three_to_one
from Bio.PDB.Residue import Residue
from Bio.Data import CodonTable
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Align import PairwiseAligner, PairwiseAlignment
from requests import RequestException

import pp5
import pp5.align
import pp5.dihedral
from pp5.dihedral import Dihedral, pp_mean_bfactor, pp_dihedral_angles
from pp5.external_dbs import pdb, unp, ena

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from Bio.Align import substitution_matrices

BLOSUM62 = substitution_matrices.load("BLOSUM62")
CODON_TABLE = CodonTable.standard_dna_table.forward_table

LOGGER = logging.getLogger(__name__)


class ProteinInitError(ValueError):
    pass


class ResidueRecord(NamedTuple):
    """
    Represents a singe residue in a protein record.
    - Sequence id: (number of this residue in the sequence)
    - Name: (single letter AA code or X for unknown)
    - Codon: three-letter nucleotide sequence
    - phi/psi/omega: dihedral angles
    - bfactor: average b-factor along of the residue's backbone atoms
    - secondary: single-letter secondary structure code
    """
    seq_id: int
    name: str
    codon: str
    phi: float
    psi: float
    omega: float
    bfactor: float
    secondary: str

    def __repr__(self):
        d = Dihedral(self.phi, self.psi, self.omega)
        return f'#{self.seq_id:03d}: {self.name} [{self.codon}] {d} ' \
               f'({self.bfactor:.2f}, {self.secondary})'


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

    def __init__(self, unp_id: str, proposed_pdb_id=None):
        """
        Initialize a protein record.
        :param unp_id: Uniprot id which uniquely identifies the protein.
        :param proposed_pdb_id: Optional PDB id in case a *specific*
        structure is desired. If not provided, OR if provided but it does
        not exist as a cross-reference in the Uniprot record, the PDB id of
        the matching structure with best X-ray resolution and best-matching
        sequence length will be selected.
        """
        LOGGER.info(f'{unp_id}: Initializing protein record...')
        self.__setstate__({})

        # First we must find a matching PDB structure and chain for the
        # Uniprot id. If a proposed_pdb_id is given we'll try to use that.
        self.unp_id = unp_id
        self.pdb_id, self.pdb_chain_id = self._find_pdb_xref(proposed_pdb_id)

        # Extract the PDB AA sequence, dihedral angles and b-factors
        # from the PDB structure.
        # Even though we're working with one PDB chain, the results is a
        # list of multiple Polypeptide objects because we split them at
        # non-standard residues (HETATM atoms in PDB).
        pdb_aa_seq, aa_idxs, angles, bfactors = '', [], [], []
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
                angles.extend([Dihedral()] * gap_len)
                bfactors.extend([None] * gap_len)

            pdb_aa_seq += str(pp.get_sequence())
            aa_idxs.extend(range(curr_start_idx, curr_end_idx + 1))
            angles.extend(pp_dihedral_angles(pp))
            bfactors.extend(pp_mean_bfactor(pp, backbone_only=True))

        # Find the best matching DNA for our AA sequence via pairwise alignment
        # between the PDB AA sequence and translated DNA sequences.
        dna_seq_record, dna_aa_alignment = self._find_dna_alignment(pdb_aa_seq)
        dna_seq = str(dna_seq_record.seq)
        self.ena_id = dna_seq_record.id

        idx_to_codon = self._find_codons(dna_seq, dna_aa_alignment, pdb_aa_seq)
        # TODO: idx to secondary structure

        residue_recs = []
        for i in range(len(pdb_aa_seq)):
            rr = ResidueRecord(
                seq_id=aa_idxs[i], name=pdb_aa_seq[i],
                codon=idx_to_codon.get(i, None),
                phi=angles[i].phi, psi=angles[i].psi, omega=angles[i].omega,
                bfactor=bfactors[i], secondary='?'
            )
            residue_recs.append(rr)

        self._protein_seq = pdb_aa_seq
        self._dna_seq = dna_seq
        self._residue_recs = residue_recs

    @property
    def unp_rec(self):
        """
        :return: Uniprot record for this protein.
        """
        if not self._unp_rec:
            self._unp_rec = unp.unp_record(self.unp_id)
        return self._unp_rec

    @property
    def pdb_rec(self):
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
        return [x.codon for x in self._residue_recs]

    @property
    def dihedral_angles(self) -> List[Dihedral]:
        return [Dihedral(x.phi, x.psi, x.omega) for x in self._residue_recs]

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

    def __iter__(self) -> Iterator[ResidueRecord]:
        return iter(self._residue_recs)

    def _find_codons(self, dna_seq, dna_aa_alignment, pdb_aa_seq) -> Dict[int,
                                                                          str]:
        # Indices in the DNA AA seq which are aligned to the PDB AA seq
        aligned_idx_pdb_aa, aligned_idx_dna_aa = dna_aa_alignment.aligned

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

                # Make sure it matches
                assert CODON_TABLE[codon] == pdb_aa_seq[pdb_idx]
                idx_to_codon[pdb_idx] = codon

        return idx_to_codon

    def _find_dna_alignment(self, pdb_aa_seq) -> Tuple[SeqRecord,
                                                       PairwiseAlignment]:
        ena_ids = []
        cross_refs = self.unp_rec.cross_references
        allowed_types = {'mrna', 'genomic_dna'}

        embl_refs = (x for x in cross_refs if x[0].lower() == 'embl')
        for dbname, id1, id2, comment, molecule_type in embl_refs:
            molecule_type = molecule_type.lower()
            if molecule_type in allowed_types and id2 and len(id2) > 3:
                ena_ids.append(id2)

        # Map id to sequence by fetching from ENA API
        ena_seqs = []
        max_enas = 50  # Limit number of ENA records we are willing to check
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

        aligner = PairwiseAligner(substitution_matrix=BLOSUM62,
                                  open_gap_score=-10, extend_gap_score=-0.5)
        alignments = []
        for seq in ena_seqs:
            translated = seq.translate(stop_symbol='')
            alignment = aligner.align(pdb_aa_seq, translated.seq)
            alignments.append(alignment)

        # Sort alignments by score
        best_ena, best_alignment = max(
            zip(ena_seqs, alignments), key=lambda x: x[1].score
        )

        LOGGER.info(f'{self}: ENA ID = {best_ena.id}')
        LOGGER.info(f'{self}: Translated DNA to PDB alignment '
                    f'(norm_score='
                    f'{best_alignment.score / len(pdb_aa_seq):.2f}, '
                    f'len={len(best_alignment)})\n'
                    f'{best_alignment[0]}')

        return best_ena, best_alignment[0]

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
    def from_pdb(cls, pdb_id: str) -> List[ProteinRecord]:
        """
        Given a PDB id, finds all the proteins it contains (usually one) in
        terms of unique Uniprot ids, and returns a sequence of ProteinRecord
        objects for each.
        :param pdb_id: The PDB id to query.
        :return: A sequence of ProteinRecord (lazily generated).
        """
        # pdb id -> mlutiple uniprot ids -> multiple ProteinRecords
        unp_ids = pdb.pdbid_to_unpids(pdb_id)
        if not unp_ids:
            raise ProteinInitError(f"Can't find Uniprot cross-reference for "
                                   f"pdb_id={pdb_id}")

        protein_recs = [cls(unp_id, pdb_id) for unp_id in unp_ids]
        return protein_recs


def collect_data():
    # Query PDB for structures
    query = pdb.PDBCompositeQuery(
        pdb.PDBExpressionSystemQuery('Escherichia Coli'),
        pdb.PDBResolutionQuery(max_res=1.0)
    )

    pdb_ids = query.execute()
    LOGGER.info(f"Got {len(pdb_ids)} structures from PDB")

    async_results = []
    with mp.pool.Pool(processes=8) as pool:
        for i, pdb_id in enumerate(pdb_ids):
            async_results.append(
                pool.apply_async(ProteinRecord.from_pdb, (pdb_id,))
            )

        start_time, counter = time.time(), 0
        for async_result in async_results:
            try:
                protein_recs = async_result.get(30)
            except TimeoutError as e:
                LOGGER.error("Timeout getting async result, skipping")
            except ProteinInitError as e:
                LOGGER.error(f"Failed to create protein: {e}")
            except Exception as e:
                LOGGER.error("Unexpected error", exc_info=e.__cause__)

            counter += len(protein_recs)
            pps = counter / (time.time() - start_time)
            LOGGER.info(f'Collected {protein_recs} ({pps:.1f} proteins/sec)')

            # TODO: Write to file

        LOGGER.info(f"Done: {counter} proteins collected.")


if __name__ == '__main__':
    # collect_data()
    # prec = ProteinRecord('P00720')
    prev = ProteinRecord('B0VB33')

    j = 3
