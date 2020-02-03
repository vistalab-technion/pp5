from __future__ import annotations

import logging
from typing import Iterable

from Bio.PDB import PPBuilder
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

import pp5
import pp5.align
import pp5.dihedral
import pp5.external_dbs

LOGGER = logging.getLogger(__name__)


class ProteinRecord:
    """
    Represents a protein in our dataset. Includes:
    - Uniprot id defining the which protein this is.
    - sequence (amino acids)
    - genetic seq (nucleutides)
    - pdb id of one structure for it.
    - Dihedral angles
    """

    def __init__(self, unp_id: str, pdb_id=None):
        """
        Initialize a protein record.
        :param unp_id: Uniprot id which uniquely identifies the protein.
        :param pdb_id: Optional PDB id in case a *specific* structure is
        desired. If not provided, the PDB id of the matching structure with
        best X-ray resolution and best-matching sequence length will be
        selected.
        """
        LOGGER.info(f'{unp_id}: Initializing protein record...')
        self.unp_id = unp_id
        self.unp_rec = pp5.external_dbs.unp_record(self.unp_id)
        self._dna_seq = self._find_dna_seq()
        self.pdb_id = self._find_pdb_id() if not pdb_id else pdb_id
        self.pdb_rec = pp5.external_dbs.pdb_struct(self.pdb_id)
        self.angles = self._calc_dihedral()

    @property
    def dna_seq(self) -> Seq:
        """
        :return: DNA nucleotide sequence.
        """
        return self._dna_seq.seq

    @property
    def protein_seq(self) -> Seq:
        """
        :return: Protein sequence as 1-letter AA names.
        """
        return Seq(self.unp_rec.sequence)

    @property
    def protein_seq_dna(self):
        """
        :return: Protein sequence based on translating DNA sequence with
        standard codon table.
        """
        return self.dna_seq.translate(stop_symbol='')

    def _find_dna_seq(self) -> SeqRecord:
        ena_ids = []
        cross_refs = self.unp_rec.cross_references
        allowed_types = {'mrna', 'genomic_dna'}

        embl_refs = (x for x in cross_refs if x[0].lower() == 'embl')
        for dbname, id1, id2, comment, molecule_type in embl_refs:
            molecule_type = molecule_type.lower()
            if molecule_type in allowed_types and id2 and len(id2) > 3:
                ena_ids.append(id2)

        if len(ena_ids) == 0:
            raise RuntimeError(f"Can't find ENA id for UNP id {self.unp_id}")

        # Map id to sequence by fetching from ENA API
        ena_seqs = map(pp5.external_dbs.ena_seq, ena_ids)

        # Take alignment with length roughly matching the protein length *3
        expected_len = 3 * self.unp_rec.sequence_length
        ena_seqs = sorted(ena_seqs,
                          key=lambda seq: abs(len(seq) - expected_len))

        LOGGER.info(f'{self.unp_id}: ENA ID = {ena_seqs[0].id}')
        return ena_seqs[0]

    def _find_pdb_id(self) -> str:
        cross_refs = self.unp_rec.cross_references

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

            # The sort key for PDB entries
            return seq_len_diff, resolution, len(chains_groups), len(chains)

        pdb_xrefs = sorted(pdb_xrefs, key=sort_key)

        # Get best match according to sort key and return its id.
        xref = pdb_xrefs[0]
        LOGGER.info(f'{self.unp_id}: PDB ID = {xref[1]}|{xref[3]}|{xref[4]}')
        return xref[1]

    def _calc_dihedral(self):
        pdb_id = self.pdb_id
        cross_refs = self.unp_rec.cross_references

        # Find the cross-ref entry for our PDB id
        pdb_xrefs = (x for x in cross_refs if x[0].lower() == 'pdb')
        pdb_xrefs = [x for x in pdb_xrefs if x[1].lower() == pdb_id.lower()]
        if len(pdb_xrefs) == 0:
            raise ValueError(f"PDB id {pdb_id} not found as cross-reference "
                             f"for protein {self.unp_id}")
        # PDB xref format is specified above
        xref = pdb_xrefs[0]
        chains_str = xref[4]

        # We just need one of the relevant chain IDs, so we'll take the first
        chain_id = chains_str[0]
        chain = self.pdb_rec[0][chain_id]

        # Build a polypeptide from the chain
        pp_builder = PPBuilder()
        polypeptides = pp_builder.build_peptides(chain, aa_only=1)
        angles = []
        for pp in polypeptides:
            angles.extend(pp5.dihedral.pp_dihedral_angles(pp, degrees=True))

        return angles

    @classmethod
    def from_pdb(cls, pdb_id: str) -> Iterable[ProteinRecord]:
        """
        Given a PDB id, finds all the proteins it contains (usually one) in
        terms of unique Uniprot ids, and returns a sequence of ProteinRecord
        objects for each.
        :param pdb_id: The PDB id to query.
        :return: A sequence of ProteinRecord (lazily generated).
        """
        # pdb id -> mlutiple uniprot ids -> multiple ProteinRecords
        unp_ids = pp5.external_dbs.pdbid_to_unpids(pdb_id)
        if not unp_ids:
            raise ValueError(f"Can't find Uniprot cross-reference for "
                             f"pdb_id={pdb_id}")

        return map(lambda unp_id: ProteinRecord(unp_id, pdb_id), unp_ids)


if __name__ == '__main__':
    # precs = list(ProteinRecord.from_pdb('6NQ3'))
    # precs = list(ProteinRecord.from_pdb('102L'))
    ProteinRecord('P02794')
