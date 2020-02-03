from __future__ import annotations

import logging
from typing import Iterable

from Bio.SeqRecord import SeqRecord

import pp5
import pp5.align
import pp5.external_dbs

LOGGER = logging.getLogger(__name__)


class ProteinRecord:
    """
    Represents a protein in our dataset. Includes:
    - Uniprot id defining the which protein this is.
    - pdb id of one structure for it.
    - sequence (amino acids)
    - genetic seq (nucleutides)
    - codon list (codons)
    - Dihedral angles
    """

    def __init__(self, unp_id: str):
        LOGGER.info(f'{unp_id}: Initializing protein record...')
        self.unp_id = unp_id
        self.unp_s = pp5.external_dbs.unp_record(self.unp_id)
        self.dna_seq = self.dna_seq()

    def dna_seq(self) -> SeqRecord:
        ena_ids = []

        embl_refs = (x for x in self.unp_s.cross_references if
                     x[0].lower() == 'embl')
        for dbname, id1, id2, comment, type in embl_refs:
            if type.lower() in {'mrna', 'genomic_dna'} and id2 and len(
                    id2) > 3:
                ena_ids.append(id2)

        if len(ena_ids) == 0:
            raise RuntimeError(f"Can't find ENA id for UNP id {self.unp_id}")

        ena_seqs = map(pp5.external_dbs.ena_seq, ena_ids)

        # Take alignment with length roughly matching the protein length *3
        expected_len = 3 * self.unp_s.sequence_length
        ena_seqs = sorted(ena_seqs,
                          key=lambda seq: abs(len(seq) - expected_len))

        LOGGER.info(f'{self.unp_id}: ENA ID = {ena_seqs[0].id}')
        return ena_seqs[0]

    @classmethod
    def from_pdb(cls, pdb_id: str) -> Iterable[ProteinRecord]:
        """
        Given a PDB id, finds all the proteins it contains (usually one) in
        terms of unique UniProt ids, and returns a sequence of ProteinRecord
        objects for each.
        :param pdb_id: The PDB id to query.
        :return: A sequence of ProteinRecord (lazily generated).
        """
        # pdb id -> mlutiple uniprot ids -> multiple ProteinRecords
        unp_ids = pp5.external_dbs.pdbid_to_unpids(pdb_id)
        if not unp_ids:
            raise ValueError(f"Can't find UniProt cross-reference for "
                             f"pdb_id={pdb_id}")

        return map(ProteinRecord, unp_ids)


if __name__ == '__main__':
    precs = list(ProteinRecord.from_pdb('6NQ3'))
    # precs = list(ProteinRecord.from_pdb('102L'))
