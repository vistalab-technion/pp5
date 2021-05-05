import re
import itertools as it
from typing import Tuple, Sequence

from Bio.Data import IUPACData, CodonTable


def codon2aac(codon: str):
    """
    Converts codon to AA-CODON, which we will use as codon identifiers.
    :param codon: a codon string.
    :return: a string formatted AA-CODON where AA is the
    corresponding amino acid.
    """
    aa = CODON_TABLE[codon]
    return f"{aa}-{codon}".upper()


CODON_TABLE = CodonTable.standard_dna_table.forward_table
START_CODONS = CodonTable.standard_dna_table.start_codons
STOP_CODONS = CodonTable.standard_dna_table.stop_codons
CODONS = sorted(CODON_TABLE.keys())
UNKNOWN_CODON = "---"
N_CODONS = len(CODONS)
AA_CODONS = sorted(codon2aac(c) for c in CODONS)

ACIDS = sorted(set([aac[0] for aac in AA_CODONS]))
N_ACIDS = len(ACIDS)
ACIDS_1TO3 = IUPACData.protein_letters_1to3
ACIDS_1TO1AND3 = {aa: f"{aa} ({ACIDS_1TO3[aa]})" for aa in ACIDS}
UNKNOWN_AA = "X"

CODON_RE = re.compile(
    rf'^(?:(?P<aa>[{str.join("", ACIDS)}])-)?' rf'(?P<codon>{str.join("|", CODONS)})$',
    re.VERBOSE | re.IGNORECASE,
)

#: Pairs of synonymous codons indices
SYN_CODON_IDX: Sequence[Tuple[int, int]] = tuple(
    (i, j)
    for i, j in it.product(range(N_CODONS), range(N_CODONS))
    if AA_CODONS[i][0] == AA_CODONS[j][0]
)

SYN_CODON_IDX_UNIQ: Sequence[Tuple[int, int]] = tuple(
    set(tuple(sorted(ij)) for ij in SYN_CODON_IDX)
)
