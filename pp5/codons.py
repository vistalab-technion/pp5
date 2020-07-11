import re

from Bio.Data import CodonTable
from Bio.Data import IUPACData


def codon2aac(codon: str):
    """
    Converts codon to AA-CODON, which we will use as codon identifiers.
    :param codon: a codon string.
    :return: a string formatted AA-CODON where AA is the
    corresponding amino acid.
    """
    aa = CODON_TABLE[codon]
    return f'{aa}-{codon}'.upper()


CODON_TABLE = CodonTable.standard_dna_table.forward_table
START_CODONS = CodonTable.standard_dna_table.start_codons
STOP_CODONS = CodonTable.standard_dna_table.stop_codons
CODONS = sorted(CODON_TABLE.keys())
N_CODONS = len(CODONS)
AA_CODONS = sorted(codon2aac(c) for c in CODONS)

ACIDS = sorted(set([aac[0] for aac in AA_CODONS]))
ACIDS_1TO3 = IUPACData.protein_letters_1to3
ACIDS_1TO1AND3 = {aa: f'{aa} ({ACIDS_1TO3[aa]})' for aa in ACIDS}

CODON_RE = re.compile(rf'^(?:(?P<aa>[{str.join("", ACIDS)}])-)?'
                      rf'(?P<codon>{str.join("|", CODONS)})$',
                      re.VERBOSE | re.IGNORECASE)
