import re
import itertools as it
from typing import Set, Tuple, Iterator, Sequence, cast

from Bio.Data import IUPACData, CodonTable

# Separates between AA and matching codon, e.g. "A-GCA"
AAC_SEP = "-"


# Separates between two (or more) AAs or two (or more) codons, e.g. "A-GCA_C-TGC"
AAC_TUPLE_SEP = "_"


def codon2aac(codon: str):
    """
    Converts codon to AA-CODON, which we will use as codon identifiers.
    :param codon: a codon string.
    :return: a string formatted AA-CODON where AA is the
    corresponding amino acid.
    """
    aa = CODON_TABLE[codon]
    return f"{aa}{AAC_SEP}{codon}".upper()


CODON_TABLE = CodonTable.standard_dna_table.forward_table
START_CODONS: Sequence[str] = CodonTable.standard_dna_table.start_codons
STOP_CODONS: Sequence[str] = CodonTable.standard_dna_table.stop_codons
CODONS: Sequence[str] = sorted(CODON_TABLE.keys())
UNKNOWN_CODON = "---"
N_CODONS = len(CODONS)
AA_CODONS: Sequence[str] = sorted(codon2aac(c) for c in CODONS)

ACIDS = sorted(set([aac[0] for aac in AA_CODONS]))
N_ACIDS = len(ACIDS)
ACIDS_1TO3 = IUPACData.protein_letters_1to3
ACIDS_1TO1AND3 = {aa: f"{aa} ({ACIDS_1TO3[aa]})" for aa in ACIDS}
UNKNOWN_AA = "X"

CODON_RE = re.compile(
    rf'^(?:(?P<aa>[{str.join("", ACIDS)}]){AAC_SEP})?(?P<codon>{str.join("|", CODONS)})$',
    re.VERBOSE | re.IGNORECASE,
)


AAC = str
AACTuple = Tuple[AAC, ...]
AACIndexedTuple = Tuple[int, AACTuple]


def aac2aa(aac: AAC) -> str:
    """
    :param aac: An AA-CODON
    :return: The AA string.
    """
    return aac.split(AAC_SEP)[0]


def aac2c(aac: AAC):
    """
    :param aac: An AA-CODON
    :return: The Codon string.
    """
    return aac.split(AAC_SEP)[1]


def is_synonymous(aac1: AAC, aac2: AAC) -> bool:
    """
    Whether two AAC's are synonymous, i.e. codons corresponding to the same AA.
    :param aac1: First AAC.
    :param aac2: Second AAC.
    :return: True iff they're synonymous.
    """
    return aac2aa(aac1) == aac2aa(aac2)


def is_synonymous_tuple(aact1: AACTuple, aact2: AACTuple):
    """
    Whether two tuples, each containing k AACs, are synonymous, i.e.,
    each pair of corresponding AACs at corresponding indices within the tuples is
    itself synonymous.

    :param aact1: First AAC tuple.
    :param aact2: Second AAC tuple.
    :return: True iff the tuples are considered synonymous.
    """
    assert len(aact1) == len(aact2)
    return all(is_synonymous(aac1, aac2) for aac1, aac2 in zip(aact1, aact2))


def aac_tuples(k: int = 1) -> Iterator[AACTuple]:
    """
    Generates k-tuples of aa-codons.
    :param k: Number of elements in each tuple.
    :return: A generator producing the tuples.
    """
    return it.product(*([AA_CODONS] * k))


def aac_tuple_pairs(
    k: int = 1, synonymous: bool = False, unique: bool = False
) -> Set[Tuple[AACIndexedTuple, AACIndexedTuple]]:
    """

    Creates pairs of AAC indexed k-tuples.
    The pairs can optionally contain only synonymous codons, and can optionally be
    unique.

    :param k: Length of each tuple in each returned pair.
    :param synonymous: Whether to only returns synonymous pairs of AAC tuples. Two
        AAC tuples A and B are synonymous if is_synonymous_tuple(A, B) is True.
    :param unique: Whether to only return unique pairs. If False, then a pair of
        tuples (A, B) and another pair (B, A) will be returned, otherwise only one of
        them.
    :return: A set of pairs ((i, A), (j, B)) where A and B are each AAC
        tuples of length k and i and j are their indices is the sorted order of AAC
        tuples.
    """

    if synonymous:
        _synonymous_fn = is_synonymous_tuple
    else:
        _synonymous_fn = lambda *args: True

    if unique:
        _reduce_fn = lambda x: tuple(sorted(x))
    else:
        _reduce_fn = lambda x: x

    indexed_pairs = it.product(enumerate(aac_tuples(k)), enumerate(aac_tuples(k)))

    indexed_filtered_pairs = (
        _reduce_fn(ij)  # ij = tuple of two AACIndexedTuple
        for ij in indexed_pairs
        if _synonymous_fn(ij[0][1], ij[1][1])
    )

    return {*indexed_filtered_pairs}
