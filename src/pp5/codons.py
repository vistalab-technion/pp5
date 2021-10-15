import re
import itertools as it
from typing import Set, Tuple, Union, Iterator, Sequence, cast

import numpy as np
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
N_CODONS = len(CODONS)
AA_CODONS: Sequence[str] = sorted(codon2aac(c) for c in CODONS)

ACIDS = sorted(set([aac[0] for aac in AA_CODONS]))
N_ACIDS = len(ACIDS)
ACIDS_1TO3 = IUPACData.protein_letters_1to3
ACIDS_1TO1AND3 = {aa: f"{aa} ({ACIDS_1TO3[aa]})" for aa in ACIDS}

UNKNOWN_NUCLEOTIDE = "Z"
MISSING_CODON = "---"
UNKNOWN_CODON = UNKNOWN_NUCLEOTIDE * 3
UNKNOWN_AA = "X"

CODON_RE = re.compile(
    rf'^(?:(?P<aa>[{str.join("", ACIDS)}]){AAC_SEP})?(?P<codon>{str.join("|", CODONS)})$',
    re.VERBOSE | re.IGNORECASE,
)


AA = str
AAC = str
AACTuple = Tuple[AAC, ...]
AATuple = Tuple[AA, ...]
AACIndexedTuple = Tuple[int, AACTuple]


def aac_join(aa: str, c: str, validate: bool = True) -> AAC:
    """
    Joins an amino acid and codon into a single string separated by AAC_SEP.
    :param aa: An amino acid.
    :param c: A codon.
    :param validate: Whether to raise an error if the resulting AAC is invalid.
    :return: The AAC string.
    """
    aac = f"{aa}{AAC_SEP}{c}"
    if validate and aac not in AA_CODONS:
        raise ValueError(f"Invalid AA={aa} or codon={c}")
    return aac


def aac_split(aac: AAC, validate: bool = True) -> Tuple[str, str]:
    """
    Splits an AAC string into its AA and codon.
    :param aac: An AAC string.
    :param validate: Whether to raise an error if the resulting AA or codon is invalid.
    :return: A tuple of (aa, codon).
    """
    aa, c, *_ = aac.split(AAC_SEP)
    if validate and (aa not in ACIDS or c not in CODONS or len(_) > 0):
        raise ValueError(f"Invalid AAC={aac}")
    return aa, c


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


def aact2aat(aact: AACTuple) -> AATuple:
    """
    Convert a tuple of AACs e.g. (A-GCA, A-GCC) to a tuple of the corresponding AAs,
    e.g. (A, A).
    :param aact: The AAC tuple.
    :return: An AA tuple.
    """
    return tuple(aac2aa(aac) for aac in aact)


def aact_str2tuple(aact_str: str) -> AACTuple:
    """
    Converts a string of separated AAC's to a tuple.
    :param aact_str: The string to convert, e.g. "A-GCA_C-CCC"
    :return: The tuple, e.g. ("A-GCA", "C-CCC").
    """
    return tuple(aact_str.split(AAC_TUPLE_SEP))


def aact_tuple2str(aact: Union[AACTuple, Sequence[str]]) -> str:
    """
    Converts a tuple of AACs to a string representation.
    :param aact: The tuple to convert, e.g. ("A-GCA", "C-CCC").
    :return: The string representation, e.g. "A-GCA_C-CCC"
    """
    return str.join(AAC_TUPLE_SEP, aact)


def is_synonymous(aac1: AAC, aac2: AAC) -> bool:
    """
    Whether two AAC's are synonymous, i.e. codons corresponding to the same AA.
    :param aac1: First AAC.
    :param aac2: Second AAC.
    :return: True iff they're synonymous.
    """
    return aac2aa(aac1) == aac2aa(aac2)


def is_synonymous_tuple(aact1: AACTuple, aact2: AACTuple) -> bool:
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
) -> Iterator[Tuple[AACIndexedTuple, AACIndexedTuple]]:
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
    indexed_pairs = it.product(enumerate(aac_tuples(k)), enumerate(aac_tuples(k)))

    if not unique and not synonymous:
        # This is an optimization for the non-unique non-synonymous case
        yield from indexed_pairs
    else:
        for (i, aact_i), (j, aact_j) in indexed_pairs:
            if unique and j < i:
                continue
            if synonymous and not is_synonymous_tuple(aact_i, aact_j):
                continue
            yield (i, aact_i), (j, aact_j)


def aac_index_pairs(
    k: int = 1, synonymous: bool = False, unique: bool = False
) -> Iterator[Tuple[int, int]]:
    """
    Returns indices of AAC tuples.
    :param k: Length of each tuple in each returned pair.
    :param synonymous: Whether to only returns synonymous pairs of AAC tuples. Two
        AAC tuples A and B are synonymous if is_synonymous_tuple(A, B) is True.
    :param unique: Whether to only return unique pairs. If False, then a pair of
        tuples (A, B) and another pair (B, A) will be returned, otherwise only one of
        them.
    :return: A set of pairs (i, j) where i and j are each indices of AACs is the sorted
        order of AAC tuples.
    """
    yield from (
        (i, j) for (i, aact1), (j, aact2) in aac_tuple_pairs(k, synonymous, unique)
    )


# Pairs of synonymous codons indices
SYN_CODON_IDX: Sequence[Tuple[int, int]] = tuple(
    aac_index_pairs(k=1, synonymous=True, unique=False)
)

SYN_CODON_IDX_UNIQ: Set[Tuple[int, int]] = set(
    aac_index_pairs(k=1, synonymous=True, unique=True)
)
