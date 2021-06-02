import itertools as it

import pytest

from pp5.codons import (
    CODONS,
    AAC_SEP,
    CODON_RE,
    N_CODONS,
    AA_CODONS,
    SYN_CODON_IDX,
    SYN_CODON_IDX_UNIQ,
    aac2c,
    aac2aa,
    is_synonymous,
    aac_tuple_pairs,
    is_synonymous_tuple,
)


class TestCodonRe:
    def test_invalid(self):
        invalids = ["AAZ", "AACD", "XAAC", "123", "AC", "!@#", "Z-GTA", "A-ZCA"]
        for s in invalids:
            m = CODON_RE.match(s)
            assert m is None, s

    def test_codon_only(self):
        for s in CODONS:
            m = CODON_RE.match(s)
            assert m is not None, s
            aa, c = m.groups()
            assert aa is None, s
            assert c == s, s

    def test_aa_codon(self):
        for s in AA_CODONS:
            m = CODON_RE.match(s)
            assert m is not None, s
            aa, c = m.groups()
            assert aa == s[0], s
            assert c == s[2:], s

    @pytest.mark.parametrize("aac", AA_CODONS)
    def test_aac2aa_aac2c(self, aac):
        m = CODON_RE.match(aac)
        aa, c = m.groups()
        assert aa == aac2aa(aac)
        assert c == aac2c(aac)


class TestSynonymous:
    @pytest.mark.parametrize("aac1", AA_CODONS)
    def test_is_synonymous(self, aac1):
        for aac2 in AA_CODONS:
            expected = aac1.split(AAC_SEP)[0] == aac2.split(AAC_SEP)[0]
            actual = is_synonymous(aac1, aac2)
            assert expected == actual

    def test_syn_codon_idx(self):
        for i, j in SYN_CODON_IDX:
            assert is_synonymous(AA_CODONS[i], AA_CODONS[j])

    def test_syn_codon_idx_uniq(self):
        assert len(set(SYN_CODON_IDX_UNIQ)) == len(SYN_CODON_IDX_UNIQ)
        for i, j in SYN_CODON_IDX_UNIQ:
            assert is_synonymous(AA_CODONS[i], AA_CODONS[j])


class TestAACTuplePairs:
    @pytest.mark.parametrize("unique", [False, True], ids=["non_uniq", "uniq"])
    @pytest.mark.parametrize("synonymous", [False, True], ids=["non_syn", "syn"])
    @pytest.mark.parametrize("k", [1, 2])
    def test_aac_tuple_pairs(self, k, synonymous, unique):
        if k > 1 and synonymous and not unique:
            return

        tuple_pairs = aac_tuple_pairs(k, synonymous=synonymous, unique=unique)

        # number of all k-tuple pairs
        n_k_tuples = N_CODONS ** k
        num_all_k_tuple_pairs = n_k_tuples ** 2
        num_unique_k_tuple_pairs = num_all_k_tuple_pairs / 2 + n_k_tuples / 2

        print(f"{n_k_tuples=}, {num_all_k_tuple_pairs=}, {num_unique_k_tuple_pairs=}")

        if not synonymous:
            if not unique:
                assert len(tuple_pairs) == num_all_k_tuple_pairs
            else:
                assert len(tuple_pairs) == num_unique_k_tuple_pairs

        for (i, aact1), (j, aact2) in tuple_pairs:
            assert len(aact1) == len(aact2) == k
            if synonymous:
                assert is_synonymous_tuple(aact1, aact2)
