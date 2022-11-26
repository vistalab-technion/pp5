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
    aact2aat,
    aac_tuples,
    is_synonymous,
    aact_str2tuple,
    aact_tuple2str,
    aac_index_pairs,
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

        # tuple_pairs = aac_tuple_pairs(k, synonymous=synonymous, unique=unique)
        tuple_pairs = aac_tuple_pairs(k, synonymous=synonymous, unique=unique)
        index_pairs = aac_index_pairs(k, synonymous=synonymous, unique=unique)

        # expected number of all k-tuple pairs
        n_k_tuples = N_CODONS**k
        num_all_k_tuple_pairs = n_k_tuples**2
        num_unique_k_tuple_pairs = num_all_k_tuple_pairs / 2 + n_k_tuples / 2

        num_tuple_pairs = 0
        num_synonymous_k_tuple_pairs = 0
        for ((i, aact1), (j, aact2)), (i_, j_) in zip(tuple_pairs, index_pairs):
            assert len(aact1) == len(aact2) == k

            if synonymous:
                assert is_synonymous_tuple(aact1, aact2)
                num_synonymous_k_tuple_pairs += 1
            num_tuple_pairs += 1

            assert i == i_
            assert j == j_
            if unique:
                assert j >= i

        if not synonymous:
            if not unique:
                assert num_tuple_pairs == num_all_k_tuple_pairs
            else:
                assert num_tuple_pairs == num_unique_k_tuple_pairs

        print(
            f"{n_k_tuples=}, {num_all_k_tuple_pairs=}, {num_unique_k_tuple_pairs=}, "
            f"{num_synonymous_k_tuple_pairs=}"
        )


class TestStringConversions:
    @pytest.mark.parametrize("k", [1, 2, 3])
    @pytest.mark.parametrize("use_aas", [False, True], ids=["codons", "aas"])
    def test_conversion(self, k, use_aas):
        for t in aac_tuples(k=k):
            if use_aas:
                t = aact2aat(t)

            t_str = aact_tuple2str(t)
            assert type(t_str) == str
            t_ = aact_str2tuple(t_str)
            assert t_ == t
