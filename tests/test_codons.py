import itertools as it

import pytest

from pp5.codons import (
    CODONS,
    AAC_SEP,
    CODON_RE,
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
