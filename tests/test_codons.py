import pytest

from pp5.codons import CODON_RE, CODONS, AA_CODONS


class TestCodonRe:
    def test_invalid(self):
        invalids = ['AAZ', 'AACD', 'XAAC', '123', 'AC', '!@#', 'Z-GTA',
                    'A-ZCA']
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
