from typing import Sequence, Tuple, Dict, Any
from pandas import DataFrame
from itertools import product, chain
from pp5.codons import CODONS, AA_CODONS, CODON_TABLE, ACIDS
from pp5.stats import categorical_histogram, relative_histogram, product_histogram, \
    ratio, factor
import numpy as np
import pandas as pd


CODON_DELIMITER = '-'


def roll(a: np.array, shifts: Sequence[int]):
    return np.stack(
        tuple(np.roll(a, -n) for n in shifts),
        axis=1
    )


def codon_tuples(
        data: DataFrame, sequence_length: int = 2, allowed_ss: Sequence[str] = ()):
    """
    :param data: Pandas dataframe with the following columns present for each residue location:
        unp_id      -- uniprot id
        unp_idx     -- location in the uniprot sequence
        codon       -- codon identity
        codon_score -- number of coding variants matching the codon identity
        secondary   -- DSSP annotation
        name        -- residue name
    :sequence_length: Size of the tuple to produce
    :allowed_ss: A list of allowed DSSP annotation. Leave empty to include all SSs.
    :return: A tuple containing a list of codon tuples and the corresponding amino acids, returned
        as delimited strings.
    """
    shifts = tuple(n for n in range(sequence_length))

    # List of allowed SS
    allowed_ss = allowed_ss if len(allowed_ss) > 0 else list(
        np.unique(data['secondary'].values))

    unp_ids = roll(data['unp_id'].values, shifts)
    unp_idxs = roll(data['unp_idx'].values, shifts)
    codons = roll(data['codon'].values, shifts)
    codon_scores = roll(data['codon_score'].values, shifts)
    sss = roll(data['secondary'].values, shifts)
    aas = roll(data['name'].values, shifts)

    idx = (np.min(codon_scores, axis=1) == 1) & \
          np.all(np.equal(unp_ids, unp_ids[:, 0][:, None]), axis=1) & \
          np.all(np.equal(unp_idxs - unp_idxs[:, 0][:, None],
                          np.array([*range(sequence_length)])), axis=1) & \
          np.array(tuple(
              all(s in allowed_ss for s in sss[i, :]) for i in range(sss.shape[0])))

    return (
        tuple(f'{CODON_DELIMITER}'.join(c) for c in codons[idx, :]),
        tuple(''.join(a) for a in aas[idx, :])
    )


def relative_codon_histogram(
        codon_hist: Dict[Any, Tuple[float, float]],
        aa_hist: Dict[Any, Tuple[float, float]]
) -> Dict[Any, Tuple[float, float]]:
    """
    :param codon_hist: A histogram approximating codon distribution given as c_1...c_n: (p(c_1,...c_n), sigma)
    :param aa_hist: A histogram approximating AA distribution given as a_1...a_n: (p(a_1,...a_n), sigma)
    :return: A histogram approximating relative codon distribution given as
        a_1...a_n: c_1...c_n: (p(c_1,...c_n|a_1,...a_n), sigma)
    """
    return relative_histogram(
        codon_hist,
        aa_hist,
        {k: ''.join([CODON_TABLE[c] for c in k.split('-')]) for k in codon_hist.keys()}
    )


def tuple_freq_analysis(
        data: DataFrame,
        sequence_length: int = 2,
        bootstraps: int = 50,
        allowed_ss=(),
) -> DataFrame:
    # Single AA and codon histograms
    codon, aa = codon_tuples(data, sequence_length=1, allowed_ss=allowed_ss)
    single_aa_hist = categorical_histogram(aa, ACIDS, bootstraps=bootstraps,
                                           normalized=True)
    single_codon_hist = categorical_histogram(codon, CODONS, bootstraps=bootstraps,
                                              normalized=True)

    # Calculate separable tuple histograms P(a1,...,an) = P(a1)....P(an)
    # and P(c1,...,an) = P(c1)....P(cn)
    model_aa_hist = {
        ''.join(a): v
        for a, v in product_histogram(single_aa_hist, n=sequence_length).items()
    }
    model_codon_hist = {
        f'{CODON_DELIMITER}'.join(c): v
        for c, v in product_histogram(single_codon_hist, n=sequence_length).items()
    }

    # Calculate relative histogram P(c1,...,cn|a1,...,an) = P(c1|a1)...P(cn|an)
    model_codon_rel_hist = relative_codon_histogram(model_codon_hist, model_aa_hist)

    # Calculate AA and codon tuple histograms
    codons, aas = codon_tuples(data, sequence_length=sequence_length,
                               allowed_ss=allowed_ss)
    aa_hist = categorical_histogram(
        samples=aas,
        bins=tuple([*model_aa_hist.keys()]),
        bootstraps=bootstraps,
        normalized=True
    )
    codon_hist = categorical_histogram(
        samples=codons,
        bins=tuple([*model_codon_hist.keys()]),
        bootstraps=bootstraps,
        normalized=True
    )

    # Calculate relative histogram P(c1,...,cn|a1,...,an)
    codon_rel_hist = relative_codon_histogram(codon_hist, aa_hist)

    # Representativeness ratio
    # R[aa][cc] = P(c1,...,cn|a1,...,an) / P(c1|a1) / ... / P(cn|an)
    rep_ratio = {
        aa:
            {
                cc: ratio(codon_rel_hist[aa][cc], model_codon_rel_hist[aa][cc])
                for cc in codon_rel_hist[aa].keys()
            }
        for aa in codon_rel_hist.keys()
    }

    # Represent ratio as dataframe
    aa_cc = [*chain(*[[(a, c) for c in rep_ratio[a].keys()] for a in rep_ratio.keys()])]
    return DataFrame(
        index=pd.MultiIndex.from_tuples(
            aa_cc,
            names=["a1...an", "c1...cn"]
        ),
        data=tuple(
            (*aa_hist[a],
             *model_aa_hist[a],
             *codon_hist[c],
             *model_codon_hist[c],
             *codon_rel_hist[a][c],
             *model_codon_rel_hist[a][c],
             *rep_ratio[a][c],
             ) for a, c in aa_cc
        ),
        columns=pd.MultiIndex.from_tuples(
            [*product(
                ("P(a1,...,an)",
                 "P(a1)...P(an)",
                 "P(c1,...,cn)",
                 "P(c1)...P(cn)",
                 "P(c1,...,cn|a1,...,an)",
                 "P(c1|a1)...P(cn|a1)",
                 "ratio"
                 ),
                ("value", "sigma")
            )],
            names=("", "")
        )
    )
