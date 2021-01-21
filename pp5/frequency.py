from typing import Sequence, Tuple, Dict, Any
from pandas import DataFrame
from itertools import product, chain
from pp5.codons import CODONS, AA_CODONS, CODON_TABLE, ACIDS
from pp5.stats import categorical_histogram, relative_histogram, product_histogram, \
    ratio, factor
import numpy as np
import pandas as pd
from scipy.stats import binom


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

    codons = codons[idx, :]
    aas = aas[idx, :]
    unps = np.array([
        f"{unp_id}:{str(int(unp_idx))}"
        for unp_id, unp_idx in zip(unp_ids[idx, 0], unp_idxs[idx, 0])
    ])
    _, idx = np.unique(unps, return_index=True)

    return (
        tuple(f'{CODON_DELIMITER}'.join(c) for c in codons[idx, :]),
        tuple(''.join(a) for a in aas[idx, :])
    )


def relative_codon_histogram(
        codon_hist: Dict[Any, Tuple[float, float]],
        aa_hist: Dict[Any, Tuple[float, float]]
) -> Dict[Any, Tuple[float, float]]:
    """
    :param codon_hist: A histogram approximating codon distribution given as c_1...c_n:
        (p(c_1...c_n), sigma)
    :param aa_hist: A histogram approximating AA distribution given as a_1...a_n:
        (p(a_1...a_n), sigma)
    :return: A histogram approximating relative codon distribution given as
        a_1...a_n: c_1...c_n: (p(c_1...c_n|a_1...a_n), sigma)
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
                                           normalized=False)
    single_codon_hist = categorical_histogram(codon, CODONS, bootstraps=bootstraps,
                                              normalized=False)

    aa_combinations = tuple(
        ''.join(a) for a in product(*([[*single_aa_hist.keys()]] * sequence_length)))
    codon_combinations = tuple(CODON_DELIMITER.join(a) for a in product(
        *([[*single_codon_hist.keys()]] * sequence_length)))

    # Calculate AA and codon tuple histograms
    codons, aas = codon_tuples(data, sequence_length=sequence_length,
                               allowed_ss=allowed_ss)
    aa_hist = categorical_histogram(
        samples=aas,
        bins=aa_combinations,
        bootstraps=1,
        normalized=False
    )
    codon_hist = categorical_histogram(
        samples=codons,
        bins=codon_combinations,
        bootstraps=1,
        normalized=False
    )

    # P(c|a)
    p_ca = {}
    for v in relative_codon_histogram(single_codon_hist, single_aa_hist).values():
        p_ca = {**p_ca, **v}

    # P(c1...cn|a1...an)
    p_cc_aa = {
        CODON_DELIMITER.join(k): v
        for k, v in product_histogram(p_ca, n=sequence_length).items()
    }

    def bin_quantile(k, n, p):
        return binom.cdf(k, n, p)

    cc_aa = {k: ''.join([CODON_TABLE[c] for c in k.split(CODON_DELIMITER)]) for k in
             codon_hist.keys()}

    quantiles = {
        cc:
            (
                bin_quantile(ncc[0], aa_hist[cc_aa[cc]][0], p_cc_aa[cc][0]),
                bin_quantile(ncc[0], aa_hist[cc_aa[cc]][0],
                             p_cc_aa[cc][0] + p_cc_aa[cc][1]),
                bin_quantile(ncc[0], aa_hist[cc_aa[cc]][0],
                             p_cc_aa[cc][0] - p_cc_aa[cc][1])
            )
        for cc, ncc in codon_hist.items()
    }

    return DataFrame(
        {
            'aa_sequence': cc_aa[k],
            'cc_sequence': k,
            'N_cc': codon_hist[k][0],
            'N_aa': aa_hist[cc_aa[k]][0],
            'P_cc_aa': p_cc_aa[k][0],
            'σP_cc_aa': p_cc_aa[k][1],
            'Q': quantiles[k][0],
            'Q-σ': quantiles[k][1],
            'Q+σ': quantiles[k][2],
        }
        for k in quantiles
    )

