from typing import Sequence, Tuple, Dict, Any
from pandas import DataFrame
from itertools import product, chain
from pp5.codons import CODONS, AA_CODONS, CODON_TABLE, ACIDS
from pp5.stats import categorical_histogram, relative_histogram, product_histogram, \
    ratio, factor
import numpy as np
import pandas as pd
from scipy.stats import binom, norm


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


def _remove_sigma(hist: Dict[str, Tuple[float, float]]):
    return {k: v[0] for k, v in hist.items()}


def tuple_freq_analysis(
        data: DataFrame,
        sequence_length: int = 2,
        allowed_ss=(),
) -> DataFrame:
    codon, aa = codon_tuples(data, sequence_length=1, allowed_ss=allowed_ss)
    single_aa_hist = categorical_histogram(aa, ACIDS, bootstraps=0,
                                           normalized=False)
    single_codon_hist = categorical_histogram(codon, CODONS, bootstraps=0,
                                              normalized=False)

    N = sum([v[0] for v in single_aa_hist.values()])

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
        bootstraps=0,
        normalized=False
    )
    codon_hist = categorical_histogram(
        samples=codons,
        bins=codon_combinations,
        bootstraps=0,
        normalized=False
    )

    # N(a,a')
    Naa = _remove_sigma(aa_hist)

    # N(c,c')
    Ncc = _remove_sigma(codon_hist)

    # N(a)N(a')
    NaNa = _remove_sigma({
        ''.join(k): v
        for k, v in product_histogram(single_aa_hist, n=sequence_length).items()
    })

    # N(c)N(c')
    NcNc = _remove_sigma({
        CODON_DELIMITER.join(k): v
        for k, v in product_histogram(single_codon_hist, n=sequence_length).items()
    })

    # cc' -> aa'
    cc_aa = {k: ''.join([CODON_TABLE[c] for c in k.split(CODON_DELIMITER)]) for k in
             codon_hist.keys()}

    # score statistic
    zs = {
        cc: (Ncc[cc] / Naa[cc_aa[cc]] - NcNc[cc] / NaNa[cc_aa[cc]]) / np.sqrt(
            NcNc[cc] / NaNa[cc_aa[cc]]) / (1. - NcNc[cc] / NaNa[cc_aa[cc]])
        for cc in NcNc.keys()
    }

    # Wald statistic
    zw = {
        cc: (Ncc[cc] / Naa[cc_aa[cc]] - NcNc[cc] / NaNa[cc_aa[cc]]) / np.sqrt(
            Ncc[cc] / Naa[cc_aa[cc]]) / (1. - Ncc[cc] / Naa[cc_aa[cc]])
        for cc in NcNc.keys()
    }

    # score statistic undoing the aa dependency
    zs_ = {
        cc: (Ncc[cc] / Naa[cc_aa[cc]] - NcNc[cc] / Naa[cc_aa[cc]] / N) / np.sqrt(
            NcNc[cc] / Naa[cc_aa[cc]] / N) / (1. - NcNc[cc] / Naa[cc_aa[cc]] / N)
        for cc in NcNc.keys()
    }

    # Wald statistic undoing the aa dependency
    zw_ = {
        cc: (Ncc[cc] / Naa[cc_aa[cc]] - NcNc[cc] / Naa[cc_aa[cc]] / N) / np.sqrt(
            Ncc[cc] / Naa[cc_aa[cc]]) / (1. - Ncc[cc] / Naa[cc_aa[cc]])
        for cc in NcNc.keys()
    }


    # P(c|a)P(c'|a')
    PcPc = {
        cc: NcNc[cc] / NaNa[cc_aa[cc]]
        for cc in NcNc.keys()
    }

    # P(cc'|aa')
    Pcc = {
        cc: Ncc[cc] / Naa[cc_aa[cc]]
        for cc in NcNc.keys()
    }

    # Probability ratio
    R = {
        cc: Pcc[cc] / PcPc[cc]
        for cc in NcNc.keys()
    }

    return DataFrame(
        {
            'aa_sequence': cc_aa[k],
            'cc_sequence': k,
            'N_c': tuple(
                single_codon_hist[c][0] for c in k.split(CODON_DELIMITER)
            ),
            'N_a': tuple(
                single_aa_hist[CODON_TABLE[c]][0] for c in k.split(CODON_DELIMITER)
            ),
            'N_cc': codon_hist[k][0],
            'N_aa': aa_hist[cc_aa[k]][0],
            'Pcc': Pcc[k],
            'PcPc': PcPc[k],
            'Pratio': R[k],
            'zs': zs[k],
            'zw': zw[k],
            'zs_p_value': norm.cdf(zs[k]),
            'zw_p_value': norm.cdf(zw[k]),
            'zs_': zs_[k],
            'zw_': zw_[k],
            'zs__p_value': norm.cdf(zs_[k]),
            'zw__p_value': norm.cdf(zw_[k]),
        }
        for k in Pcc.keys()
    )

