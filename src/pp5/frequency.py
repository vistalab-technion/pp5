from typing import Any, Dict, Tuple, Sequence
from itertools import product

import numpy as np
from pandas import DataFrame
from scipy.stats import norm

from pp5.codons import ACIDS, CODONS, CODON_TABLE
from pp5.stats.histograms import histogram, relative_histogram

CODON_DELIMITER = "-"


def roll(a: np.array, shifts: Sequence[int]):
    return np.stack(tuple(np.roll(a, -n) for n in shifts), axis=1)


def codon_tuples(
    data: DataFrame, sequence_length: int = 2, allowed_ss: Sequence[str] = ()
):
    """
    :param data: Pandas dataframe with the following columns present for each residue
        location:
        unp_id      -- uniprot id
        unp_idx     -- location in the uniprot sequence
        codon       -- codon identity
        codon_score -- number of coding variants matching the codon identity
        secondary   -- DSSP annotation
        name        -- residue name
    :param sequence_length: Size of the tuple to produce.
    :param allowed_ss: A list of allowed DSSP annotation. Leave empty to include all
        SSs.
    :return: A tuple containing a list of codon tuples and the corresponding amino acids,
         returned as delimited strings.
    """
    shifts = tuple(n for n in range(sequence_length))

    # List of allowed SS
    allowed_ss = (
        allowed_ss if len(allowed_ss) > 0 else list(np.unique(data["secondary"].values))
    )

    unp_ids = roll(data["unp_id"].values, shifts)
    unp_idxs = roll(data["unp_idx"].values, shifts)
    codons = roll(data["codon"].values, shifts)
    codon_scores = roll(data["codon_score"].values, shifts)
    sss = roll(data["secondary"].values, shifts)
    aas = roll(data["name"].values, shifts)

    idx = (
        (np.min(codon_scores, axis=1) == 1)
        & np.all(np.equal(unp_ids, unp_ids[:, 0][:, None]), axis=1)
        & np.all(
            np.equal(
                unp_idxs - unp_idxs[:, 0][:, None], np.array([*range(sequence_length)])
            ),
            axis=1,
        )
        & np.array(
            tuple(all(s in allowed_ss for s in sss[i, :]) for i in range(sss.shape[0]))
        )
    )

    codons = codons[idx, :]
    aas = aas[idx, :]
    unps = np.array(
        [
            f"{unp_id}:{str(int(unp_idx))}"
            for unp_id, unp_idx in zip(unp_ids[idx, 0], unp_idxs[idx, 0])
        ]
    )
    _, idx = np.unique(unps, return_index=True)

    return (
        tuple(f"{CODON_DELIMITER}".join(c) for c in codons[idx, :]),
        tuple("".join(a) for a in aas[idx, :]),
    )


def relative_codon_histogram(
    codon_hist: Dict[Any, Tuple[float, float]], aa_hist: Dict[Any, Tuple[float, float]]
) -> Dict[Any, Dict[Any, Tuple[float, float]]]:
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
        {k: "".join([CODON_TABLE[c] for c in k.split("-")]) for k in codon_hist.keys()},
    )


def _remove_sigma(hist: Dict[str, Tuple[float, float]]):
    return {k: v[0] for k, v in hist.items()}


def tuple_freq_analysis(
    data: DataFrame,
    sequence_length: int = 2,
    allowed_ss=(),
) -> DataFrame:
    codon, aa = codon_tuples(data, sequence_length=1, allowed_ss=allowed_ss)
    Na = histogram(aa, ACIDS, normalized=False)
    Nc = histogram(codon, CODONS, normalized=False)
    N = sum(Nc.values())

    codons, aas = codon_tuples(
        data, sequence_length=sequence_length, allowed_ss=allowed_ss
    )
    aa_combinations = tuple(
        "".join(a) for a in product(*([[*Na.keys()]] * sequence_length))
    )
    codon_combinations = tuple(
        CODON_DELIMITER.join(a) for a in product(*([[*Nc.keys()]] * sequence_length))
    )
    Naa = histogram(aas, aa_combinations, normalized=False)
    Ncc = histogram(codons, codon_combinations, normalized=False)

    c_aa = tuple(
        tuple(
            cc.split(CODON_DELIMITER)[i] + CODON_DELIMITER + aa
            for cc, aa in zip(codons, aas)
        )
        for i in range(sequence_length)
    )
    # cc' -> aa'
    cc_aa = {
        k: "".join([CODON_TABLE[c] for c in k.split(CODON_DELIMITER)])
        for k in Ncc.keys()
    }
    all_c_aas = tuple(
        tuple(
            cc.split(CODON_DELIMITER)[i] + CODON_DELIMITER + cc_aa[cc]
            for cc in codon_combinations
        )
        for i in range(sequence_length)
    )

    Nc_aa = tuple(
        histogram(c, bins, normalized=False) for bins, c in zip(all_c_aas, c_aa)
    )

    Pc_Pc = {
        cc: tuple(
            np.float64(Nc_aa[n][c + CODON_DELIMITER + cc_aa[cc]]) / Naa[cc_aa[cc]]
            for n, c in enumerate(cc.split(CODON_DELIMITER))
        )
        for cc in codon_combinations
    }
    PcPc = {k: np.product(v) for k, v in Pc_Pc.items()}

    Pcc = {cc: np.float64(Ncc[cc]) / Naa[cc_aa[cc]] for cc in codon_combinations}

    R = {cc: np.float64(Pcc[cc]) / PcPc[cc] for cc in codon_combinations}

    zs = {
        cc: np.float64(Pcc[cc] - PcPc[cc])
        / np.sqrt(PcPc[cc] * (1.0 - PcPc[cc]) / Naa[cc_aa[cc]])
        for cc in codon_combinations
    }
    zw = {
        cc: np.float64(Pcc[cc] - PcPc[cc])
        / np.sqrt(Pcc[cc] * (1.0 - Pcc[cc]) / Naa[cc_aa[cc]])
        for cc in codon_combinations
    }

    return DataFrame(
        {
            "aa_sequence": cc_aa[k],
            "cc_sequence": k,
            #            'Nc_aa': tuple(nc[k] for nc in Nc_aa),
            #            'Pc_aa': tuple(pc[k] for pc in Pc_Pc),
            "Naa": Naa[cc_aa[k]],
            "Ncc": Ncc[k],
            "Pcc": Pcc[k],
            "PcPc": PcPc[k],
            "Pratio": R[k],
            "zs": zs[k],
            "zw": zw[k],
            "zs_p_value": norm.cdf(zs[k]),
            "zw_p_value": norm.cdf(zw[k]),
        }
        for k in codon_combinations
    )
