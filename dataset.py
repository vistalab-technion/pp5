import pickle
import os.path
from typing import List

import pandas as pd

PREC_FOLDER = "./data/prec/"
OUT_FOLDER = "./out/prec-collected"
RE_COLLECTION = (
    f"{OUT_FOLDER}/20230423_093836-floria-ex_EC-src_ALL-r1.8_s0.7-re-no-res-filter"
)


def get_valid_structures() -> List[str]:
    """
    :return: A list of PDB IDs that fulfill the following criteria:
        - The resolution is <= 1.8 Å
        - The R_free is <= 0.24
    """
    selected_pdbs = []
    filtered_structs = pd.read_csv(f"{RE_COLLECTION}/meta-structs_filtered.csv")
    for unp_id, group in filtered_structs.groupby("unp_id"):
        group.sort_values(by="resolution", inplace=True)
        selected_pdbs.append(group.iloc[[0], :])  # highest resolution

    selected_pdbs = pd.concat(selected_pdbs)
    return list(selected_pdbs["pdb_id"].values)


def check_if_structures_exist(pdb_ids: List[str]):
    missing_re, missing_af = 0, 0
    missing_af_structs, missing_re_structs = [], []

    for pdb_id in pdb_ids:
        pdb_id = pdb_id.replace(":", "_")

        if not os.path.exists(PREC_FOLDER + f"{pdb_id}-re.prec"):
            missing_re += 1
            missing_re_structs.append(pdb_id)

        if not os.path.exists(PREC_FOLDER + f"{pdb_id}-af.prec"):
            missing_af += 1
            missing_af_structs.append(pdb_id)

    print("#PDB-redo structures missing from the list: ", missing_re)
    if missing_re > 0:
        print("Missing PDB-redo structures:", missing_re_structs)

    print("#AlphaFold structures missing from the list: ", missing_af)
    if missing_af > 0:
        print("Missing AlphaFold structures:", missing_af_structs)


class ProteinDataset:
    def __init__(self, pdb_ids: List[str], source: str = "re"):
        """
        :param pdb_ids: A list of PDB IDs
        :param source: The source of the data. Can be either "re" or "af".
        """
        # Load all p-recs
        precs = []
        for pdb_id in pdb_ids:
            pdb_id = pdb_id.replace(":", "_")
            file_name = PREC_FOLDER + f"{pdb_id}-{source}.prec"
            try:
                with open(file_name, "rb") as f:
                    prec = pickle.load(f)
                    precs.append(prec)
            except FileNotFoundError:
                continue

        self.precs = precs

    def __len__(self):
        return len(self.precs)

    def __getitem__(self, item):
        return self.precs[item]

    def __iter__(self):
        return iter(self.precs)


if __name__ == "__main__":
    available_pdbs = get_valid_structures()
    check_if_structures_exist(available_pdbs)
    re_dataset = ProteinDataset(available_pdbs, source="re")
    af_dataset = ProteinDataset(available_pdbs, source="af")
