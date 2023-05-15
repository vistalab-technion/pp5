import os
import re
import csv
import pickle
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

OUT_FOLDER = "/home/sanketh/proteins/out/prec-collected"
PDB_FOLDER = (
    f"{OUT_FOLDER}/20230422_172357-floria-ex_EC-src_ALL-r1.8_s0.7-no-res-filter"
)
ALPHAFOLD_FOLDER = f"{OUT_FOLDER}/20230412_115756-floria-ex_EC-src_ALL-r1.8_s0.7-af"
ANALYSIS_FOLDER = "./analysis-non-redundant"


# Make directories and subdirectories
os.makedirs(ANALYSIS_FOLDER, exist_ok=True)
os.makedirs(f"{ANALYSIS_FOLDER}/turn_type_frequency", exist_ok=True)
os.makedirs(f"{ANALYSIS_FOLDER}/seq_frequency", exist_ok=True)
os.makedirs(f"{ANALYSIS_FOLDER}/cross_bond_angles/turn_type/", exist_ok=True)
os.makedirs(f"{ANALYSIS_FOLDER}/cross_bond_angles/sequence/", exist_ok=True)


def load_data(folder_path: str):
    """
    Reads the data-precs.csv file from the folder using pandas and loads it.
    """
    file_path = f"{folder_path}/data-precs.csv"
    data = []

    with open(file_path, "r", errors="replace") as file:
        reader = csv.reader(file)
        for row in reader:
            data.append(row)

    # Now, you can convert the data list into a pandas DataFrame
    df = pd.DataFrame(data[1:], columns=data[0])
    return df


pdb = load_data(PDB_FOLDER)
af = None  # load_data(ALPHAFOLD_FOLDER)

# structures per unp
pdb_freq = {
    group_name: len(group["pdb_id"].unique())
    for group_name, group in pdb.groupby("unp_id")
}

# sort pdb_freq by value
pdb_freq = dict(sorted(pdb_freq.items(), key=lambda item: item[1], reverse=True))

# bar plot the first 50 pdb_freq keys
plt.figure(figsize=(20, 10))
plt.bar(list(pdb_freq.keys())[:100], list(pdb_freq.values())[:100])
plt.xticks(rotation=90)
plt.xlim(-0.5, 99.5)
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(f"{ANALYSIS_FOLDER}/pdbs_per_unp.pdf", dpi=300, format="pdf")

# Pick one structure per unp_id
selected_pdbs = []
for unp_id, group in pdb.groupby("unp_id"):
    available_pdbs = group["pdb_id"].unique()
    selected_pdb_id = available_pdbs[0]
    selected_pdb = group[group["pdb_id"] == selected_pdb_id]
    selected_pdbs.append(selected_pdb)

# This removes all PDB structures based on the UNP ID
# TODO: selection is arbitrary, must be based on resolution.
pdb = pd.concat(selected_pdbs)


def diagnose_data(pdb_data: pd.DataFrame, af_data: pd.DataFrame = None):
    pdb_data_ids = pdb_data["pdb_id"].unique()
    if af_data is not None:
        af_data_ids = af_data["pdb_id"].unique()

    print(f"#sequences in PDB: {len(pdb_data_ids)}")
    if af_data is not None:
        print(f"#sequences in AF: {len(af_data_ids)}")
        common_pdb_ids = set(pdb_data_ids).intersection(set(af_data_ids))
        print(f"#common sequences in PDB and AF: {len(common_pdb_ids)}")

        missing_ids = set(pdb_data_ids).difference(set(af_data_ids))
        print(f"#ids present in PDB but missing in AF: {len(missing_ids)}")

        extra_ids = set(af_data_ids).difference(set(pdb_data_ids))
        print(f"#ids present in AF but missing in PDB: {len(extra_ids)}")


def plot_dihedral_angles(df: pd.DataFrame, title: str):
    phi_psi = df[["phi", "psi"]].apply(pd.to_numeric, errors="coerce").values
    n = len(phi_psi)

    plt.figure()
    plt.scatter(phi_psi[1:-1, 0], phi_psi[1:-1, 1], s=0.001)
    plt.xlim([-180, 180])
    plt.ylim([-180, 180])
    plt.xlabel(r"$\phi_{k}$")
    plt.ylabel(r"$\psi_{k}$")
    plt.title(f"Ramachandran plot: {title} ({n=})")
    plt.tight_layout()
    plt.show()

    plt.figure()
    plt.scatter(phi_psi[1:, 0], phi_psi[:-1, 1], s=0.001)
    plt.xlim([-180, 180])
    plt.ylim([-180, 180])
    plt.xlabel(r"$\phi_{k+1}$")
    plt.ylabel(r"$\psi_{k}$")
    plt.title(f"Cross-bond angles: {title} ({n=})")
    plt.tight_layout()
    plt.show()


diagnose_data(pdb, None)
plot_dihedral_angles(pdb, "PDB")
# plot_dihedral_angles(af, "AlphaFold")

# Regex pattern to find different types of turns.
pattern = r"[H|E](-{1,5}T{2}-{1,5}|TTTT)[H|E]"


def get_turns_and_frequencies(df: pd.DataFrame, title: str, save: bool = True):
    turns = []
    turn_type_freq = {}
    seq_freq = {}

    # Loop over each sequence in the dataframe and search the following subpattern.
    for name, group in df.groupby("pdb_id"):
        sequence = group["secondary"].sum()
        matches = re.finditer(pattern, sequence)

        for match in matches:
            start, end = match.start(), match.end()
            matched_substring = match.group()

            # Save the frequency of each turn type.
            if matched_substring in turn_type_freq.keys():
                turn_type_freq[matched_substring] += 1
            else:
                turn_type_freq[matched_substring] = 1

            # Save the turn.
            turn_ = group.iloc[start:end]
            seq = turn_[turn_["secondary"] == "T"]["name"].sum()
            if seq in seq_freq.keys():
                seq_freq[seq] += 1
            else:
                seq_freq[seq] = 1

            turns.append(turn_)

    print(f"Total number of turns: {len(turns)}")
    print(f"Turn type frequency: {turn_type_freq}")

    # Sort turns based on their frequency
    turn_type_freq = dict(sorted(turn_type_freq.items(), key=lambda item: -item[1]))
    seq_freq = dict(sorted(seq_freq.items(), key=lambda item: -item[1])[:100])
    print(f"Turn type frequency: {turn_type_freq}")

    # Make bar plot for turn type frequency
    plt.figure(figsize=(20, 5))
    plt.bar(turn_type_freq.keys(), turn_type_freq.values())
    plt.title(f"Turn type frequency: {title}")
    plt.xticks(rotation=45)
    plt.xlim(-0.5, len(turn_type_freq.keys()) - 0.5)
    plt.tight_layout()
    if save:
        plt.savefig(f"{ANALYSIS_FOLDER}/turn_type_frequency/{title}.png", dpi=200)
    else:
        plt.show()
    plt.close()

    plt.figure(figsize=(25, 5))
    plt.bar(list(seq_freq.keys()), list(seq_freq.values()))
    plt.title(f"Sequence frequency: {title}")
    plt.xticks()
    plt.xlim(-0.5, 100 - 0.5)
    plt.tight_layout()
    if save:
        plt.savefig(f"{ANALYSIS_FOLDER}/seq_frequency/{title}.png", dpi=200)
    else:
        plt.show()
    plt.close()

    return turns


save_plots = True
pdb_turns = get_turns_and_frequencies(pdb, "PDB", save_plots)
# af_turns = get_turns_and_frequencies(af, "AlphaFold", save_plots)

# For each turn, extract the phi and psi when secondary structure is T.

# Group all angles by the secondary structure type.
angles_by_secondary_structure = {}

# Group all angles by the sequence type.
angles_by_sequence = {}

WD_turns = []
WK_turns = []

for turn in pdb_turns:
    subsequence = turn["secondary"].sum()
    dihedral_angles = (
        turn[turn["secondary"] == "T"][["phi", "psi"]]
        .apply(pd.to_numeric, errors="coerce")
        .values
    )
    res_sequence = turn[turn["secondary"] == "T"]["name"].sum()

    phi_k_plus_1, psi_k = dihedral_angles[1, 0], dihedral_angles[0, 1]

    if subsequence in angles_by_secondary_structure.keys():
        angles_by_secondary_structure[subsequence].append((phi_k_plus_1, psi_k))
    else:
        angles_by_secondary_structure[subsequence] = [(phi_k_plus_1, psi_k)]

    if res_sequence in angles_by_sequence.keys():
        angles_by_sequence[res_sequence].append((phi_k_plus_1, psi_k))
    else:
        angles_by_sequence[res_sequence] = [(phi_k_plus_1, psi_k)]

    # if res_sequence == "WD":
    #     WD_turns.append(turn)
    #
    # if res_sequence == "WK":
    #     WK_turns.append(turn)
    #
#
# with open("analysis/WD_WK_turns.pkl", "wb") as f:
#     pickle.dump({"WK": WK_turns, "WD": WD_turns}, f)
#     f.close()
#


def plot_angles_by_dictkeys(
    angles_dict: Dict[str, List[Tuple[float, float]]],
    folder_name: str,
    save: bool = False,
):
    # Plot angles for based on keys in the dictionary.
    for key, angles in angles_dict.items():
        # Convert to numpy array
        angles = np.array(angles)

        # Plot cross-bond angles
        plt.scatter(angles[:, 0], angles[:, 1], s=4)
        plt.xlim([-180, 180])
        plt.ylim([-180, 180])
        plt.xlabel(r"$\phi_{k+1}$")
        plt.ylabel(r"$\psi_{k}$")
        plt.title(f"Cross-bond angles: {key} (n={angles.shape[0]}).")
        plt.tight_layout()
        if save:
            plt.savefig(f"{folder_name}/{key}.png", dpi=200)
        else:
            plt.show()
        plt.close()


plot_angles_by_dictkeys(
    angles_by_secondary_structure,
    f"{ANALYSIS_FOLDER}/cross_bond_angles/turn_type/",
    save_plots,
)
plot_angles_by_dictkeys(
    angles_by_sequence,
    f"{ANALYSIS_FOLDER}/cross_bond_angles/sequence/",
    save_plots,
)
