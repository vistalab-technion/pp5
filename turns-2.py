import os
import re
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from dataset import ProteinDataset, get_valid_structures
from src.pp5.prec import ProteinRecord

now = datetime.now()
dt_string = now.strftime("%d-%m-%Y-%H-%M-%S")

# Folder for analysis
ANALYSIS_FOLDER = "./analysis-2-refined"

os.makedirs(ANALYSIS_FOLDER, exist_ok=True)
os.makedirs(f"{ANALYSIS_FOLDER}/turn_type_frequency", exist_ok=True)
os.makedirs(f"{ANALYSIS_FOLDER}/seq_frequency", exist_ok=True)
os.makedirs(f"{ANALYSIS_FOLDER}/cross_bond_angles/turn_type/pdb", exist_ok=True)
os.makedirs(f"{ANALYSIS_FOLDER}/cross_bond_angles/turn_type/af", exist_ok=True)
os.makedirs(f"{ANALYSIS_FOLDER}/cross_bond_angles/sequence/pdb", exist_ok=True)
os.makedirs(f"{ANALYSIS_FOLDER}/cross_bond_angles/sequence/af", exist_ok=True)

# Load PDB dataset
available_pdbs = get_valid_structures()
pdb_dataset = ProteinDataset(available_pdbs, "re")
af_dataset = ProteinDataset(available_pdbs, "af")


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
    plt.savefig(f"{ANALYSIS_FOLDER}/dihedral-angles-{title}.png")

    plt.figure()
    plt.scatter(phi_psi[1:, 0], phi_psi[:-1, 1], s=0.001)
    plt.xlim([-180, 180])
    plt.ylim([-180, 180])
    plt.xlabel(r"$\phi_{k+1}$")
    plt.ylabel(r"$\psi_{k}$")
    plt.title(f"Cross-bond angles: {title} ({n=})")
    plt.tight_layout()
    plt.savefig(f"{ANALYSIS_FOLDER}/cross-bond-angles-{title}.png")


def _get_phi_psi_from_prec(prec_: ProteinRecord):
    phi_psi_omega = []
    dihedral_angles = prec_.dihedral_angles
    for _, dihedral_angle in dihedral_angles.items():
        phi, psi, omega = (
            dihedral_angle.phi_deg,
            dihedral_angle.psi_deg,
            dihedral_angle.omega_deg,
        )
        phi_psi_omega.append({"phi": phi, "psi": psi, "omega": omega})
    return pd.DataFrame(phi_psi_omega)


def _get_sequence(prec_: ProteinRecord):
    for _, residue_record in prec_._residue_recs.items():
        yield residue_record.name, residue_record.secondary


def plot_angles_by_dictkeys(
    angles_dict: dict,
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


# dses = {"pdb": pdb_dataset, "af": af_dataset}
# for ds_name, ds in dses.items():
#     phi_psis = []
#     for prec in ds:
#         phi_psis.append(_get_phi_psi_from_prec(prec)[["phi", "psi"]])
#     phi_psi_vals = pd.concat(phi_psis)
#     plot_dihedral_angles(phi_psi_vals, ds_name)

# Regex pattern to find different types of turns.
# pattern = r"[H|E](-{1,5}T{2}-{1,5}|TTTT)[H|E]"
pattern = r"[E](-{1,3}T{2}-{1,3}|TTTT)[E]"
# pattern = r"[E](-T{2}-|TTTT|-TT--|--TT-)[E]"

# Select turns in pdb_dataset
for ds_name, ds in {"pdb": pdb_dataset, "af": af_dataset}.items():
    turn_type_freq = {}
    seq_freq = {}
    turn_type_angles = {}
    turn_seq_omega_angles = {}
    turn_seq_angles = {}

    for prec in ds:
        sequence = "".join([s for s, _ in _get_sequence(prec)])
        secondary_structure = "".join([s for _, s in _get_sequence(prec)])
        matches = re.finditer(pattern, secondary_structure)
        for match in matches:
            start, end = match.start(), match.end()
            matched_substring = match.group()

            # Save the frequency of each turn type.
            if matched_substring in turn_type_freq.keys():
                turn_type_freq[matched_substring] += 1
            else:
                turn_type_freq[matched_substring] = 1

            # Get locations of "T" in the matched substring.
            t_locations = [i for i, c in enumerate(matched_substring) if c == "T"]
            turn_sequence_list = []
            if len(t_locations) == 2:
                core_turn_location_start = start + t_locations[0]
                core_turn_location_end = start + t_locations[0] + 2
                turn_sequence_list.append(
                    sequence[core_turn_location_start:core_turn_location_end]
                )
            elif len(t_locations) == 4:
                core_turn_location_start = start + t_locations[1]
                core_turn_location_end = start + t_locations[1] + 2
                turn_sequence_list.append(
                    sequence[core_turn_location_start:core_turn_location_end]
                )
            else:
                raise RuntimeError("Unexpected number of T's in the matched substring.")

            # Dihedral angles of the turn.
            turn_angles = _get_phi_psi_from_prec(prec).iloc[
                core_turn_location_start:core_turn_location_end
            ]
            turn_cross_bond_angle = [
                turn_angles["phi"].values[1],
                turn_angles["psi"].values[0],
            ]
            turn_omega = turn_angles["omega"].values[1]  # Omega_k+1

            # Save the turn.
            turn_sequence = "".join(turn_sequence_list)
            turn_secondary = secondary_structure[start:end]

            if turn_sequence.endswith("P"):
                print(
                    prec.pdb_id,
                    turn_sequence,
                    f"start={core_turn_location_start}",
                    f"end={core_turn_location_end}",
                    f"cross_bond_angle={turn_cross_bond_angle}",
                    f"omega={turn_omega}",
                )

            if turn_sequence in seq_freq.keys():
                seq_freq[turn_sequence] += 1
            else:
                seq_freq[turn_sequence] = 1

            if turn_sequence in turn_seq_angles.keys():
                turn_seq_angles[turn_sequence].append(turn_cross_bond_angle)
            else:
                turn_seq_angles[turn_sequence] = [turn_cross_bond_angle]

            if turn_sequence in turn_seq_omega_angles.keys():
                turn_seq_omega_angles[turn_sequence].append(turn_omega)
            else:
                turn_seq_omega_angles[turn_sequence] = [turn_omega]

            if matched_substring in turn_type_angles.keys():
                turn_type_angles[matched_substring].append(turn_cross_bond_angle)
            else:
                turn_type_angles[matched_substring] = [turn_cross_bond_angle]

    plot_angles_by_dictkeys(
        turn_seq_angles, f"{ANALYSIS_FOLDER}/cross_bond_angles/sequence/", save=True
    )
    plot_angles_by_dictkeys(
        turn_type_angles, f"{ANALYSIS_FOLDER}/cross_bond_angles/turn_type/", save=True
    )

    # Sort turns based on their frequency
    turn_type_freq = dict(sorted(turn_type_freq.items(), key=lambda item: -item[1]))
    seq_freq = dict(sorted(seq_freq.items(), key=lambda item: -item[1])[:100])
    print(f"Turn type frequency: {turn_type_freq}")

    # Make bar plot for turn type frequency
    plt.figure(figsize=(20, 5))
    plt.bar(turn_type_freq.keys(), turn_type_freq.values())
    plt.title(f"Turn type frequency: PDB-redo")
    plt.xticks(rotation=45)
    plt.xlim(-0.5, len(turn_type_freq.keys()) - 0.5)
    plt.tight_layout()
    plt.savefig(f"{ANALYSIS_FOLDER}/turn_type_frequency/{ds_name}.png", dpi=200)
    plt.show()
    plt.close()

    plt.figure(figsize=(25, 5))
    plt.bar(list(seq_freq.keys()), list(seq_freq.values()))
    plt.title(f"Sequence frequency: PDB-redo")
    plt.xticks()
    plt.xlim(-0.5, 100 - 0.5)
    plt.tight_layout()
    plt.savefig(f"{ANALYSIS_FOLDER}/seq_frequency/{ds_name}.png", dpi=200)
    plt.show()
    plt.close()

    # Save turn_seq_angles and turn_type_angles into a pickle file
    with open(f"{ANALYSIS_FOLDER}/turn_seq_angles_{ds_name}.pkl", "wb") as f:
        pickle.dump(turn_seq_angles, f)

    with open(f"{ANALYSIS_FOLDER}/turn_type_angles_{ds_name}.pkl", "wb") as f:
        pickle.dump(turn_type_angles, f)

    with open(f"{ANALYSIS_FOLDER}/turn_seq_angles_w_omega_{ds_name}.pkl", "wb") as f:
        pickle.dump({"omega": turn_seq_omega_angles, "cross_bond": turn_seq_angles}, f)
