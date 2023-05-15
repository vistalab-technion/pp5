import os
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load turn_type_angles, and sequence_angles
ANALYSIS_FOLDER = "./analysis-2-refined"
CROSS_BOND_ANGLE_PLOTS = f"{ANALYSIS_FOLDER}/cross_bond_angle_plots"
os.makedirs(CROSS_BOND_ANGLE_PLOTS, exist_ok=True)

dataset_name = "pdb"

os.makedirs(f"{CROSS_BOND_ANGLE_PLOTS}/{dataset_name}", exist_ok=True)


def plot_cross_bond_angles(
    cross_bond: np.ndarray, omega: np.ndarray, title: str, ds_name: str
):
    plt.figure()
    plt.scatter(cross_bond[:, 0], cross_bond[:, 1], s=1, c=omega, cmap="Dark2")
    plt.xlim([-180, 180])
    plt.ylim([-180, 180])
    plt.xlabel(r"$\phi_{k+1}$")
    plt.ylabel(r"$\psi_{k}$")
    plt.title(f"Cross-bond angles. (n={cross_bond.shape[0]})")
    plt.tight_layout()
    plt.colorbar()
    plt.savefig(f"{CROSS_BOND_ANGLE_PLOTS}/{ds_name}/{title}.png")


def classify_turn_angle(angles: np.ndarray) -> [float, float, float]:
    angles = np.array(angles)
    # Conditions to be in turn 1:
    # - phi_k+1 < 0, psi_k < 45
    turn_1 = angles[(angles[:, 0] < 0) & (angles[:, 1] < 45)]

    # Conditions to be in turn 2:
    # - phi_k+1 > 45, psi_k > 90
    turn_2 = angles[(angles[:, 0] > 45) & (angles[:, 1] > 90)]

    # Conditions to be in turn 3:
    # Turns that do not belong to turn 1 or turn 2
    turn_3 = angles[
        ~((angles[:, 0] < 0) & (angles[:, 1] < 45))
        & ~((angles[:, 0] > 45) & (angles[:, 1] > 90))
    ]

    turn_1_pct = 100 * (turn_1.shape[0] / angles.shape[0])
    turn_2_pct = 100 * (turn_2.shape[0] / angles.shape[0])
    turn_3_pct = 100 * (turn_3.shape[0] / angles.shape[0])
    return turn_1_pct, turn_2_pct, turn_3_pct


with open(f"{ANALYSIS_FOLDER}/turn_seq_angles_w_omega_{dataset_name}.pkl", "rb") as f:
    all_angles = pickle.load(f)
    sequence_angles = all_angles["cross_bond"]
    omega_angles = all_angles["omega"]
    f.close()

all_turn_angles = np.concatenate(list(sequence_angles.values()), axis=0)
all_omega_angles = np.concatenate(list(omega_angles.values()), axis=0)

# Plot cross-bond angles
plot_cross_bond_angles(all_turn_angles, all_omega_angles, "all", dataset_name)

pattern_frequency = []

all_turn_1_pct, all_turn_2_pct, all_turn_3_pct = classify_turn_angle(all_turn_angles)
pattern_frequency.append(
    {
        "pattern": "**",
        "n": len(all_turn_angles),
        "turn_1": all_turn_1_pct,
        "turn_2": all_turn_2_pct,
        "turn_3": all_turn_3_pct,
    }
)
print(
    f"pattern='**'."
    f" Turn 1: {all_turn_1_pct:.2f}%, Turn 2: {all_turn_2_pct:.2f}%, "
    f" Turn 3:  {all_turn_3_pct:.2f}%."
)


# sort sequence_angles by length
sequence_angles = dict(sorted(sequence_angles.items(), key=lambda item: -len(item[1])))
for sequence, angles_ in sequence_angles.items():
    cross_bond_angles_ = np.array(angles_)
    curr_omega_angles = np.array(omega_angles[sequence])

    plot_cross_bond_angles(
        cross_bond_angles_, curr_omega_angles, sequence, dataset_name
    )

    turn_1_pct_, turn_2_pct_, turn_3_pct_ = classify_turn_angle(cross_bond_angles_)

    print(
        f"{sequence=} (n={len(angles_)}). "
        f"Turn 1: {turn_1_pct_:.2f}%,  "
        f"Turn 2: {turn_2_pct_:.2f}%, "
        f"Turn 3:  {turn_3_pct_:.2f}%"
    )
    pattern_frequency.append(
        {
            "pattern": sequence,
            "n": len(angles_),
            "turn_1": turn_1_pct_,
            "turn_2": turn_2_pct_,
            "turn_3": turn_3_pct_,
        }
    )

# Define 20 available amino acids
unique_aas = set(i for i in "".join(sequence_angles.keys()))

# Find the keys that end with a specific amino acid
for aa in unique_aas:
    pattern = f"*{aa}"
    pattern_cross_bond_angles = []
    pattern_omega_angles = []
    for sequence, cross_bond_angles_ in sequence_angles.items():
        if sequence.endswith(aa):
            pattern_cross_bond_angles.append(cross_bond_angles_)
            pattern_omega_angles.append(omega_angles[sequence])

    pattern_cross_bond_angles = np.concatenate(pattern_cross_bond_angles, axis=0)
    pattern_omega_angles = np.concatenate(pattern_omega_angles, axis=0)

    plot_cross_bond_angles(
        pattern_cross_bond_angles, pattern_omega_angles, pattern, dataset_name
    )
    pattern_turn_1_pct, pattern_turn_2_pct, pattern_turn_3_pct = classify_turn_angle(
        pattern_cross_bond_angles
    )
    print(
        f"{pattern=} (n={pattern_cross_bond_angles.shape[0]}). "
        f"Turn 1: {pattern_turn_1_pct:.2f}%,  "
        f"Turn 2: {pattern_turn_2_pct:.2f}%, "
        f"Turn 3:  {pattern_turn_3_pct:.2f}%"
    )
    pattern_frequency.append(
        {
            "pattern": pattern,
            "n": pattern_cross_bond_angles.shape[0],
            "turn_1": pattern_turn_1_pct,
            "turn_2": pattern_turn_2_pct,
            "turn_3": pattern_turn_3_pct,
        }
    )

# Find the keys that start with a specific amino acid
for aa in unique_aas:
    pattern = f"{aa}*"
    pattern_cross_bond_angles = []
    pattern_omega_angles = []
    for sequence, cross_bond_angles_ in sequence_angles.items():
        if sequence.startswith(aa):
            pattern_cross_bond_angles.append(cross_bond_angles_)
            pattern_omega_angles.append(omega_angles[sequence])

    pattern_cross_bond_angles = np.concatenate(pattern_cross_bond_angles, axis=0)
    pattern_omega_angles = np.concatenate(pattern_omega_angles, axis=0)

    plot_cross_bond_angles(
        pattern_cross_bond_angles, pattern_omega_angles, pattern, dataset_name
    )
    pattern_turn_1_pct, pattern_turn_2_pct, pattern_turn_3_pct = classify_turn_angle(
        pattern_cross_bond_angles
    )
    print(
        f"{pattern=} (n={pattern_cross_bond_angles.shape[0]}). "
        f"Turn 1: {pattern_turn_1_pct:.2f}%,  "
        f"Turn 2: {pattern_turn_2_pct:.2f}%, "
        f"Turn 3:  {pattern_turn_3_pct:.2f}%"
    )
    pattern_frequency.append(
        {
            "pattern": pattern,
            "n": pattern_cross_bond_angles.shape[0],
            "turn_1": pattern_turn_1_pct,
            "turn_2": pattern_turn_2_pct,
            "turn_3": pattern_turn_3_pct,
        }
    )

pd.DataFrame(pattern_frequency).set_index("pattern").to_csv(f"{dataset_name}.csv")
