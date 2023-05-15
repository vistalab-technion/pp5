import pickle

import pandas as pd

pickle_path = "/home/sanketh/proteins/analysis/WD_WK_turns.pkl"

# Load pickle file
with open(pickle_path, "rb") as f:
    turns = pickle.load(f)
    f.close()


def extract_turn_info(df: pd.DataFrame, label: str):
    print(f"========={label}=========")
    print("PDB ID: ", df["pdb_id"].values[0])
    print("Residue Sequence: ", df["name"].sum())
    print("Secondary Structure: ", df["secondary"].sum())
    turn_location = df[df["secondary"] == "T"]
    print("Locations: ", turn_location["res_id"].values.tolist())
    print("\n")


def extract_cross_bond_angles(df: pd.DataFrame):
    ...


for turn in turns["WD"]:
    extract_turn_info(turn, "WD")

print("\n\n")


for turn in turns["WK"]:
    extract_turn_info(turn, "WK")
