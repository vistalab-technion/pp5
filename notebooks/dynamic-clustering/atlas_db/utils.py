import warnings
warnings.filterwarnings("ignore")

from Bio.Data import IUPACData
from Bio.PDB import PDBParser
from Bio.PDB.DSSP import dssp_dict_from_pdb_file
from Bio.PDB.Polypeptide import PPBuilder

import os
import mdtraj
import tempfile
import numpy as np
import pandas as pd
from tqdm import tqdm


DSSP_PATH = "/home/sanketh/miniconda3/envs/proteins/bin/mkdssp"

ACIDS_1TO3 = {
    a.upper(): aaa.upper() for a, aaa in IUPACData.protein_letters_1to3.items()
}
ACIDS_3TO1 = {aaa: a for a, aaa in ACIDS_1TO3.items()}
ACIDS_3TO1 = {
    **ACIDS_3TO1,
    **{
        'HID': 'H',
        'HIE': 'H',
        'HIP': 'H',
        'CYX': 'C',
        'CYM': 'C',
        'GLH': 'E',
        'ARN': 'R',
        'ASH': 'D',
        'LYN': 'K',
    }
}
def pdb_to_secondary_structure(pdb_path: str):
    """
    Uses DSSP to determine secondary structure for a PDB record.
    The DSSP codes for secondary structure used here are:
     H        Alpha helix (4-12)
     B        Isolated beta-bridge residue
     E        Strand
     G        3-10 helix
     I        Pi helix
     T        Turn
     S        Bend
     -        None
    :param pdb_path: path to the pdb structure file.
    :return: A tuple of
        ss_dict: maps from a residue id to the 1-char string denoting the
        type of secondary structure at that residue.
        keys: The residue ids.
    """
    try:
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter("ignore")
            dssp_dict, keys = dssp_dict_from_pdb_file(
                str(pdb_path), DSSP=DSSP_PATH
            )
            if len(ws) > 0:
                for w in ws:
                    print(f"Got DSSP warning for {pdb_path}: " f"{w.message}")
    except Exception as e:
        raise RuntimeError(f"Failed to get secondary structure for {pdb_path}")

    # dssp_dict maps a reisdue id to a tuple containing various things about
    # that residue. We take only the secondary structure info.
    ss_dict = {k: v[1] for k, v in dssp_dict.items()}

    return ss_dict, keys


def get_pdb_angles(file, chain=None):
    parser = PDBParser()
    struct = parser.get_structure('struct', file)
    models = [*struct.get_models()]
    assert len(models) == 1
    chains = [c for c in models[0].get_chains() if c.id == chain or chain is None]
    assert len(chains) == 1
    polypeptides = PPBuilder().build_peptides(chains[0], aa_only=True)[0]
    angles = {}
    # Get secondary structure
    ss = pdb_to_secondary_structure(file)[0]
    dihedrals = np.rad2deg(np.array(polypeptides.get_phi_psi_list()).astype(np.float32))

    for i, poly in enumerate(polypeptides):
        if i == 0 or i == len(polypeptides) - 1:
            continue

        phi_1, psi_1 = dihedrals[i]
        ss_i = ss.get((chains[0].id, (" ", poly.id[1], " ")))

        angles[i] = (
            poly.id[1],
            ACIDS_3TO1[poly.resname],
            ss_i,
            phi_1,
            psi_1,
        )

    return angles


def flatten_md_data(md_traj):
    """
    Flatten a md trajectory into a single dataframe.
    :param md_traj: MD Trajectory
    :return: static data, Angles DF.
    """
    f, temp_path = tempfile.mkstemp()
    os.close(f)
    md_traj[0].save_pdb(temp_path)
    static_angles = get_pdb_angles(temp_path)
    static_angles_df = pd.DataFrame(static_angles, index=["res_id", "AA", "SS", "phi", "psi"])
    static_data = static_angles_df.loc[["res_id", "AA", "SS"]].T.set_index("res_id")
    phi = np.rad2deg(mdtraj.compute_phi(md_traj)[1])[1:, :-1]
    psi = np.rad2deg(mdtraj.compute_psi(md_traj)[1])[1:, 1:]
    all_data = np.concatenate([phi, psi], axis=1)
    all_data_df = pd.DataFrame(all_data, columns=[
        *[f"phi:{i}" for i in static_angles_df.loc["res_id"]],
        *[f"psi:{i}" for i in static_angles_df.loc["res_id"]]
        ]
    )
    return static_data, all_data_df
