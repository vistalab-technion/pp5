import math
from typing import NamedTuple, List

import numpy as np
import pandas as pd
import Bio.PDB as PDB
from Bio.PDB.Residue import Residue
from Bio.PDB.Polypeptide import Polypeptide
from Bio.PDB.Atom import Atom

import pp5.external_dbs.pdb
from pp5 import external_dbs
from pp5 import PDB_DIR, DATA_DIR

BACKBONE_ATOMS = {'N', 'CA', 'C'}


class Dihedral(NamedTuple):
    """
    Holds the three dihedral angles associated with adjacent AAs.
    Values are stored in radians.
    """
    phi: float = math.nan
    psi: float = math.nan
    omega: float = math.nan

    def __repr__(self, degrees=True):
        phi = math.degrees(self.phi) if degrees else self.phi
        psi = math.degrees(self.psi) if degrees else self.psi
        omega = math.degrees(self.omega) if degrees else self.omega

        u = '°' if degrees else 'rad'
        return f'(ɸ={phi:3.2f}{u}, ψ={psi:3.2f}{u}, ω={omega:3.2f}{u})'


def pp_dihedral_angles(pp: Polypeptide) -> List[Dihedral]:
    """
    Return a list of phi/psi/omega dihedral angles from a Polypeptide object.
    http://proteopedia.org/wiki/index.php/Phi_and_Psi_Angles
    :param pp: Polypeptide to calcalate dihedral angles for.
    :return: A list of tuples (phi, psi, omega), with the same length as the
    polypeptide chain. Calculated as radians in range (-pi, pi].
    """
    nan = math.nan
    angles = []

    # Loop over amino acids (AAs) in the polypeptide
    for i in range(len(pp)):
        aa_curr: Residue = pp[i]
        try:
            # Get the locations (x, y, z) of backbone atoms
            n = aa_curr['N'].get_vector()
            ca = aa_curr['CA'].get_vector()  # Alpha-carbon
            c = aa_curr['C'].get_vector()
        except KeyError:
            # Phi/Psi cannot be calculated for this AA
            angles.append(Dihedral())
            continue

        # Phi
        if i > 0:
            aa_prev = pp[i - 1]
            try:
                c_prev = aa_prev['C'].get_vector()
                phi = PDB.calc_dihedral(c_prev, n, ca, c)
            except KeyError:
                phi = nan
        else:  # No phi for first AA
            phi = nan

        # Psi
        if i < (len(pp) - 1):
            aa_next = pp[i + 1]
            try:
                n_next = aa_next['N'].get_vector()
                psi = PDB.calc_dihedral(n, ca, c, n_next)
            except KeyError:
                psi = nan
        else:  # No psi for last AA
            psi = nan

        # Omega
        if i > 0:
            aa_prev = pp[i - 1]
            try:
                c_prev = aa_prev['C'].get_vector()
                ca_prev = aa_prev['CA'].get_vector()
                omega = PDB.calc_dihedral(ca_prev, c_prev, n, ca)
            except KeyError:
                omega = nan
        else:  # No omega for first AA
            omega = nan

        angles.append(Dihedral(phi, psi, omega))

    return angles


def pp_mean_bfactor(pp: PDB.Polypeptide, backbone_only=False) -> List[float]:
    """
    Calculates the average b-factor for each residue in a polypeptide chain.
    :param pp: The polypeptide.
    :param backbone_only: Whether to only average over backbone atoms: N CA
    C (where CA means alpha-carbon).
    http://proteopedia.org/wiki/index.php/Backbone
    :return:
    """
    mean_bfactors = []
    for res in pp:
        bfactors = []
        for atom in res:
            atom: Atom
            if backbone_only and atom.get_name() not in BACKBONE_ATOMS:
                continue
            bfactors.append(atom.get_bfactor())
        mean_bfactors.append(np.mean(bfactors))

    return mean_bfactors


def pdb_dihedral(pdb_id: str) -> pd.DataFrame:
    """
    Calculate dihedral angles for a protein based on its PDB id.

    :param pdb_id: The ID of the protein in PDB.
    :return: a dataframe with columns ('Chain', 'AA', 'Phi', 'Psi', 'Omega')
    containing the dihedral angles for all AAs in all chains.
    """
    struct = pp5.external_dbs.pdb.pdb_struct(pdb_id)

    # Create polypeptide objects for each chain
    pp_builder = PDB.PPBuilder()
    polypeptide_chains = pp_builder.build_peptides(struct, aa_only=1)

    df = pd.DataFrame(columns=('Chain', 'AA', 'Phi', 'Psi', 'Omega'))

    # From each chain, calculate dihedral angles
    chains = list(struct.get_chains())
    for chain_idx, polypeptide in enumerate(polypeptide_chains):
        chain = chains[chain_idx].id
        seq = polypeptide.get_sequence()
        angles = pp_dihedral_angles(polypeptide, degrees=True)

        data = [(chain, seq[i], *angles[i]) for i in range(len(seq))]
        df = df.append(pd.DataFrame(data, columns=df.columns))

    return df


if __name__ == '__main__':
    pid = '5jkk'
    df = pdb_dihedral(pid)

    filename = DATA_DIR.joinpath(f'{pid}.angles.csv')
    print(f'Writing {filename}...')
    df.to_csv(filename, index=None)
