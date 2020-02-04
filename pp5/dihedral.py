import math
import warnings
from typing import NamedTuple, List

import pandas as pd
import Bio.PDB as pdb
from Bio.PDB import Model, Chain, Residue, Atom

import pp5.external_dbs.pdb
from pp5 import external_dbs
from pp5 import PDB_DIR, DATA_DIR


class Dihedral(NamedTuple):
    """
    Holds the three dihedral angles associated with adjacent AAs.
    """
    phi: float
    psi: float
    omega: float
    degrees: bool

    def __repr__(self):
        u = '°' if self.degrees else 'rad'
        return f'(ɸ={self.phi:.2f}{u}, ψ={self.psi:.2f}{u}, ' \
               f'ω={self.omega:.2f}{u})'


def pp_dihedral_angles(pp: pdb.Polypeptide, degrees=False) -> List[Dihedral]:
    """
    Return a list of phi/psi/omega dihedral angles from a Polypeptide object.
    :param pp: Polypeptide to calcalate dihedral angles for.
    :param degrees: Whther output should be degrees in range (-180, 180] or
    radians in range (-pi, pi].
    :return: A list of tuples (phi, psi, omega), with the same length as the
    polypeptide chain.
    """
    nan = math.nan
    angles = []

    # Loop over amino acids (AAs) in the polypeptide
    for i in range(len(pp)):
        aa_curr = pp[i]
        try:
            # Get the locations (x, y, z) of relevant atoms
            n = aa_curr['N'].get_vector()
            ca = aa_curr['CA'].get_vector()
            c = aa_curr['C'].get_vector()
        except KeyError:
            # Phi/Psi cannot be calculated for this AA
            angles.append(Dihedral(nan, nan, nan, degrees))
            continue

        # Phi
        if i > 0:
            aa_prev = pp[i - 1]
            try:
                c_prev = aa_prev['C'].get_vector()
                phi = pdb.calc_dihedral(c_prev, n, ca, c)
            except KeyError:
                phi = nan
        else:  # No phi for first AA
            phi = nan

        # Psi
        if i < (len(pp) - 1):
            aa_next = pp[i + 1]
            try:
                n_next = aa_next['N'].get_vector()
                psi = pdb.calc_dihedral(n, ca, c, n_next)
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
                omega = pdb.calc_dihedral(ca_prev, c_prev, n, ca)
            except KeyError:
                omega = nan
        else:  # No omega for first AA
            omega = nan

        if degrees:
            phi = math.degrees(phi) if phi else nan
            psi = math.degrees(psi) if psi else nan
            omega = math.degrees(omega) if omega else nan

        angles.append(Dihedral(phi, psi, omega, degrees))

    return angles


def pdb_dihedral(pdb_id: str) -> pd.DataFrame:
    """
    Calculate dihedral angles for a protein based on its PDB id.

    :param pdb_id: The ID of the protein in PDB.
    :return: a dataframe with columns ('Chain', 'AA', 'Phi', 'Psi', 'Omega')
    containing the dihedral angles for all AAs in all chains.
    """
    struct = pp5.external_dbs.pdb.pdb_struct(pdb_id)

    # Create polypeptide objects for each chain
    pp_builder = pdb.PPBuilder()
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
