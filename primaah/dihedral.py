import math
import warnings

import pandas as pd
import Bio.PDB as pdb
from Bio.PDB import Model, Chain, Residue, Atom
from Bio.PDB.PDBExceptions import PDBConstructionWarning

from primaah import PDB_DIR

warnings.simplefilter('ignore', PDBConstructionWarning)


def pp_dihedral_angles(pp: pdb.Polypeptide, degrees=False):
    """
    Return a list of phi/psi/omega dihedral angles from a Polypeptide object.
    :param pp: Polypeptide to calcalate dihedral angles for.
    :param degrees: Whther output should be degrees in range (-180, 180] or
    radians in range (-pi, pi].
    :return: A list of tuples (phi, psi, omega), with the same length as the
    polypeptide chain.
    """
    angles = []
    for i in range(len(pp)):
        aa_curr = pp[i]
        try:
            n = aa_curr['N'].get_vector()
            ca = aa_curr['CA'].get_vector()
            c = aa_curr['C'].get_vector()
        except KeyError:
            # Phi/Psi cannot be calculated for this AA
            angles.append((None, None, None))
            continue

        # Phi
        if i > 0:
            aa_prev = pp[i - 1]
            try:
                c_prev = aa_prev['C'].get_vector()
                phi = pdb.calc_dihedral(c_prev, n, ca, c)
            except KeyError:
                phi = None
        else:  # No phi for first AA
            phi = None

        # Psi
        if i < (len(pp) - 1):
            aa_next = pp[i + 1]
            try:
                n_next = aa_next['N'].get_vector()
                psi = pdb.calc_dihedral(n, ca, c, n_next)
            except KeyError:
                psi = None
        else:  # No psi for last AA
            psi = None

        # Omega
        if i > 0:
            aa_prev = pp[i - 1]
            try:
                c_prev = aa_prev['C'].get_vector()
                ca_prev = aa_prev['CA'].get_vector()
                omega = pdb.calc_dihedral(ca_prev, c_prev, n, ca)
            except KeyError:
                omega = None
        else:  # No omega for first AA
            omega = None

        if degrees:
            phi = math.degrees(phi) if phi else None
            psi = math.degrees(psi) if psi else None
            omega = math.degrees(omega) if omega else None

        angles.append((phi, psi, omega))

    return angles


def pdb_dihedral(pdb_id: str) -> pd.DataFrame:
    """
    Calculate dihedral angles for a protein based on its PDB id.

    :param pdb_id: The ID of the protein in PDB.
    :return: a dataframe with columns ('Chain', 'AA', 'Phi', 'Psi', 'Omega')
    containing the dihedral angles for all AAs in all chains.
    """
    pdb_list = pdb.PDBList(verbose=False)
    pdb_filename = pdb_list.retrieve_pdb_file(
        pdb_id, file_format='mmCif', pdir=PDB_DIR
    )

    parser = pdb.MMCIFParser()
    struct = parser.get_structure(pdb_id, pdb_filename)
    chains = list(struct.get_chains())

    pp_builder = pdb.PPBuilder()
    polypeptide_chains = pp_builder.build_peptides(struct, aa_only=1)

    df = pd.DataFrame(columns=('Chain', 'AA', 'Phi', 'Psi', 'Omega'))

    for chain_idx, polypeptide in enumerate(polypeptide_chains):
        chain = chains[chain_idx].id
        seq = polypeptide.get_sequence()
        angles = pp_dihedral_angles(polypeptide, degrees=True)

        data = [(chain, seq[i], *angles[i]) for i in range(len(seq))]
        df = df.append(pd.DataFrame(data, columns=df.columns))

    return df


if __name__ == '__main__':
    df = pdb_dihedral('6s61')

    groups = df.groupby('Chain')
    print(list(groups)[-1])
