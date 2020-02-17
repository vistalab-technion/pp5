import math
from math import nan
from typing import List

import Bio.PDB as PDB
import numpy as np
import pandas as pd
from Bio.PDB.Atom import Atom
from Bio.PDB.Polypeptide import Polypeptide
from Bio.PDB.Residue import Residue

import pp5.external_dbs.pdb
from pp5 import DATA_DIR

BACKBONE_ATOMS = {'N', 'CA', 'C'}


class Dihedral(object):
    """
    Holds the three dihedral angles associated with adjacent AAs.
    Values are stored in radians.
    """
    def __init__(self, phi: float = nan, psi: float = nan, omega: float = nan):
        """
        All angles should be specified in radians between (-pi, pi].
        """
        self._phi = phi
        self._psi = psi
        self._omega = omega

    @property
    def phi(self):
        return self._phi

    @property
    def psi(self):
        return self._psi

    @property
    def omega(self):
        return self._omega

    @property
    def phi_deg(self):
        return math.degrees(self._phi)

    @property
    def psi_deg(self):
        return math.degrees(self._psi)

    @property
    def omega_deg(self):
        return math.degrees(self._omega)

    @classmethod
    def from_deg(cls, phi: float = nan, psi: float = nan, omega: float = nan):
        return cls(math.radians(phi), math.radians(psi), math.radians(omega))

    @classmethod
    def from_rad(cls, phi: float = nan, psi: float = nan, omega: float = nan):
        return cls(phi, psi, omega)

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

        angles.append(Dihedral.from_rad(phi, psi, omega))

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

