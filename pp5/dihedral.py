import math
from math import nan
from typing import List

import Bio.PDB as PDB
import numba
import numpy as np
from numpy import ndarray
import pandas as pd
from Bio.PDB.Atom import Atom
from Bio.PDB.Polypeptide import Polypeptide
from Bio.PDB.Residue import Residue

import pp5.external_dbs.pdb
from pp5 import DATA_DIR
from pp5.external_dbs.pdb import PDBUnitCell

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


class DihedralAnglesEstimator(object):
    """
    Calculates dihedral angles for a polypeptide chain of a Protein.
    """

    def __init__(self, ):
        pass

    @staticmethod
    def _calc_fn(a1: Atom, a2: Atom, a3: Atom, a4: Atom):
        return calc_dihedral2(
            a1.get_vector().get_array(), a2.get_vector().get_array(),
            a3.get_vector().get_array(), a4.get_vector().get_array()
        )

    def estimate(self, pp: Polypeptide):
        angles = []

        # Loop over amino acids (AAs) in the polypeptide
        for i in range(len(pp)):
            phi, psi, omega = nan, nan, nan

            aa_curr: Residue = pp[i]
            aa_prev = pp[i - 1] if i > 0 else {}
            aa_next = pp[i + 1] if i < len(pp) - 1 else {}

            try:
                # Get the locations (x, y, z) of backbone atoms
                n = aa_curr['N']
                ca = aa_curr['CA']  # Alpha-carbon
                c = aa_curr['C']
            except KeyError:
                # Phi/Psi cannot be calculated for this AA
                angles.append(Dihedral())
                continue

            # Phi
            if 'C' in aa_prev:
                c_prev = aa_prev['C']
                phi = self._calc_fn(c_prev, n, ca, c)

            # Psi
            if 'N' in aa_next:
                n_next = aa_next['N']
                psi = self._calc_fn(n, ca, c, n_next)

            # Omega
            if 'C' in aa_prev and 'CA' in aa_prev:
                c_prev, ca_prev = aa_prev['C'], aa_prev['CA']
                omega = self._calc_fn(ca_prev, c_prev, n, ca)

            angles.append(Dihedral.from_rad(phi, psi, omega))

        return angles


@numba.jit(nopython=True)
def calc_dihedral2(v1: ndarray, v2: ndarray, v3: ndarray, v4: ndarray):
    """
    Calculates the dihedral angle defined by four 3d points.
    This is the angle between the plane defined by the first three
    points and the plane defined by the last three points.

    Uses faster approach, based on https://stackoverflow.com/a/34245697/1230403
    """
    b0 = v1 - v2
    b1 = v3 - v2
    b2 = v4 - v3

    # normalize b1 so that it does not influence magnitude of vector
    # rejections that come next
    b1 /= np.linalg.norm(b1)

    # vector rejections
    # v = projection of b0 onto plane perpendicular to b1
    #   = b0 minus component that aligns with b1
    # w = projection of b2 onto plane perpendicular to b1
    #   = b2 minus component that aligns with b1
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1

    # angle between v and w in a plane is the torsion angle
    # v and w may not be normalized but that's fine since tan is y/x
    x = np.dot(v, w)
    y = np.dot(np.cross(b1, v), w)
    return np.arctan2(y, x)


def calc_dihedral_montecarlo(mu_sigma, n_samples):
    assert len(mu_sigma) == 4

    angles = np.empty(n_samples)
    samples = [np.random.multivariate_normal(mu, S, size=n_samples)
               for mu, S in mu_sigma]
    for i in range(n_samples):
        # Sample atom coordinates from multivariate Gaussian
        xs = [sample[i] for sample in samples]
        angles[i] = calc_dihedral2(*xs)

    return np.mean(angles), np.std(angles)


def atoms_dihedral(a1: Atom, a2: Atom, a3: Atom, a4: Atom):
    return calc_dihedral2(
        a1.get_vector().get_array(), a2.get_vector().get_array(),
        a3.get_vector().get_array(), a4.get_vector().get_array()
    )


def atoms_dihedral_montecarlo(a1: Atom, a2: Atom, a3: Atom, a4: Atom,
                              unit_cell: PDBUnitCell, isotonic=False,
                              n_samples=100):
    """
    Dihedral angle calculation based on monte carlo sampling of atom locations.
    :param n_samples: Number of sample to draw.
    :return: Tuple of average dihedral angle and standard deviation.
    """

    def mvn_mu_sigma(a: Atom):
        u = a.get_anisou()
        b = a.get_bfactor()
        if u is None or isotonic:
            u = np.zeros((6,))
            u[0:3] = a.get_bfactor() / (8 * math.pi * math.pi)

        U = np.zeros((3, 3))
        U[[0, 1, 2], [0, 1, 2]] = u[0:3]
        U[[0, 0, 1], [1, 2, 2]] = u[3:6]
        U[[1, 2, 2], [0, 0, 1]] = u[3:6]

        B = unit_cell.B
        S = np.dot(B, np.dot(U, B.T))
        mu = a.get_vector().get_array()
        return mu, S

    mu_sigma = [mvn_mu_sigma(a) for a in [a1, a2, a3, a4]]
    return calc_dihedral_montecarlo(mu_sigma, n_samples)


def pp_dihedral_angles(pp: Polypeptide, mc_n_samples: int = 0,
                       mc_unit_cell=None, mc_isotonic=False) -> List[Dihedral]:
    """
    Return a list of phi/psi/omega dihedral angles from a Polypeptide object.
    http://proteopedia.org/wiki/index.php/Phi_and_Psi_Angles
    :param pp: Polypeptide to calcalate dihedral angles for.
    :return: A list of tuples (phi, psi, omega), with the same length as the
    polypeptide chain. Calculated as radians in range (-pi, pi].
    """

    if mc_n_samples:
        assert mc_unit_cell is not None

    def calc_fn(*atoms):
        std = nan
        if not mc_n_samples:
            d = atoms_dihedral(*atoms)
        else:
            d, std = atoms_dihedral_montecarlo(*atoms, unit_cell=mc_unit_cell,
                                               isotonic=mc_isotonic,
                                               n_samples=mc_n_samples)
            std = math.degrees(std)
        return d, std

    angles = []

    # Loop over amino acids (AAs) in the polypeptide
    for i in range(len(pp)):
        phi, phi_std = nan, nan
        psi, psi_std = nan, nan
        omega, omega_std = nan, nan

        aa_curr: Residue = pp[i]
        aa_prev = pp[i - 1] if i > 0 else {}
        aa_next = pp[i + 1] if i < len(pp) - 1 else {}

        try:
            # Get the locations (x, y, z) of backbone atoms
            n = aa_curr['N']
            ca = aa_curr['CA']  # Alpha-carbon
            c = aa_curr['C']
        except KeyError:
            # Phi/Psi cannot be calculated for this AA
            angles.append(Dihedral())
            continue

        # Phi
        if 'C' in aa_prev:
            c_prev = aa_prev['C']
            phi, phi_std = calc_fn(c_prev, n, ca, c)

        # Psi
        if 'N' in aa_next:
            n_next = aa_next['N']
            psi, psi_std = calc_fn(n, ca, c, n_next)

        # Omega
        if 'C' in aa_prev and 'CA' in aa_prev:
            c_prev, ca_prev = aa_prev['C'], aa_prev['CA']
            omega, omega_std = calc_fn(ca_prev, c_prev, n, ca)

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
