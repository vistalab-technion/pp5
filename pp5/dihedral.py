from __future__ import annotations

import math
import warnings
from math import nan
from typing import List

import Bio.PDB as PDB
import numba
import numpy as np
import uncertainties
import uncertainties.unumpy as unumpy
import uncertainties.umath as umath
from Bio.PDB.Atom import Atom
from Bio.PDB.Polypeptide import Polypeptide
from Bio.PDB.Residue import Residue
from numpy import ndarray

from pp5.external_dbs.pdb import PDBUnitCell

CONST_8PI2 = math.pi * math.pi * 8
BACKBONE_ATOMS = {'N', 'CA', 'C'}


class Dihedral(object):
    """
    Holds the three dihedral angles associated with adjacent AAs.
    Values are stored in radians.
    """

    NAMES = ('phi', 'psi', 'omega')
    SYMBOLS = dict(phi='ɸ', psi='ψ', omega='ω', deg='°', rad='ʳ', pm='±')

    def __init__(self, phi, psi, omega):
        """
        All angles should be specified in radians between (-pi, pi].
        They ir type can be either a float, or an tuple (val,
        std) containing a nominal value and standard deviation.
        """

        cls = self.__class__
        loc = locals()
        for name in self.NAMES:
            a = loc[name]
            name_std = f'{name}_std'

            if isinstance(a, float):
                val, std = a, None
            elif hasattr(a, '__len__') and len(a) == 2:
                val, std = float(a[0]), float(a[1])
            else:
                raise ValueError('Input angles must be either a float or a '
                                 'tuple (value, std)')

            setattr(self, name, val)
            setattr(self, name_std, std)
            val_deg = math.degrees(val)
            std_deg = math.degrees(std) if std else None
            setattr(cls, f'{name}_deg',
                    property(lambda s=self, n=name: s._deg(n)))
            setattr(cls, f'{name_std}_deg',
                    property(lambda s=self, n=name_std: s._deg(n)))

    def _deg(self, name):
        r = getattr(self, name)
        return math.degrees(r) if r is not None else None

    def as_dict(self, degrees=False, skip_omega=False, with_std=False):
        """
        Convert this instance into a dict.
        :param degrees: Whether to output as degrees.
        :param skip_omega: Whether to discard omega from output.
        :return: A dict with keys phi, psi and possibly omega.
        """
        names = Dihedral.NAMES
        if skip_omega:
            names = names[0:2]

        if with_std:
            names += tuple(map(lambda n: f'{n}_std', names))

        if degrees:
            attrs = map(lambda n: f'{n}_deg', names)
        else:
            attrs = names

        return {name: getattr(self, attr)
                for name, attr in zip(names, attrs)}

    @classmethod
    def from_deg(cls, phi, psi, omega):
        return cls(np.deg2rad(phi), np.deg2rad(psi), np.deg2rad(omega))

    @classmethod
    def from_rad(cls, phi, psi, omega):
        return cls(phi, psi, omega)

    @classmethod
    def empty(cls):
        return cls(nan, nan, nan)

    @staticmethod
    @numba.jit(nopython=True)
    def wraparound_diff(a1: float, a2: float):
        d = math.fabs(a1-a2)
        return min(d, 2*math.pi - d)

    def __repr__(self, deg=True):
        reprs = []
        unit_sym = self.SYMBOLS["deg" if deg else "rad"]
        for name in self.NAMES:
            val_attr = f'{name}_deg' if deg else name
            std_attr = f'{name}_std_deg' if deg else f'{name}_std'
            val = getattr(self, val_attr)
            std = getattr(self, std_attr)
            std_str = f'{self.SYMBOLS["pm"]}{std:.1f}' if std else ''
            reprs.append(f'{self.SYMBOLS[name]}={val:.1f}{std_str}{unit_sym}')
        return f'({str.join(",", reprs)})'


class DihedralAnglesEstimator(object):
    """
    Calculates dihedral angles for a polypeptide chain of a Protein.
    """

    def __init__(self, **kw):
        pass

    def estimate(self, pp: Polypeptide) -> List[Dihedral]:
        """
        Estimate the dihedral angles of a polypeptide chain.
        :param pp: A polypeptide.
        :return: A list of Dihedral objects containing the angles. The
        length of the list will be identical to the length of the
        polypeptide in AAs.
        """
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
                angles.append(Dihedral.empty())
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

    @staticmethod
    @numba.jit(nopython=True)
    def calc_dihedral2(v1: ndarray, v2: ndarray, v3: ndarray, v4: ndarray):
        """
        Calculates the dihedral angle defined by four 3d points.
        This is the angle between the plane defined by the first three
        points and the plane defined by the last three points.
        Fast approach, based on https://stackoverflow.com/a/34245697/1230403
        """
        b0 = v1 - v2
        b1 = v3 - v2
        b2 = v4 - v3

        # normalize b1 so that it does not influence magnitude of vector
        # rejections that come next
        b1 /= np.linalg.norm(b1)

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

    def _calc_fn(self, a1: Atom, a2: Atom, a3: Atom, a4: Atom):
        return self.calc_dihedral2(
            a1.get_vector().get_array(), a2.get_vector().get_array(),
            a3.get_vector().get_array(), a4.get_vector().get_array()
        )


class DihedralAnglesUncertaintyEstimator(DihedralAnglesEstimator):
    """
    Calculates dihedral angles with uncertainty estimation based on
    b-factors and error propagation.
    """

    def __init__(self, unit_cell: PDBUnitCell = None, isotropic=True, **kw):
        """
        :param unit_cell: Unit-cell of the PDB structure. Optional,
        but must be provided if isotropic=False.
        :param isotropic: Whether to use isotropic b-factor or anisotropic
        temperature factors.
        """
        super().__init__(**kw)
        self.bfactor_est = BFactorEstimator(True, unit_cell, isotropic)

    def _calc_fn(self, *atoms: Atom):
        assert len(atoms) == 4

        # Create vectors with uncertainty based on b-factor
        vs = [a.get_vector().get_array() for a in atoms]
        es = [np.sqrt(self.bfactor_est.atom_xyz(a)) for a in atoms]
        uvs = [unumpy.uarray(vs[i], es[i]) for i in range(4)]
        v1, v2, v3, v4 = uvs
        b0 = v1 - v2
        b1 = v3 - v2
        b2 = v4 - v3
        b1 /= umath.sqrt(np.sum(b1 ** 2))
        v = b0 - np.dot(b0, b1) * b1
        w = b2 - np.dot(b2, b1) * b1
        x = np.dot(v, w)
        y = np.dot(np.cross(b1, v), w)
        ang = umath.atan2(y, x)
        return ang.nominal_value, ang.std_dev


class DihedralAnglesMonteCarloEstimator(DihedralAnglesUncertaintyEstimator):
    """
    Calculates dihedral angles with uncertainty estimation based on
    monte-carlo sampling of atom locations with anisotropic temperature
    factors.
    """

    def __init__(self, unit_cell: PDBUnitCell = None, isotropic=True,
                 n_samples=100, skip_omega=False, **kw):
        """
        :param unit_cell: Unit-cell of the PDB structure. Optional,
        but must be provided if isotropic=False.
        :param isotropic: Whether to use isotropic b-factor or anisotropic
        temperature factors.
        :param n_samples: How many monte carlo samples per angle.
        """
        super().__init__(unit_cell, isotropic, **kw)
        self.n_samples = n_samples
        self.skip_omega = skip_omega

    def _calc_fn(self, *atoms: Atom):
        assert len(atoms) == 4

        # For Omega, don't do mc sampling.
        if self.skip_omega and atoms[0].get_name() == 'CA':
            return super()._calc_fn(*atoms)

        # Calculate mu and 3x3 covariance matrix of atom positions
        mu_sigma = [self.bfactor_est.mvn_mu_sigma(a) for a in atoms]

        # Sample atom coordinates from multivariate Gaussian
        samples = [np.random.multivariate_normal(mu, S, size=self.n_samples)
                   for mu, S in mu_sigma]

        angles = np.empty(self.n_samples)
        for i in range(self.n_samples):
            # Take one sample of each atom
            xs = [sample[i] for sample in samples]
            angles[i] = self.calc_dihedral2(*xs)

        return np.mean(angles), np.std(angles)


class BFactorEstimator(object):
    def __init__(self, backbone_only=True, unit_cell: PDBUnitCell = None,
                 isotropic=True, sigma_factor=1.):
        """
        :param backbone_only: Whether to only average over backbone atoms,
        i.e. N CA C (where CA means alpha-carbon).
        http://proteopedia.org/wiki/index.php/Backbone
        :param unit_cell: Unit-cell of the PDB structure. Optional,
        but must be provided if isotropic=False.
        :param isotropic: Whether to use isotropic b-factor or anisotropic
        temperature factors.
        :param sigma_factor: Constant factor to apply to covariance. For
        debugging.
        """
        super().__init__()
        if unit_cell is None and isotropic is False:
            raise ValueError("unit_cell must be provided if isotropic=False")
        self.bb_only = backbone_only
        self.unit_cell = unit_cell
        self.isotropic = isotropic
        self.sigma_factor = sigma_factor

    def average_bfactors(self, pp: PDB.Polypeptide) -> List[float]:
        """
        Calculates the average b-factor for each residue in a polypeptide
        chain. The b-factors will be returned in units of Angstroms^2.
        :param pp: The polypeptide.
        :return: A list of b-factors, the same length as the polypeptide.
        Each b-factor in the list is an average of b-factors from each atom
        (or only backbone atoms) in each residue.
        """
        mean_bfactors = []
        for res in pp:
            bfactors = []
            for atom in res:
                atom: Atom
                if self.bb_only and atom.get_name() not in BACKBONE_ATOMS:
                    continue
                bfactors.append(self.atom_avg(atom))

            mean_bfactors.append(np.mean(bfactors).item())

        return mean_bfactors

    def mvn_mu_sigma(self, a: Atom):
        """
        Calculates the mean and covariance matrix for an atom's location,
        assuming a multivariate normal (MVN) distribution, by using the
        b-factors.
        :param a: The atom.
        :return: Tuple of (mu, Sigma). Note that physical units are returned;
        mu is in Angstroms and Sigma is in Angstroms^2.
        """
        mu = a.get_vector().get_array()
        sigma = self.atom_cov_matrix(a)
        return mu, sigma

    def atom_cov_matrix(self, a: Atom):
        """
        Calculates the covariance matrix for an atom's location,
        by using the b-factors.
        :param a: The atom.
        :return: A 3x3 matrix. Note that physical units of Angstrom^2 are
        returned.
        """
        u = a.get_anisou()
        if self.isotropic or u is None:
            u = a.get_bfactor() / CONST_8PI2
            U = np.diag([u, u, u]).astype(np.float32)
            # No need for coordinate transform in this case
            S = U
        else:
            # NOTE: The order in u is U11, U12, U13, U22, U23, U33
            # This is different from the order in the mmCIF file
            U = np.zeros((3, 3), dtype=np.float32)
            i, j = np.triu_indices(3, 0)
            U[i, j] = U[j, i] = u
            # Change from direct lattice to cartesian coordinates
            S = self.unit_cell.direct_lattice_to_cartesian(U)
            S[j, i] = S[i, j]  # fix slight asymmetry due to rounding
        return S * self.sigma_factor

    def atom_avg(self, a: Atom) -> float:
        """
        :param a: An atom.
        :return: Average b-factor along X, Y, Z directions.
        """
        sigma = self.atom_cov_matrix(a)
        return np.trace(sigma) / 3.

    def atom_xyz(self, a: Atom) -> np.ndarray:
        """
        :param a: An atom.
        :return: b-factor along X, Y, Z directions.
        """
        sigma = self.atom_cov_matrix(a)
        return np.diagonal(sigma)
