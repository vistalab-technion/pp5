import math
import warnings
from math import nan
from typing import List

import Bio.PDB as PDB
import numba
import numpy as np
from Bio.PDB.Atom import Atom
from Bio.PDB.Polypeptide import Polypeptide
from Bio.PDB.Residue import Residue
from numpy import ndarray
from uncertainties import unumpy, umath

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
        return math.degrees(r) if r else None

    @classmethod
    def from_deg(cls, phi, psi, omega):
        return cls(np.deg2rad(phi), np.deg2rad(psi), np.deg2rad(omega))

    @classmethod
    def from_rad(cls, phi, psi, omega):
        return cls(phi, psi, omega)

    @classmethod
    def empty(cls):
        return cls(nan, nan, nan)

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

    def _calc_fn(self, a1: Atom, a2: Atom, a3: Atom, a4: Atom):
        return calc_dihedral2(
            a1.get_vector().get_array(), a2.get_vector().get_array(),
            a3.get_vector().get_array(), a4.get_vector().get_array()
        )

    def estimate(self, pp: Polypeptide) -> List[Dihedral]:
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


class DihedralAnglesUncertaintyEstimator(DihedralAnglesEstimator):
    """
    Calculates dihedral angles with uncertainty estimation based on
    b-factors and error propagation.
    """

    def __init__(self, **kw):
        super().__init__(**kw)

    def _calc_fn(self, *atoms: Atom):
        assert len(atoms) == 4

        # Create vectors with uncertainty based on b-factor
        vs = [a.get_vector().get_array() for a in atoms]
        es = [[math.sqrt(a.get_bfactor() / CONST_8PI2)] * 3 for a in atoms]
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


class DihedralAnglesMonteCarloEstimator(DihedralAnglesEstimator):
    """
    Calculates dihedral angles with uncertainty estimation based on
    monte-carlo sampling of atom locations with anisotropic temperature
    factors.
    """

    def __init__(self, unit_cell: PDBUnitCell, isotropic=False,
                 n_samples=100, skip_omega=False):
        """
        :param unit_cell: Unit-cell of the PDB structure.
        :param isotropic: Whether to use isotropic b-factor or anisotropic
        temperature factors.
        :param n_samples: How many monte carlo samples per angle.
        """
        super().__init__()
        self.unit_cell = unit_cell
        self.isotropic = isotropic
        self.n_samples = n_samples
        self.skip_omega = skip_omega

    def mvn_mu_sigma(self, a: Atom):
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

        mu = a.get_vector().get_array()
        return mu, S

    def _calc_fn(self, *atoms: Atom):
        assert len(atoms) == 4

        # For Omega, don't do mc sampling.
        if self.skip_omega and atoms[0].get_name() == 'CA':
            return super()._calc_fn(*atoms)

        mu_sigma = [self.mvn_mu_sigma(a) for a in atoms]

        # Sample atom coordinates from multivariate Gaussian
        samples = [np.random.multivariate_normal(mu, S, size=self.n_samples)
                   for mu, S in mu_sigma]

        angles = np.empty(self.n_samples)
        for i in range(self.n_samples):
            # Take one sample of each atom
            xs = [sample[i] for sample in samples]
            angles[i] = calc_dihedral2(*xs)

        return np.mean(angles), np.std(angles)


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
