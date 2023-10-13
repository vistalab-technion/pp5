from __future__ import annotations

import math
from math import nan
from typing import Dict, List, Union, Optional, Sequence

import numba
import numpy as np
import Bio.PDB as PDB
import uncertainties.umath as umath
import uncertainties.unumpy as unumpy
from numpy import ndarray
from pytest import approx
from Bio.PDB.Atom import Atom, DisorderedAtom
from scipy.optimize import minimize_scalar
from Bio.PDB.Residue import Residue
from Bio.PDB.Polypeptide import Polypeptide

from pp5.backbone import (
    NO_ALTLOC,
    CONST_8PI2,
    BACKBONE_ATOMS,
    BACKBONE_ATOM_C,
    BACKBONE_ATOM_N,
    BACKBONE_ATOM_CA,
    AltlocAtom,
    verify_altloc,
    altloc_ctx_all,
    atom_altloc_ids,
    residue_altloc_ids,
    residue_backbone_atoms,
)
from pp5.external_dbs.pdb import PDBUnitCell


class Dihedral(object):
    """
    Holds the three dihedral angles associated with adjacent AAs.
    Values are stored in radians in the range [-pi, pi].
    """

    NAMES = ("phi", "psi", "omega")
    SYMBOLS = dict(phi="ɸ", psi="ψ", omega="ω", deg="°", rad="ʳ", pm="±")

    def __init__(self, phi, psi, omega):
        """
        All angles should be specified in radians.
        Their type can be either a float, or a tuple (val, std) containing a
        nominal value and standard deviation.
        """

        loc = locals()
        for name in self.NAMES:
            a = loc[name]

            if isinstance(a, (float, int, np.float32, np.float64)):
                val, std = float(a), None
            elif hasattr(a, "__len__") and len(a) == 2:
                val = float(a[0])
                std = float(a[1]) if a[1] is not None else None
            else:
                raise ValueError(
                    "Input angles must be either a float or a " "tuple (value, std)"
                )

            # Shift to [-pi, pi]
            if val < -math.pi or val > math.pi:
                val = math.atan2(math.sin(val), math.cos(val))

            setattr(self, f"_{name}", val)
            setattr(self, f"_{name}_std", std)

    @property
    def phi(self):
        """
        :return: phi in radians [-pi, pi]
        """
        return self._phi

    @property
    def psi(self):
        """
        :return: psi in radians [-pi, pi]
        """
        return self._psi

    @property
    def omega(self):
        """
        :return: omega in radians [-pi, pi]
        """
        return self._omega

    @property
    def phi_std(self):
        return self._phi_std

    @property
    def psi_std(self):
        return self._psi_std

    @property
    def omega_std(self):
        return self._omega_std

    @property
    def phi_deg(self):
        return self._deg(self._phi)

    @property
    def psi_deg(self):
        return self._deg(self._psi)

    @property
    def omega_deg(self):
        return self._deg(self._omega)

    @property
    def phi_std_deg(self):
        return self._deg(self._phi_std)

    @property
    def psi_std_deg(self):
        return self._deg(self._psi_std)

    @property
    def omega_std_deg(self):
        return self._deg(self._omega_std)

    def as_dict(
        self,
        degrees=False,
        skip_omega=False,
        with_std=False,
        prefix: str = "",
        postfix: str = "",
    ):
        """
        Convert this instance into a dict.
        :param degrees: Whether to output as degrees.
        :param skip_omega: Whether to discard omega from output.
        :param with_std: Whether to include std values in the output.
        :param prefix: A prefix to add to each key.
        :return: A dict with keys phi, psi and possibly omega.
        """
        names = Dihedral.NAMES
        if skip_omega:
            names = names[0:2]

        if with_std:
            names += tuple(map(lambda n: f"{n}_std", names))

        if degrees:
            attrs = map(lambda n: f"{n}_deg", names)
        else:
            attrs = names

        return {
            f"{prefix}{name}{postfix}": getattr(self, attr)
            for name, attr in zip(names, attrs)
        }

    def __getstate__(self):
        return self.as_dict(degrees=False, skip_omega=False, with_std=True)

    def __setstate__(self, state: dict):
        init_args = {}
        for n in self.NAMES:
            init_args[n] = (state[n], state[f"{n}_std"])
        self.__init__(**init_args)

    @classmethod
    def from_deg(cls, phi, psi, omega=180.0) -> Dihedral:
        return cls(np.deg2rad(phi), np.deg2rad(psi), np.deg2rad(omega))

    @classmethod
    def from_rad(cls, phi, psi, omega=np.pi) -> Dihedral:
        return cls(phi, psi, omega)

    @classmethod
    def empty(cls) -> Dihedral:
        return cls(nan, nan, nan)

    @classmethod
    def cross_bond(cls, a0: Dihedral, a1: Dihedral) -> Dihedral:
        """
        Calculates the cross-bond dihedral angle between two adjacent residues.
        Given (phi_0, psi_0, omega_0) and (phi_1, psi_1, omega_1)
        Returns (phi_1, psi_0, omega_0).

        :param a0: First residue dihedral angles.
        :param a1: Second residue dihedral angles.
        :return: The cross-bond dihedral angles.
        """

        return cls((a1.phi, a1.phi_std), (a0.psi, a0.psi_std), (a0.omega, a0.omega_std))

    @staticmethod
    def _deg(rad: Optional[float]):
        if rad is None:
            return None
        return math.degrees(rad)

    @staticmethod
    def flat_torus_distance(
        a1: Dihedral, a2: Dihedral, degrees=False, squared=False
    ) -> float:
        """
        Computes the distance between two dihedral angles as if they were on a
        "flat torus" (also a Ramachandran Plot). Calculates a euclidean
        distance, but with a "wrap-around" at +-180, so e.g. the distance
        between -178 and 178 degrees is actually 4 degrees.
        :param a1: first angle.
        :param a2: second angle.
        :param degrees: Whether to return degrees (True) or radians (False)
        :param squared: Whether to return squared-distance.
        :return: The angle difference.
        """

        dist = flat_torus_distance_sq(
            np.array([a1.phi, a1.psi]).reshape(-1, 2),
            np.array([a2.phi, a2.psi]).reshape(-1, 2),
        )[0]
        if not squared:
            dist = math.sqrt(dist)
        if degrees:
            dist = math.degrees(dist)
        return dist

    @staticmethod
    def s1_distance(a1: Dihedral, a2: Dihedral, degrees=False):
        """
        Computes the distance between two dihedral angles using the S1
        distance metric (arc length difference) in each direction separately.
        Equivalent to flat torus distance.
        :param a1: first angle.
        :param a2: second angle.
        :param degrees: Whether to return degrees (True) or radians (False)
        :return: The angle difference.
        """
        return Dihedral._s1_distance(a1.phi, a2.phi, a1.psi, a2.psi, degrees)

    @staticmethod
    @numba.jit(nopython=True)
    def _s1_distance(phi0, phi1, psi0, psi1, degrees=False):
        dist = np.sqrt(
            np.arccos(np.cos(phi0 - phi1)) ** 2 + np.arccos(np.cos(psi0 - psi1)) ** 2
        )
        return np.degrees(dist) if degrees else dist

    @staticmethod
    def frechet_centroid(*angles: Dihedral, metric_fn=None):
        """
        Calculates the approximate centroid point of a set of points on a
        torus.
        Finds the Frechet mean of a simple distance metric on S1 (circle)
        for each angle separately.
        :param angles: Sequence of angles. Only phi, psi will be used.
        :return: A Dihedral object with phi, psi representing the centroid
        location.
        """
        if not metric_fn:
            metric_fn = Dihedral._mean_sq_metric_s1

        def frechet_mean_s1(phi):
            res = minimize_scalar(
                fun=metric_fn,
                args=(phi,),
                bounds=(-math.pi, math.pi),
                method="Bounded",
                options=dict(xatol=1e-4, maxiter=50),
            )
            # Return optimum and function value at optimum (Frechet variance)
            return res.x, math.sqrt(res.fun)

        phi_psi = np.array([(a.phi, a.psi) for a in angles], dtype=np.float32)
        m_phi = frechet_mean_s1(phi_psi[:, 0])
        m_psi = frechet_mean_s1(phi_psi[:, 1])

        return Dihedral.from_rad(m_phi, m_psi, math.pi)

    @staticmethod
    def circular_centroid(*angles: Dihedral):
        """
        Calculates the approximate centroid point of a set of points on a
        torus. Calculates the average angle with circular-wrapping for each
        direction (phi, psi) separately.
        :param angles: Sequence of angles. Only phi, psi will be used.
        :return: A Dihedral object with phi, psi representing the centroid
        location.
        """
        # Create (N,2) array of phi,psi pairs
        phi_psi = np.array([(a.phi, a.psi) for a in angles], dtype=np.float32)
        n = len(phi_psi)
        if n == 1:
            phi, psi = phi_psi[0]
            return Dihedral.from_rad((phi, 0), (psi, 0), math.pi)

        sin, cos = np.sin(phi_psi), np.cos(phi_psi)

        sigma_sin = np.nansum(sin, axis=0)  # (2, )
        sigma_cos = np.nansum(cos, axis=0)  # (2, )
        circmean = np.arctan2(sigma_sin, sigma_cos)  # output is in [-pi, pi]

        # Handle missing data: If all phi/psi angles were NaN then
        # sigma_sin/cos for them will be zero due to nansum.
        # This causes r to be zero for this angle, and then -log(0) = inf.
        # Replace mean=0, with mean=nan mark missing data, then log(nan)=nan,
        # and the std will also be nan.
        n = np.sum(~np.isnan(phi_psi), axis=0, dtype=np.float32)
        n[n == 0.0] = np.nan  # prevent 0/0 runtime warning
        r = np.hypot(sigma_sin / n, sigma_cos / n)  # will be nan for missing
        r = np.minimum(r, 1.0)  # sometimes hypot returns a value slightly above 1.0
        circstd = np.sqrt(-2 * np.log(r))

        m_phi = (circmean[0], circstd[0])
        m_psi = (circmean[1], circstd[1])
        return Dihedral.from_rad(m_phi, m_psi, math.pi)

    @staticmethod
    @numba.jit(nopython=True)
    def _mean_sq_metric_s1(phi0, phi1):
        # A metric function for S1 (mean squared)
        # Note: arccos returns values in [0, pi]
        return np.mean(np.arccos(np.cos(phi0 - phi1)) ** 2)

    def to_str(self, deg=True, with_omega=True) -> str:
        """
        Creates a string representation of this object.
        :param deg: Whether to write angles in degrees (True) or radians (False).
        :param with_omega: Whether to include the omega angle.
        :return: A string representation.
        """
        return self.__repr__(deg=deg, with_omega=with_omega)

    def __repr__(self, deg=True, with_omega=True):
        reprs = []
        unit_sym = self.SYMBOLS["deg" if deg else "rad"]
        for name in self.NAMES:
            if not with_omega and name == "omega":
                continue
            val_attr = f"{name}_deg" if deg else name
            std_attr = f"{name}_std_deg" if deg else f"{name}_std"
            val = getattr(self, val_attr)
            std = getattr(self, std_attr)
            std_str = f'{self.SYMBOLS["pm"]}{std:.1f}' if std is not None else ""
            reprs.append(f"{self.SYMBOLS[name]}={val:.1f}{std_str}{unit_sym}")
        return f'({str.join(",", reprs)})'

    def __eq__(self, other, delta=1e-10):
        if self is other:
            return True
        if not isinstance(other, Dihedral):
            return False

        self_d = self.as_dict()
        other_d = other.as_dict()
        for k, v in self_d.items():
            other_v = other_d.get(k, math.inf)
            if not v == approx(other_v, nan_ok=True):
                return False

        return True


@numba.jit(nopython=True)
def flat_torus_distance_sq(phi_psi0: np.ndarray, phi_psi1: np.ndarray):
    """
    Computes the **squared** distance between pairs of dihedral angles as if they were
    on a "flat torus" (also a Ramachandran Plot).
    Calculates a squared-euclidean distance, but with a "wrap-around" at +-180, so e.g.
    the distance between -178 and 178 degrees is actually 4 degrees.
    :param phi_psi0: (N,2) containing N (phi, psi) pairs in radians within [-pi, pi].
    :param phi_psi1: Angles corresponding to phi_psi0, must be same shape.
    :return: An array of shape (N,) containing the flat torus squared-distances.
    """
    absdiff = np.fabs(phi_psi0.reshape(-1, 2) - phi_psi1.reshape(-1, 2))
    absdiff_2pi = 2 * np.pi - absdiff
    absdiff_min = np.minimum(absdiff, absdiff_2pi)
    dist = np.sum(absdiff_min**2, axis=1)
    return dist


@numba.jit(nopython=True)
def flat_torus_distance2_sq(phi_psi0: np.ndarray, phi_psi1: np.ndarray):
    """
    Another way to calculate the flat-torus distance. Should be equivalent.

    Computes the **squared** distance between pairs of dihedral angles as if they were
    on a "flat torus" (also a Ramachandran Plot).
    Calculates a squared-euclidean distance, but with a "wrap-around" at +-180, so e.g.
    the distance between -178 and 178 degrees is actually 4 degrees.
    :param phi_psi0: (N,2) containing N (phi, psi) pairs in radians within [-pi, pi].
    :param phi_psi1: Angles corresponding to phi_psi0, must be same shape.
    :return: An array of shape (N,) containing the flat torus squared-distances.
    """
    dphi = phi_psi0[:, 0] - phi_psi1[:, 0]
    dpsi = phi_psi0[:, 1] - phi_psi1[:, 1]
    d2 = np.arccos(np.cos(dphi)) ** 2 + np.arccos(np.cos(dpsi)) ** 2
    return d2


@numba.jit(nopython=True)
def wraparound_diff(
    a1: Union[float, np.ndarray], a2: Union[float, np.ndarray], deg: bool = False
):
    d = np.fabs(a1 - a2)
    if deg:
        return np.minimum(d, 2 * 180.0 - d)
    else:
        return np.minimum(d, 2 * np.pi - d)


@numba.jit(nopython=True)
def wraparound_mean(angles: Union[np.ndarray, Sequence[float]], deg: bool = False):
    """
    Calculates an average of multiple angles with wrap-around at pi.
    :param angles: An (N,) array of angles in any value range.
    :param deg: Whether the input is in degrees (True) or radians (False). Also
        determines output units.
    :return: The mean angle, wrapped around to ±pi.
    """
    if deg:
        angles_rad = np.deg2rad(angles)
    else:
        angles_rad = angles

    angles_sin = np.sin(angles_rad)
    angles_cos = np.cos(angles_rad)
    atan_rad = np.arctan2(np.nansum(angles_sin), np.nansum(angles_cos))

    if deg:
        return np.rad2deg(atan_rad)
    return atan_rad


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


class DihedralAngleCalculator(object):
    """
    Calculates dihedral angles for a polypeptide chain of a Protein.
    """

    def __init__(self, **kw):
        pass

    def process_poly(self, pp: Polypeptide) -> List[Dihedral]:
        """
        Calculate the dihedral angles from a polypeptide chain.

        :param pp: A polypeptide.
        :return: A list of Dihedral objects containing the angles. The
        length of the list will be identical to the length of the
        polypeptide in AAs.
        """
        return [d[NO_ALTLOC] for d in self.process_poly_altlocs(pp, with_altlocs=False)]

    def process_poly_altlocs(
        self, pp: Polypeptide, with_altlocs: bool = False
    ) -> List[Dict[str, Dihedral]]:
        """
        Calculate the dihedral angles from a polypeptide chain, with optional support
        for alternate locations (altlocs).

        :param pp: A polypeptide.
        :param with_altlocs: Whether to calculate angles for each alternate location.
        :return: A list with a dict per residue in the polypepdide. Each dict maps
        from the altloc id to the dihedral angles for that altloc (if with_altlocs=True)
        Each dict will always contain the key NO_ALTLOC, which is mapped to the
        dihedral angles obtained without performing an altloc selection.
        """
        angles = []

        # Loop over amino acids (AAs) in the polypeptide
        for i in range(len(pp)):
            r_curr: Residue = pp[i]
            r_prev = pp[i - 1] if i > 0 else None
            r_next = pp[i + 1] if i < len(pp) - 1 else None

            d: Dict[str, Dihedral] = self.process_residues(
                r_curr, r_prev, r_next, with_altlocs=with_altlocs
            )
            angles.append(d)

        return angles

    def process_residues(
        self,
        r_curr: Residue,
        r_prev: Optional[Residue],
        r_next: Optional[Residue],
        with_altlocs: bool = False,
    ) -> Dict[str, Dihedral]:
        """
        Calculate the dihedral angles for a single residue.
        :param r_curr: The residue for which to calculate angles.
        :param r_prev: The previous residue.
        :param r_next: The next residue.
        :param with_altlocs: Whether to calculate angles for each alternate
        location (altloc) that exists in r_curr.
        :return: A dict containing the dihedral angles for the current residue,
        per altloc. The result will always contain the key NO_ALTLOC, which is mapped to
        the dihedral angles obtained without performing an altloc selection. Other keys
        will correspond to the altloc ids exising in r_curr (only if with_altlocs=True).
        """

        r_prev_atoms = r_prev.child_dict if r_prev is not None else {}
        r_next_atoms = r_next.child_dict if r_next is not None else {}

        try:
            # Get the backbone atoms
            n: Atom = r_curr[BACKBONE_ATOM_N]
            ca: Atom = r_curr[BACKBONE_ATOM_CA]
            c: Atom = r_curr[BACKBONE_ATOM_C]
        except KeyError:
            # Phi/Psi cannot be calculated for this AA
            return {NO_ALTLOC: Dihedral.empty()}

        c_prev: Optional[Atom] = r_prev_atoms.get(BACKBONE_ATOM_C)
        ca_prev: Optional[Atom] = r_prev_atoms.get(BACKBONE_ATOM_CA)
        n_next: Optional[Atom] = r_next_atoms.get(BACKBONE_ATOM_N)

        return self.process_disordered_atoms(
            n, ca, c, c_prev, ca_prev, n_next, with_altlocs=with_altlocs
        )

    def process_disordered_atoms(
        self,
        n: AltlocAtom,
        ca: AltlocAtom,
        c: AltlocAtom,
        c_prev: Optional[AltlocAtom],
        ca_prev: Optional[AltlocAtom],
        n_next: Optional[AltlocAtom],
        with_altlocs: bool = False,
    ) -> Dict[str, Dihedral]:

        # Calculate the dihedral angles using the default altloc conformation
        dihedrals = {NO_ALTLOC: self.process_atoms(n, ca, c, c_prev, ca_prev, n_next)}

        # Check whether any of the atoms are disordered
        curr_atoms = (n, ca, c)
        all_atoms = (n, ca, c, c_prev, ca_prev, n_next)
        disorder = (isinstance(a, DisorderedAtom) for a in curr_atoms)

        # If we don't need to account for altlocs, or there are no disordered atoms,
        # skip altloc processing by simply treating each atom as non-disordered.
        if not with_altlocs or not any(disorder):
            return dihedrals

        # Get the ids of all altloc in the backbone of the current residue
        curr_altloc_ids = atom_altloc_ids(curr_atoms)

        # For each altloc id, set the altloc id of all atoms to the current altloc,
        # and calculate dihedral angles using the resulting atom locations. If the
        # prev/next atoms also have this id, we'll use that id with them as well.
        for altloc_id in curr_altloc_ids:
            # Select current altloc_id for all associated atoms
            with altloc_ctx_all(atoms=all_atoms, altloc_id=altloc_id):
                # Make sure all disordered atoms were set to the same altloc id
                verify_altloc(all_atoms, altloc_id)

                # Calculate dihedral angles for this altloc
                d: Dihedral = self.process_atoms(n, ca, c, c_prev, ca_prev, n_next)

            dihedrals[altloc_id] = d

        return dihedrals

    def process_atoms(
        self,
        n: AltlocAtom,
        ca: AltlocAtom,
        c: AltlocAtom,
        c_prev: Optional[AltlocAtom],
        ca_prev: Optional[AltlocAtom],
        n_next: Optional[AltlocAtom],
    ) -> Dihedral:
        phi, psi, omega = nan, nan, nan

        if c_prev is not None:
            phi = self._calc_fn(c_prev, n, ca, c)

            if ca_prev is not None:
                omega = self._calc_fn(ca_prev, c_prev, n, ca)

        if n_next is not None:
            psi = self._calc_fn(n, ca, c, n_next)

        return Dihedral.from_rad(phi, psi, omega)

    def _calc_fn(self, a1: Atom, a2: Atom, a3: Atom, a4: Atom):
        """Calculates the dihedral angle between four atoms."""
        return calc_dihedral2(
            a1.get_vector().get_array(),
            a2.get_vector().get_array(),
            a3.get_vector().get_array(),
            a4.get_vector().get_array(),
        )


class DihedralAnglesUncertaintyEstimator(DihedralAngleCalculator):
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
        self.uncertainty = AtomLocationUncertainty(True, unit_cell, isotropic)

    def _calc_fn(self, *atoms: Atom):
        assert len(atoms) == 4

        # Create vectors with uncertainty based on b-factor
        vs = [a.get_vector().get_array() for a in atoms]
        es = [np.sqrt(self.uncertainty.atom_xyz(a)) for a in atoms]
        uvs = [unumpy.uarray(vs[i], es[i]) for i in range(4)]
        v1, v2, v3, v4 = uvs
        b0 = v1 - v2
        b1 = v3 - v2
        b2 = v4 - v3
        b1 /= umath.sqrt(np.sum(b1**2))
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

    def __init__(
        self,
        unit_cell: PDBUnitCell = None,
        isotropic=True,
        n_samples=100,
        skip_omega=False,
        **kw,
    ):
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
        if self.skip_omega and atoms[0].get_name() == BACKBONE_ATOM_CA:
            return super()._calc_fn(*atoms)

        # Calculate mu and 3x3 covariance matrix of atom positions
        mu_sigma = [self.uncertainty.mvn_mu_sigma(a) for a in atoms]

        # Sample atom coordinates from multivariate Gaussian
        samples = [
            np.random.multivariate_normal(mu, S, size=self.n_samples)
            for mu, S in mu_sigma
        ]

        angles = np.empty(self.n_samples)
        for i in range(self.n_samples):
            # Take one sample of each atom
            xs = [sample[i] for sample in samples]
            angles[i] = calc_dihedral2(*xs)

        return np.mean(angles), np.std(angles)


class AtomLocationUncertainty(object):
    """
    Calculates uncertainty in atom locations, based on isotropic or
    anisotropic b-factors.
    """

    def __init__(
        self,
        backbone_only=True,
        unit_cell: PDBUnitCell = None,
        isotropic=True,
        sigma_factor=1.0,
        scale_as_bfactor=False,
    ):
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
        :param scale_as_bfactor: Whether to return the uncertainties in
        regular physical units of A^2, or scaled by 8pi^2 as for b-factors
        (i.e., B = 8pi^2 U).
        """
        super().__init__()
        if unit_cell is None and isotropic is False:
            raise ValueError("unit_cell must be provided if isotropic=False")
        self.bb_only = backbone_only
        self.unit_cell = unit_cell
        self.isotropic = isotropic
        self.sigma_factor = sigma_factor
        self.scale_as_bfactor = scale_as_bfactor

    def process_poly(self, pp: PDB.Polypeptide) -> Sequence[float]:
        """
        Calculates the average uncertainties for each residue in a polypeptide
        chain. The uncertainties will be returned in units of Angstroms^2.

        :param pp: The polypeptide.
        :return: A list of uncertainties the same length as the polypeptide.
        Each value in the list is calculated based on b-factors from each
        atom (or only backbone atoms) in each residue.
        """
        return tuple(self.process_residue(res) for res in pp)

    def process_residue(self, res: Residue) -> float:
        """
        Calculates the average uncertainty over atoms of a residue.

        :param res: The residue.
        """
        bfactors = []
        for atom in res:
            atom: Atom
            if self.bb_only and atom.get_name() not in BACKBONE_ATOMS:
                continue
            bfactors.append(self.atom_avg(atom))

        res_mean = np.mean(bfactors).item()
        if self.scale_as_bfactor:
            res_mean *= CONST_8PI2

        return res_mean

    def process_residue_altlocs(self, res: Residue) -> Dict[str, float]:
        """
        Calculates the average uncertainty over atoms of a residue, for each altloc.

        :param res: The residue.
        :return: A dict mapping from altloc id to the uncertainty. The special id
        NO_ALTLOC will always be included in the result.
        """
        bfactors = {NO_ALTLOC: self.process_residue(res)}

        atoms = residue_backbone_atoms(res) if self.bb_only else tuple(res.get_atoms())
        for altloc_id in residue_altloc_ids(res):
            with altloc_ctx_all(atoms, altloc_id):
                verify_altloc(atoms, altloc_id)
                bfactors[altloc_id] = self.process_residue(res)

        return bfactors

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
        :return: Average uncertainty along X, Y, Z directions.
        """
        sigma = self.atom_cov_matrix(a)
        return np.trace(sigma) / 3.0

    def atom_xyz(self, a: Atom) -> np.ndarray:
        """
        :param a: An atom.
        :return: vector of uncertainties along X, Y, Z directions.
        """
        sigma = self.atom_cov_matrix(a)
        return np.diagonal(sigma)
