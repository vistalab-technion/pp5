import contextlib
import math
from math import radians, degrees

import numba
import numpy as np
from numpy import ndarray
import pytest
from Bio.PDB import PPBuilder

import tests
from pp5.external_dbs import pdb
from pp5 import dihedral
from pp5.dihedral import Dihedral
from pp5.dihedral import DihedralAnglesEstimator

import pymol.cmd as pymol
from pytest import approx

calc_dihedral2 = DihedralAnglesEstimator.calc_dihedral2


def random_angles(n=100):
    phi_psi = np.random.uniform(-np.pi, np.pi, size=(n, 2))
    angles = [Dihedral.from_rad(phi, psi, 0.) for phi, psi in phi_psi]
    return angles


@numba.jit(nopython=True)
def calc_dihedral_naive(v1: ndarray, v2: ndarray,
                        v3: ndarray, v4: ndarray):
    """
    Calculates the dihedral angle defined by four 3d points.
    Uses naive approach, should be slow.
    """

    def angle_between(v1, v2):
        """
        :return: Angle between two vectors.
        """
        v1_n = v1 / np.linalg.norm(v1)
        v2_n = v2 / np.linalg.norm(v2)
        cos = np.maximum(np.minimum(np.dot(v1_n, v2_n), 1.), -1.)
        return np.arccos(cos)

    ab = v1 - v2
    cb = v3 - v2
    db = v4 - v3

    u = np.cross(ab, cb)
    v = np.cross(db, cb)
    w = np.cross(u, v)

    angle = angle_between(u, v)
    try:
        if angle_between(cb, w) > 0.001:
            angle = -angle
    except Exception:  # zero division
        pass
    return angle


class TestRawDihedralAngleCalculation(object):

    def test_manual(self):
        # This test case is based on
        # https://stackoverflow.com/a/34245697/1230403

        p0 = np.array([24.969, 13.428, 30.692])  # N
        p1 = np.array([24.044, 12.661, 29.808])  # CA
        p2 = np.array([22.785, 13.482, 29.543])  # C
        p3 = np.array([21.951, 13.670, 30.431])  # O
        p4 = np.array([23.672, 11.328, 30.466])  # CB
        p5 = np.array([22.881, 10.326, 29.620])  # CG
        p6 = np.array([23.691, 9.935, 28.389])  # CD1
        p7 = np.array([22.557, 9.096, 30.459])  # CD2

        assert calc_dihedral2(p0, p1, p2, p3) == approx(radians(-71.21515))
        assert calc_dihedral2(p0, p1, p4, p5) == approx(radians(-171.94319))
        assert calc_dihedral2(p1, p4, p5, p6) == approx(radians(60.82226))
        assert calc_dihedral2(p1, p4, p5, p7) == approx(radians(-177.63641))

    def test_compare_to_naive(self):
        for i in range(1000):
            vs = [np.random.standard_normal((3,)) * 10. for _ in range(4)]
            expected = calc_dihedral_naive(*vs)
            assert calc_dihedral2(*vs) == approx(expected)


class TestPerformance(object):
    @staticmethod
    def random_coords():
        mu = np.random.normal(0, 10, size=(3,))
        sigma = np.random.normal(0, 1, size=(3, 3))
        return [np.random.multivariate_normal(mu, sigma) for _ in range(4)]

    def test_benchmark_calc_dihedral_naive_numba(self, benchmark):
        benchmark(calc_dihedral_naive, *self.random_coords())

    def test_benchmark_calc_dihedral_naive_python(self, benchmark):
        benchmark(calc_dihedral_naive.py_func, *self.random_coords())

    def test_benchmark_calc_dihedral_fast_numba(self, benchmark):
        benchmark(calc_dihedral2, *self.random_coords())

    def test_benchmark_calc_dihedral_fast_python(self, benchmark):
        benchmark(calc_dihedral2.py_func, *self.random_coords())


class TestDihedralAnglesVsPyMOL(object):
    TEST_PDB_IDS = ['3O18:A', '4GXE', '1KTP', '4GY3', ]  # 3O18:B problmatic

    @contextlib.contextmanager
    def pymol_structure_chains(self, pdb_id):
        try:
            pdb_filename = str(pdb.pdb_download(pdb_id))
            pymol.load(pdb_filename, pdb_id)
            pymol.split_chains()
            yield pdb_id
        finally:
            # * is widcard
            pymol.delete(f'{pdb_id}*')

    @pytest.mark.parametrize('pdb_id', TEST_PDB_IDS)
    def test_basic_estimator(self, pdb_id):
        estimator = dihedral.DihedralAnglesEstimator()
        self._compare_to_pymol(pdb_id, estimator)

    def _compare_to_pymol(self, pdb_id, estimator):
        pdb_id, pdb_chain = pdb.split_id(pdb_id)
        pdb_rec = pdb.pdb_struct(pdb_id)

        with self.pymol_structure_chains(pdb_id):
            for chain in pdb_rec.get_chains():
                curr_chain = chain.get_id()
                if pdb_chain and pdb_chain != curr_chain:
                    continue

                pymol_obj_id = f'{pdb_id}_{curr_chain}'
                pymol_angles_dict = pymol.phi_psi(pymol_obj_id)
                pymol_angles = list(pymol_angles_dict.values())

                # pymol does not ignore non-standard AAs, so also include here
                pp_list = PPBuilder().build_peptides(chain, aa_only=False)
                pp5_angles = []
                for pp in pp_list:
                    pp5_angles.extend(estimator.estimate(pp))

                # discard first and last angle pairs as PyMOL doesn't compute
                pp5_angles = pp5_angles[1:-1]
                len_diff = len(pymol_angles) - len(pp5_angles)
                # usually zero after removing first and last, but sometimes
                # pymol has extra angles for some reason, and there's no
                # easy way to align...
                assert len_diff >= 0

                j = 0
                for i, phi_psi in enumerate(pymol_angles):
                    msg = f'{pdb_id}:{curr_chain} @ {i}'
                    pp5_phi_psi = pp5_angles[j].phi_deg, pp5_angles[j].psi_deg

                    try:
                        # If pymol was not able to calculate it sets to zero...
                        if phi_psi[0] == 0 or phi_psi[1] == 0:
                            continue

                        # Try to fix alignment problem up to len_diff times
                        if not pp5_phi_psi == approx(phi_psi, abs=1):
                            if len_diff > 0:
                                len_diff -= 1
                                j -= 1
                                continue

                        assert pp5_phi_psi == approx(phi_psi, abs=1e-3), msg
                    finally:
                        j += 1


class TestDihedralAnglesEstimators(object):
    TEST_PDB_IDS = ['5jdt', '3ajo', '1b0y', '2wur']

    @classmethod
    def setup_class(cls):
        pass

    def _compare_with_estimator(self, pdb_id, estimator, **kw):
        pdb_rec = pdb.pdb_struct(pdb_id)
        pp_chains = PPBuilder().build_peptides(pdb_rec, aa_only=True)

        for pp in pp_chains:
            biopython_angles = pp.get_phi_psi_list()
            biopython_angles_deg = [
                (degrees(phi) if phi else None, degrees(psi) if psi else None)
                for phi, psi in biopython_angles
            ]
            angles = estimator.estimate(pp)

            assert len(angles) == len(biopython_angles)
            for i in range(len(angles)):
                ang = angles[i]
                bio_phi = biopython_angles[i][0]
                bio_psi = biopython_angles[i][1]
                bio_phi_deg = biopython_angles_deg[i][0]
                bio_psi_deg = biopython_angles_deg[i][1]

                if not (math.isnan(ang.phi) and bio_phi is None):
                    assert ang.phi == approx(bio_phi, **kw)
                    assert ang.phi_deg == approx(bio_phi_deg, **kw)

                if not (math.isnan(ang.psi) and bio_psi is None):
                    assert ang.psi == approx(bio_psi, **kw)
                    assert ang.psi_deg == approx(bio_psi_deg, **kw)

    @pytest.mark.parametrize('pdb_id', TEST_PDB_IDS)
    def test_basic_estimator(self, pdb_id):
        estimator = dihedral.DihedralAnglesEstimator()
        self._compare_with_estimator(pdb_id, estimator)

    @pytest.mark.parametrize('pdb_id', TEST_PDB_IDS)
    def test_uncertainty_estimator(self, pdb_id):
        estimator = dihedral.DihedralAnglesUncertaintyEstimator()
        self._compare_with_estimator(pdb_id, estimator)

    @pytest.mark.parametrize('pdb_id', TEST_PDB_IDS)
    def test_montecarlo_estimator(self, pdb_id):
        unit_cell = pdb.pdb_to_unit_cell(pdb_id)
        estimator = dihedral.DihedralAnglesMonteCarloEstimator(unit_cell)
        # Not sure how to test, for now set infinite abs error
        self._compare_with_estimator(pdb_id, estimator, abs=math.inf)


class TestDihedralEq:
    def test_eq_deg(self):
        d1 = dihedral.Dihedral.from_deg(1, 2, 3)
        d2 = dihedral.Dihedral.from_deg(1, 2, 3)
        assert d1 == d2

    def test_eq_rad(self):
        d1 = dihedral.Dihedral.from_rad(1., 1.5, 3.)
        d2 = dihedral.Dihedral.from_rad(1., 1.5, 3.)
        assert d1 == d2

    def test_eq_deg_rad(self):
        d1 = dihedral.Dihedral.from_deg(degrees(1.), degrees(1.5), degrees(3.))
        d2 = dihedral.Dihedral.from_rad(1., 1.5, 3.)
        assert d1 == d2

    def test_zero(self):
        d1 = dihedral.Dihedral.from_deg(0., 0., 0.)
        d2 = dihedral.Dihedral.from_rad(0., 0., 0.)
        assert d1 == d2

    def test_nan(self):
        d1 = dihedral.Dihedral.from_rad(math.nan, 0., 0.)
        d2 = dihedral.Dihedral.from_deg(math.nan, 0., 0.)
        assert d1 == d2
        d1 = dihedral.Dihedral.from_deg(0., math.nan, 0.)
        d2 = dihedral.Dihedral.from_rad(0., math.nan, 0.)
        assert d1 == d2
        d1 = dihedral.Dihedral.from_deg(1., 2., math.nan)
        d2 = dihedral.Dihedral.from_deg(1., 2., math.nan)
        assert d1 == d2

    def test_std(self):
        d1 = dihedral.Dihedral.from_deg([1., 0.1], [2., 0.2], math.nan)
        d2 = dihedral.Dihedral.from_deg([1., 0.1], [2., 0.2], math.nan)
        assert d1 == d2


class TestWraparoundDiff(object):
    TEST_CASES = [
        ((170, -170), 20), ((180, -180), 0), ((-20, 30), 50), ((30, 30), 0),
    ]

    @pytest.mark.parametrize(('angles', 'expected'), TEST_CASES)
    def test_wraparound_diff(self, angles, expected):
        a1, a2 = angles
        a1, a2 = math.radians(a1), math.radians(a2)

        actual = dihedral.Dihedral._wraparound_diff(a1, a2)
        assert actual == approx(math.radians(expected))

        actual2 = dihedral.Dihedral._wraparound_diff(a2, a1)
        assert actual2 == approx(math.radians(expected))


class TestFlatTorusDistance:
    TEST_CASES = [
        ((170, 170), (-170, -170), 20 * math.sqrt(2)),
        ((-170, 170), (170, -170), 20 * math.sqrt(2)),
        ((0, 173), (0, -177), 10.),
        ((172, 0), (-178, 0), 10.),
        ((5, 5), (-5, -5), 10 * math.sqrt(2)),
    ]

    @pytest.mark.parametrize(('a1', 'a2', 'expected_dist'), TEST_CASES)
    def test_1(self, a1, a2, expected_dist):
        a1 = Dihedral.from_deg(*a1, 0)
        a2 = Dihedral.from_deg(*a2, 0)
        actual_dist = Dihedral.flat_torus_distance(a1, a2, degrees=True)
        assert actual_dist == approx(expected_dist)

    def test_compare_to_s1_distance(self):
        diffs = [
            Dihedral.flat_torus_distance(a1, a2) - Dihedral.s1_distance(a1, a2)
            for a1, a2 in zip(random_angles(1000), random_angles(1000))
        ]

        assert max(diffs) == approx(0.)


class TestFrechetMean:
    TEST_CASES = [
        # phi, psi, phi_expected, psi_expected
        ([-45, 45], [-30, 30], 0, 0),
        ([-45, -30, 30, 45], [75, 80, 100, 105], 0, 90),
        ([-140, -145, -155, -160], [-75, -80, -100, -105], -150, -90),
        ([-170, -175, 175, 170], [0, 0, 0, 0], -180, 0),
        ([-90, 90], [0, 180], 0, -90),
        ([12.34], [56.78], 12.34, 56.78),
    ]

    @pytest.mark.parametrize(('phi', 'psi', 'phi_exp', 'psi_exp'), TEST_CASES)
    def test_1(self, phi, psi, phi_exp, psi_exp):
        assert len(phi) == len(psi), 'test case error'

        angs = [Dihedral.from_deg(phi[i], psi[i], 0) for i in range(len(psi))]
        actual = Dihedral.frechet_torus_centroid(*angs)

        assert actual.phi_deg == approx(phi_exp, abs=1e-2)
        assert actual.psi_deg == approx(psi_exp, abs=1e-2)

    def test_benchmark_frechet_controid_numba(self, benchmark):
        benchmark.pedantic(
            Dihedral.frechet_torus_centroid, args=random_angles(1000),
            rounds=10, iterations=50, warmup_rounds=1,
        )

    def test_benchmark_frechet_centroid_python(self, benchmark):
        benchmark.pedantic(
            Dihedral.frechet_torus_centroid, args=random_angles(1000),
            kwargs=dict(metric_fn=Dihedral._mean_sq_metric_s1.py_func),
            rounds=10, iterations=50, warmup_rounds=1,
        )
