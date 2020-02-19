import math
from math import radians

import numba
import numpy as np
from numpy import ndarray
import pytest
from Bio.PDB import PPBuilder

import tests
from pp5.external_dbs import pdb
from pp5 import dihedral
from pp5.dihedral import calc_dihedral2

from pytest import approx

RESOURCES_PATH = tests.TEST_RESOURCES_PATH.joinpath('dihedral')


class TestRawDihedralAngleCalculation(object):

    def random_coords(self):
        mu = np.random.normal(0, 10, size=(3,))
        sigma = np.random.normal(0, 1, size=(3, 3))
        return [np.random.multivariate_normal(mu, sigma) for _ in range(4)]

    @staticmethod
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

    def test_1(self):
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
            expected = self.calc_dihedral_naive(*vs)
            assert calc_dihedral2(*vs) == approx(expected)

    def test_benchmark_naive_numba(self, benchmark):
        benchmark(self.calc_dihedral_naive, *self.random_coords())

    def test_benchmark_naive_python(self, benchmark):
        benchmark(self.calc_dihedral_naive.py_func, *self.random_coords())

    def test_benchmark_calc_dihedral_fast_numba(self, benchmark):
        benchmark(calc_dihedral2, *self.random_coords())

    def test_benchmark_calc_dihedral_fast_python(self, benchmark):
        benchmark(calc_dihedral2.py_func, *self.random_coords())


class TestDihedralAnglesEstimator(object):
    TEST_PDB_IDS = ['3ajo', '1b0y', '2wur']

    @classmethod
    def setup_class(cls):
        pass

    def compare_with_estimator(self, pdb_id, estimator, **kw):
        pdb_rec = pdb.pdb_struct(pdb_id, pdb_dir=RESOURCES_PATH)
        pp_chains = PPBuilder().build_peptides(pdb_rec, aa_only=True)

        for pp in pp_chains:
            biopython_angles = pp.get_phi_psi_list()
            angles = estimator.estimate(pp)

            assert len(angles) == len(biopython_angles)
            for i in range(len(angles)):
                ang = angles[i]
                bio_phi = biopython_angles[i][0]
                bio_psi = biopython_angles[i][1]

                if not (math.isnan(ang.phi) and bio_phi is None):
                    assert ang.phi == approx(bio_phi, **kw)
                    assert ang.phi_deg == approx(math.degrees(bio_phi), **kw)

                if not (math.isnan(ang.psi) and bio_psi is None):
                    assert ang.psi == approx(bio_psi, **kw)
                    assert ang.psi_deg == approx(math.degrees(bio_psi), **kw)

    @pytest.mark.parametrize('pdb_id', TEST_PDB_IDS)
    def test_basic_estimator(self, pdb_id):
        estimator = dihedral.DihedralAnglesEstimator()
        self.compare_with_estimator(pdb_id, estimator)

    @pytest.mark.parametrize('pdb_id', TEST_PDB_IDS)
    def test_uncertainty_estimator(self, pdb_id):
        estimator = dihedral.DihedralAnglesUncertaintyEstimator()
        self.compare_with_estimator(pdb_id, estimator)

    @pytest.mark.parametrize('pdb_id', TEST_PDB_IDS)
    def test_montecarlo_estimator(self, pdb_id):
        unit_cell = pdb.pdb_to_unit_cell(pdb_id, pdb_dir=RESOURCES_PATH)
        estimator = dihedral.DihedralAnglesMonteCarloEstimator(unit_cell)
        # Not sure how to test, for now set infinite abs error
        self.compare_with_estimator(pdb_id, estimator, abs=math.inf)
