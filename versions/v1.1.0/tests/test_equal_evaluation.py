"""Check equal-evaluation counts, transport, and solver independence."""

from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from euler_path import equal_evaluation, nonlinear, nonlinear_recursive, solver_dependence


class EqualEvaluationTests(unittest.TestCase):
    def test_even_budget_gaussian_identity_and_independent_covariance(self):
        covariance = np.array([[1.5, 0.25], [0.25, 0.75]])
        for budget in (2, 4, 8, 16, 64, 512, 4096):
            identity = equal_evaluation.gaussian_identity(budget)
            self.assertLess(float(identity["absolute_difference"]), 1e-74)
            euler = solver_dependence.gaussian_matrix(np.linspace(0, 1, budget + 1), 0.6, 0.6, covariance, "euler")
            heun = solver_dependence.gaussian_matrix(np.linspace(0, 1, budget // 2 + 1), 0.6, 0, covariance, "heun")
            np.testing.assert_allclose(euler["w2"][-1], heun["w2"][-1], rtol=0, atol=2e-11)

    def test_invalid_or_odd_budget_is_rejected(self):
        for budget in (0, 1, 3):
            with self.assertRaises(ValueError):
                equal_evaluation.gaussian_identity(budget)
            with self.assertRaises(ValueError):
                equal_evaluation.terminal_polar([1], budget, "base_heun")

    def test_field_counts_and_independent_euler_updates(self):
        radii = np.array([0.07, 0.4, 1, 3, 8])
        for budget in (2, 16, 128):
            for method, corrected in (("base_euler", False), ("corrected_euler", True)):
                actual = equal_evaluation.terminal_polar(radii, budget, method)
                expected = nonlinear_recursive.polar(np.linspace(0, 1, budget + 1), radii, corrected)
                np.testing.assert_allclose(actual["radius"], expected["radius"][-1], rtol=2e-14)
                np.testing.assert_allclose(actual["angle"], expected["angle"][-1], atol=2e-14)
                self.assertEqual(actual["field_evaluations"], budget)
            heun = equal_evaluation.terminal_polar(radii, budget, "base_heun")
            self.assertEqual(heun["steps"], budget // 2)
            self.assertEqual(heun["field_evaluations"], budget)

    def test_independent_cartesian_heun_and_particle_transport(self):
        radii = np.array([0, 0.07, 0.4, 1, 3, 8])
        angles = np.array([0, 0.3, 1, 2.1, 4, 5])
        initial = radii[:, None] * np.column_stack((np.cos(angles), np.sin(angles)))
        reference = np.array([nonlinear.implicit_radius(r, 1) for r in radii])
        weights = np.full(len(radii), 1 / len(radii))
        for budget in (2, 4, 16, 256):
            polar = equal_evaluation.terminal_polar(radii, budget, "base_heun")
            cartesian = equal_evaluation.terminal_cartesian(initial, budget, "base_heun")
            predicted = polar["radius"][:, None] * np.column_stack((np.cos(angles), np.sin(angles)))
            np.testing.assert_allclose(cartesian, predicted, atol=3e-12, rtol=2e-14)
            exact = reference[:, None] * np.column_stack((np.cos(angles), np.sin(angles)))
            independent_rms = np.sqrt(weights @ np.sum((cartesian - exact)**2, axis=1))
            w2, particle = nonlinear.radial_errors(polar["radius"], reference, weights)
            self.assertEqual(float(w2), float(particle))
            self.assertAlmostEqual(independent_rms, float(w2), places=12)

    def test_radius_maps_are_increasing(self):
        radii = np.geomspace(1e-8, 1e4, 1001)
        for method in equal_evaluation.METHODS:
            for budget in (2, 4, 16, 128):
                numerical = equal_evaluation.terminal_polar(radii, budget, method)
                self.assertTrue(np.all(np.diff(numerical["radius"]) > 0))

    def test_euler_substitution_and_restarted_heun_are_detectable(self):
        radii = np.array([0.07, 0.4, 1, 3, 8])
        heun = equal_evaluation.terminal_polar(radii, 4, "base_heun")["radius"]
        euler = equal_evaluation.terminal_polar(radii, 4, "base_euler")["radius"]
        self.assertGreater(np.max(abs(heun - euler)), 0.1)
        exact_mid = np.array([nonlinear.implicit_radius(r, 0.5) for r in radii])
        restarted = equal_evaluation.terminal_polar(exact_mid, 2, "base_heun", horizon=0.5)["radius"]
        self.assertGreater(np.max(abs(heun - restarted)), 0.1)


if __name__ == "__main__":
    unittest.main()
