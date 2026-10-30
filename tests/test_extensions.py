"""Check independent numerical routes and reject incorrect updates."""

from pathlib import Path
import sys
import unittest

import mpmath as mp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from euler_path import checks, nonlinear, nonlinear_recursive, solver_dependence
from run_nonlinear import metrics, validate_gaussian


class NonlinearTests(unittest.TestCase):
    def setUp(self):
        self.radii = np.array([0.05, 0.4, 1, 3, 8])
        self.nodes = np.array([0, 0.04, 0.17, 0.43, 0.72, 1])

    def test_laguerre_moments_and_mixture_second_moment(self):
        for order in (64, 128, 256):
            data = nonlinear.mixture_quadrature(order, [0.6, 0.4], [0.5, 1.5])
            self.assertTrue(np.all(data["laguerre_nodes"] > 0))
            self.assertTrue(np.all(data["weights"] >= 0))
            self.assertAlmostEqual(data["weights"].sum(), 1, places=13)
            self.assertAlmostEqual(data["weights"] @ data["radii"]**2, 2.1, places=12)
            for power in (1, 2, 3):
                self.assertAlmostEqual(data["laguerre_weights"] @ data["laguerre_nodes"]**power,
                                       float(mp.factorial(power)), places=11)

    def test_implicit_reference_matches_time_ode_and_separated_angle(self):
        solution = nonlinear.reference(self.radii)
        values = nonlinear.reference_values(solution, [0, 0.25, 0.5, 1], self.radii)
        for index, time in enumerate([0, 0.25, 0.5, 1]):
            implicit = np.array([nonlinear.implicit_radius(r, time) for r in self.radii])
            np.testing.assert_allclose(values["radius"][index], implicit, rtol=5e-13, atol=2e-13)
        independent = np.array([nonlinear.radius_integrals(r) for r in self.radii])
        np.testing.assert_allclose(values["angle"][-1], independent[:, 1], rtol=5e-12, atol=2e-13)
        for column, name in ((2, "action_base"), (3, "action_corrected"), (4, "coefficient_base"),
                             (5, "coefficient_corrected"), (6, "coefficient_angle")):
            np.testing.assert_allclose(values[name], independent[:, column], rtol=5e-12, atol=2e-13)

    def test_initial_node_errors_are_exact_zero(self):
        solution = nonlinear.reference(self.radii)
        reference = nonlinear.reference_values(solution, self.nodes, self.radii)
        np.testing.assert_array_equal(reference["radius"][0], self.radii)
        base = nonlinear_recursive.polar(self.nodes, self.radii, False)
        corrected = nonlinear_recursive.polar(self.nodes, self.radii, True)
        errors = metrics(base, corrected, reference, np.full(len(self.radii), 1 / len(self.radii)))
        self.assertTrue(all(error[0] == 0 for error in errors.values()))

    def test_constant_limit_has_known_coefficients(self):
        a, time = 0.6, 1.0
        for radius in self.radii:
            values = nonlinear.radius_integrals(radius, time, a, 0)
            end = radius * np.exp(a * time)
            self.assertAlmostEqual(values[0], end, places=12)
            self.assertAlmostEqual(values[1], a * time, places=12)
            self.assertAlmostEqual(values[4], end * a**2 * time / 2, places=12)
            self.assertAlmostEqual(values[5], end * 2 * a**3 * time / 3, places=12)
            self.assertAlmostEqual(values[6], a**2 * time, places=12)
            action = a * radius**2 * np.expm1(2 * a * time) / 2
            self.assertAlmostEqual(values[2] / action, 1, places=12)
            self.assertAlmostEqual(values[3] / action, 2, places=12)

    def test_cartesian_and_polar_updates_agree_at_every_node(self):
        angles = np.array([0, 0.3, 1, 2.1, 4])
        initial = self.radii[:, None] * np.column_stack((np.cos(angles), np.sin(angles)))
        for corrected in (False, True):
            polar = nonlinear_recursive.polar(self.nodes, self.radii, corrected)
            paths = nonlinear_recursive.cartesian(self.nodes, initial, corrected)
            checks.compare_cartesian_polar(paths, polar["radius"], polar["angle"], angles)

    def test_strict_all_node_order_for_recorded_nonuniform_grid(self):
        base = nonlinear_recursive.polar(self.nodes, self.radii, False)
        corrected = nonlinear_recursive.polar(self.nodes, self.radii, True)
        exact = np.array([[nonlinear.implicit_radius(r, t) for r in self.radii] for t in self.nodes])
        self.assertTrue(np.all(base["radius"][1:] < corrected["radius"][1:]))
        self.assertTrue(np.all(corrected["radius"][1:] < exact[1:]))

    def test_radius_maps_are_monotone_on_a_dense_radius_grid(self):
        source = np.geomspace(1e-6, 1e3, 1001)
        for corrected in (False, True):
            radii = nonlinear_recursive.polar(self.nodes, source, corrected)["radius"]
            self.assertTrue(np.all(np.diff(radii, axis=1) > 0))

    def test_transport_is_not_a_gaussian_second_moment_surrogate(self):
        weights = np.array([0.5, 0.5])
        first, second = np.array([1, 3]), np.array([np.sqrt(2), np.sqrt(8)])
        w2, _ = nonlinear.radial_errors(first, second, weights)
        self.assertAlmostEqual(weights @ first**2, weights @ second**2, places=14)
        self.assertGreater(w2, 0.2)

    def test_particle_metric_includes_angle_error(self):
        radius = np.array([2.0])
        w2, particle = nonlinear.radial_errors(radius, radius, np.array([1.0]), np.array([0.1]), np.array([0.0]))
        self.assertEqual(w2, 0)
        self.assertAlmostEqual(particle, 4 * np.sin(0.05), places=14)

    def test_shape_changes_after_normalization_by_median(self):
        radii = nonlinear.source_quantiles([0.1, 0.5, 0.9], [0.6, 0.4], [0.5, 1.5])
        end = np.array([nonlinear.implicit_radius(r, 1) for r in radii])
        self.assertTrue(np.all(np.diff(end / radii) > 0))
        self.assertGreater(np.max(abs(end / end[1] - radii / radii[1])), 0.1)

    def test_checker_rejects_restart_and_missing_nonlinear_term(self):
        initial = np.column_stack((self.radii, np.zeros_like(self.radii)))
        polar = nonlinear_recursive.polar(self.nodes, self.radii, True)
        wrong_restart = initial.copy()
        paths = [initial.copy()]
        for left, right in zip(self.nodes[:-1], self.nodes[1:]):
            if left:
                independent = np.array([nonlinear.radius_integrals(r, left) for r in self.radii])
                wrong_restart = independent[:, 0, None] * np.column_stack((np.cos(independent[:, 1]), np.sin(independent[:, 1])))
            paths.append(nonlinear_recursive.cartesian([0, right - left], wrong_restart, True)[-1])
        with self.assertRaisesRegex(AssertionError, "Node 2: Cartesian update mismatch"):
            checks.compare_cartesian_polar(np.array(paths), polar["radius"], polar["angle"], np.zeros(len(self.radii)))
        current, paths = initial.copy(), [initial.copy()]
        for step in np.diff(self.nodes):
            r = np.linalg.norm(current, axis=1)
            alpha = 0.6 + 0.3 * r**2 / (1 + r**2)
            current += step * alpha[:, None] * (current + np.column_stack((-current[:, 1], current[:, 0])))
            paths.append(current.copy())
        with self.assertRaisesRegex(AssertionError, "Node 1: Cartesian update mismatch"):
            checks.compare_cartesian_polar(np.array(paths), polar["radius"], polar["angle"], np.zeros(len(self.radii)))

    def test_checker_rejects_missing_nodes_and_nonfinite_paths(self):
        polar = nonlinear_recursive.polar(self.nodes, self.radii, True)
        paths = nonlinear_recursive.cartesian(self.nodes, np.column_stack((self.radii, np.zeros(5))), True)
        with self.assertRaisesRegex(AssertionError, "coverage"):
            checks.compare_cartesian_polar(paths[:-1], polar["radius"], polar["angle"], np.zeros(5))
        paths[2, 0, 0] = np.nan
        with self.assertRaisesRegex(AssertionError, "nonfinite"):
            checks.compare_cartesian_polar(paths, polar["radius"], polar["angle"], np.zeros(5))


class SolverTests(unittest.TestCase):
    def setUp(self):
        mp.mp.dps = 80
        self.covariance = np.array([[1.5, 0.25], [0.25, 0.75]])

    def test_heun_matrix_matches_independent_stability_polynomial(self):
        times = [0, 0.04, 0.17, 0.43, 0.72, 1]
        for b in (0, 0.6, 0.6 / np.sqrt(3)):
            formula = solver_dependence.gaussian_formula(times, 0.6, b, 2.25, "heun")
            direct = solver_dependence.gaussian_matrix(times, 0.6, b, self.covariance, "heun")
            np.testing.assert_allclose(direct["w2"], np.array([float(x) for x in formula]), rtol=0, atol=2e-14)

    def test_checker_detects_euler_used_in_place_of_heun(self):
        times = [0, 0.5, 1]
        heun = np.array([float(x) for x in solver_dependence.gaussian_formula(times, 0.6, 0.6, 2.25, "heun")])
        wrong = solver_dependence.gaussian_matrix(times, 0.6, 0.6, self.covariance, "euler")
        self.assertGreater(np.max(abs(heun - wrong["w2"])), 0.02)
        with self.assertRaisesRegex(AssertionError, "disagrees"):
            validate_gaussian(heun, wrong, 2.25, 0.6)

    def test_high_precision_heun_cancellation_error(self):
        b = mp.mpf("0.6") / mp.sqrt(3)
        formula = solver_dependence.gaussian_formula(np.linspace(0, 1, 33), "0.6", b, "2.25", "heun")[-1]
        matrix = solver_dependence.gaussian_matrix_high_precision(32, "0.6", b, 1, self.covariance, "heun", 50)
        self.assertLess(abs(mp.mpf(matrix["w2"]) - formula), mp.mpf("1e-40"))

    def test_heun_rotation_doubles_leading_error(self):
        base = solver_dependence.gaussian_leading_coefficient(0.6, 0, 1, 2.25, "heun")[1]
        corrected = solver_dependence.gaussian_leading_coefficient(0.6, 0.6, 1, 2.25, "heun")[1]
        self.assertAlmostEqual(corrected / base, 2, places=14)

    def test_fixed_gaussian_sensitivity_coefficient(self):
        base = solver_dependence.gaussian_leading_coefficient(0.6, 0, 1, 2.25, "euler")[1]
        for ratio in (0.9, 0.98, 1.02, 1.1):
            order, coefficient = solver_dependence.gaussian_leading_coefficient(0.6, 0.6 * ratio, 1, 2.25, "euler")
            self.assertEqual(order, 1)
            self.assertAlmostEqual(coefficient / base, abs(1 - ratio**2), places=13)

    def test_flow_matching_complete_covariance_and_euler_particle_exactness(self):
        for times in ([0, 0.04, 0.17, 0.43, 0.72, 1], np.linspace(0, 1, 33)):
            run = solver_dependence.flow_matching(times)
            exact_covariance = run["exact_maps"] @ np.transpose(run["exact_maps"], (0, 2, 1))
            np.testing.assert_allclose(exact_covariance, run["target_covariances"], rtol=0, atol=2e-14)
            self.assertLess(run["particle_rms"].max(), 2e-14)
            base = solver_dependence.flow_matching(times, corrected=False)
            self.assertGreater(base["w2"][-1], 1e-3)

    def test_flow_matching_missing_rotation_fails_exact_map_check(self):
        wrong = solver_dependence.flow_matching([0, 0.3, 0.8, 1], corrected=False)
        correct = solver_dependence.flow_matching([0, 0.3, 0.8, 1])
        self.assertGreater(np.linalg.norm(wrong["maps"][-1] - correct["exact_maps"][-1]), 1)

    def test_anisotropic_skew_coupling_requires_paired_eigenvalues(self):
        eigenvalues = [1, 4, 1, 4]
        matrix = solver_dependence.paired_skew(eigenvalues)
        np.testing.assert_allclose(matrix + matrix.T, 0, rtol=0, atol=1e-14)
        np.testing.assert_allclose(matrix @ matrix.T, np.diag(eigenvalues), rtol=0, atol=1e-14)
        for values in ([1, 4], [1, 1, 4], [1, 1, 4, 5]):
            with self.assertRaises(ValueError):
                solver_dependence.paired_skew(values)


if __name__ == "__main__":
    unittest.main()
