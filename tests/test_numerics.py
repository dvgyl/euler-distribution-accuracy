"""Test exact identities and reject changes that invalidate the comparison."""

import ast
import copy
from pathlib import Path
import sys
import unittest

import mpmath as mp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from euler_path import checks, population, recursive


class NumericalTests(unittest.TestCase):
    def setUp(self):
        mp.mp.dps = 80
        self.covariance = np.array([[1.5, 0.25], [0.25, 0.75]])
        self.nodes = ["0", "0.04", "0.17", "0.43", "0.72", "1"]

    def test_elementary_action_and_particle_exactness(self):
        base = recursive.radial([0, 1], 0, 1, 2)
        fixed = recursive.radial([0, 1], 1, 1, 2)
        self.assertAlmostEqual(base["action"], 2 - np.pi / 2, places=13)
        self.assertAlmostEqual(fixed["action"], 2, places=13)
        self.assertAlmostEqual(base["w2"][-1]**2, 2 * (np.sqrt(2) - 1)**2, places=13)
        np.testing.assert_allclose(fixed["maps"][-1], [[1, -1], [1, 1]], rtol=0, atol=1e-14)
        self.assertLess(fixed["particle_rms"][-1], 1e-14)

    def test_radial_all_node_order_and_telescope(self):
        amplitudes = [0, 0.25, 0.5, 0.75, 1]
        runs = [recursive.radial(self.nodes, alpha, 1, 2) for alpha in amplitudes]
        for first, second in zip(runs[:-1], runs[1:]):
            self.assertLess(first["action"], second["action"])
            self.assertTrue(np.all(first["w2"][1:] > second["w2"][1:]))
        np.testing.assert_allclose(runs[-1]["maps"], runs[-1]["exact_maps"], rtol=0, atol=2e-14)

    def test_gaussian_affine_mean_and_all_node_comparison(self):
        mean, drift = np.array([0.3, -0.7]), np.array([0.2, 0.4])
        for rotation in (0, 0.6):
            formula = population.gaussian(self.nodes, "0.6", rotation, np.trace(self.covariance), np.dot(drift, drift))
            direct = recursive.gaussian(self.nodes, 0.6, rotation, self.covariance, mean, drift)
            checks.compare(formula, direct, self.covariance)
            np.testing.assert_allclose(direct["means"], mean + np.asarray(self.nodes, dtype=float)[:, None] * drift, rtol=0, atol=1e-14)

    def test_anisotropic_dimensions_and_weighted_rotation(self):
        for dimension in (2, 4, 8):
            factor = np.diag(np.linspace(0.7, 1.6, dimension))
            factor[np.arange(1, dimension), np.arange(dimension - 1)] = 0.12
            covariance = factor @ factor.T
            direct = recursive.gaussian(self.nodes, 0.6, 0.6, covariance, np.zeros(dimension), np.zeros(dimension))
            rotation = direct["rotation_matrix"]
            np.testing.assert_allclose(rotation @ covariance + covariance @ rotation.T, 0, atol=2e-14)
            np.testing.assert_allclose(rotation @ covariance @ rotation.T, covariance, atol=2e-14)
            np.testing.assert_allclose(rotation @ rotation, -np.eye(dimension), atol=2e-14)

    def test_fixed_field_improves_nonuniform_nodes(self):
        base = recursive.gaussian(self.nodes, 0.6, 0, self.covariance, [0, 0], [0, 0])
        fixed = recursive.gaussian(self.nodes, 0.6, 0.6, self.covariance, [0, 0], [0, 0])
        self.assertTrue(np.all(0 < fixed["w2"][1:]))
        self.assertTrue(np.all(fixed["w2"][1:] < base["w2"][1:]))
        h_max = max(np.diff(np.asarray(self.nodes, dtype=float)))
        constant = np.sqrt(np.trace(self.covariance)) * np.exp(0.6)
        self.assertLess(max(base["w2"]), constant * 0.6**2 / 2 * h_max)
        self.assertLess(max(fixed["w2"]), constant * 2 * 0.6**3 / 3 * h_max**2)

    def test_contraction_control_is_worse_at_one_step(self):
        base = recursive.gaussian([0, 1], -0.6, 0, self.covariance, [0, 0], [0, 0])
        fixed = recursive.gaussian([0, 1], -0.6, -0.6, self.covariance, [0, 0], [0, 0])
        self.assertLess(base["w2"][-1], fixed["w2"][-1])

    def test_calibrated_coefficient_matches_recursive_node_distributions(self):
        for rate in (-1, -0.6, 0.6):
            rotation = float(mp.sqrt(population.calibration(rate)))
            run = recursive.gaussian([0, 1, 2, 3, 4], rate, rotation, self.covariance, [0.4, -0.3], [0.2, 0.1])
            np.testing.assert_allclose(run["covariances"], run["target_covariances"], rtol=3e-14, atol=3e-14)
            self.assertLess(max(run["w2"]), 1e-12)
            self.assertGreater(run["particle_rms"][-1], 1e-3)

    def test_calibration_boundaries_and_infeasibility(self):
        root = population.negative_root()
        self.assertLess(abs(mp.exp(root) + 1 + root), mp.mpf("1e-75"))
        self.assertEqual(population.calibration(0), 0)
        self.assertGreater(population.calibration(-1), 0)
        self.assertLess(population.calibration(root - mp.mpf("1e-8")), 0)
        self.assertGreater(population.calibration(root + mp.mpf("1e-8")), 0)
        run = recursive.gaussian([0, 1], float(root), 0, self.covariance, [0, 0], [0, 0])
        self.assertLess(run["w2"][-1], 2e-14)
        self.assertGreater(run["particle_rms"][-1], 0.5)

    def test_intermediate_calibration_orders_node_errors(self):
        delta = population.calibration("0.6")
        runs = [recursive.gaussian([0, 1, 2], 0.6, float(s * mp.sqrt(delta)), self.covariance, [0, 0], [0, 0]) for s in (0, 0.25, 0.5, 0.75, 1)]
        for first, second in zip(runs[:-1], runs[1:]):
            self.assertTrue(np.all(first["w2"][1:] > second["w2"][1:]))
            self.assertLess(first["action"], second["action"])

    def test_calibration_reuse_does_not_give_exact_other_step(self):
        rotation = float(mp.sqrt(population.calibration("0.6")))
        run = recursive.gaussian([0, 0.5, 1], 0.6, rotation, self.covariance, [0, 0], [0, 0])
        self.assertGreater(run["w2"][-1], 0.01)

    def test_restart_control_is_rejected_at_the_second_node(self):
        formula = population.gaussian(self.nodes, "0.6", "0.6", np.trace(self.covariance), 0)
        wrong = recursive.gaussian(self.nodes, 0.6, 0.6, self.covariance, [0, 0], [0, 0], restart=True)
        message = checks.reject_expected(formula, wrong, self.covariance)
        self.assertIn("Node 2: covariance mismatch", message)

    def test_grid_retuning_control_is_rejected(self):
        nodes = [0, 0.25, 0.5, 0.75, 1]
        formula = population.gaussian(nodes, "0.6", "0.6", np.trace(self.covariance), 0)
        rotation = float(mp.sqrt(population.calibration("0.15")) / mp.mpf("0.25"))
        wrong = recursive.gaussian(nodes, 0.6, rotation, self.covariance, [0, 0], [0, 0])
        self.assertIn("Node 1: covariance mismatch", checks.reject_expected(formula, wrong, self.covariance))

    def test_ordinary_rotation_is_not_weighted_flux_null_for_ellipses(self):
        wrong = recursive.quarter_turn(2)
        residual = np.linalg.norm(wrong @ self.covariance + self.covariance @ wrong.T)
        self.assertGreater(residual, 0.5)

    def test_odd_dimension_cannot_have_invertible_real_skew_rotation(self):
        skew = np.array([[0, -1, 2], [1, 0, -3], [-2, 3, 0]])
        self.assertAlmostEqual(np.linalg.det(skew), 0, places=13)
        with self.assertRaisesRegex(ValueError, "even integer"):
            recursive.quarter_turn(3)

    def test_stable_w2_preserves_tiny_proportional_covariance_distance(self):
        distance = recursive.gaussian_w2(np.zeros(2), self.covariance * (1 - 1e-9)**2, np.zeros(2), self.covariance)
        self.assertAlmostEqual(distance / (1e-9 * np.sqrt(np.trace(self.covariance))), 1, delta=2e-6)

    def test_gaussian_w2_general_covariance_and_mean(self):
        first, second = np.diag([1.0, 9.0]), np.diag([4.0, 1.0])
        self.assertAlmostEqual(recursive.gaussian_w2([0, 0], first, [1, 2], second), np.sqrt(10), places=13)

    def test_gaussian_w2_noncommuting_covariances(self):
        first = self.covariance
        second = np.array([[3.0, 0.8], [0.8, 0.7]])
        expected_squared = np.trace(first) + np.trace(second) - 2 * np.sqrt(
            np.trace(first @ second) + 2 * np.sqrt(np.linalg.det(first) * np.linalg.det(second)))
        self.assertAlmostEqual(recursive.gaussian_w2([0, 0], first, [0, 0], second)**2, expected_squared, places=13)

    def test_recursive_module_has_no_population_formula_dependency(self):
        source = (ROOT / "src" / "euler_path" / "recursive.py").read_text()
        imports = [node for node in ast.walk(ast.parse(source)) if isinstance(node, (ast.Import, ast.ImportFrom))]
        allowed = {"numpy", "scipy.integrate", "scipy.linalg"}
        modules = {node.module for node in imports if isinstance(node, ast.ImportFrom)}
        modules.update(alias.name for node in imports if isinstance(node, ast.Import) for alias in node.names)
        self.assertLessEqual(modules, allowed)

    def test_grid_validation_rejects_restart_or_negative_time_grids(self):
        for nodes in ([0, 0.5, 0.5, 1], [1, 2], [0, -0.1, 1], [0, np.nan, 1]):
            with self.assertRaises(ValueError):
                recursive.checked_nodes(nodes)


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        mp.mp.dps = 80
        self.covariance = np.array([[1.5, 0.25], [0.25, 0.75]])
        nodes = [0, 0.25, 0.5, 0.75, 1]
        self.formula = population.gaussian(nodes, "0.6", "0.6", np.trace(self.covariance), 0)
        self.direct = recursive.gaussian(nodes, 0.6, 0.6, self.covariance, [0, 0], [0, 0])

    def test_rejects_nonfinite_recursive_comparison_quantities(self):
        locations = [("covariances", (1, 0, 0)), ("w2", 1), ("particle_rms", 1),
                     ("means", (1, 0)), ("target_means", (1, 0)), ("action", None)]
        for key, index in locations:
            for value in (np.nan, np.inf, -np.inf):
                with self.subTest(key=key, value=value), np.errstate(invalid="ignore"):
                    direct = copy.deepcopy(self.direct)
                    if index is None:
                        direct[key] = value
                    else:
                        direct[key][index] = value
                    with self.assertRaisesRegex(AssertionError, "nonfinite"):
                        checks.compare(self.formula, direct, self.covariance)

    def test_rejects_nonfinite_population_comparison_quantities(self):
        for key in ("scale", "target_scale", "w2", "particle_rms", "action"):
            for value in ("nan", "inf", "-inf"):
                with self.subTest(key=key, value=value), np.errstate(invalid="ignore"):
                    formula = copy.deepcopy(self.formula)
                    if key == "action":
                        formula[key] = value
                    else:
                        formula["nodes"][1][key] = value
                    with self.assertRaisesRegex(AssertionError, "nonfinite"):
                        checks.compare(formula, self.direct, self.covariance)

    def test_rejects_nonfinite_source_covariance(self):
        for value in (np.nan, np.inf, -np.inf):
            with self.subTest(value=value), np.errstate(invalid="ignore"):
                covariance = self.covariance.copy()
                covariance[0, 1] = value
                with self.assertRaisesRegex(AssertionError, "nonfinite"):
                    checks.compare(self.formula, self.direct, covariance)

    def test_rejects_nonfinite_allowance_parameters(self):
        for key in ("multiplier", "action_absolute"):
            for value in (np.nan, np.inf, -np.inf):
                with self.subTest(key=key, value=value), np.errstate(invalid="ignore"):
                    with self.assertRaisesRegex(AssertionError, "nonfinite"):
                        checks.compare(self.formula, self.direct, self.covariance, **{key: value})

    def test_rejects_nonfinite_derived_allowances(self):
        with self.subTest(quantity="covariance"), np.errstate(over="ignore"):
            with self.assertRaisesRegex(AssertionError, "nonfinite"):
                checks.compare(self.formula, self.direct, self.covariance * 1e160)
        with self.subTest(quantity="distance"), np.errstate(over="ignore"):
            formula = copy.deepcopy(self.formula)
            formula["nodes"][1]["target_scale"] = "1.79e308"
            with self.assertRaisesRegex(AssertionError, "nonfinite"):
                checks.compare(formula, self.direct, self.covariance)
        with self.subTest(quantity="mean"), np.errstate(over="ignore"):
            direct = copy.deepcopy(self.direct)
            direct["means"][1] = direct["target_means"][1] = [1e200, 0]
            with self.assertRaisesRegex(AssertionError, "nonfinite"):
                checks.compare(self.formula, direct, self.covariance)
        with self.subTest(quantity="action"), np.errstate(over="ignore"):
            with self.assertRaisesRegex(AssertionError, "nonfinite"):
                checks.compare(self.formula, self.direct, self.covariance, action_absolute=1e308)

    def test_rejects_missing_or_duplicate_population_node_coverage(self):
        selections = [[], [0, 1, 2, 3], [1, 2, 3, 4], [0, 1, 3, 4],
                      [0, 1, 1, 3, 4], [0, 1, 2, 3, 4, 4]]
        for selection in selections:
            with self.subTest(nodes=selection):
                formula = copy.deepcopy(self.formula)
                formula["nodes"] = [formula["nodes"][k] for k in selection]
                with self.assertRaisesRegex(AssertionError, "nodes"):
                    checks.compare(formula, self.direct, self.covariance)

    def test_rejects_incomplete_recursive_node_coverage(self):
        for key in ("times", "covariances", "w2", "particle_rms", "means", "target_means"):
            for count_change in (-1, 1):
                with self.subTest(key=key, count_change=count_change):
                    direct = copy.deepcopy(self.direct)
                    if count_change < 0:
                        direct[key] = direct[key][:-1]
                    else:
                        direct[key] = np.concatenate([direct[key], direct[key][-1:]])
                    with self.assertRaisesRegex(AssertionError, "nodes"):
                        checks.compare(self.formula, direct, self.covariance)

    def test_rejects_synchronized_missing_node_coverage(self):
        formula = copy.deepcopy(self.formula)
        direct = copy.deepcopy(self.direct)
        formula["nodes"] = formula["nodes"][:-1]
        for key in ("covariances", "w2", "particle_rms", "means", "target_means"):
            direct[key] = direct[key][:-1]
        with self.assertRaisesRegex(AssertionError, "nodes"):
            checks.compare(formula, direct, self.covariance)

    def test_rejects_finite_mismatches_at_each_comparison(self):
        locations = [("covariances", (1, 0, 0), "Node 1: covariance mismatch"),
                     ("w2", 1, "Node 1: w2 mismatch"),
                     ("particle_rms", 1, "Node 1: particle_rms mismatch"),
                     ("means", (1, 0), "Node 1: mean mismatch"),
                     ("action", None, "The action mismatch")]
        for key, index, message in locations:
            with self.subTest(key=key):
                direct = copy.deepcopy(self.direct)
                if index is None:
                    direct[key] += 0.01
                else:
                    direct[key][index] += 0.01
                with self.assertRaisesRegex(AssertionError, message):
                    checks.compare(self.formula, direct, self.covariance)


if __name__ == "__main__":
    unittest.main()
