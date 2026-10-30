"""Check actual solver stages, terminal maps, and retained Gaussian evidence."""

import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import mpmath as mp
import numpy as np
from scipy.linalg import expm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from euler_path import gaussian_higher_order as comparison


CONFIG = ROOT / "experiments" / "gaussian_higher_order_config.json"
COVARIANCE = [["1.5", "0.25"], ["0.25", "0.75"]]


class GaussianHigherOrderTests(unittest.TestCase):
    def test_matrix_maps_and_particle_error_against_independent_exponential(self):
        covariance = np.array(COVARIANCE, dtype=float)
        factor = np.linalg.cholesky(covariance)
        skew = factor @ np.array([[0, -1], [1, 0]]) @ np.linalg.inv(factor)
        for budget in (12, 48):
            for method in comparison.METHODS:
                matrix = comparison.matrix_terminal(budget, method, COVARIANCE)
                scalar = comparison.polynomial_terminal(budget, method)
                numerical = np.array(matrix["terminal_map"], dtype=float)
                b = 0.6 if method == "corrected_euler" else 0.6 / np.sqrt(3) if method == "corrected_heun" else 0
                exact = expm(0.6 * np.eye(2) + b * skew)
                difference = numerical - exact
                particle = np.sqrt(np.trace(difference @ covariance @ difference.T))
                self.assertAlmostEqual(particle, float(scalar["particle_rms"]), delta=2e-15)
                self.assertEqual(matrix["field_evaluations"], budget)
                self.assertEqual(matrix["steps"], budget // comparison.METHODS[method][1])
                self.assertLess(float(matrix["covariance_residual"]), 1e-70)
                self.assertAlmostEqual(float(matrix["w2"]), float(scalar["w2"]), delta=1e-70)

    def test_rk3_stages_on_nonlinear_field_have_third_order(self):
        errors = []
        for step in (mp.mpf("0.02"), mp.mpf("0.01")):
            calls = []

            def field(current):
                calls.append(current)
                return current**2

            current = comparison._matrix_step(mp.mpf(1), field, step, "rk3")
            errors.append(abs(current - 1 / (1 - step)))
            self.assertEqual(len(calls), 3)
            self.assertEqual(calls[0], 1)
            self.assertGreater(calls[2], calls[1])
        self.assertGreater(float(errors[0] / errors[1]), 15)
        self.assertLess(float(errors[0] / errors[1]), 17)

    def test_fixed_fields_attain_distribution_and_particle_orders(self):
        fine = {method: comparison.polynomial_terminal(384, method) for method in comparison.METHODS}
        coarse = {method: comparison.polynomial_terminal(192, method) for method in comparison.METHODS}
        for method in ("corrected_heun", "base_rk3"):
            self.assertAlmostEqual(float(coarse[method]["w2"]) / float(fine[method]["w2"]), 8, delta=0.04)
        self.assertAlmostEqual(float(coarse["corrected_heun"]["particle_rms"]) / float(fine["corrected_heun"]["particle_rms"]), 4, delta=0.02)
        with mp.workdps(80):
            for values in (coarse, fine):
                self.assertLess(abs(mp.mpf(values["corrected_euler"]["w2"]) / mp.mpf(values["base_heun"]["w2"]) - 1), mp.mpf("1e-68"))
                self.assertLess(abs(mp.mpf(values["base_rk3"]["w2"]) / mp.mpf(values["base_rk3"]["particle_rms"]) - 1), mp.mpf("1e-68"))

    def test_invalid_parameters_are_rejected_at_both_entry_points(self):
        cases = [{"budget": value} for value in (0, -6, 3, 12.0, True, float("nan"))]
        cases += [{"method": value} for value in ("unknown", [])]
        cases += [{name: value} for name in ("rate", "horizon") for value in (0, -1, "nan", "inf", "-inf", None)]
        cases += [{"digits": value} for value in (0, 29, 50.0, True, float("nan"))]
        for change in cases:
            arguments = {"budget": 12, "method": "corrected_heun", **change}
            with self.subTest(change=change):
                with self.assertRaises(ValueError):
                    comparison.polynomial_terminal(**arguments)
                with self.assertRaises(ValueError):
                    comparison.matrix_terminal(covariance=COVARIANCE, **arguments)
        for trace in (0, -1, "nan", "inf"):
            with self.assertRaises(ValueError):
                comparison.polynomial_terminal(12, "base_rk3", covariance_trace=trace)
        for covariance in ([[1, 2], [2, 1]], [[1, 0], [1, 1]], [[1, 0], [0, "nan"]], [[1]], None):
            with self.assertRaises(ValueError):
                comparison.matrix_terminal(12, "base_rk3", covariance)

    def test_generation_retains_precision_evidence_and_is_reproducible(self):
        with tempfile.TemporaryDirectory() as directory:
            first, second = Path(directory) / "first", Path(directory) / "second"
            summary = comparison.generate(CONFIG, first)
            comparison.generate(CONFIG, second)
            self.assertEqual(summary["rows"], 4)
            self.assertLess(summary["maximum_relative_check_difference"], 1e-35)
            for path in first.iterdir():
                self.assertEqual(path.read_bytes(), (second / path.name).read_bytes())
            with (first / "gaussian_higher_order_raw.csv").open() as handle:
                raw = list(csv.DictReader(handle))
            self.assertEqual(len(raw), 32)
            self.assertEqual({int(row["precision_digits"]) for row in raw}, {50, 80})
            for row in raw:
                self.assertEqual(row["field_evaluations"], row["actual_field_evaluations"])
                self.assertEqual(int(row["matrix_arithmetic_precision_digits"]), int(row["precision_digits"]) + 20)

    def test_wrong_rk3_stage_is_rejected_by_independent_check(self):
        original = comparison._matrix_step

        def wrong_stage(current, field, step, solver):
            if solver != "rk3":
                return original(current, field, step, solver)
            first = field(current)
            second = field(current + step * first / 2)
            third = field(current + step * second)
            return current + step * (first + 4 * second + third) / 6

        with tempfile.TemporaryDirectory() as directory, patch.object(comparison, "_matrix_step", wrong_stage):
            output = Path(directory) / "output"
            with self.assertRaisesRegex(AssertionError, "matrix check"):
                comparison.generate(CONFIG, output)
            self.assertFalse(output.exists())

    def test_invalid_configuration_produces_no_output(self):
        original = json.loads(CONFIG.read_text())
        changes = [{"rate": "nan"}, {"horizon": "inf"}, {"relative_allowance": "nan"},
                   {"relative_allowance": 1}, {"evaluation_budgets": [12, 12]},
                   {"evaluation_budgets": [12, 13]}, {"precision_digits": [80, 50]},
                   {"precision_digits": [50, float("nan")]}, {"covariance": [[1, 0], [0, "nan"]]}]
        with tempfile.TemporaryDirectory() as directory:
            config, output = Path(directory) / "config.json", Path(directory) / "output"
            for change in changes:
                config.write_text(json.dumps({**original, **change}))
                with self.subTest(change=change), self.assertRaises(ValueError):
                    comparison.generate(config, output)
                self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
