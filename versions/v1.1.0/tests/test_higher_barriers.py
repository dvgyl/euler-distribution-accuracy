"""Regression and exact-arithmetic checks; not proof-assistant verification."""
from fractions import Fraction
import math
from pathlib import Path
import sys
import unittest

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from euler_path.higher_barriers import (
    complex_structure, gaussian_coefficients, gaussian_euler_covariance,
    even_product_errors, gaussian_risk_frontier, distorted_grid,
    odd_budget_field, secant_coefficient, secant_terminal_bias,
    rational_sqrt_bounds, rational_gaussian_certificate,
    covariance_relaxation_bound, mixture_loss_check, mixture_euler_errors,
    symmetric_mixture_quantile,
)


class HigherBarrierTests(unittest.TestCase):
    def test_complex_structure(self):
        for d in [2, 4, 8]:
            j = complex_structure(d)
            np.testing.assert_array_equal(j.T, -j)
            np.testing.assert_array_equal(j @ j, -np.eye(d))
        with self.assertRaises(ValueError):
            complex_structure(3)

    def test_matrix_product_agree_on_nonuniform_grids(self):
        for n in [16, 63, 128]:
            t = distorted_grid(n, 0.45)
            j = complex_structure(4)
            for alpha in [0., 0.25, 0.8, 1.]:
                def matrix(s):
                    a, b = gaussian_coefficients(s, 1.25)
                    return a * np.eye(4) + np.sqrt(alpha) * b * j
                actual, _ = gaussian_euler_covariance(matrix, t, 4, 1.25)
                expected = even_product_errors(t, alpha, 4, 1.25)
                np.testing.assert_allclose(actual, expected, atol=4e-14, rtol=1e-9)

    def test_gaussian_risk_frontier_is_sharp(self):
        for alpha in [0., .4, 1., 2.]:
            vals = gaussian_risk_frontier(np.linspace(0, 1, 19), alpha, 4, 1.4)
            np.testing.assert_allclose(vals['residual'], vals['moment_bound'], atol=1e-13)
            np.testing.assert_allclose(vals['loss'], vals['bayes_loss'] + vals['gauge_energy'])

    def test_budget_dependent_even_rates(self):
        for beta in [0., .5, 1.]:
            errors = [even_product_errors(np.linspace(0, 1, n+1), 1-n**(-beta)).max()
                      for n in [512, 1024]]
            order = np.log2(errors[0] / errors[1])
            self.assertLess(abs(order - (1+beta)), .01)

    def test_doubled_loss_does_not_imply_zero_residual(self):
        b = np.zeros((3, 3)); b[1, 0] = np.sqrt(1.5); b[0, 1] = -np.sqrt(1.5)
        c = b @ b.T
        self.assertAlmostEqual(np.trace(c), 3.)
        self.assertAlmostEqual(np.linalg.norm(np.eye(3) - c), np.sqrt(1.5))

    def test_odd_budget_family_refines_at_first_order(self):
        scaled = []
        for n in [128, 256, 512]:
            grid = distorted_grid(n, .4 * np.cos(np.log(n)))
            err, _ = gaussian_euler_covariance(odd_budget_field(n), grid, 3)
            scaled.append(n * err.max())
        self.assertGreater(min(scaled), .2)
        self.assertLess(max(scaled), 8.)

    def test_secant_field_exact_target_grid(self):
        for n in [8, 32, 128]:
            grid = np.linspace(0, 1, n+1)
            error, _ = gaussian_euler_covariance(
                lambda t: secant_coefficient(t, 1/n) * np.eye(3), grid, 3)
            self.assertLess(error.max(), 2e-13)

    def test_secant_field_continuous_bias_is_first_order(self):
        for n in [256, 1024]:
            bias, coefficient, estimate = secant_terminal_bias(n)
            self.assertGreater(bias, 0.)
            self.assertLess(abs(n * bias / coefficient - 1), .01)
            self.assertLess(estimate, 1e-10)

    def test_square_root_enclosures_exact_integer_comparisons(self):
        for q in [Fraction(0), Fraction(1, 9), Fraction(2), Fraction(137, 19)]:
            lo, hi = rational_sqrt_bounds(q, 24)
            self.assertLessEqual(lo**2, q)
            self.assertGreaterEqual(hi**2, q)
            self.assertLessEqual(hi-lo, Fraction(1, 10**24))

    def test_rational_full_path_certificate(self):
        for n in [16, 32, 64]:
            for alpha in [Fraction(0), Fraction(1, 2), Fraction(1)]:
                cert = rational_gaussian_certificate(n, alpha)
                lo, hi = cert['maximum_node']
                self.assertLessEqual(lo, hi)
                self.assertLess(hi-lo, Fraction(1, 10**32))
                approximate = even_product_errors(np.linspace(0, 1, n+1), float(alpha)).max()
                self.assertLess(abs(float((lo+hi)/2)-approximate), 1e-13)
                if alpha < 1:
                    self.assertGreater(lo, 0)
                else:
                    self.assertEqual(lo, 0)

    def test_spectral_relaxation_limits(self):
        eig = [0.5, 1., 2.]
        self.assertAlmostEqual(covariance_relaxation_bound(eig, 0), np.linalg.norm(eig))
        self.assertEqual(covariance_relaxation_bound(eig, sum(eig)), 0)
        values = [covariance_relaxation_bound(eig, b) for b in [0., 1., 2., 3.]]
        self.assertTrue(all(a > b for a, b in zip(values, values[1:])))

    def test_non_gaussian_risk_acceleration_identity(self):
        for t in [0., .1, .3, .5, .8, 1.]:
            m, pairing = mixture_loss_check(t, .8, .6, 128)
            self.assertLess(abs(m-pairing), 2e-10)

    def test_non_gaussian_quadrature_refinement(self):
        for t in [.2, .5, .8]:
            old = mixture_loss_check(t, 1.2, .5, 128)
            new = mixture_loss_check(t, 1.2, .5, 192)
            self.assertLess(max(abs(a-b) for a, b in zip(old, new)), 3e-8)

    def test_mixture_quantile_symmetry_and_gaussian_limit(self):
        z = np.array([-16., -3., -1., 0., 1., 3., 16.])
        q = symmetric_mixture_quantile(z, 1.2, .7)
        np.testing.assert_allclose(q, -q[::-1], atol=1e-13)
        np.testing.assert_allclose(symmetric_mixture_quantile(z, 0., .7), .7*z, atol=1e-13)

    def test_non_gaussian_full_path_order(self):
        first = mixture_euler_errors(128, quadrature_nodes=32).max()
        second = mixture_euler_errors(256, quadrature_nodes=32).max()
        self.assertLess(abs(np.log2(first/second)-1), .025)
        refined = mixture_euler_errors(128, quadrature_nodes=48).max()
        self.assertLess(abs(first-refined), 1e-10)

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            even_product_errors([0, .5, .3], .5)
        with self.assertRaises(ValueError):
            rational_gaussian_certificate(16, Fraction(2))
        with self.assertRaises(ValueError):
            secant_coefficient(.5, 0)


if __name__ == '__main__':
    unittest.main()
