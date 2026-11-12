"""Independent checks for the endpoint-cancellation and schedule counterexample."""
from pathlib import Path
import sys
import unittest

import numpy as np
from scipy.integrate import quad

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from euler_path.global_validation import (
    CosineGaussian, covariance_reference, euler_profile, high_precision_reference)


class GlobalValidationTests(unittest.TestCase):
    def test_material_acceleration_projection_is_nonzero(self):
        path = CosineGaussian()
        for t in (.07, .31, .77):
            h = 1e-5
            matrix = path.matrix(t)
            derivative = (path.matrix(t+h)-path.matrix(t-h))/(2*h)
            acceleration = derivative+matrix@matrix
            np.testing.assert_allclose((acceleration+acceleration.T)/2,
                                       path.defect(t)*np.eye(2), atol=2e-12)
            np.testing.assert_allclose(matrix+matrix.T, 2*path.a*np.eye(2), atol=1e-15)
        self.assertGreater(path.projected_acceleration_norm(0.), .1)
        self.assertEqual(CosineGaussian(epsilon=0.).projected_acceleration_norm(.3), 0.)

    def test_quantile_schedule_is_the_declared_probability_mesh(self):
        path = CosineGaussian()
        nodes = path.nodes(257, "quantile")
        cdf = nodes+path.delta*np.sin(2*np.pi*nodes)/(2*np.pi)
        np.testing.assert_allclose(cdf, np.linspace(0, 1, 258), rtol=0, atol=6e-16)
        self.assertTrue(np.all(np.diff(nodes) > 0))
        self.assertGreater(np.ptp(np.diff(nodes)), .001)

    def test_scalar_metric_agrees_with_independent_covariance_recursion(self):
        for epsilon in (.08, 0.):
            path = CosineGaussian(epsilon=epsilon)
            for schedule in ("uniform", "quantile"):
                for n in (17, 63):
                    scalar = euler_profile(path, n, schedule)
                    matrix = covariance_reference(path, n, schedule)
                    np.testing.assert_allclose(matrix["covariance"],
                                               scalar["euler_scale"][-1]**2*np.eye(2), atol=5e-14, rtol=0)
                    self.assertLess(abs(matrix["w2"]-scalar["w2"][-1]), 3e-14)

    def test_weighted_coefficients_match_independent_quadrature(self):
        path = CosineGaussian()
        for schedule in ("uniform", "quantile"):
            for t in (.13, .25, .75, 1.):
                first = quad(lambda s: path.defect(s)/path.density(s, schedule), 0, t,
                             epsabs=1e-13, epsrel=1e-13)[0]
                second = quad(lambda s: path.density(s, schedule)**-2, 0, t,
                              epsabs=1e-13, epsrel=1e-13)[0]
                self.assertAlmostEqual(first, path.weighted_defect_integral(t, schedule), places=13)
                self.assertAlmostEqual(second, path.reciprocal_density_squared_integral(t, schedule), places=13)

    def test_endpoint_cancels_but_interior_does_not(self):
        path = CosineGaussian()
        coarse, fine = [euler_profile(path, n) for n in (1024, 2048)]
        endpoint_order = np.log2(coarse["w2"][-1]/fine["w2"][-1])
        interior_order = np.log2(coarse["w2"][256]/fine["w2"][512])
        self.assertGreater(endpoint_order, 1.99)
        self.assertLess(abs(interior_order-1), .012)
        predicted = path.constants()
        self.assertLess(abs(2048**2*fine["w2"][-1]/predicted["endpoint_coefficient"]-1), .001)
        self.assertLess(abs(2048*fine["w2"][512]/predicted["interior_coefficient"]-1), .01)
        self.assertGreater(np.max(fine["w2"]), 100*fine["w2"][-1])

    def test_changed_schedule_destroys_cosine_endpoint_cancellation(self):
        for epsilon, expected in ((.08, 1), (0., 2)):
            path = CosineGaussian(epsilon=epsilon)
            coarse, fine = [euler_profile(path, n, "quantile") for n in (2048, 4096)]
            order = np.log2(coarse["w2"][-1]/fine["w2"][-1])
            self.assertLess(abs(order-expected), .02)
            coefficient = path.constants("quantile")["endpoint_coefficient"]
            self.assertLess(abs(4096**expected*fine["w2"][-1]/coefficient-1), .01)

    def test_uniform_signed_third_order_term_matches_expansion(self):
        for epsilon in (.08, 0.):
            path = CosineGaussian(epsilon=epsilon)
            n = 2048
            log_error = euler_profile(path, n)["log_scale_error"][-1]
            measured = n**3*(log_error+2*path.a**3/(3*n**2))
            expected = path.a**4-epsilon**2/8
            self.assertLess(abs(measured/expected-1), .002)

    def test_finest_endpoint_agrees_with_100_digit_product(self):
        for epsilon, schedule, n in ((.08, "uniform", 4096), (0., "quantile", 128)):
            path = CosineGaussian(epsilon=epsilon)
            reference = high_precision_reference(path, n, schedule, digits=100)
            expected = float(reference["w2"])
            computed = euler_profile(path, n, schedule)["w2"][-1]
            self.assertLess(abs(computed/expected-1), 2e-9)


if __name__ == "__main__":
    unittest.main()
