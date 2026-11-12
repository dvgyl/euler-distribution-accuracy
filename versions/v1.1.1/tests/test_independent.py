import math
import unittest
import numpy as np
from euler_path.independent_interpolation import (
    gaussian_diagonal_errors, even_straight_complex_error,
    canonical_terminal_coefficient, independent_mixture_cosine_second_derivative,
    source_residual_for_gaussian_target,
)


class IndependentGaussianStartTests(unittest.TestCase):
    def test_anisotropic_three_dimensional_first_order_limit(self):
        eig=[0.64,1.0,2.25]
        limit=canonical_terminal_coefficient(eig)
        err,_=gaussian_diagonal_errors(eig,1024)
        self.assertGreater(limit,0.4)
        self.assertLess(abs(err*1024-limit)/limit,0.003)

    def test_isotropic_three_dimensional_first_order_limit(self):
        eig=[1.5625]*3
        limit=canonical_terminal_coefficient(eig)
        err,_=gaussian_diagonal_errors(eig,1024)
        self.assertLess(abs(err*1024-limit)/limit,0.003)

    def test_even_correlated_exactness_each_node(self):
        for n in [1,2,8,32,256]:
            term,maximum=even_straight_complex_error(1.25,n)
            self.assertLess(maximum,1e-12)
            self.assertLess(term,1e-12)

    def test_mixture_weak_two_jet(self):
        for mu in [0,0.3,0.75,2.0]:
            for k in [0.3,1.0,2.5]:
                a,b=independent_mixture_cosine_second_derivative(k,1.,mu,.49)
                self.assertAlmostEqual(a,b,places=12)

    def test_exact_canonical_residual_at_gaussian_source(self):
        s=np.diag([0.64,1.,2.25])
        self.assertAlmostEqual(source_residual_for_gaussian_target(s),math.sqrt(0.64**2+1+2.25**2),places=14)

    def test_rational_positive_certificate(self):
        self.assertGreater(math.exp(-2)/(2*math.sqrt(2*math.pi)),1/54)


if __name__=='__main__':
    unittest.main()
