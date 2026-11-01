"""Independent mathematical and numerical checks for the revision."""
from pathlib import Path
import sys
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from euler_path.revision_experiments import (
    HarmonicPath,polar_quadrature,discover_correction,integrate,errors,gaussian_infeasibility)


class RevisionTests(unittest.TestCase):
    def setUp(self):
        self.path=HarmonicPath()

    def test_global_inverse_recovers_nonradial_labels(self):
        z=np.array([0,.03+.04j,.7+.2j,-1.1+.5j,2-3j,15+8j])
        for t in (0,.25,1):
            recovered=self.path.inverse(t,self.path.map(t,z))
            np.testing.assert_allclose(recovered,z,atol=4e-14,rtol=0)

    def test_acceleration_includes_nonlinear_map_hessian(self):
        z=np.array([.2+.3j,.7-.1j,-1+.8j,2+1j]);t=.43;h=2e-4
        for omega in (0.,.4,.55):
            # Independent symmetric second time difference of the exact trajectory.
            label=z*np.exp(-1j*(omega-self.path.a)*t)
            plus=self.path.exact(t+h,label,omega);center=self.path.exact(t,label,omega)
            minus=self.path.exact(t-h,label,omega)
            numerical=(plus-2*center+minus)/h**2
            exact=self.path.acceleration_labels(t,z,omega)
            np.testing.assert_allclose(numerical,exact,rtol=3e-6,atol=5e-8)
            if omega==self.path.a:
                np.testing.assert_allclose(exact,2*self.path.a**2*self.path.rotation_labels(t,z),atol=2e-16)

    def test_exact_residual_matches_independent_quadrature(self):
        z,weights=polar_quadrature(128,64)
        for omega,c in ((0,.1),(.4,.32),(.39,.3),(-.4,-.32)):
            residual=self.path.acceleration_labels(1,z,omega)-c*self.path.rotation_labels(1,z)
            quadrature=np.sqrt(weights@np.abs(residual)**2)
            self.assertAlmostEqual(quadrature,self.path.residual_upper(omega,c),places=12)

    def test_global_search_checks_all_stationary_points(self):
        result=discover_correction(self.path)
        best=result["global_minimizer"]
        self.assertAlmostEqual(best["omega"],.4,places=12)
        self.assertAlmostEqual(best["c"],.32,places=12)
        self.assertLess(best["residual_upper"],1e-12)
        self.assertEqual(len(result["stationary_points"]),3)
        self.assertGreater(result["local_runs"][1]["residual_upper"],.01)

    def test_second_moment_matches_exact_angular_integral(self):
        z,weights=polar_quadrature(128,64)
        for t in (0,.5,1):
            self.assertAlmostEqual(weights@np.abs(self.path.map(t,z))**2,
                                   self.path.target_second_moment(t),places=12)

    def test_rotation_coupling_changes_distribution_rate(self):
        z,weights=polar_quadrature(48,48)
        pairs=[]
        for n in (64,128):
            out,_,_=integrate(self.path,z,n,"Euler",.4)
            pairs.append(errors(self.path,out,z,weights,.4,n,.32))
        self.assertGreater(np.log2(pairs[0]["w2_upper_coupling"]/pairs[1]["w2_upper_coupling"]),1.95)
        self.assertLess(np.log2(pairs[0]["particle_rms"]/pairs[1]["particle_rms"]),1.05)
        for row in pairs:
            self.assertLess(row["w2_lower_second_moment"],row["w2_upper_coupling"])

    def test_infeasible_direction_even_when_scalar_budget_positive(self):
        rows=gaussian_infeasibility()
        self.assertGreater(min(row["scalar_budget"] for row in rows),.3)
        for row in rows:
            self.assertGreater(row["all_field_residual_lower"],0)
            self.assertGreaterEqual(row["best_affine_projection_residual"]+1e-10,row["all_field_residual_lower"])

    def test_vector_field_not_free_of_evaluation_cost(self):
        z=np.array([.1+.2j,.7+.3j])
        for method,stages in [("Euler",1),("Heun",2),("RK3",3)]:
            _,counts,_=integrate(self.path,z,12,method,.4)
            self.assertEqual(counts["field_evaluations"],12*stages)
            self.assertGreater(counts["inverse_iterations"],counts["field_evaluations"])
            self.assertLessEqual(counts["max_inverse_bound"],2e-14)

    def test_learned_path_reference_and_moment_are_independent(self):
        from scipy.integrate import solve_ivp
        z=np.array([.2+.3j,.8-.5j,2.+.7j]);delta=.017
        solution=solve_ivp(lambda t,x:self.path.field(t,x,.4,delta=delta),[0,1],self.path.map(0,z),
                           method="DOP853",rtol=2e-12,atol=2e-13)
        np.testing.assert_allclose(solution.y[:,-1],self.path.exact(1,z,.4,delta),rtol=3e-12,atol=3e-12)
        q,w=polar_quadrature(128,64)
        self.assertAlmostEqual(w@np.abs(self.path.exact(1,q,.4,delta))**2,
                               self.path.learned_second_moment(1,delta),places=12)


if __name__=="__main__":unittest.main()
