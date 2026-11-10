"""Independent finite-dimensional falsification checks for step-count transfer.

These do not certify the infinite-dimensional analytic theorems.
"""
from pathlib import Path
import sys, unittest
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.integrate import quad
from scipy.special import ndtri
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from euler_path.transfer import (bisect_grid,gaussian_transfer,stable_secant,gaussian_scale,
    constants,jet_modulus,source_risk_upper_bound,source_risk_gap_barrier,scalar_euler_scale)
from euler_path.neural_diagnostic import MixturePath,NumpyField,quantile_w2,integrate_fixed

class TransferTests(unittest.TestCase):
    def test_exact_bisection_not_a_redistributed_power_grid(self):
        grid=np.linspace(0,1,14)**1.7; fine=bisect_grid(grid)
        np.testing.assert_array_equal(fine[::2],grid)
        np.testing.assert_allclose(fine[1::2],(grid[:-1]+grid[1:])/2)
        self.assertFalse(np.allclose(fine,np.linspace(0,1,27)**1.7))
        with self.assertRaises(ValueError):bisect_grid(np.array([0,.2,.1,1]))

    def test_secant_exactness_is_coarse_only(self):
        for d in [1,2,3,5]:
            z=gaussian_transfer(64,d=d)
            self.assertLess(z['coarse_max'],2e-14)
            self.assertGreater(z['fine_common_max'],.005)

    def test_frozen_field_transfer_has_first_order(self):
        x=gaussian_transfer(512);y=gaussian_transfer(1024)
        self.assertAlmostEqual(x['fine_common_max']/y['fine_common_max'],2,delta=.005)
        predicted=(1+np.pi/2)/4
        self.assertAlmostEqual(1024*y['fine_terminal'],predicted,delta=.001)

    def test_scalar_balancing_cannot_remove_both_leading_terms(self):
        opt=minimize_scalar(lambda q:gaussian_transfer(512,q)['two_grid_max'],bounds=(0,1),method='bounded')
        self.assertTrue(opt.success);self.assertAlmostEqual(opt.x,.75,delta=.001)
        row=gaussian_transfer(512,opt.x)
        self.assertAlmostEqual(row['coarse_max'],row['fine_common_max'],delta=1e-7)
        self.assertAlmostEqual(512*row['two_grid_max'],(np.pi+2)/16,delta=.002)

    def test_nonuniform_recursive_contrast_coefficient(self):
        c=.7
        for n in [40,80,160]:
            grid=np.linspace(0,1,n+1)**1.4; fine=bisect_grid(grid);h=np.diff(grid)
            coarse=np.r_[1,np.cumprod(1+c*h)]
            refined=np.r_[1,np.cumprod(1+c*np.diff(fine))][::2]
            # Quadratic test transported by the exact exponential flow.
            z=.5*np.exp(-2*c*grid)*(coarse**2-refined**2)
            i,j=n//4,3*n//4
            predicted=-.25*c*c*np.sum(h[i:j]**2)
            remainder=z[j]-z[i]-predicted
            self.assertLess(abs(remainder),2*np.max(h)*np.sum(h[i:j]**2))
            self.assertLess(z[-1],0)

    def test_even_straight_coupling_is_exact_on_both_grids(self):
        j=np.array([[0,-1],[1,0.]])
        for grid in [np.linspace(0,1,22)**1.3,bisect_grid(np.linspace(0,1,22)**1.3)]:
            m=np.eye(2)
            for t,h in zip(grid[:-1],np.diff(grid)):
                a=(1-t)*np.eye(2)+t*j
                velocity=(j-np.eye(2))@np.linalg.inv(a)
                m=(np.eye(2)+h*velocity)@m
                expected=gaussian_scale(t+h)**2*np.eye(2)
                np.testing.assert_allclose(m@m.T,expected,atol=8e-15)

    def test_source_moment_identity_for_nonsymmetric_affine_fields(self):
        rng=np.random.default_rng(9)
        for d in [1,2,3,6]:
            for _ in range(8):
                m=rng.normal(size=(d,d));md=rng.normal(size=(d,d));b=rng.normal(size=d);nu=float(rng.uniform(.2,8))
                excess=np.sum((m+np.eye(d))**2)+np.sum(b*b)
                fp=2*np.trace(m)+2*d
                fpp=2*np.sum(m*m)+2*np.sum(b*b)+2*np.trace(md+m@m)-2*(d+nu)
                rhs=nu-excess+fp+.5*fpp
                self.assertAlmostEqual(rhs,np.trace(md+m@m),places=11)

    def test_doubled_risk_not_sufficient(self):
        j=np.array([[0,-1,0],[1,0,0],[0,0,0.]])*np.sqrt(1.5)
        c=j@j.T;residual=np.eye(3)-c
        self.assertAlmostEqual(np.trace(c),3)
        self.assertAlmostEqual(np.linalg.norm(residual,'fro'),np.sqrt(1.5))
        self.assertGreater(residual[2,2],.99)

    def test_explicit_jet_modulus_matches_one_variable_optimization(self):
        c=constants(1,1,4,T=.3)
        for b in [1e-8,1e-5,.1,20]:
            value,s=jet_modulus(b,c)
            self.assertGreater(s,0);self.assertLessEqual(s,c.T/2)
            # Compare in log radius to avoid a poorly resolved tiny minimizer.
            opt=minimize_scalar(lambda l:4*c.C_m*b/np.exp(2*l)+1.5*c.J_m*np.exp(l),bounds=(np.log(1e-12),np.log(c.T/2)),method='bounded')
            self.assertLessEqual(value,opt.fun*(1+1e-8))
        self.assertEqual(jet_modulus(0,c),(0,0))

    def test_explicit_risk_gap_barrier_has_uniform_margin(self):
        c=constants(1,1,4,T=.3)
        for gap in [.01,.1,1,4]:
            lower,h=source_risk_gap_barrier(gap,c)
            self.assertGreater(lower,0);self.assertGreater(h,0)
            # At the largest allowed coarse error and mesh, all three terms
            # sum to <= 3/4 of the excluded gap, independently of N.
            rhs=jet_modulus(lower+c.C_path*h,c)[0]+c.C_H*h+(64*c.D_q*lower/c.c_l**2)**(1/3)
            self.assertLessEqual(rhs,.75*gap*(1+1e-12))
        with self.assertRaises(ValueError):source_risk_gap_barrier(0,c)

    def test_source_risk_bound_zero_field_valid_budget(self):
        c=constants(1,1,4)
        for n in [8,32,128]:
            grid=np.linspace(0,1,n+1)
            e=float(np.max(np.abs(1-gaussian_scale(grid,2))))
            bound=source_risk_upper_bound(e,e,n,1/n,c,uniform=True)
            # u=0 has Delta=1 and nu=4 and belongs to B_1.
            self.assertGreaterEqual(bound,3)

    def test_initial_jet_finite_differences(self):
        rng=np.random.default_rng(13)
        for _ in range(30):
            a,b,c=rng.normal(size=3);t=np.linspace(0,1,10001);f=a*t+b*t*t+c*t**3
            v=float(np.abs(f).max())+1e-10;j=6*abs(c)
            for s in [.03,.12,.5]:
                self.assertLessEqual(abs(a),2.5*v/s+j*s*s+1e-9)
                self.assertLessEqual(abs(2*b),3*v/s**2+j*s+1e-9)

    def test_terminal_cancellation_does_not_give_whole_path_order(self):
        errors=[]
        for n in [256,512,1024]:
            t=np.linspace(0,1,n+1);scale=scalar_euler_scale(lambda s:-5*s*s,t)
            exact=np.exp(-5*t**3/3);e=np.abs(scale-exact)
            errors.append((e.max(),e[-1]))
        self.assertAlmostEqual(errors[-2][0]/errors[-1][0],2,delta=.02)
        self.assertAlmostEqual(errors[-2][1]/errors[-1][1],4,delta=.04)

    def test_mixture_quantile_inversion_and_centering(self):
        p=MixturePath();t=np.array([0,.2,.5,.9,1.])[:,None];q=np.linspace(.0001,.9999,101)
        x=p.quantile(t,q)
        np.testing.assert_allclose(p.cdf(t,x),np.broadcast_to(q,x.shape),atol=2e-14)
        self.assertAlmostEqual(p.weight*p.mean_left+(1-p.weight)*p.mean_right,0)
        self.assertAlmostEqual(p.target_second_moment,4)

    def test_mixture_canonical_is_quantile_time_derivative(self):
        p=MixturePath();t=np.array([.07,.31,.6,.9])[:,None];q=np.array([.01,.1,.4,.8,.97]);x=p.quantile(t,q);eps=1e-6
        deriv=(p.quantile(t+eps,q)-p.quantile(t-eps,q))/(2*eps)
        np.testing.assert_allclose(p.canonical(t,x),deriv,atol=3e-8,rtol=2e-7)

    def test_quantile_metric_sorts_not_particle_rms(self):
        x=np.array([1.,2.,3.]);y=x[::-1]
        self.assertEqual(float(quantile_w2(x,y)),0)
        self.assertGreater(np.sqrt(np.mean((x-y)**2)),1)

    def test_checkpoint_frozen_export_and_target_refinement_failure(self):
        p=MixturePath();q=(np.arange(512)+.5)/512;initial=ndtri(q)
        for seed in [17,29,43]:
            field=NumpyField(ROOT/'results'/'transfer'/'neural'/f'secant_seed{seed}.npz')
            coarse=integrate_fixed(field,initial,16);fine=integrate_fixed(field,initial,32)[::2]
            target=p.quantile(np.linspace(0,1,17)[:,None],q)
            self.assertGreater(quantile_w2(fine,target).max(),2*quantile_w2(coarse,target).max())

if __name__=='__main__':unittest.main()
