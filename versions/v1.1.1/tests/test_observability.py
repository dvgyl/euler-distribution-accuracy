import math,sys,unittest
from pathlib import Path
from fractions import Fraction
import numpy as np
from numpy.polynomial.hermite_e import hermeval,hermegauss
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from euler_path.observability import *
from euler_path.transfer import stable_secant,gaussian_scale,scalar_euler_scale

class ObservabilityTests(unittest.TestCase):
    def test_balanced_factor(self):
        for m in range(2,8):self.assertAlmostEqual(refinement_energy([1/m]*m),1-1/m)
    def test_degenerate_energy(self):
        for n in [10,100]:self.assertAlmostEqual(refinement_energy([1/n,1-1/n]),2/n*(1-1/n))
    def test_bad_fractions(self):
        for a in [[],[.3,.3],[0,1],[-.1,1.1],[float('nan')]]:
            with self.assertRaises(ValueError):refinement_energy(a)
    def test_multicount(self):
        for d in [1,2,4]:
            for m in [0,1,3]:self.assertEqual(len(list(multiindices(d,m))),math.comb(d+m-1,m))
    def test_moment(self):
        self.assertAlmostEqual(gaussian_radial_moment(1,0),1)
        self.assertAlmostEqual(gaussian_radial_moment(1,2),2+2*math.sqrt(2/math.pi))
        self.assertAlmostEqual(gaussian_radial_moment(3,1),1+2*math.sqrt(2/math.pi))
    def test_envelopes(self):
        self.assertEqual(hermite_envelopes((1,0)),(1.,0.))
        p1,p2=hermite_envelopes((2,));self.assertAlmostEqual(p1,1);self.assertAlmostEqual(p2,1)
        p1,p2=hermite_envelopes((1,1));self.assertAlmostEqual(p1,math.sqrt(2));self.assertAlmostEqual(p2,math.sqrt(2))
    def test_gradient_orthogonality(self):
        x,w=hermegauss(16);w/=math.sqrt(2*math.pi)
        for a in range(1,7):
            pa=hermeval(x,[0]*(a-1)+[1/math.sqrt(math.factorial(a-1))])
            for b in range(1,7):
                pb=hermeval(x,[0]*(b-1)+[1/math.sqrt(math.factorial(b-1))])
                self.assertAlmostEqual(float(w@(pa*pb)),float(a==b),places=11)
    def test_gaussian_sobolev_tail(self):
        rng=np.random.default_rng(4);v=rng.normal(size=12)
        for q in [0,2,5]:self.assertLessEqual(sum(v[q+1:]**2),sum(np.arange(12)*v*v)/(q+1))
    def test_observer_synthetic(self):
        for n in [8,32,128]:
            h=1/n;t=np.linspace(0,1,n+1);I=np.sin(t*4);Z=np.r_[0,-.25*np.cumsum(h*h*I[:-1])]
            E=float(max(abs(Z)));b=signed_observer(E,n,h,1,1,4,1,0,.5,True)
            self.assertGreaterEqual(b,1)
    def test_spectral_tail_required(self):
        j=spectral_residual_bound(0,10**30,1e-30,1,1,1,uniform=True)
        self.assertGreater(j['tail_square'],0);self.assertGreaterEqual(j['bound'],j['outer_F']*math.sqrt(j['tail_square']))
    def test_weak_strong_separation(self):
        for n in [10,100]:
            full=math.sqrt((1-math.exp(-2*n*n))/2)/n**2
            weak_upper=(1+math.sqrt(2/math.pi))/n**3
            self.assertGreater(full/weak_upper,.3*n)
    def test_sharp_constant(self):
        n=4096
        for m in [2,3,4]:
            grid=np.linspace(0,1,n*m+1);q=scalar_euler_scale(lambda t:float(stable_secant(t,1/n)),grid)[::m]
            e=max(abs(q-gaussian_scale(grid[::m])))
            self.assertAlmostEqual(n*e,.5*(1-1/m)*(1+math.pi/2),delta=.002)
    def test_degenerate_refinement_rate(self):
        es=[]
        for n in [64,128]:
            coarse=np.linspace(0,1,n+1);grid=np.empty(2*n+1);grid[::2]=coarse;grid[1::2]=coarse[:-1]+1/n**2
            q=scalar_euler_scale(lambda t:float(stable_secant(t,1/n)),grid)[::2];es.append(max(abs(q-gaussian_scale(coarse))))
        self.assertGreater(es[0]/es[1],3.8)
    def test_temporal_alias(self):
        n=128;h=1/n
        def c(t):
            a=float(stable_secant(t,h));p=float(stable_secant(t-h/2,h));b=p/(1+h*p/2)
            return a+(b-a)*math.sin(math.pi*t/h)**2
        grid=np.linspace(0,1,2*n+1);q=scalar_euler_scale(c,grid)[::2]
        self.assertLess(max(abs(q-gaussian_scale(grid[::2]))),1e-12)
    def test_endpoint_boundary(self):
        errs=[]
        for n in [128,256]:
            t=np.linspace(0,1,n+1);q=scalar_euler_scale(lambda s:-5*s*s,t);er=abs(q-np.exp(-5*t**3/3));errs.append((er[-1],max(er)))
        self.assertGreater(errs[0][0]/errs[1][0],3.8);self.assertTrue(1.9<errs[0][1]/errs[1][1]<2.1)
    def test_interval(self):
        l,h=witness_interval(.2,1000,2000,10);self.assertTrue(0<=l<.2<h<=2)
        self.assertGreater(witness_radius(100,10),witness_radius(1000,10))
    def test_exact_norm(self):
        self.assertEqual(exact_frobenius_ceiling([3.,4.]),5)
        self.assertEqual(exact_frobenius_ceiling([3.,4.,2**-30]),6)
        self.assertEqual(exact_frobenius_ceiling([0.,0.]),0)
    def test_network_budget(self):
        j=tanh_network_budget([np.eye(2),np.eye(2),np.eye(2)],np.zeros(2),2)
        self.assertEqual(j['frobenius_integer_upper'],[2,2,2]);self.assertEqual(j['K_integer_upper'],96)
    def test_input_guards(self):
        with self.assertRaises(ValueError):witness_radius(0,1)
        with self.assertRaises(ValueError):hermite_envelopes((0,0))
        with self.assertRaises(ValueError):spectral_residual_bound(0,2,.5,1,0,1)

class BernsteinTests(unittest.TestCase):
    def test_explicit_radius(self):
        from euler_path.observability import empirical_bernstein_radius
        n, q, delta, v = 262144, 3456, .025, .7
        expected = math.sqrt(2*v*math.log(4*q/delta)/n) + 28*math.log(4*q/delta)/(3*(n-1))
        self.assertAlmostEqual(empirical_bernstein_radius(v,n,q,delta),expected,places=15)

    def test_sample_size_decreases_radius(self):
        from euler_path.observability import empirical_bernstein_radius
        self.assertGreater(empirical_bernstein_radius(.1,100,10,.05), empirical_bernstein_radius(.1,1000,10,.05))

    def test_invalid_radius_input(self):
        from euler_path.observability import empirical_bernstein_radius
        for args in [(-1,100,10,.05),(.1,1,10,.05),(.1,100,0,.05),(.1,100,10,1.)]:
            with self.assertRaises(ValueError): empirical_bernstein_radius(*args)

if __name__=='__main__':unittest.main()
