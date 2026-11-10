"""Explicit transfer-observability formulas and bounded-witness diagnostics.

The analytic formulas implement the stated theorems. Except for the integer
network-norm routine, floating evaluation is not an interval certificate.
Full W2 errors must not be replaced by SW2 or finite-feature lower bounds.
"""
from __future__ import annotations
import itertools
import math
from fractions import Fraction
from typing import Iterable, Iterator, Sequence
import numpy as np


def refinement_energy(fractions: Sequence[float]) -> float:
    x=np.asarray(fractions,dtype=float)
    if x.ndim!=1 or len(x)<1 or not np.all(np.isfinite(x)) or np.any(x<=0):
        raise ValueError('Refinement fractions must be a positive finite vector')
    if not math.isclose(float(x.sum()),1.,rel_tol=0.,abs_tol=1e-12):
        raise ValueError('Fractions must sum to one')
    return max(0.,1.-float(x@x))


def signed_observer(error: float,n: int,H: float,T: float,A: float,M: float,
                    comparison: float,remainder: float,beta: float,
                    uniform: bool=False) -> float:
    vals=[error,H,T,A,M,comparison,remainder,beta]
    if not all(math.isfinite(v) for v in vals) or error<0 or n<1 or H<=0 or T<=0 or min(A,M,comparison)<=0 or remainder<0 or not 0<beta<=1:
        raise ValueError('Invalid observer parameters')
    if uniform and not math.isclose(H,T/n,rel_tol=1e-10,abs_tol=1e-14):
        raise ValueError('Uniform grid requires H=T/N')
    c=min(T/(2*A),1/(2*M));L=max(4/c,8*remainder/beta)
    term=math.sqrt(32*comparison*n*error/(beta*T*c)) if uniform else (64*comparison*n*error/(beta*c*c))**(1/3)
    return L*H+term


def gaussian_radial_moment(d: int,r: int) -> float:
    if d<1 or int(d)!=d or r<0 or int(r)!=r:raise ValueError('Require d>=1, integer r>=0')
    return sum(math.comb(r,j)*2**(j/2)*math.exp(math.lgamma((d+j)/2)-math.lgamma(d/2)) for j in range(r+1))


def multiindices(d: int,degree: int) -> Iterator[tuple[int,...]]:
    if d<1 or degree<0:raise ValueError('Invalid dimension/degree')
    if d==1:yield (degree,);return
    for k in range(degree+1):
        for tail in multiindices(d-1,degree-k):yield (k,)+tail


def hermite_envelopes(alpha: Sequence[int]) -> tuple[float,float]:
    if not alpha or any(a<0 or int(a)!=a for a in alpha) or sum(alpha)==0:
        raise ValueError('Nonzero nonnegative integer multiindex required')
    # Absolute coefficients grouped by total degree, avoiding symbolic packages.
    terms={0:1.}
    for a in alpha:
        uni={a-2*k:math.sqrt(math.factorial(a))/(2**k*math.factorial(k)*math.factorial(a-2*k)) for k in range(a//2+1)}
        out={}
        for i,x in terms.items():
            for j,y in uni.items():out[i+j]=out.get(i+j,0.)+x*y
        terms=out
    m=sum(alpha);return (sum(k*v for k,v in terms.items())/math.sqrt(m),sum(k*(k-1)*v for k,v in terms.items())/math.sqrt(m))


def spectral_residual_bound(error: float,n: int,H: float,d: int,K: float,q: int,
                            T: float=1.,factor: int=2,uniform: bool=False) -> dict:
    """The full own-law projected residual upper bound for equal m refinement.

    Finite q only; the returned tail is essential. This is a worst-case theorem
    evaluation, not a learned model's measured residual.
    """
    if K<1 or not 0<T<=1 or q<0 or int(q)!=q or factor<2 or int(factor)!=factor:
        raise ValueError('Invalid budget, horizon, degree, or refinement factor')
    F=math.exp(K*T);A=K*(1+K);B=K*(1+K)**2;J=B+K*A;D=K+2*K*K
    Q=2*(1+d);Q4=8*(1+d*(d+2));CE=A*F**3
    Ao=A*F*F*math.sqrt(Q);M=B*F**3*math.sqrt(Q4);V1=K*T*A*F**5+D*F**3
    beta=1-1/factor;squares=0.;details=[]
    for m in range(1,q+2):
        for alpha in multiindices(d,m):
            P1,P2=hermite_envelopes(alpha);G=P1*F**m;HH=(P2+K*T*P1*F)*F**m
            DC=G*2**(m-1)*F**(m-1)*math.sqrt(gaussian_radial_moment(d,2*m-2))
            C0=J*F**3*G/6+K*A*F*G/2+T*HH*A*A*F**(m+1)/8
            CB=HH*A+G*D
            CW=gaussian_radial_moment(d,m+1)*(C0*F**(m+1)+.5*CB*F**m*CE*T)
            CX=(1+factor**-2)*CW+(factor-1)*M/(4*factor**2)
            b=signed_observer(error,n,H,T,Ao,M,DC,CX,beta,uniform)
            squares+=b*b;details.append({'alpha':alpha,'bound':b,'comparison':DC,'weak_remainder':CW})
    tail=d*Q*V1*V1/(q+1);bound=F*math.sqrt(squares+tail)
    if not math.isfinite(bound):raise OverflowError('Constants overflow; use arbitrary precision')
    return {'bound':bound,'coefficient_square_sum':squares,'tail_square':tail,'outer_F':F,'coefficients':details}


def witness_radius(n: int,total_means: int,delta: float=.05) -> float:
    if n<1 or total_means<1 or not 0<delta<1:raise ValueError('Invalid confidence arguments')
    return math.sqrt(2*math.log(2*total_means/delta)/n)


def witness_interval(empirical: float,numerical_n: int,target_n: int,
                     total_means: int,delta: float=.05) -> tuple[float,float]:
    if not 0<=empirical<=2:raise ValueError('A [-1,1] witness discrepancy lies in [0,2]')
    r=witness_radius(numerical_n,total_means,delta)+witness_radius(target_n,total_means,delta)
    return max(0.,empirical-r),min(2.,empirical+r)


def exact_frobenius_ceiling(values: Iterable[float]) -> int:
    """Integer upper bound using exact rational values of stored binary floats."""
    squared=Fraction(0)
    for x in values:
        if not math.isfinite(float(x)):raise ValueError('Nonfinite coefficient')
        f=Fraction.from_float(float(x));squared+=f*f
    k=math.isqrt(squared.numerator//squared.denominator)
    if k*k*squared.denominator<squared.numerator:k+=1
    return k


def tanh_network_budget(weights: Sequence[np.ndarray],last_bias: np.ndarray,
                        width: int) -> dict:
    if len(weights)!=3 or width<1:raise ValueError('Require three weight matrices and positive width')
    s=[exact_frobenius_ceiling(np.asarray(w).ravel()) for w in weights]
    b=exact_frobenius_ceiling(np.asarray(last_bias).ravel());s1,s2,s3=s
    U=s3*width+b;P=s3*s2*s1;R=2*s3*s2*(s2+1)*s1*s1
    return {'frobenius_integer_upper':s,'last_bias_integer_upper':b,'output_upper':U,'first_derivative_upper':P,'second_derivative_upper':R,'K_integer_upper':max(1,U,P,R),'meaning':'Exact integer a priori bound for real tanh network with stored coefficients; not a numerical inference enclosure.'}


def empirical_bernstein_radius(variance: float, n: int, comparisons: int, delta: float, value_range: float = 4.0) -> float:
    """Two-sided simultaneous radius; iid bounded differences of given range.

    Applies Maurer--Pontil (2009), Theorem 4, with a union bound.
    This is for a finite witness mean, not a Wasserstein upper bound.
    """
    if n < 2 or comparisons < 1 or not 0 < delta < 1:
        raise ValueError("Require n >= 2, positive comparisons, and delta in (0,1).")
    if not math.isfinite(variance) or variance < 0 or not math.isfinite(value_range) or value_range <= 0:
        raise ValueError("Require nonnegative finite variance and positive finite range.")
    logterm = math.log(4.0 * comparisons / delta)
    return math.sqrt(2.0 * variance * logterm / n) + 7.0 * value_range * logterm / (3.0 * (n - 1))
