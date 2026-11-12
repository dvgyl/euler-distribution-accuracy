"""Step-count transfer controls and explicit constants.

These routines evaluate analytical identities and conservative theorem
constants. Floating-point evaluation is not an interval certificate.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
import math
from typing import Callable
import numpy as np


@dataclass(frozen=True)
class TransferConstants:
    d: int
    K: float
    T: float
    target_trace: float
    F: float
    A: float
    B: float
    J: float
    Q: float
    Q3: float
    Q4: float
    C_E: float
    D_q: float
    M_q: float
    C_q: float
    C_transfer: float
    D0: float
    c_l: float
    C_H: float
    C_path: float
    C_m: float
    J_m: float

    def to_dict(self) -> dict:
        return asdict(self)


def constants(d: int, K: float, target_trace: float, T: float = 1.) -> TransferConstants:
    """Constants used in the finite-N initial-risk inequality, for 0<T<=1.

    Raises OverflowError rather than silently returning meaningless infinities.
    A certified application additionally needs a proved global budget K.
    """
    if isinstance(d, bool) or int(d) != d or d < 1:
        raise ValueError('d must be a positive integer')
    if not (math.isfinite(K) and K >= 1 and 0 < T <= 1
            and math.isfinite(target_trace) and target_trace >= 0):
        raise ValueError('Require K>=1, 0<T<=1 and nonnegative target trace')
    d = int(d)
    F = math.exp(K*T)
    A, B = K*(1+K), K*(1+K)**2
    J, D = B + K*A, K + 2*K*K
    Q, Q4 = 2*(1+d), 8*(1+d*(d+2))
    Q3 = Q4**.75
    C_E = A*F**3
    H_q = (1+K*T)*F**3
    D_q, M_q = 3*F**3*math.sqrt(Q), B*F**4*math.sqrt(Q*Q4)
    Cq0 = J*F**5/6 + K*A*F**3/2 + T*H_q*A*A*F**3/4
    Cqb = H_q*A + F*F*D
    C_q = Q3*(Cq0*F**3 + .5*Cqb*F*F*C_E*T)
    C_transfer = 1.25*C_q + M_q/16
    D0 = A*F*F*Q
    c_l = min(T/D0, 1/(2*M_q))
    C_H = max(2/c_l, 16*C_transfer)
    C_path = C_E*math.sqrt(Q)*T + K*F*math.sqrt(Q) + math.sqrt(d+target_trace)
    C_m = F*math.sqrt(Q) + math.sqrt(d) + math.sqrt(target_trace)
    J_m = 6*K*A*F*F*Q + 2*J*F**3*Q3
    result = TransferConstants(d, K, T, target_trace, F, A, B, J, Q, Q3,
                               Q4, C_E, D_q, M_q, C_q, C_transfer, D0,
                               c_l, C_H, C_path, C_m, J_m)
    if not all(math.isfinite(v) for v in asdict(result).values()):
        raise OverflowError('Conservative constants overflow; use arbitrary precision')
    return result


def jet_modulus(bias: float, c: TransferConstants) -> tuple[float, float]:
    """(omega_K(bias), selected finite-difference radius), with omega_K(0)=0."""
    if not math.isfinite(bias) or bias < 0:
        raise ValueError('bias must be finite and nonnegative')
    if bias == 0:
        return 0., 0.
    s = min(c.T/2, (16*c.C_m*bias/(3*c.J_m))**(1/3))
    return 4*c.C_m*bias/(s*s) + 1.5*c.J_m*s, s


def source_risk_upper_bound(coarse_error: float, fine_error: float,
                           n: int, max_step: float,
                           c: TransferConstants, uniform: bool = False) -> float:
    """Upper bound for |Delta_0-tr(S)|, not a numerical residual estimator."""
    if n < 1 or max_step <= 0 or coarse_error < 0 or fine_error < 0:
        raise ValueError('Invalid error or grid input')
    omega, _ = jet_modulus(coarse_error + c.C_path*max_step, c)
    e = coarse_error + fine_error
    if uniform:
        if not math.isclose(max_step, c.T/n, rel_tol=1e-10, abs_tol=1e-14):
            raise ValueError('Uniform-grid bound requires H=T/N')
        term = math.sqrt(32*c.D_q*n*e/(c.T*c.c_l))
    else:
        term = (64*c.D_q*n*e/(c.c_l*c.c_l))**(1/3)
    return omega + c.C_H*max_step + term



def source_risk_gap_barrier(gap: float, c: TransferConstants) -> tuple[float, float]:
    """Return (c_gap, H_gap) with two-grid error >= c_gap/N for gap >= gap.

    These conservative constants apply when H <= H_gap, provided all theorem
    hypotheses hold. Floating-point evaluation is not an interval certificate.
    """
    if not math.isfinite(gap) or gap <= 0:
        raise ValueError('gap must be finite and positive')
    s = min(c.T/2, gap/(12*c.J_m))
    b = gap*s*s/(32*c.C_m)
    h = min(c.T, gap/(4*c.C_H), b/(2*c.C_path))
    lower = min(b/2, c.c_l*c.c_l*gap**3/(4096*c.D_q))
    if not (0 < lower < math.inf and 0 < h < math.inf):
        raise OverflowError('Constants underflow or overflow; use arbitrary precision')
    return lower, h

def bisect_grid(grid: np.ndarray) -> np.ndarray:
    grid = np.asarray(grid, dtype=float)
    if grid.ndim != 1 or len(grid) < 2 or not np.all(np.diff(grid) > 0):
        raise ValueError('grid must be a strictly increasing vector')
    out = np.empty(2*len(grid)-1)
    out[::2], out[1::2] = grid, (grid[:-1]+grid[1:])/2
    return out


def gaussian_scale(t: np.ndarray | float, sigma: float = 1.) -> np.ndarray:
    if sigma <= 0:
        raise ValueError('sigma must be positive')
    return np.sqrt((1-np.asarray(t))**2 + sigma*sigma*np.asarray(t)**2)


def canonical_coefficient(t: np.ndarray | float, sigma: float = 1.) -> np.ndarray:
    t = np.asarray(t)
    return ((1+sigma*sigma)*t-1)/((1-t)**2+sigma*sigma*t*t)


def stable_secant(t: np.ndarray | float, h: float, sigma: float = 1.) -> np.ndarray:
    """Rationalized Gaussian scale secant; no subtractive cancellation."""
    if h <= 0:
        raise ValueError('h must be positive')
    t = np.asarray(t)
    g, gp = gaussian_scale(t, sigma), gaussian_scale(t+h, sigma)
    return (2*((1+sigma*sigma)*t-1)+(1+sigma*sigma)*h)/(g*(gp+g))


def scalar_euler_scale(coefficient: Callable[[float], float], grid: np.ndarray) -> np.ndarray:
    grid = np.asarray(grid, dtype=float)
    if not np.all(np.diff(grid) > 0):
        raise ValueError('grid must increase')
    multipliers = 1+np.diff(grid)*np.asarray([coefficient(t) for t in grid[:-1]])
    # A negative scalar multiplier changes orientation but not a Gaussian scale.
    return np.r_[1., np.cumprod(np.abs(multipliers))]


def gaussian_transfer(n: int, theta: float = 1., sigma: float = 1.,
                      d: int = 1, power: float = 1.) -> dict:
    """One frozen scalar field, on the base grid and its exact bisection.

    The secant parameter h is the base uniform step, even on distorted grids;
    no exact-fit claim is made there. theta=0 is the canonical velocity;
    theta=1 fits every node of the uniform base grid exactly.
    """
    if n < 1 or d < 1 or power <= 0:
        raise ValueError('Invalid n, d, or grid power')
    h = 1/n
    grid = np.linspace(0, 1, n+1)**power
    refined = bisect_grid(grid)
    coefficient = lambda t: float((1-theta)*canonical_coefficient(t, sigma)
                                  + theta*stable_secant(t, h, sigma))
    coarse = scalar_euler_scale(coefficient, grid)
    fine = scalar_euler_scale(coefficient, refined)[::2]
    target = gaussian_scale(grid, sigma)
    ec, ef = np.sqrt(d)*np.abs(coarse-target), np.sqrt(d)*np.abs(fine-target)
    return {'N': n, 'theta': float(theta), 'sigma': sigma, 'd': d,
            'grid_power': power, 'coarse_max': float(ec.max()),
            'fine_common_max': float(ef.max()),
            'coarse_terminal': float(ec[-1]), 'fine_terminal': float(ef[-1]),
            'two_grid_sum': float(ec.max()+ef.max()),
            'two_grid_max': float(max(ec.max(), ef.max())),
            'cross_grid_max': float(np.sqrt(d)*np.abs(coarse-fine).max()),
            'source_excess': float(d*(coefficient(0)+1)**2),
            'H': float(np.diff(grid).max())}
