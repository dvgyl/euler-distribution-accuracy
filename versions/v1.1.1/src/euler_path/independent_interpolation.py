"""Population Gaussian tests and exact weak-jet identities for independent interpolation.

No learned model, sampled Wasserstein estimator, or random Monte Carlo result is
used in this module.  All W2 values use diagonal Gaussian optimal transport.
"""
from __future__ import annotations

import math
from typing import Sequence
import numpy as np
from scipy.integrate import quad


def scale(t: float, eigenvalues: np.ndarray) -> np.ndarray:
    eig = np.asarray(eigenvalues, dtype=float)
    return np.sqrt((1.0 - t)**2 + t**2 * eig)


def canonical_velocity(t: float, eigenvalues: np.ndarray) -> np.ndarray:
    eig = np.asarray(eigenvalues, dtype=float)
    return ((1.0 + eig) * t - 1.0) / ((1.0 - t)**2 + t**2 * eig)


def gaussian_diagonal_errors(eigenvalues: Sequence[float], n: int) -> tuple[float, float]:
    """Exact W2 at terminal and maximum nodes for canonical linear Euler.

    Z~N(0,I), Y~N(0,diag(eigenvalues)) are INDEPENDENT in the path definition.
    The Euler field is the conditional-mean flow matching field, i.e. the
    unique symmetric/kinetic-action-minimizing Gaussian representative.
    """
    eig = np.asarray(eigenvalues, dtype=float)
    if eig.ndim != 1 or np.any(eig <= 0) or n < 1:
        raise ValueError("positive eigenvalues and n>=1 required")
    h = 1.0 / n
    m = np.ones_like(eig)
    max_w2 = 0.0
    for k in range(n):
        t = k * h
        m *= 1.0 + h * canonical_velocity(t, eig)
        g = scale((k + 1) * h, eig)
        max_w2 = max(max_w2, float(np.linalg.norm(m - g)))
    return float(np.linalg.norm(m - scale(1.0, eig))), max_w2


def canonical_terminal_coefficient(eigenvalues: Sequence[float]) -> float:
    """Analytic limit of N*W2(base Gaussian Euler terminal, exact Gaussian).

    For each eigendirection g''/g = lambda/g^4.  The scalar log expansion
    gives (1/2)*g(1)*integral_0^1 lambda/g(t)^4 dt.
    """
    eig = np.asarray(eigenvalues, dtype=float)
    coeffs = []
    for lam in eig:
        integral, _ = quad(lambda t: lam / ((1-t)**2 + t*t*lam)**2,
                           0.0, 1.0, epsabs=2e-13, epsrel=2e-13)
        coeffs.append(0.5 * math.sqrt(lam) * integral)
    return float(np.linalg.norm(coeffs))


def even_straight_complex_error(sigma: float, n: int) -> tuple[float, float]:
    """Floating-point check of exact Euler transport by an even-dimensional
    correlated straight coupling with the INDEPENDENT path's marginals.

    The theoretical error is identically zero on EVERY grid. Values here
    measure rounding and are not interpreted as positive W2 evidence.
    """
    if sigma <= 0 or n < 1:
        raise ValueError("sigma>0 and n>=1 required")
    h = 1.0/n
    m = 1.0+0.0j
    max_error = 0.0
    for k in range(n):
        t = k*h
        a = (1.0-t)+1j*sigma*t
        beta = (-1.0+1j*sigma)/a
        m *= (1.0+h*beta)
        target = math.sqrt((1.0-(k+1)*h)**2+sigma**2*((k+1)*h)**2)
        max_error = max(max_error, math.sqrt(2)*abs(abs(m)-target))
    return math.sqrt(2)*abs(abs(m)-sigma), max_error


def independent_mixture_cosine_second_derivative(
        wave_number: float, source_projection: float, mixture_shift_projection: float,
        target_noise_variance: float) -> tuple[float, float]:
    """Two exactly equal expressions for the t=0 weak second derivative.

    Source is N(0,1) in the chosen direction. Target is the symmetric
    mixture (1/2)N(+mu,tau^2)+(1/2)N(-mu,tau^2); shift and source
    projections are independent. source_projection is only a directional
    normalization check and must be 1 for the normalized theorem.
    """
    k = float(wave_number)
    s = float(source_projection)
    m = float(mixture_shift_projection)
    tau2 = float(target_noise_variance)
    if s <= 0 or tau2 < 0:
        raise ValueError("positive initial variance required")
    # F(t) = exp(-k^2*((1-t)^2*s+t^2*tau2)/2)*cos(k*t*m)
    closed = math.exp(-k*k*s/2) * (k**4*s*s-k*k*(s+tau2+m*m))
    # Reference straight coupling: E[phi''(G)(G^2+E H^2)].
    # For G~N(0,s), E[G^2 cos(kG)] = (s-k^2*s^2)*exp(-k^2*s/2).
    by_chain = -k*k * math.exp(-k*k*s/2) * ((s-k*k*s*s)+tau2+m*m)
    return closed, by_chain


def source_residual_for_gaussian_target(target_covariance: np.ndarray) -> float:
    """For the Gaussian canonical field, a_0(x)=S*x is a gradient;
    the exact projected L2 residual is sqrt(tr(S^2)).
    """
    s = np.asarray(target_covariance, dtype=float)
    if s.ndim != 2 or s.shape[0] != s.shape[1] or not np.allclose(s,s.T):
        raise ValueError("target covariance must be symmetric matrix")
    return float(np.sqrt(np.trace(s @ s)))
