"""Population Gaussian checks and exact rational enclosures for the new barriers.

The enclosures use integer/rational arithmetic; other functions are floating-point
population calculations, not finite-sample estimates or formal proof certificates.
"""
from __future__ import annotations

from fractions import Fraction
from math import isqrt
from typing import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.integrate import quad
from scipy.special import roots_hermitenorm


def checked_grid(grid: ArrayLike) -> NDArray[np.float64]:
    t = np.asarray(grid, dtype=float)
    if t.ndim != 1 or len(t) < 2 or not np.all(np.isfinite(t)):
        raise ValueError("grid must be a finite one-dimensional array")
    if abs(t[0]) > 1e-14 or np.any(np.diff(t) <= 0):
        raise ValueError("grid must start at zero and strictly increase")
    return t


def gaussian_scale_squared(t: ArrayLike, sigma: float = 1.0):
    if sigma <= 0 or not np.isfinite(sigma):
        raise ValueError("sigma must be positive and finite")
    t = np.asarray(t)
    return (1 - t) ** 2 + sigma**2 * t**2


def gaussian_coefficients(t: ArrayLike, sigma: float = 1.0):
    q = gaussian_scale_squared(t, sigma)
    return ((1 + sigma**2) * np.asarray(t) - 1) / q, sigma / q


def complex_structure(d: int) -> NDArray[np.float64]:
    if d <= 0 or d % 2:
        raise ValueError("an orthogonal complex structure requires positive even d")
    out = np.zeros((d, d))
    for k in range(0, d, 2):
        out[k, k + 1] = -1.0
        out[k + 1, k] = 1.0
    return out


def gaussian_euler_covariance(
    field_matrix: Callable[[float], NDArray[np.float64]],
    grid: ArrayLike,
    d: int,
    sigma: float = 1.0,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Independent matrix implementation of recursive Euler's Gaussian law."""
    t = checked_grid(grid)
    if d < 1:
        raise ValueError("d must be positive")
    covariance = np.eye(d)
    errors = [0.0]
    covariances = [covariance.copy()]
    for start, end in zip(t[:-1], t[1:]):
        matrix = np.asarray(field_matrix(float(start)), dtype=float)
        if matrix.shape != (d, d) or not np.all(np.isfinite(matrix)):
            raise ValueError("field matrix has incorrect shape or nonfinite entries")
        step = np.eye(d) + (end - start) * matrix
        covariance = step @ covariance @ step.T
        covariance = (covariance + covariance.T) / 2
        eigenvalues = np.linalg.eigvalsh(covariance)
        if eigenvalues.min() < -1e-11:
            raise ArithmeticError("covariance lost positive semidefiniteness")
        g = np.sqrt(gaussian_scale_squared(end, sigma))
        errors.append(float(np.linalg.norm(np.sqrt(np.maximum(eigenvalues, 0)) - g)))
        covariances.append(covariance.copy())
    return np.asarray(errors), np.asarray(covariances)


def even_product_errors(grid: ArrayLike, alpha: float, d: int = 2,
                        sigma: float = 1.0) -> NDArray[np.float64]:
    """Exact population formula evaluated in floating point; stable near alpha=1."""
    if d < 1 or d % 2 or not 0 <= alpha <= 1:
        raise ValueError("require positive even d and alpha in [0,1]")
    t = checked_grid(grid)
    q = gaussian_scale_squared(t, sigma)
    defects = (1 - alpha) * np.diff(t)**2 * sigma**2 / (q[:-1] * q[1:])
    if np.any(defects >= 1):
        # Zero or singular coarse-grid steps are not needed in the refinement tests.
        raise ValueError("grid too coarse for positive product factors")
    log_ratio = np.r_[0.0, np.cumsum(np.log1p(-defects))]
    return np.sqrt(d * q) * (-np.expm1(log_ratio / 2))


def gaussian_risk_frontier(t: ArrayLike, alpha: float, d: int = 2,
                           sigma: float = 1.0) -> dict[str, NDArray]:
    if d <= 0 or d % 2 or alpha < 0:
        raise ValueError("require positive even d and nonnegative alpha")
    q = gaussian_scale_squared(t, sigma)
    minimum = d * sigma**2 / q
    return {"bayes_loss": minimum, "loss": (1 + alpha) * minimum,
            "gauge_energy": alpha * minimum,
            "residual": abs(1 - alpha) * np.sqrt(d) * sigma**2 / q**1.5,
            "moment_bound": abs(1 - alpha) * minimum / np.sqrt(d * q)}


def skew_cross(axis: ArrayLike) -> NDArray[np.float64]:
    x, y, z = np.asarray(axis, dtype=float)
    return np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])


def odd_budget_field(n: int) -> Callable[[float], NDArray[np.float64]]:
    """A 3D smooth, exactly path-preserving, N-dependent bounded family.

    The N dependence changes parameters, not temporal frequencies.  Consequently
    the field and its first two time derivatives have a common finite bound.
    """
    if n < 1:
        raise ValueError("n must be positive")
    amplitude = 1 + 0.15 * np.cos(np.log(n))
    angle_amplitude = 0.4 + 0.1 * np.sin(np.log(n))
    def matrix(t: float) -> NDArray[np.float64]:
        angle = angle_amplitude * np.sin(2 * np.pi * t)
        axis = np.array([np.sin(angle), 0., np.cos(angle)])
        a, b = gaussian_coefficients(t)
        return a * np.eye(3) + amplitude * b * skew_cross(axis)
    return matrix


def distorted_grid(n: int, distortion: float) -> NDArray[np.float64]:
    if n < 1 or abs(distortion) >= 1:
        raise ValueError("n must be positive and abs(distortion)<1")
    r = np.linspace(0, 1, n + 1)
    return r + distortion * np.sin(2 * np.pi * r) / (2 * np.pi)


def secant_coefficient(t: float, h: float, sigma: float = 1.0) -> float:
    """Stable algebraic form of (g(t+h)/g(t)-1)/h."""
    if h <= 0:
        raise ValueError("h must be positive")
    g = np.sqrt(gaussian_scale_squared(t, sigma))
    gh = np.sqrt(gaussian_scale_squared(t + h, sigma))
    return float((2 * ((1 + sigma**2) * t - 1) + (1 + sigma**2) * h)
                 / (g * (gh + g)))


def secant_terminal_bias(n: int, d: int = 3, sigma: float = 1.0):
    """Continuous-model terminal bias; quadrature estimate is not an interval bound."""
    if n < 1 or d < 1:
        raise ValueError("n and d must be positive")
    h = 1 / n
    def integrand(t):
        a, _ = gaussian_coefficients(t, sigma)
        return secant_coefficient(t, h, sigma) - a
    log_scale_ratio, quadrature_estimate = quad(integrand, 0., 1., epsabs=1e-13,
                                               epsrel=1e-12, limit=100)
    bias = np.sqrt(d) * sigma * np.expm1(log_scale_ratio)
    coefficient = np.sqrt(d) * sigma / 2 * quad(
        lambda t: sigma**2 / gaussian_scale_squared(t, sigma)**2, 0., 1.,
        epsabs=1e-13, epsrel=1e-13)[0]
    return float(bias), float(coefficient), float(quadrature_estimate)


def rational_sqrt_bounds(value: Fraction, digits: int = 35) -> tuple[Fraction, Fraction]:
    """Certified sqrt enclosure, using only integer arithmetic and exact comparisons."""
    value = Fraction(value)
    if value < 0 or digits < 1:
        raise ValueError("value must be nonnegative and digits positive")
    scale = 10**digits
    k = isqrt(value.numerator * scale**2 // value.denominator)
    lower = Fraction(k, scale)
    upper = lower if lower * lower == value else Fraction(k + 1, scale)
    assert lower * lower <= value <= upper * upper
    return lower, upper


def rational_gaussian_certificate(n: int, alpha: Fraction, d: int = 2,
                                  digits: int = 35) -> dict:
    """Exact-arithmetic terminal and maximum-node W2 enclosures, sigma=1.

    All numerical integration is eliminated: the scalar covariance recursion is
    rational.  Irrational square roots are enclosed by integer-square tests.
    The returned Fraction endpoints are the authoritative certificate.
    """
    alpha = Fraction(alpha)
    if n < 2 or d < 1 or d % 2 or not 0 <= alpha <= 1:
        raise ValueError("require n>=2, positive even d, alpha in [0,1]")
    h = Fraction(1, n)
    variance = Fraction(1)
    rootd = rational_sqrt_bounds(Fraction(d), digits)
    max_lower, max_upper = Fraction(0), Fraction(0)
    terminal = (Fraction(0), Fraction(0))
    for k in range(n):
        t = k * h
        q = (1 - t)**2 + t**2
        a = (2 * t - 1) / q
        variance *= (1 + h * a)**2 + alpha * h**2 / q**2
        nxt = (k + 1) * h
        target_q = (1 - nxt)**2 + nxt**2
        assert 0 <= variance <= target_q
        g_lower, g_upper = rational_sqrt_bounds(target_q, digits)
        v_lower, v_upper = rational_sqrt_bounds(variance, digits)
        gap_lower = max(Fraction(0), g_lower - v_upper)
        gap_upper = max(Fraction(0), g_upper - v_lower)
        terminal = (rootd[0] * gap_lower, rootd[1] * gap_upper)
        max_lower = max(max_lower, terminal[0])
        max_upper = max(max_upper, terminal[1])
    return {"terminal": terminal, "maximum_node": (max_lower, max_upper),
            "terminal_variance": variance, "digits": digits}


def covariance_relaxation_bound(eigenvalues: ArrayLike, budget: float) -> float:
    """PSD trace-budget relaxation; no general gauge attainability is asserted."""
    eig = np.asarray(eigenvalues, dtype=float)
    if eig.ndim != 1 or np.any(eig <= 0) or budget < 0:
        raise ValueError("positive eigenvalues and nonnegative budget required")
    if budget >= eig.sum():
        return 0.0
    if budget == 0:
        return float(np.linalg.norm(eig))
    lo, hi = 0., float(eig.max())
    for _ in range(100):
        mid = (lo + hi) / 2
        if np.maximum(eig - mid, 0).sum() > budget:
            lo = mid
        else:
            hi = mid
    return float(np.linalg.norm(np.minimum(eig, (lo + hi) / 2)))


def mixture_loss_check(t: float, mu: float, within_variance: float,
                       nodes: int = 128) -> tuple[float, float]:
    """Non-Gaussian population check of E[X*a]=M for the canonical velocity.

    Y is a balanced N(+/-mu, within_variance) mixture.  Two distinct formulas
    are integrated: posterior Bayes variance and the differentiated velocity.
    Gauss-Hermite quadrature is not certified; convergence must be reported.
    """
    if not 0 <= t <= 1 or mu < 0 or within_variance <= 0 or nodes < 4:
        raise ValueError("invalid mixture parameters")
    roots, weights = roots_hermitenorm(nodes)
    weights = weights / np.sqrt(2 * np.pi)
    v = (1 - t)**2 + t**2 * within_variance
    x = np.concatenate([t * mu + np.sqrt(v) * roots, -t * mu + np.sqrt(v) * roots])
    w = np.r_[weights, weights] / 2
    def velocity(tt):
        vv = (1 - tt)**2 + tt**2 * within_variance
        c = tt * within_variance - (1 - tt)
        return c / vv * x + (1 - tt) * mu / vv * np.tanh(tt * mu * x / vv)
    tanh = np.tanh(t * mu * x / v)
    sech2 = 1 - tanh**2
    minimum = within_variance / v + mu**2 * (1 - t)**2 / v**2 * np.dot(w, sech2)
    dx_velocity = ((t * within_variance - (1 - t)) / v
                   + (1 - t) * t * mu**2 / v**2 * sech2)
    dt_velocity = np.imag(velocity(t + 1e-25j)) / 1e-25
    acceleration = dt_velocity + dx_velocity * velocity(t)
    return float(minimum), float(np.dot(w, x * acceleration))


def symmetric_mixture_quantile(z: ArrayLike, mean: float, sd: float,
                               iterations: int = 60) -> NDArray[np.float64]:
    """Quantile transport of N(0,1) to a symmetric two-normal mixture.

    Survival probabilities avoid loss of Gaussian tail accuracy when Phi(z)
    rounds to one. Bisection is a floating-point calculation, not an interval
    certificate for the Gaussian CDF.
    """
    from scipy.special import ndtr
    z = np.asarray(z, dtype=float)
    if mean < 0 or sd <= 0:
        raise ValueError("mean must be nonnegative and sd positive")
    absolute = np.abs(z)
    target = ndtr(-absolute)
    lower, upper = sd * absolute - mean, sd * absolute + mean
    for _ in range(iterations):
        middle = (lower + upper) / 2
        survival = (ndtr(-(middle - mean) / sd) + ndtr(-(middle + mean) / sd)) / 2
        lower = np.where(survival > target, middle, lower)
        upper = np.where(survival > target, upper, middle)
    return np.sign(z) * ((lower + upper) / 2)


def mixture_euler_errors(n: int, mu: float = 1.2, sigma: float = .7,
                         d: int = 3, quadrature_nodes: int = 32) -> NDArray[np.float64]:
    """Full W2 product-quantile formula evaluated by composite quadrature.

    Target Y = mu*sign*e1 + sigma*Z. The canonical field separates into one
    mixture coordinate and d-1 Gaussian coordinates. The scalar Euler maps
    are monotone on the checked grids, so shared quantiles give optimal OT.
    Composite Gauss-Legendre integrates the Gaussian-weighted formula on
    [-10,10]; this quadrature/truncation is not an interval certificate.
    """
    if n < 2 or d < 1 or mu < 0 or sigma <= 0:
        raise ValueError("invalid parameters")
    from numpy.polynomial.legendre import leggauss
    roots, raw_weights = leggauss(quadrature_nodes)
    breaks = [-10., -4., -2., -1., -.5, 0., .5, 1., 2., 4., 10.]
    z_chunks, w_chunks = [], []
    for left, right in zip(breaks[:-1], breaks[1:]):
        zz = (left+right)/2 + (right-left)/2 * roots
        ww = (right-left)/2 * raw_weights * np.exp(-zz**2/2) / np.sqrt(2*np.pi)
        z_chunks.append(zz); w_chunks.append(ww)
    z, weights = np.concatenate(z_chunks), np.concatenate(w_chunks)
    x = z.copy(); scalar = 1.; h = 1/n
    errors = [0.]
    for k in range(n):
        t = k/n
        q = gaussian_scale_squared(t, sigma)
        a = ((1 + sigma**2)*t - 1)/q
        c = (1-t)*mu/q; slope = t*mu/q
        if 1+h*a <= 0:
            raise ValueError("grid too coarse to guarantee monotone Euler maps")
        x = x + h*(a*x + c*np.tanh(slope*x))
        scalar *= 1+h*a
        end = (k+1)/n; g = np.sqrt(gaussian_scale_squared(end, sigma))
        reference = symmetric_mixture_quantile(z, end*mu, g)
        square = np.dot(weights, (x-reference)**2)+(d-1)*(scalar-g)**2
        errors.append(float(np.sqrt(max(0., square))))
    return np.asarray(errors)
