"""Population integrals for the expanding nonlinear radial path."""

import numpy as np
from scipy.integrate import quad, solve_ivp
from scipy.linalg import eigh_tridiagonal
from scipy.optimize import brentq


def coefficients(radius, a=0.6, eta=0.3):
    radius = np.asarray(radius)
    s = radius**2
    alpha = a + eta * s / (1 + s)
    q = 2 * eta * s / (1 + s)**2
    dq = 4 * eta * s * (1 - s) / (1 + s)**3
    p = alpha**2 + alpha * q
    beta = np.sqrt(p)
    dp = 2 * alpha * q + q**2 + alpha * dq
    gamma = alpha * (2 * beta + dp / (2 * beta))
    k = alpha**3 + 3 * alpha**2 * q + alpha * q**2 + alpha**2 * dq
    return alpha, beta, p, gamma, k, alpha + q


def mixture_quadrature(order, weights, scales):
    """Use the symmetric Laguerre recurrence to avoid unstable root evaluation."""
    weights, scales = np.asarray(weights, dtype=float), np.asarray(scales, dtype=float)
    if order < 2 or weights.shape != scales.shape or weights.ndim != 1:
        raise ValueError("The quadrature dimensions are incorrect.")
    if np.any(weights <= 0) or np.any(scales <= 0) or not np.isclose(weights.sum(), 1):
        raise ValueError("The mixture must have positive weights and scales.")
    indices = np.arange(order, dtype=float)
    nodes, vectors = eigh_tridiagonal(2 * indices + 1, np.arange(1, order, dtype=float), lapack_driver="stev")
    laguerre_weights = vectors[0]**2
    radii = np.concatenate([scale * np.sqrt(2 * nodes) for scale in scales])
    probability_weights = np.concatenate([weight * laguerre_weights for weight in weights])
    return {"radii": radii, "weights": probability_weights, "laguerre_nodes": nodes,
            "laguerre_weights": laguerre_weights, "component": np.repeat(np.arange(len(scales)), order)}


def implicit_radius(radius, time, a=0.6, eta=0.3):
    """Solve the separated scalar equation in log radius with a fixed bracket."""
    radius, time = float(radius), float(time)
    if radius < 0 or time < 0 or a <= 0 or eta < 0:
        raise ValueError("The radius, time, and field parameters are invalid.")
    if radius == 0 or time == 0:
        return radius
    if eta == 0:
        return radius * np.exp(a * time)
    c, s = a + eta, radius**2

    def residual(log_growth):
        log_ratio = np.log1p(c * s * np.expm1(2 * log_growth) / (a + c * s))
        return log_growth / a - eta * log_ratio / (2 * a * c) - time

    growth = brentq(residual, a * time, c * time, xtol=5e-15, rtol=9e-16)
    return radius * np.exp(growth)


def radius_integrals(radius, horizon=1, a=0.6, eta=0.3):
    """Compute angle, action, and error coefficients without solving an ordinary differential equation in time."""
    end = implicit_radius(radius, horizon, a, eta)
    if radius == 0:
        return np.zeros(7)
    end_field = coefficients(end, a, eta)[0] * end

    def integrands(r):
        alpha, beta, p, gamma, k, _ = coefficients(r, a, eta)
        field = alpha * r
        d0 = p * r / 2
        d1 = r * (alpha * p / 2 + k / 6)
        return (beta / field, alpha * r, (alpha**2 + p) * r / alpha,
                end_field * d0 / field**2, end_field * d1 / field**2, gamma / (2 * field))

    values = [quad(lambda r: integrands(r)[j], radius, end, epsabs=2e-12, epsrel=2e-12)[0]
              for j in range(6)]
    return np.array([end, *values])


def reference(radii, horizon=1, a=0.6, eta=0.3, rtol=2e-13, atol=2e-14):
    """Integrate log radius, angle, action, and coefficient equations in time."""
    radii = np.asarray(radii, dtype=float)
    if np.any(radii <= 0):
        raise ValueError("The reference requires positive source radii.")
    count = len(radii)
    initial = np.zeros((7, count))
    initial[0] = np.log(radii)

    def derivative(time, state):
        log_radius, angle, action0, action1, c0, c1, q_angle = state.reshape(7, count)
        radius = np.exp(log_radius)
        alpha, beta, p, gamma, k, fprime = coefficients(radius, a, eta)
        return np.array([alpha, beta, alpha**2 * radius**2, (alpha**2 + p) * radius**2,
                         fprime * c0 + p * radius / 2,
                         fprime * c1 + radius * (alpha * p / 2 + k / 6), gamma / 2]).ravel()

    solution = solve_ivp(derivative, (0, horizon), initial.ravel(), method="DOP853",
                         rtol=rtol, atol=atol, dense_output=True)
    if not solution.success:
        raise ArithmeticError(solution.message)
    return solution


def reference_values(solution, times, initial_radii):
    times = np.asarray(times, dtype=float)
    values = solution.sol(times).reshape(7, len(initial_radii), -1)
    radius = np.exp(values[0].T)
    radius[times == 0] = initial_radii
    return {"radius": radius, "angle": values[1].T,
            "action_base": values[2, :, -1], "action_corrected": values[3, :, -1],
            "coefficient_base": values[4, :, -1], "coefficient_corrected": values[5, :, -1],
            "coefficient_angle": values[6, :, -1]}


def radial_errors(numerical_radius, exact_radius, weights, numerical_angle=None, exact_angle=None):
    """Evaluate radial quantile transport and the same-start particle coupling."""
    deficit = numerical_radius - exact_radius
    w2 = np.sqrt(np.sum(weights * deficit**2, axis=-1))
    if numerical_angle is None:
        return w2, w2.copy()
    angle_deficit = numerical_angle - exact_angle
    squared = deficit**2 + 4 * numerical_radius * exact_radius * np.sin(angle_deficit / 2)**2
    return w2, np.sqrt(np.sum(weights * squared, axis=-1))


def source_quantiles(probabilities, weights, scales):
    weights, scales = np.asarray(weights), np.asarray(scales)
    return np.array([brentq(lambda r: np.sum(weights * (-np.expm1(-r**2 / (2 * scales**2)))) - p,
                           0, 20 * max(scales), xtol=1e-14) for p in probabilities])
