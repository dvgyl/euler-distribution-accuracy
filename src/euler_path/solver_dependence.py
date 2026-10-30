"""Constant Gaussian fields and the known rotated Gaussian interpolation."""

import mpmath as mp
import numpy as np

from .recursive import checked_nodes, gaussian_w2, quarter_turn


def gaussian_formula(nodes, a, b, covariance_trace, solver):
    times = [mp.mpf(str(value)) for value in nodes]
    z = mp.mpc(str(a), str(b))
    factor = mp.mpc(1)
    distances = []
    for index, time in enumerate(times):
        if index:
            step_z = (time - times[index - 1]) * z
            if solver == "euler":
                factor *= 1 + step_z
            elif solver == "heun":
                factor *= 1 + step_z + step_z**2 / 2
            else:
                raise ValueError("The solver must be euler or heun.")
        distances.append(mp.sqrt(str(covariance_trace)) * abs(mp.exp(mp.mpf(str(a)) * time) - abs(factor)))
    return distances


def gaussian_matrix(nodes, a, b, covariance, solver):
    times = checked_nodes(nodes)
    covariance = np.asarray(covariance, dtype=float)
    factor = np.linalg.cholesky(covariance)
    weighted_rotation = factor @ quarter_turn(len(covariance)) @ np.linalg.inv(factor)
    field = a * np.eye(len(covariance)) + b * weighted_rotation
    current = np.eye(len(covariance))
    maps, covariances, distances = [], [], []
    zero = np.zeros(len(covariance))
    for index, time in enumerate(times):
        if index:
            step = time - times[index - 1]
            first = field @ current
            if solver == "euler":
                current = current + step * first
            elif solver == "heun":
                predictor = current + step * first
                second = field @ predictor
                current = current + step * (first + second) / 2
            else:
                raise ValueError("The solver must be euler or heun.")
        numerical_covariance = current @ covariance @ current.T
        target_covariance = np.exp(2 * a * time) * covariance
        distances.append(gaussian_w2(zero, numerical_covariance, zero, target_covariance))
        maps.append(current.copy())
        covariances.append(numerical_covariance)
    return {"times": times, "maps": np.array(maps), "covariances": np.array(covariances), "w2": np.array(distances)}


def gaussian_matrix_high_precision(count, a, b, horizon, covariance, solver, digits=50):
    """Use matrix field evaluations to resolve errors below binary64 accuracy."""
    with mp.workdps(digits):
        covariance = mp.matrix([[mp.mpf(str(value)) for value in row] for row in covariance])
        factor = mp.cholesky(covariance)
        skew = mp.matrix([[0, -1], [1, 0]])
        field = mp.mpf(str(a)) * mp.eye(2) + mp.mpf(str(b)) * factor * skew * factor**-1
        current = mp.eye(2)
        step = mp.mpf(str(horizon)) / count
        for index in range(count):
            first = field * current
            if solver == "euler":
                current += step * first
            elif solver == "heun":
                second = field * (current + step * first)
                current += step * (first + second) / 2
            else:
                raise ValueError("The solver must be euler or heun.")
        numerical_covariance = current * covariance * current.T
        trace = covariance[0, 0] + covariance[1, 1]
        scale = mp.sqrt((numerical_covariance[0, 0] + numerical_covariance[1, 1]) / trace)
        covariance_residual = mp.norm(numerical_covariance - scale**2 * covariance)
        if covariance_residual > mp.mpf(10)**(-digits + 5):
            raise ArithmeticError("The matrix covariance is not proportional to the source covariance.")
        distance = mp.sqrt(trace) * abs(scale - mp.exp(mp.mpf(str(a)) * mp.mpf(str(horizon))))
        return {"w2": mp.nstr(distance, digits - 5), "covariance_residual": mp.nstr(covariance_residual, digits - 5)}


def gaussian_leading_coefficient(a, b, horizon, covariance_trace, solver):
    scale = np.sqrt(covariance_trace) * np.exp(a * horizon) * horizon
    if solver == "euler":
        if b**2 == a**2:
            return 2, scale * 2 * a**3 / 3
        return 1, scale * abs(a**2 - b**2) / 2
    if solver == "heun":
        return 2, scale * abs(a**3 - 3 * a * b**2) / 6
    raise ValueError("The solver must be euler or heun.")


def flow_matching(nodes, sigma=1.5, corrected=True):
    """Apply Euler to the marginal field of a deterministic rotated coupling."""
    times = checked_nodes(nodes)
    identity, skew = np.eye(2), quarter_turn(2)
    current = identity.copy()
    maps, targets, exact_maps, distances, particles = [], [], [], [], []
    for index, time in enumerate(times):
        if index:
            previous = times[index - 1]
            q = (1 - previous)**2 + sigma**2 * previous**2
            alpha = (-(1 - previous) + sigma**2 * previous) / q
            beta = sigma / q if corrected else 0
            velocity = (alpha * identity + beta * skew) @ current
            current = current + (time - previous) * velocity
        q = (1 - time)**2 + sigma**2 * time**2
        exact = (1 - time) * identity + sigma * time * skew if corrected else np.sqrt(q) * identity
        targets.append(q * identity)
        exact_maps.append(exact)
        distances.append(gaussian_w2(np.zeros(2), current @ current.T, np.zeros(2), q * identity))
        particles.append(np.linalg.norm(current - exact, "fro"))
        maps.append(current.copy())
    return {"times": times, "maps": np.array(maps), "target_covariances": np.array(targets),
            "exact_maps": np.array(exact_maps), "w2": np.array(distances), "particle_rms": np.array(particles)}


def paired_skew(eigenvalues, tolerance=1e-12):
    """Construct B with B+B.T=0 and B B.T=diag(eigenvalues), if pairs agree."""
    values = np.asarray(eigenvalues, dtype=float)
    if values.ndim != 1 or len(values) % 2 or np.any(values <= 0):
        raise ValueError("Positive eigenvalues in an even dimension are required.")
    order = np.argsort(values)
    matrix = np.zeros((len(values), len(values)))
    for left, right in order.reshape(-1, 2):
        if abs(values[left] - values[right]) > tolerance * max(values[left], values[right]):
            raise ValueError("This deterministic skew coupling requires paired eigenvalues.")
        matrix[right, left] = np.sqrt(values[left])
        matrix[left, right] = -np.sqrt(values[left])
    return matrix
