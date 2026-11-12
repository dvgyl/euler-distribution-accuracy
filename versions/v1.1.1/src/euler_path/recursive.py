"""Advance Euler matrices without population product formulas."""

import numpy as np
from scipy.integrate import quad
from scipy.linalg import expm


def quarter_turn(dimension):
    if dimension < 2 or dimension % 2:
        raise ValueError("The dimension must be a positive even integer.")
    matrix = np.zeros((dimension, dimension))
    for j in range(0, dimension, 2):
        matrix[j + 1, j] = 1
        matrix[j, j + 1] = -1
    return matrix


def checked_nodes(nodes):
    times = np.asarray(nodes, dtype=float)
    if times.ndim != 1 or len(times) < 2 or not np.all(np.isfinite(times)):
        raise ValueError("The grid must contain finite times.")
    if times[0] != 0 or np.any(np.diff(times) <= 0):
        raise ValueError("The grid must start at zero and increase strictly.")
    return times


def covariance_root(covariance):
    symmetric = (covariance + covariance.T) / 2
    values, vectors = np.linalg.eigh(symmetric)
    allowance = 64 * np.finfo(float).eps * max(1.0, np.linalg.norm(symmetric, 2))
    if values.min() < -allowance:
        raise ValueError("The covariance has a negative eigenvalue beyond arithmetic tolerance.")
    return (vectors * np.sqrt(np.maximum(values, 0))) @ vectors.T


def gaussian_w2(mean, covariance, target_mean, target_covariance):
    """Compute the Gaussian distance through optimal square-root alignment."""
    mean, target_mean = np.asarray(mean, dtype=float), np.asarray(target_mean, dtype=float)
    left = covariance_root(covariance)
    right = covariance_root(target_covariance)
    u, _, vt = np.linalg.svd(right.T @ left)
    alignment = u @ vt
    squared = np.sum((left - right @ alignment)**2) + np.sum((mean - target_mean)**2)
    return float(np.sqrt(squared))


def radial(nodes, alpha, kappa, second_moment, dimension=2, restart=False):
    times = checked_nodes(nodes)
    skew = quarter_turn(dimension)
    identity = np.eye(dimension)
    current = identity.copy()
    maps, covariances, exact_maps, distances, particle_errors = [], [], [], [], []
    source_covariance = second_moment / dimension * identity
    for k, t in enumerate(times):
        if k:
            previous = times[k - 1]
            if restart:
                current = np.sqrt(1 + (kappa * previous)**2) * expm(alpha * np.arctan(kappa * previous) * skew)
            field = (kappa**2 * previous * identity + alpha * kappa * skew) / (1 + (kappa * previous)**2)
            current = current + (t - previous) * field @ current
        exact = np.sqrt(1 + (kappa * t)**2) * expm(alpha * np.arctan(kappa * t) * skew)
        covariance = current @ source_covariance @ current.T
        difference = current - exact
        scale = np.linalg.norm(current, "fro") / np.sqrt(dimension)
        distances.append(np.sqrt(second_moment) * abs(np.sqrt(1 + (kappa * t)**2) - scale))
        particle_errors.append(np.sqrt(max(0.0, np.trace(difference @ source_covariance @ difference.T))))
        maps.append(current.copy())
        covariances.append(covariance)
        exact_maps.append(exact)
    action = quad(lambda t: np.trace(
        ((kappa**2 * t * identity + alpha * kappa * skew) / (1 + (kappa * t)**2))
        @ ((1 + (kappa * t)**2) * source_covariance)
        @ ((kappa**2 * t * identity + alpha * kappa * skew) / (1 + (kappa * t)**2)).T
    ), 0, times[-1], epsabs=1e-11, epsrel=1e-11)[0]
    return {"times": times, "maps": np.asarray(maps), "covariances": np.asarray(covariances),
            "exact_maps": np.asarray(exact_maps), "w2": np.asarray(distances),
            "particle_rms": np.asarray(particle_errors), "action": action}


def gaussian(nodes, rate, rotation, covariance, initial_mean, drift, restart=False):
    times = checked_nodes(nodes)
    covariance = np.asarray(covariance, dtype=float)
    initial_mean, drift = np.asarray(initial_mean, dtype=float), np.asarray(drift, dtype=float)
    dimension = len(initial_mean)
    if covariance.shape != (dimension, dimension) or drift.shape != (dimension,):
        raise ValueError("The covariance and mean dimensions must agree.")
    factor = np.linalg.cholesky(covariance)
    skew = quarter_turn(dimension)
    rotation_matrix = factor @ skew @ np.linalg.inv(factor)
    field_matrix = rate * np.eye(dimension) + rotation * rotation_matrix
    current_map = np.eye(dimension)
    current_covariance = covariance.copy()
    current_mean = initial_mean.copy()
    maps, covariances, means, exact_maps, target_covariances, target_means, distances, particle_errors = [], [], [], [], [], [], [], []
    for k, t in enumerate(times):
        if k:
            previous = times[k - 1]
            if restart:
                current_map = expm(previous * field_matrix)
                current_covariance = np.exp(2 * rate * previous) * covariance
                current_mean = initial_mean + previous * drift
            step = t - previous
            update = np.eye(dimension) + step * field_matrix
            current_map = current_map + step * field_matrix @ current_map
            current_covariance = update @ current_covariance @ update.T
            current_mean = current_mean + step * (drift + field_matrix @ (current_mean - initial_mean - previous * drift))
        target_mean = initial_mean + t * drift
        target_covariance = np.exp(2 * rate * t) * covariance
        exact_map = expm(t * field_matrix)
        difference = current_map - exact_map
        distances.append(gaussian_w2(current_mean, current_covariance, target_mean, target_covariance))
        particle_errors.append(np.sqrt(max(0.0, np.trace(difference @ covariance @ difference.T)
                                          + np.sum((current_mean - target_mean)**2))))
        maps.append(current_map.copy())
        covariances.append(current_covariance.copy())
        means.append(current_mean.copy())
        exact_maps.append(exact_map)
        target_covariances.append(target_covariance)
        target_means.append(target_mean)
    action = quad(lambda t: np.dot(drift, drift) + np.trace(field_matrix @ (
        np.exp(2 * rate * t) * covariance) @ field_matrix.T), 0, times[-1], epsabs=1e-11, epsrel=1e-11)[0]
    return {"times": times, "maps": np.asarray(maps), "covariances": np.asarray(covariances),
            "means": np.asarray(means), "exact_maps": np.asarray(exact_maps),
            "target_covariances": np.asarray(target_covariances), "target_means": np.asarray(target_means), "w2": np.asarray(distances),
            "particle_rms": np.asarray(particle_errors), "action": action,
            "rotation_matrix": rotation_matrix}
