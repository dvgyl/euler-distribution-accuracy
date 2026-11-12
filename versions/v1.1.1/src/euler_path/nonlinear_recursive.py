"""Independent Cartesian and polar Euler updates for the nonlinear field."""

import numpy as np

from .recursive import checked_nodes


def polar(nodes, radii, corrected, a=0.6, eta=0.3):
    times = checked_nodes(nodes)
    current = np.asarray(radii, dtype=np.longdouble).copy()
    radius = np.empty((len(times), len(current)))
    angle = np.zeros_like(radius)
    accumulated_angle = np.zeros_like(current)
    radius[0] = current
    for index, step in enumerate(np.diff(times), 1):
        h = np.longdouble(step)
        s = current**2
        expansion = a + eta * s / (1 + s)
        tangent = np.sqrt(expansion**2 + expansion * 2 * eta * s / (1 + s)**2) if corrected else 0
        real, imaginary = 1 + h * expansion, h * tangent
        current *= np.sqrt(real**2 + imaginary**2)
        accumulated_angle += np.arctan2(imaginary, real)
        radius[index], angle[index] = current, accumulated_angle
    return {"times": times, "radius": radius, "angle": angle}


def cartesian(nodes, initial, corrected, a=0.6, eta=0.3):
    times = checked_nodes(nodes)
    current = np.asarray(initial, dtype=float).copy()
    if current.ndim != 2 or current.shape[1] != 2:
        raise ValueError("The particles must have two Cartesian coordinates.")
    paths = np.empty((len(times), *current.shape))
    paths[0] = current
    for index, step in enumerate(np.diff(times), 1):
        norm = np.linalg.norm(current, axis=1)
        alpha = a + eta * norm**2 / (1 + norm**2)
        derivative = 2 * eta * norm / (1 + norm**2)**2
        beta = np.sqrt(alpha**2 + alpha * norm * derivative) if corrected else np.zeros_like(norm)
        perpendicular = np.column_stack((-current[:, 1], current[:, 0]))
        velocity = alpha[:, None] * current + beta[:, None] * perpendicular
        current = current + step * velocity
        paths[index] = current
    return paths
