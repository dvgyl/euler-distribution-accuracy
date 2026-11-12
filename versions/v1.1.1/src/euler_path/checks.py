"""Compare independently computed node quantities."""

import numpy as np


def compare(population, recursion, covariance_source, multiplier=200, action_absolute=1e-10):
    """Reject a mismatch beyond a node-dependent binary64 tolerance."""
    node_count = len(population["nodes"])
    if not node_count or {row["node"] for row in population["nodes"]} != set(range(node_count)):
        raise AssertionError("The population nodes must cover every recursion node exactly once.")
    node_arrays = ["times", "covariances", "w2", "particle_rms"]
    if "means" in recursion:
        node_arrays.extend(["means", "target_means"])
    if any(len(recursion[key]) != node_count for key in node_arrays):
        raise AssertionError("The population and recursion nodes must have equal coverage.")
    if not np.isfinite(multiplier) or not np.isfinite(action_absolute):
        raise AssertionError("The comparison tolerance parameters are nonfinite.")
    eps = np.finfo(float).eps
    maxima = {"covariance_absolute": 0.0, "w2_absolute": 0.0, "particle_rms_absolute": 0.0,
              "mean_absolute": 0.0, "residual_over_allowance": 0.0}
    source = np.asarray(covariance_source, dtype=float)
    if not np.all(np.isfinite(source)):
        raise AssertionError("The source covariance is nonfinite.")
    for row in population["nodes"]:
        k = row["node"]
        covariance_factor = float(row["scale"])**2
        expected_covariance = covariance_factor * source
        covariance_scale = max(1.0, np.linalg.norm(expected_covariance, "fro"))
        allowance = multiplier * eps * (k + 1) * covariance_scale
        residual = np.linalg.norm(recursion["covariances"][k] - expected_covariance, "fro")
        if not np.isfinite(residual) or not np.isfinite(allowance):
            raise AssertionError(f"Node {k}: the covariance comparison is nonfinite.")
        maxima["covariance_absolute"] = max(maxima["covariance_absolute"], residual)
        maxima["residual_over_allowance"] = max(maxima["residual_over_allowance"], residual / allowance)
        if residual > allowance:
            raise AssertionError(f"Node {k}: covariance mismatch {residual:.6g} exceeds tolerance {allowance:.6g}.")
        distance_scale = np.sqrt(np.trace(source)) * float(row["target_scale"])
        if not np.isfinite(distance_scale):
            raise AssertionError(f"Node {k}: the distance tolerance scale is nonfinite.")
        distance_allowance = multiplier * eps * (k + 1) * max(1.0, distance_scale)
        if not np.isfinite(distance_allowance):
            raise AssertionError(f"Node {k}: the distance tolerance is nonfinite.")
        for key in ("w2", "particle_rms"):
            residual = abs(recursion[key][k] - float(row[key]))
            if not np.isfinite(residual):
                raise AssertionError(f"Node {k}: the {key} comparison is nonfinite.")
            maxima[key + "_absolute"] = max(maxima[key + "_absolute"], residual)
            maxima["residual_over_allowance"] = max(maxima["residual_over_allowance"], residual / distance_allowance)
            if residual > distance_allowance:
                raise AssertionError(f"Node {k}: {key} mismatch {residual:.6g} exceeds tolerance {distance_allowance:.6g}.")
        if "means" in recursion:
            expected_mean = recursion["target_means"][k]
            residual = np.linalg.norm(recursion["means"][k] - expected_mean)
            mean_allowance = multiplier * eps * (k + 1) * max(1.0, np.linalg.norm(expected_mean))
            if not np.isfinite(residual) or not np.isfinite(mean_allowance):
                raise AssertionError(f"Node {k}: the mean comparison is nonfinite.")
            maxima["mean_absolute"] = max(maxima["mean_absolute"], residual)
            if residual > mean_allowance:
                raise AssertionError(f"Node {k}: mean mismatch {residual:.6g} exceeds tolerance {mean_allowance:.6g}.")
    action_residual = abs(recursion["action"] - float(population["action"]))
    action_allowance = action_absolute * max(1.0, abs(recursion["action"]))
    if not np.isfinite(action_residual) or not np.isfinite(action_allowance):
        raise AssertionError("The action comparison is nonfinite.")
    if action_residual > action_allowance:
        raise AssertionError(f"The action mismatch is {action_residual:.6g}.")
    maxima["action_absolute"] = action_residual
    maxima["checked_nodes"] = len(population["nodes"])
    return maxima


def reject_expected(population, recursion, covariance_source):
    try:
        compare(population, recursion, covariance_source)
    except AssertionError as error:
        return str(error)
    raise AssertionError("The checker accepted a deliberately incorrect recursion.")


def compare_cartesian_polar(cartesian, radii, angles, initial_angles, multiplier=200):
    """Check independent numerical updates at every recorded node."""
    cartesian = np.asarray(cartesian)
    radii, angles = np.asarray(radii), np.asarray(angles)
    if radii.shape != angles.shape or cartesian.shape != (*radii.shape, 2):
        raise AssertionError("The Cartesian and polar node coverage differs.")
    if not all(np.all(np.isfinite(array)) for array in (cartesian, radii, angles, initial_angles)):
        raise AssertionError("The trajectory comparison is nonfinite.")
    theta = angles + np.asarray(initial_angles)[None, :]
    expected = radii[..., None] * np.stack((np.cos(theta), np.sin(theta)), axis=-1)
    residual = np.linalg.norm(cartesian - expected, axis=-1)
    allowance = multiplier * np.finfo(float).eps * np.arange(1, len(radii) + 1)[:, None] * np.maximum(1, radii)
    if np.any(residual > allowance):
        node, particle = np.argwhere(residual > allowance)[0]
        raise AssertionError(f"Node {node}: Cartesian update mismatch for particle {particle}.")
    return {"maximum_absolute": float(residual.max()), "maximum_allowance_fraction": float((residual / allowance).max()),
            "checked_nodes": len(radii), "checked_particles": radii.shape[1]}
