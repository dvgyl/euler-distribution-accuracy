"""Generate the deterministic data and check every recursive node."""

import csv
import json
from pathlib import Path

import mpmath as mp
import numpy as np

from euler_path import checks, population, recursive


def uniform_nodes(count, horizon="1"):
    return [population.text(mp.mpf(horizon) * k / count) for k in range(count + 1)]


def covariance_for(dimension, config):
    if dimension == 2:
        return np.asarray(config["gaussian"]["covariance_2d"], dtype=float)
    factor = np.diag(np.linspace(0.7, 1.6, dimension))
    for j in range(1, dimension):
        factor[j, j - 1] = 0.12
    return factor @ factor.T


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def generate(config_path, output):
    config = json.loads(config_path.read_text(encoding="utf-8"))
    mp.mp.dps = config["precision_digits"]
    output.mkdir(parents=True, exist_ok=True)
    raw = output / "raw"
    raw.mkdir(exist_ok=True)
    node_rows, run_rows, validation_rows = [], [], []
    parameters = []

    def record(run_id, family, method, grid_name, inputs, formula, direct, covariance):
        comparison = checks.compare(formula, direct, covariance,
                                    config["tolerance"]["roundoff_multiplier"],
                                    config["tolerance"]["action_absolute"])
        validation_rows.append({"run_id": run_id, **comparison})
        arrays = {key: value for key, value in direct.items() if isinstance(value, np.ndarray)}
        np.savez_compressed(raw / (run_id + ".npz"), **arrays)
        parameters.append({"run_id": run_id, "family": family, "method": method, "grid": grid_name, **inputs})
        for row in formula["nodes"]:
            k = row["node"]
            node_rows.append({"run_id": run_id, "family": family, "method": method, "grid": grid_name,
                              "dimension": len(covariance), "node": k, "time": row["time"],
                              "action": formula["action"], "w2": row["w2"], "w2_squared": row["w2_squared"],
                              "particle_rms": row["particle_rms"], "algebraic_zero": row["algebraic_zero"],
                              "recursive_w2": direct["w2"][k], "recursive_particle_rms": direct["particle_rms"][k]})
        w2_values = [mp.mpf(row["w2"]) for row in formula["nodes"]]
        terminal = formula["nodes"][-1]
        run_rows.append({"run_id": run_id, "family": family, "method": method, "grid": grid_name,
                         "dimension": len(covariance), "steps": len(formula["nodes"]) - 1,
                         "max_step": population.text(max(mp.mpf(b) - mp.mpf(a) for a, b in zip(inputs["nodes"][:-1], inputs["nodes"][1:]))),
                         "action": formula["action"], "terminal_w2": terminal["w2"],
                         "terminal_w2_squared": terminal["w2_squared"], "maximum_node_w2": population.text(max(w2_values)),
                         "maximum_node_w2_squared": population.text(max(w2_values)**2),
                         "sum_node_w2_squared": population.text(sum(value**2 for value in w2_values)),
                         "terminal_particle_rms": terminal["particle_rms"]})

    radial_config = config["radial"]
    radial_grids = [(f"uniform_{n}", uniform_nodes(n, config["horizon"])) for n in radial_config["step_counts"]]
    radial_grids.append(("nonuniform", config["nonuniform_nodes"]))
    for grid_name, nodes in radial_grids:
        errors, actions = [], []
        for alpha in radial_config["amplitudes"]:
            inputs = {"nodes": nodes, "alpha": alpha, "kappa": radial_config["kappa"],
                      "second_moment": radial_config["second_moment"], "dimension": 2}
            formula = population.radial(nodes, alpha, inputs["kappa"], inputs["second_moment"])
            direct = recursive.radial(nodes, float(alpha), float(inputs["kappa"]), float(inputs["second_moment"]))
            record(f"radial_{grid_name}_alpha_{alpha.replace('.', 'p')}", "radial", alpha, grid_name,
                   inputs, formula, direct, np.eye(2))
            actions.append(mp.mpf(formula["action"]))
            errors.append([mp.mpf(row["w2"]) for row in formula["nodes"][1:]])
        if not all(a < b for a, b in zip(actions[:-1], actions[1:])):
            raise AssertionError("The radial action ordering failed.")
        if not all(all(a > b for a, b in zip(left, right)) for left, right in zip(errors[:-1], errors[1:])):
            raise AssertionError("The radial node-error ordering failed.")

    gaussian_config = config["gaussian"]
    a = gaussian_config["rate"]
    for dimension in gaussian_config["dimensions"]:
        covariance = covariance_for(dimension, config)
        initial_mean = np.linspace(-0.4, 0.6, dimension)
        drift = np.linspace(0.2, -0.3, dimension)
        counts = gaussian_config["step_counts"] if dimension == 2 else gaussian_config["additional_dimension_counts"]
        grids = [(f"uniform_{n}", uniform_nodes(n, config["horizon"])) for n in counts]
        grids.append(("nonuniform", config["nonuniform_nodes"]))
        for grid_name, nodes in grids:
            pair = []
            for method, b in (("base", "0"), ("corrected", gaussian_config["corrected_rotation"])):
                inputs = {"nodes": nodes, "rate": a, "rotation": b, "covariance": covariance.tolist(),
                          "initial_mean": initial_mean.tolist(), "drift": drift.tolist()}
                formula = population.gaussian(nodes, a, b, np.trace(covariance), np.dot(drift, drift))
                direct = recursive.gaussian(nodes, float(a), float(b), covariance, initial_mean, drift)
                record(f"gaussian_d{dimension}_{grid_name}_{method}", "gaussian", method, grid_name,
                       inputs, formula, direct, covariance)
                pair.append(formula)
                expected_means = initial_mean + np.asarray(nodes, dtype=float)[:, None] * drift
                residual = np.max(np.abs(direct["means"] - expected_means))
                allowance = 200 * np.finfo(float).eps * len(nodes) * max(1.0, np.linalg.norm(expected_means[-1]))
                if residual > allowance:
                    raise AssertionError("The recursive affine mean is incorrect.")
                validation_rows[-1]["mean_absolute"] = float(residual)
                rotation_matrix = direct["rotation_matrix"]
                if not np.allclose(rotation_matrix @ covariance + covariance @ rotation_matrix.T, 0, atol=1e-13):
                    raise AssertionError("The weighted Gaussian flux identity failed.")
            if not all(0 < mp.mpf(corrected["w2"]) < mp.mpf(base["w2"]) for base, corrected in zip(pair[0]["nodes"][1:], pair[1]["nodes"][1:])):
                raise AssertionError("The corrected Gaussian node-error ordering failed.")

    control = config["contraction"]
    covariance = covariance_for(2, config)
    for method, b in (("base", "0"), ("corrected", control["rotation"])):
        nodes = uniform_nodes(control["step_count"], config["horizon"])
        inputs = {"nodes": nodes, "rate": control["rate"], "rotation": b, "covariance": covariance.tolist(),
                  "initial_mean": [0, 0], "drift": [0, 0]}
        formula = population.gaussian(nodes, control["rate"], b, np.trace(covariance), 0)
        direct = recursive.gaussian(nodes, float(control["rate"]), float(b), covariance, [0, 0], [0, 0])
        record(f"contraction_{method}", "contraction", method, "uniform_1", inputs, formula, direct, covariance)
    contraction_rows = [row for row in run_rows if row["family"] == "contraction"]
    if not mp.mpf(contraction_rows[0]["terminal_w2"]) < mp.mpf(contraction_rows[1]["terminal_w2"]):
        raise AssertionError("The contraction negative control failed.")

    calibration_rows = []
    root = population.negative_root()
    with mp.workdps(160):
        refined_root = population.negative_root()
        root_agreement = abs(refined_root - root)
        refined_residual = abs(mp.exp(refined_root) + 1 + refined_root)
    offset = mp.mpf(config["calibration"]["root_offset"])
    points = [("specified", mp.mpf(z)) for z in config["calibration"]["rate_steps"]]
    points += [("root_below", root - offset), ("root_above", root + offset), ("symbolic_root", root)]
    for label, z in points:
        residual_delta = population.calibration(population.text(z))
        symbolic_boundary = label == "symbolic_root" or z == 0
        delta = mp.mpf(0) if symbolic_boundary else residual_delta
        feasible = delta >= 0
        row = {"label": label, "rate_step": population.text(z), "delta": population.text(delta),
               "feasible": feasible, "symbolic_boundary": symbolic_boundary,
               "high_precision_delta_residual": population.text(residual_delta),
               "binary64_delta_residual": float(np.exp(2 * float(z)) - (1 + float(z))**2),
               "rotation": "", "maximum_node_covariance_residual": "", "maximum_node_recursive_w2": "",
               "terminal_particle_rms": "", "algebraic_w2": ""}
        if feasible:
            step = mp.mpf(config["calibration"]["step"])
            rotation = mp.sqrt(delta) / step
            nodes = [population.text(k * step) for k in range(5)]
            direct = recursive.gaussian(nodes, float(z / step), float(rotation), covariance, [0, 0], [0, 0])
            residual = float(np.max(np.linalg.norm(direct["covariances"] - direct["target_covariances"], axis=(1, 2))))
            allowance = 200 * np.finfo(float).eps * 5 * max(1.0, np.linalg.norm(direct["target_covariances"][-1]))
            if residual > allowance:
                raise AssertionError(f"The calibrated covariance mismatch is {residual}.")
            row.update({"rotation": population.text(rotation), "maximum_node_covariance_residual": residual,
                        "maximum_node_recursive_w2": float(max(direct["w2"])),
                        "terminal_particle_rms": float(direct["particle_rms"][-1]), "algebraic_w2": "0"})
            np.savez_compressed(raw / f"calibration_{label}_{population.text(z).replace('.', 'p')[:24]}.npz",
                                **{key: value for key, value in direct.items() if isinstance(value, np.ndarray)})
        calibration_rows.append(row)
    if not root_agreement < mp.mpf("1e-75"):
        raise AssertionError("The feasibility root did not stabilize across precision levels.")

    nodes = uniform_nodes(4)
    formula = population.gaussian(nodes, a, a, np.trace(covariance), 0)
    restarted = recursive.gaussian(nodes, float(a), float(a), covariance, [0, 0], [0, 0], restart=True)
    retuned_b = mp.sqrt(population.calibration(mp.mpf(a) / 4)) * 4
    retuned = recursive.gaussian(nodes, float(a), float(retuned_b), covariance, [0, 0], [0, 0])
    radial_formula = population.radial(nodes, "0", "1", "2")
    radial_restart = recursive.radial(nodes, 0, 1, 2, restart=True)
    negative_controls = {"gaussian_restart": checks.reject_expected(formula, restarted, covariance),
                         "gaussian_grid_retuning": checks.reject_expected(formula, retuned, covariance),
                         "radial_restart": checks.reject_expected(radial_formula, radial_restart, np.eye(2))}

    trace = np.trace(covariance)
    constants = population.asymptotic_constants(a, config["horizon"], trace)
    horizon = mp.mpf(config["horizon"])
    constants["corrected_particle_rms_over_h"] = population.text(mp.sqrt(trace) * mp.exp(mp.mpf(a) * horizon) * mp.mpf(a)**2 * horizon)
    asymptotic_rows = []
    for row in run_rows:
        if row["family"] != "gaussian" or row["dimension"] != 2 or row["grid"] == "nonuniform":
            continue
        h = mp.mpf(config["horizon"]) / row["steps"]
        order = 1 if row["method"] == "base" else 2
        asymptotic_rows.append({"method": row["method"], "steps": row["steps"], "h": population.text(h),
                               "terminal_w2_scaled": population.text(mp.mpf(row["terminal_w2"]) / h**order),
                               "maximum_node_w2_scaled": population.text(mp.mpf(row["maximum_node_w2"]) / h**order),
                               "sum_node_w2_squared_scaled": population.text(mp.mpf(row["sum_node_w2_squared"]) / h**(2 * order - 1)),
                               "particle_rms_over_h": population.text(mp.mpf(row["terminal_particle_rms"]) / h)})
    asymptotic_checks = []
    for method in ("base", "corrected"):
        last = [row for row in asymptotic_rows if row["method"] == method][-1]
        terminal_key = "base_terminal_w2_over_h" if method == "base" else "corrected_terminal_w2_over_h_squared"
        sum_key = "base_sum_squared_over_h" if method == "base" else "corrected_sum_squared_over_h_cubed"
        for key, target_key in (("terminal_w2_scaled", terminal_key), ("sum_node_w2_squared_scaled", sum_key)):
            relative = abs(mp.mpf(last[key]) / mp.mpf(constants[target_key]) - 1)
            if relative > mp.mpf("0.002"):
                raise AssertionError(f"The asymptotic constant check failed for {method}: {key}.")
            asymptotic_checks.append({"method": method, "quantity": key, "steps": last["steps"],
                                      "relative_difference": population.text(relative), "allowance": "0.002"})
    corrected_last = [row for row in asymptotic_rows if row["method"] == "corrected"][-1]
    particle_relative = abs(mp.mpf(corrected_last["particle_rms_over_h"]) / mp.mpf(constants["corrected_particle_rms_over_h"]) - 1)
    if particle_relative > mp.mpf("0.002"):
        raise AssertionError("The particle error asymptotic constant check failed.")

    convergence_checks = []
    for method, expected_order in (("base", 1), ("corrected", 2)):
        tail = [row for row in run_rows if row["family"] == "gaussian" and row["dimension"] == 2
                and row["method"] == method and row["steps"] in (2048, 4096)]
        tail.sort(key=lambda row: row["steps"])
        for quantity, target in (("terminal_w2", expected_order), ("maximum_node_w2", expected_order),
                                 ("terminal_w2_squared", 2 * expected_order),
                                 ("maximum_node_w2_squared", 2 * expected_order),
                                 ("sum_node_w2_squared", 2 * expected_order - 1), ("terminal_particle_rms", 1)):
            observed = mp.log(mp.mpf(tail[0][quantity]) / mp.mpf(tail[1][quantity]), 2)
            if abs(observed - target) > mp.mpf("0.005"):
                raise AssertionError(f"The refinement order failed for {method}: {quantity}.")
            convergence_checks.append({"method": method, "quantity": quantity, "coarse_steps": 2048,
                                       "fine_steps": 4096, "expected_order": target,
                                       "observed_order": population.text(observed), "order_allowance": "0.005"})

    bound_checks = []
    for row in run_rows:
        if row["family"] != "gaussian":
            continue
        trace = np.trace(covariance_for(row["dimension"], config))
        coefficient = mp.sqrt(trace) * mp.exp(mp.mpf(a) * horizon)
        h_max = mp.mpf(row["max_step"])
        if row["method"] == "base":
            bound = coefficient * mp.mpf(a)**2 * horizon * h_max / 2
        else:
            bound = coefficient * 2 * mp.mpf(a)**3 * horizon * h_max**2 / 3
        ratio = mp.mpf(row["maximum_node_w2"]) / bound
        if ratio > 1:
            raise AssertionError("The maximum-node bound failed.")
        bound_checks.append({"run_id": row["run_id"], "maximum_node_bound": population.text(bound),
                             "error_over_bound": population.text(ratio)})

    elementary = {"base_action": population.text(2 - mp.pi / 2), "corrected_action": "2",
                  "base_terminal_w2_squared": population.text(2 * (mp.sqrt(2) - 1)**2),
                  "corrected_terminal_w2_squared": "0", "zero_kind": "algebraic"}
    for method in ("0", "1"):
        row = next(r for r in run_rows if r["family"] == "radial" and r["grid"] == "uniform_1" and r["method"] == method)
        prefix = "base" if method == "0" else "corrected"
        if abs(mp.mpf(row["action"]) - mp.mpf(elementary[prefix + "_action"])) > mp.mpf("1e-45"):
            raise AssertionError("The elementary action check failed.")
        if abs(mp.mpf(row["terminal_w2_squared"]) - mp.mpf(elementary[prefix + "_terminal_w2_squared"])) > mp.mpf("1e-45"):
            raise AssertionError("The elementary distribution check failed.")
    write_json(output / "elementary.json", elementary)
    write_json(output / "parameters.json", {"config": config, "runs": parameters})
    write_json(output / "asymptotic_constants.json", constants)
    write_json(output / "feasibility_root.json", {"root_80_digits": mp.nstr(root, 80),
               "root_160_digits": mp.nstr(refined_root, 160), "precision_difference": mp.nstr(root_agreement, 80),
               "root_equation_residual_160_digits": mp.nstr(refined_residual, 80),
               "boundary_rule": "The symbolic root has zero gap. Finite-precision residuals are recorded separately."})
    write_json(output / "negative_controls.json", negative_controls)
    write_csv(output / "nodes.csv", node_rows)
    write_csv(output / "runs.csv", run_rows)
    write_csv(output / "validation.csv", validation_rows)
    write_csv(output / "calibration.csv", calibration_rows)
    write_csv(output / "asymptotics.csv", asymptotic_rows)
    write_json(output / "convergence_orders.json", convergence_checks)
    write_csv(output / "maximum_node_bounds.csv", bound_checks)
    maxima = {key: max(row[key] for row in validation_rows) for key in validation_rows[0] if key != "run_id" and key != "checked_nodes"}
    summary = {"precision_digits": mp.mp.dps, "runs": len(run_rows), "node_comparisons": len(node_rows),
               "maximum_residuals": maxima, "asymptotic_checks": asymptotic_checks,
               "corrected_particle_constant_relative_difference": population.text(particle_relative),
               "negative_controls_rejected": len(negative_controls), "calibration_points": len(calibration_rows),
               "calibrated_node_comparisons": 5 * sum(row["feasible"] for row in calibration_rows),
               "convergence_order_checks": len(convergence_checks), "maximum_node_bound_checks": len(bound_checks)}
    write_json(output / "numerical_summary.json", summary)
    return summary
