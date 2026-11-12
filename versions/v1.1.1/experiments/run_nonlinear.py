"""Generate nonlinear transport, solver, and Gaussian coupling experiments."""

import json
from pathlib import Path

import mpmath as mp
import numpy as np

from euler_path import checks, nonlinear, nonlinear_recursive, solver_dependence
from run import write_csv, write_json


def metrics(base, corrected, reference, weights):
    base_w2, _ = nonlinear.radial_errors(base["radius"], reference["radius"], weights)
    fixed_w2, particle = nonlinear.radial_errors(corrected["radius"], reference["radius"], weights,
                                               corrected["angle"], reference["angle"])
    return {"base_w2": base_w2, "corrected_w2": fixed_w2, "corrected_particle_rms": particle}


def generate_nonlinear(config, base_config, output):
    raw = output / "raw" / "nonlinear"
    raw.mkdir(parents=True, exist_ok=True)
    counts = config["step_counts"]
    grids = [(f"uniform_{n}", np.linspace(0, config["horizon"], n + 1)) for n in counts]
    grids.append(("nonuniform", np.asarray(base_config["nonuniform_nodes"], dtype=float)))
    a, eta, horizon = config["a"], config["eta"], config["horizon"]
    run_rows, node_rows, validation_rows, coefficient_rows = [], [], [], []
    summaries = {}
    primary_order = None
    quadrature_rows = []
    previous = None
    for order in config["quadrature_orders"]:
        quadrature = nonlinear.mixture_quadrature(order, config["mixture_weights"], config["mixture_scales"])
        radii, weights = quadrature["radii"], quadrature["weights"]
        tolerance = config["reference_tolerances"][1]
        solution = nonlinear.reference(radii, horizon, a, eta, **tolerance)
        independent = np.array([nonlinear.radius_integrals(r, horizon, a, eta) for r in radii])
        terminal_reference = nonlinear.reference_values(solution, [0, horizon], radii)
        independent_names = ("radius", "angle", "action_base", "action_corrected",
                             "coefficient_base", "coefficient_corrected", "coefficient_angle")
        agreement = {}
        for column, name in enumerate(independent_names):
            time_value = terminal_reference[name][-1] if name in ("radius", "angle") else terminal_reference[name]
            agreement[name] = float(np.sqrt(weights @ (time_value - independent[:, column])**2))
            scale = max(1, float(np.sqrt(weights @ independent[:, column]**2)))
            if agreement[name] > 2e-11 * scale:
                raise AssertionError(f"The independent nonlinear reference check failed for {name}.")
        values = {"action_base": float(weights @ independent[:, 2]),
                  "action_corrected": float(weights @ independent[:, 3]),
                  "coefficient_base": float(np.sqrt(weights @ independent[:, 4]**2)),
                  "coefficient_corrected": float(np.sqrt(weights @ independent[:, 5]**2)),
                  "coefficient_particle": float(np.sqrt(weights @ (independent[:, 0] * independent[:, 6])**2))}
        values["action_difference"] = values["action_corrected"] - values["action_base"]
        np.savez_compressed(raw / f"quadrature_{order}.npz", **quadrature, independent_integrals=independent,
                            ode_terminal_radius=terminal_reference["radius"][-1],
                            ode_terminal_angle=terminal_reference["angle"][-1],
                            **{"ode_" + name: terminal_reference[name] for name in independent_names[2:]})
        for grid_name, times in grids:
            base = nonlinear_recursive.polar(times, radii, False, a, eta)
            corrected = nonlinear_recursive.polar(times, radii, True, a, eta)
            reference = nonlinear.reference_values(solution, times, radii)
            ode_radius = reference["radius"][-1].copy()
            ode_angle = reference["angle"][-1].copy()
            reference["radius"][-1], reference["angle"][-1] = independent[:, 0], independent[:, 1]
            errors = metrics(base, corrected, reference, weights)
            if not np.all((base["radius"][1:] < corrected["radius"][1:]) &
                          (corrected["radius"][1:] < reference["radius"][1:])):
                raise AssertionError("The recorded nonlinear radii have incorrect strict ordering.")
            if not np.all((errors["corrected_w2"][1:] > 0) &
                          (errors["corrected_w2"][1:] < errors["base_w2"][1:])):
                raise AssertionError("The recorded nonlinear node errors have incorrect strict ordering.")
            np.savez_compressed(raw / f"q{order}_{grid_name}.npz", times=times, initial_radii=radii, weights=weights,
                                base_radius=base["radius"], base_angle=base["angle"],
                                corrected_radius=corrected["radius"], corrected_angle=corrected["angle"],
                                reference_radius=reference["radius"], reference_angle=reference["angle"],
                                ode_terminal_radius=ode_radius, ode_terminal_angle=ode_angle, **errors)
            for quantity, error in errors.items():
                values[f"{grid_name}_{quantity}"] = float(error[-1])
            run_rows.append({"quadrature_order": order, "grid": grid_name, "steps": len(times) - 1,
                             "base_w2": float(errors["base_w2"][-1]),
                             "corrected_w2": float(errors["corrected_w2"][-1]),
                             "corrected_particle_rms": float(errors["corrected_particle_rms"][-1]),
                             "action_base": values["action_base"], "action_corrected": values["action_corrected"],
                             "action_difference": values["action_difference"]})
            for index, time in enumerate(times):
                node_rows.append({"quadrature_order": order, "grid": grid_name, "node": index, "time": time,
                                  **{quantity: float(error[index]) for quantity, error in errors.items()}})
        summaries[order] = values
        validation_rows.append({"quadrature_order": order, **{name + "_rms_difference": value for name, value in agreement.items()},
                                "ode_field_evaluations": solution.nfev})
        if previous is not None:
            changes = {name: abs(value / previous[name] - 1) for name, value in values.items()}
            stable = max(changes.values()) < config["quadrature_relative_allowance"]
            quadrature_rows.extend({"coarse_order": order // 2, "fine_order": order, "quantity": name,
                                    "coarse_value": previous[name], "fine_value": values[name], "relative_change": change,
                                    "allowance": config["quadrature_relative_allowance"], "passed": change < config["quadrature_relative_allowance"]}
                                   for name, change in changes.items())
            if primary_order is None and stable:
                primary_order = order
        previous = values
        coefficient_rows.append({"quadrature_order": order, **{key: value for key, value in values.items()
                                                               if key.startswith(("action", "coefficient"))}})
    if primary_order is None:
        raise AssertionError("No quadrature order meets the specified refinement condition.")
    finest = config["quadrature_orders"][-1]
    displayed_changes = [abs(summaries[primary_order][key] / summaries[finest][key] - 1) for key in summaries[finest]]
    if max(displayed_changes) >= config["quadrature_relative_allowance"]:
        raise AssertionError("The displayed quadrature results differ from the finest results.")
    primary = summaries[primary_order]
    orders = []
    fine_grid, coarse_grid = f"uniform_{counts[-1]}", f"uniform_{counts[-2]}"
    for quantity, target, coefficient_name in (("base_w2", 1, "coefficient_base"),
                                              ("corrected_w2", 2, "coefficient_corrected"),
                                              ("corrected_particle_rms", 1, "coefficient_particle")):
        coarse, fine = primary[f"{coarse_grid}_{quantity}"], primary[f"{fine_grid}_{quantity}"]
        observed = np.log(coarse / fine) / np.log(counts[-1] / counts[-2])
        scaled = fine / (horizon / counts[-1])**target
        relative = abs(scaled / primary[coefficient_name] - 1)
        if abs(observed - target) >= config["order_absolute_allowance"] or relative >= config["coefficient_relative_allowance"]:
            raise AssertionError(f"The nonlinear refinement condition failed for {quantity}.")
        orders.append({"quantity": quantity, "coarse_steps": counts[-2], "fine_steps": counts[-1],
                       "expected_order": target, "observed_order": float(observed), "scaled_fine_error": scaled,
                       "independent_coefficient": primary[coefficient_name], "coefficient_relative_difference": relative})
    reference_rows = reference_refinement(config, primary_order, grids, raw, primary)
    cartesian_rows = cartesian_validation(config, grids, raw)
    if not config.get("retain_refinement_trajectories", False):
        compact_refinement_trajectories(raw, primary_order)
    write_csv(output / "nonlinear_runs.csv", run_rows)
    write_csv(output / "nonlinear_nodes.csv", node_rows)
    write_csv(output / "nonlinear_quadrature_refinement.csv", quadrature_rows)
    write_csv(output / "nonlinear_reference_refinement.csv", reference_rows)
    write_csv(output / "nonlinear_independent_reference.csv", validation_rows)
    write_csv(output / "nonlinear_cartesian_validation.csv", cartesian_rows)
    write_csv(output / "nonlinear_coefficients.csv", coefficient_rows)
    write_json(output / "nonlinear_orders.json", orders)
    shape_data(config, output)
    summary = {"horizon": horizon, "primary_quadrature_order": primary_order, "mixture_component_count": len(config["mixture_weights"]),
               "runs": len(run_rows), "node_comparisons": len(node_rows), "orders": orders,
               "action_base": primary["action_base"], "action_corrected": primary["action_corrected"],
               "action_difference": primary["action_difference"], "maximum_displayed_quadrature_change": max(displayed_changes),
               "maximum_reference_fraction": max(row["reference_error_fraction"] for row in reference_rows),
               "maximum_cartesian_absolute_difference": max(row["maximum_absolute"] for row in cartesian_rows)}
    write_json(output / "nonlinear_summary.json", summary)
    return summary


def compact_refinement_trajectories(raw, primary_order):
    """Keep full primary paths and terminal arrays for quadrature refinements."""
    trajectory_names = ("base_radius", "base_angle", "corrected_radius",
                        "corrected_angle", "reference_radius", "reference_angle")
    for path in raw.glob("q[0-9]*_*.npz"):
        if path.name.startswith(f"q{primary_order}_"):
            continue
        with np.load(path, allow_pickle=False) as saved:
            arrays = {name: saved[name] for name in saved.files}
        for name in trajectory_names:
            arrays["terminal_" + name] = arrays.pop(name)[-1]
        np.savez_compressed(path, **arrays)


def reference_refinement(config, order, grids, raw, primary):
    with np.load(raw / f"quadrature_{order}.npz") as data:
        radii, weights, independent = data["radii"], data["weights"], data["independent_integrals"]
    rows = []
    for index, tolerance in enumerate(config["reference_tolerances"]):
        solution = nonlinear.reference(radii, config["horizon"], config["a"], config["eta"], **tolerance)
        values = nonlinear.reference_values(solution, [0, config["horizon"]], radii)
        np.savez_compressed(raw / f"reference_tolerance_{index}.npz", rtol=tolerance["rtol"], atol=tolerance["atol"], **values)
        radial_residual = float(np.sqrt(weights @ (values["radius"][-1] - independent[:, 0])**2))
        angle_residual = float(np.sqrt(weights @ (independent[:, 0] * (values["angle"][-1] - independent[:, 1]))**2))
        for grid_name, times in grids:
            with np.load(raw / f"q{order}_{grid_name}.npz") as data:
                base = {"radius": data["base_radius"][-1]}
                corrected = {"radius": data["corrected_radius"][-1], "angle": data["corrected_angle"][-1]}
            errors = metrics(base, corrected, {"radius": values["radius"][-1], "angle": values["angle"][-1]}, weights)
            for quantity, error in errors.items():
                target = primary[f"{grid_name}_{quantity}"]
                reference_bound = radial_residual + (angle_residual if quantity == "corrected_particle_rms" else 0)
                fraction = reference_bound / target
                rows.append({"quadrature_order": order, "tolerance_index": index, **tolerance,
                             "grid": grid_name, "quantity": quantity, "ode_metric": float(error), "independent_metric": target,
                             "absolute_metric_difference": abs(float(error) - target), "radial_reference_rms": radial_residual,
                             "angle_reference_rms": angle_residual, "reference_error_fraction": fraction,
                             "allowance": config["reference_fraction_allowance"], "passed": fraction < config["reference_fraction_allowance"]})
        for name, column in (("action_base", 2), ("action_corrected", 3)):
            difference = abs(weights @ values[name] - weights @ independent[:, column])
            fraction = float(difference / primary[name])
            rows.append({"quadrature_order": order, "tolerance_index": index, **tolerance,
                         "grid": "continuous", "quantity": name, "ode_metric": float(weights @ values[name]),
                         "independent_metric": primary[name], "absolute_metric_difference": float(difference),
                         "radial_reference_rms": radial_residual, "angle_reference_rms": angle_residual,
                         "reference_error_fraction": fraction, "allowance": config["reference_fraction_allowance"],
                         "passed": fraction < config["reference_fraction_allowance"]})
    primary_failures = [row for row in rows if row["tolerance_index"] >= 1 and not row["passed"]]
    if primary_failures:
        raise AssertionError("The primary or refined reference is too close to the measured numerical error.")
    return rows


def cartesian_validation(config, grids, raw):
    radii, angles = np.array(config["cartesian_radii"]), np.array(config["cartesian_angles"])
    initial = radii[:, None] * np.column_stack((np.cos(angles), np.sin(angles)))
    rows = []
    for grid_name, times in grids:
        for corrected in (False, True):
            polar = nonlinear_recursive.polar(times, radii, corrected, config["a"], config["eta"])
            cartesian = nonlinear_recursive.cartesian(times, initial, corrected, config["a"], config["eta"])
            validation = checks.compare_cartesian_polar(cartesian, polar["radius"], polar["angle"], angles)
            rows.append({"grid": grid_name, "method": "corrected" if corrected else "base", **validation})
            np.savez_compressed(raw / f"cartesian_{grid_name}_{rows[-1]['method']}.npz", initial=initial,
                                cartesian=cartesian, initial_angles=angles, **polar)
    return rows


def shape_data(config, output):
    probabilities = np.linspace(0.005, 0.995, 199)
    source = nonlinear.source_quantiles(probabilities, config["mixture_weights"], config["mixture_scales"])
    times = np.array(config["shape_times"])
    radii = np.array([[nonlinear.implicit_radius(r, t, config["a"], config["eta"]) for r in source] for t in times])
    median = nonlinear.source_quantiles([0.5], config["mixture_weights"], config["mixture_scales"])[0]
    medians = np.array([nonlinear.implicit_radius(median, t, config["a"], config["eta"]) for t in times])
    np.savez_compressed(output / "nonlinear_shape_data.npz", probabilities=probabilities, times=times,
                        radii=radii, source_radii=source, median_radii=medians,
                        normalized_quantiles=radii / medians[:, None], dilation_ratios=radii / source[None, :])


def generate_gaussian_extensions(config, base_config, output):
    raw = output / "raw" / "solver_dependence"
    raw.mkdir(parents=True, exist_ok=True)
    a, horizon = config["a"], config["horizon"]
    covariance = np.array(base_config["gaussian"]["covariance_2d"])
    trace = np.trace(covariance)
    grids = [(f"uniform_{n}", np.linspace(0, horizon, n + 1)) for n in config["step_counts"]]
    grids.append(("nonuniform", np.asarray(base_config["nonuniform_nodes"], dtype=float)))
    sensitivity, solvers, flow_rows = [], [], []
    sensitivity_grids = [(f"uniform_{n}", np.linspace(0, horizon, n + 1)) for n in config["sensitivity_step_counts"]]
    sensitivity_grids.append(grids[-1])
    for ratio in config["sensitivity_rotation_ratios"]:
        b = mp.mpf(str(a)) * mp.mpf(str(ratio))
        for grid_name, times in sensitivity_grids:
            formula = solver_dependence.gaussian_formula(times, a, b, trace, "euler")
            direct = solver_dependence.gaussian_matrix(times, a, float(b), covariance, "euler")
            residual = validate_gaussian(formula, direct, trace, a)
            expected_order, coefficient = solver_dependence.gaussian_leading_coefficient(a, float(b), horizon, trace, "euler")
            row = {"rotation_ratio": ratio, "rotation": float(b), "grid": grid_name, "steps": len(times) - 1,
                   "terminal_w2": mp.nstr(formula[-1], 50), "expected_order": expected_order,
                   "predicted_coefficient": coefficient, "maximum_matrix_difference": residual}
            sensitivity.append(row)
            np.savez_compressed(raw / f"sensitivity_ratio_{str(ratio).replace('.', 'p')}_{grid_name}.npz",
                                formula_w2=np.array([float(value) for value in formula]), **direct)
    sensitivity_refinement = []
    sensitivity_orders = []
    for ratio in config["sensitivity_rotation_ratios"]:
        data = [row for row in sensitivity if row["rotation_ratio"] == ratio and row["grid"] != "nonuniform"]
        for row in data:
            scaled = float(row["terminal_w2"]) / (horizon / row["steps"])**row["expected_order"]
            relative = abs(scaled / row["predicted_coefficient"] - 1)
            sensitivity_refinement.append({"rotation_ratio": ratio, "steps": row["steps"],
                                           "expected_order": row["expected_order"], "scaled_error": scaled,
                                           "predicted_coefficient": row["predicted_coefficient"],
                                           "relative_difference": relative,
                                           "allowance": config["coefficient_relative_allowance"],
                                           "passed": relative < config["coefficient_relative_allowance"]})
        # Each row records whether its scaled error meets the coefficient tolerance.
        # The run requires this test only at the finest grid for each ratio.
        coarse, fine = data[-2:]
        observed = float(mp.log(mp.mpf(coarse["terminal_w2"]) / mp.mpf(fine["terminal_w2"]), 2))
        last = sensitivity_refinement[-1]
        if not last["passed"] or abs(observed - fine["expected_order"]) >= config["order_absolute_allowance"]:
            raise AssertionError("The fixed Gaussian sensitivity refinement did not meet the specified accuracy.")
        sensitivity_orders.append({"rotation_ratio": ratio, "coarse_steps": coarse["steps"], "fine_steps": fine["steps"],
                                   "observed_order": observed, **{key: last[key] for key in
                                       ("expected_order", "scaled_error", "predicted_coefficient", "relative_difference")}})
    for solver in ("euler", "heun"):
        for method, b in (("base", mp.mpf(0)), ("corrected", mp.mpf(str(a))),
                          ("heun_cubic_cancellation", mp.mpf(str(a)) / mp.sqrt(3))):
            for grid_name, times in grids:
                formula = solver_dependence.gaussian_formula(times, a, b, trace, solver)
                direct = solver_dependence.gaussian_matrix(times, a, float(b), covariance, solver)
                residual = validate_gaussian(formula, direct, trace, a)
                expected_order, coefficient = solver_dependence.gaussian_leading_coefficient(a, float(b), horizon, trace, solver)
                if solver == "heun" and method == "heun_cubic_cancellation":
                    expected_order, coefficient = 3, np.sqrt(trace) * np.exp(a * horizon) * horizon * a**4 / 9
                solvers.append({"solver": solver, "method": method, "rotation": float(b), "grid": grid_name,
                                "steps": len(times) - 1, "field_evaluations": (1 if solver == "euler" else 2) * (len(times) - 1),
                                "terminal_w2": mp.nstr(formula[-1], 50), "expected_order": expected_order,
                                "predicted_coefficient": coefficient, "maximum_matrix_difference": residual})
                np.savez_compressed(raw / f"{solver}_{method}_{grid_name}.npz",
                                    formula_w2=np.array([float(value) for value in formula]), **direct)
    orders = []
    precision_rows = []
    for solver in ("euler", "heun"):
        for method in ("base", "corrected", "heun_cubic_cancellation"):
            rows = [row for row in solvers if row["solver"] == solver and row["method"] == method and
                    row["grid"] in (f"uniform_{config['step_counts'][-2]}", f"uniform_{config['step_counts'][-1]}")]
            coarse, fine = rows
            observed = float(mp.log(mp.mpf(coarse["terminal_w2"]) / mp.mpf(fine["terminal_w2"]), 2))
            relative = abs(float(mp.mpf(fine["terminal_w2"])) / (horizon / fine["steps"])**fine["expected_order"] /
                           fine["predicted_coefficient"] - 1)
            if abs(observed - fine["expected_order"]) >= config["order_absolute_allowance"] or relative >= config["coefficient_relative_allowance"]:
                raise AssertionError("The solver error coefficient or order check failed.")
            orders.append({"solver": solver, "method": method, "expected_order": fine["expected_order"],
                           "observed_order": observed, "predicted_coefficient": fine["predicted_coefficient"],
                           "coefficient_relative_difference": relative})
            b = {"base": mp.mpf(0), "corrected": mp.mpf(str(a)),
                 "heun_cubic_cancellation": mp.mpf(str(a)) / mp.sqrt(3)}[method]
            for row in rows:
                for digits in (50, 80):
                    high_precision = solver_dependence.gaussian_matrix_high_precision(row["steps"], a, b, horizon,
                                                                                     covariance, solver, digits)
                    residual = abs(mp.mpf(high_precision["w2"]) - mp.mpf(row["terminal_w2"]))
                    if residual > mp.mpf("1e-40"):
                        raise AssertionError("The high-precision Gaussian matrix recurrence disagrees with the formula.")
                    precision_rows.append({"solver": solver, "method": method, "steps": row["steps"], "digits": digits,
                                           "matrix_w2": high_precision["w2"], "formula_w2": row["terminal_w2"],
                                           "absolute_difference": mp.nstr(residual, 50),
                                           "covariance_proportionality_residual": high_precision["covariance_residual"]})
    heun_rows = [row for row in solvers if row["solver"] == "heun" and row["steps"] == config["step_counts"][-1]]
    heun_base, heun_corrected = [row for row in heun_rows if row["method"] in ("base", "corrected")]
    heun_ratio = float(mp.mpf(heun_corrected["terminal_w2"]) / mp.mpf(heun_base["terminal_w2"]))
    if abs(heun_ratio / 2 - 1) > config["coefficient_relative_allowance"]:
        raise AssertionError("The Heun asymptotic ratio differs from two.")
    for grid_name, times in grids:
        for corrected in (False, True):
            direct = solver_dependence.flow_matching(times, config["flow_matching_sigma"], corrected)
            covariance_residual = np.linalg.norm(direct["maps"] @ np.transpose(direct["maps"], (0, 2, 1)) - direct["target_covariances"], axis=(1, 2))
            arithmetic = 200 * np.finfo(float).eps * np.arange(1, len(times) + 1) * max(1, config["flow_matching_sigma"]**2)
            if corrected and (np.any(covariance_residual > arithmetic) or np.any(direct["particle_rms"] > arithmetic)):
                raise AssertionError("The rotated Gaussian coupling is not Euler exact to arithmetic tolerance.")
            flow_rows.append({"grid": grid_name, "method": "rotated_coupling" if corrected else "base", "steps": len(times) - 1,
                              "algebraic_w2_zero": corrected, "algebraic_particle_zero": corrected,
                              "terminal_w2_residual": float(direct["w2"][-1]),
                              "maximum_covariance_residual": float(covariance_residual.max()),
                              "maximum_particle_rms_residual": float(direct["particle_rms"].max()),
                              "maximum_allowance_fraction": float(max((covariance_residual / arithmetic).max(),
                                                                       (direct["particle_rms"] / arithmetic).max()))})
            np.savez_compressed(raw / f"flow_matching_{flow_rows[-1]['method']}_{grid_name}.npz", **direct)
    write_csv(output / "gaussian_sensitivity.csv", sensitivity)
    write_csv(output / "gaussian_sensitivity_refinement.csv", sensitivity_refinement)
    write_csv(output / "solver_dependence.csv", solvers)
    write_csv(output / "flow_matching.csv", flow_rows)
    write_json(output / "solver_orders.json", orders)
    write_json(output / "gaussian_sensitivity_orders.json", sensitivity_orders)
    write_csv(output / "solver_precision_refinement.csv", precision_rows)
    summary = {"sensitivity_runs": len(sensitivity), "solver_runs": len(solvers),
               "horizon": horizon,
               "flow_matching_runs": len(flow_rows), "heun_corrected_over_base_finest": heun_ratio,
               "solver_orders": orders, "maximum_matrix_difference": max(row["maximum_matrix_difference"] for row in solvers),
               "sensitivity_orders": sensitivity_orders,
               "maximum_high_precision_matrix_difference": mp.nstr(max(mp.mpf(row["absolute_difference"]) for row in precision_rows), 50),
               "flow_matching_maximum_particle_residual": max(row["maximum_particle_rms_residual"] for row in flow_rows
                                                               if row["method"] == "rotated_coupling")}
    write_json(output / "solver_summary.json", summary)
    return summary


def validate_gaussian(formula, direct, trace, a):
    expected = np.array([float(value) for value in formula])
    allowance = 200 * np.finfo(float).eps * np.arange(1, len(expected) + 1) * max(1, np.sqrt(trace) * np.exp(a * direct["times"][-1]))
    residual = abs(expected - direct["w2"])
    if not np.all(np.isfinite(residual)) or np.any(residual > allowance):
        raise AssertionError("The Gaussian matrix recurrence disagrees with the stability polynomial.")
    return float(residual.max())


def generate(config_path, base_config_path, output):
    config = json.loads(Path(config_path).read_text())
    base_config = json.loads(Path(base_config_path).read_text())
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    mp.mp.dps = base_config["precision_digits"]
    nonlinear_summary = generate_nonlinear(config, base_config, output)
    solver_summary = generate_gaussian_extensions(config, base_config, output)
    return {"nonlinear": nonlinear_summary, "solver": solver_summary}
