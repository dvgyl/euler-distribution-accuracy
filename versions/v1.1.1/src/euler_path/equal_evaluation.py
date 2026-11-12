"""Compare recursive solvers at equal numbers of field evaluations.

Each row contains one NFE value and one quadrature order. NFE means number of
function evaluations. The fields stay fixed across all NFE values. Only terminal
arrays are saved.
"""

import csv
import json
from pathlib import Path

import mpmath as mp
import numpy as np

from . import nonlinear


METHODS = ("base_euler", "corrected_euler", "base_heun")


def terminal_polar(radii, budget, method, horizon=1, a=0.6, eta=0.3):
    """Apply radial updates from the current numerical state.

    For Heun, f(r)=alpha(r)r has f'(r)>0. Its radius map is
    H(r)=r+h[f(r)+f(r+h*f(r))]/2. Thus
    H'(r)=1+h[f'(r)+f'(r+h*f(r))*(1+h*f'(r))]/2>0.
    This increasing map is the radial quantile transport. It preserves the
    angle, so its particle RMS equals the population Wasserstein distance.
    """
    if method not in METHODS or budget < 2 or budget % 2:
        raise ValueError("Use a known method and a positive even NFE value.")
    if horizon <= 0 or a <= 0 or eta < 0 or np.any(np.asarray(radii) < 0):
        raise ValueError("The radii, horizon, or field parameters are invalid.")
    current = np.asarray(radii, dtype=np.longdouble).copy()
    angle = np.zeros_like(current)
    steps = budget // 2 if method == "base_heun" else budget
    h = np.longdouble(horizon) / steps
    for _ in range(steps):
        s = current**2
        alpha = a + eta * s / (1 + s)
        if method == "base_heun":
            first = alpha * current
            predicted = current + h * first
            s_predicted = predicted**2
            second = (a + eta * s_predicted / (1 + s_predicted)) * predicted
            current += h * (first + second) / 2
        else:
            beta = np.sqrt(alpha**2 + 2 * eta * alpha * s / (1 + s)**2) if method == "corrected_euler" else 0
            real, imaginary = 1 + h * alpha, h * beta
            current *= np.hypot(real, imaginary)
            angle += np.arctan2(imaginary, real)
    return {"radius": np.asarray(current, dtype=float), "angle": np.asarray(angle, dtype=float),
            "steps": steps, "field_evaluations": steps * (2 if method == "base_heun" else 1)}


def terminal_cartesian(initial, budget, method, horizon=1, a=0.6, eta=0.3):
    """Check the solver with independent two-dimensional field evaluations."""
    if method not in METHODS or budget < 2 or budget % 2:
        raise ValueError("Use a known method and a positive even NFE value.")
    current = np.asarray(initial, dtype=float).copy()
    if current.ndim != 2 or current.shape[1] != 2:
        raise ValueError("The particles must have two Cartesian coordinates.")
    steps = budget // 2 if method == "base_heun" else budget
    h = horizon / steps

    def field(points):
        squared_norm = np.einsum("ij,ij->i", points, points)
        alpha = a + eta * squared_norm / (1 + squared_norm)
        velocity = alpha[:, None] * points
        if method == "corrected_euler":
            radial_derivative = 2 * eta * squared_norm / (1 + squared_norm)**2
            beta = np.sqrt(alpha * (alpha + radial_derivative))
            velocity += beta[:, None] * np.column_stack((-points[:, 1], points[:, 0]))
        return velocity

    for _ in range(steps):
        first = field(current)
        if method == "base_heun":
            current += h * (first + field(current + h * first)) / 2
        else:
            current += h * first
    return current


def gaussian_identity(budget, rate="0.6", horizon="1", precision=80):
    """Check the exact Gaussian identity at even NFE values in high precision.

    With z=a*T/N, a corrected Euler pair has scale 1+2z+2z^2.
    A base Heun step of size 2T/N has the same scale. Raising both
    factors to N/2 proves equality for each positive even N. The common
    affine mean update is exact, and the covariance scales are equal.
    """
    if budget < 2 or budget % 2:
        raise ValueError("The Gaussian identity requires a positive even NFE value.")
    with mp.workdps(precision):
        z = mp.mpf(rate) * mp.mpf(horizon) / budget
        euler_scale = mp.sqrt((1 + z)**2 + z**2)**budget
        heun_scale = (1 + 2 * z + (2 * z)**2 / 2)**(budget // 2)
        return {"euler_scale": str(euler_scale), "heun_scale": str(heun_scale),
                "absolute_difference": str(abs(euler_scale - heun_scale))}


def write_csv(path, data):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(data[0]))
        writer.writeheader()
        writer.writerows(data)


def generate(comparison_config, nonlinear_config, output):
    """Save equal-evaluation errors and checks without full trajectories."""
    config = json.loads(Path(comparison_config).read_text())
    field = json.loads(Path(nonlinear_config).read_text())
    output = Path(output)
    raw = output / "raw" / "equal_evaluation"
    raw.mkdir(parents=True, exist_ok=True)
    budgets = config["evaluation_budgets"]
    a, eta, horizon = field["a"], field["eta"], field["horizon"]
    run_rows, reference_rows = [], []
    for order in config["quadrature_orders"]:
        rule = nonlinear.mixture_quadrature(order, field["mixture_weights"], field["mixture_scales"])
        radii, weights = rule["radii"], rule["weights"]
        independent = np.array([nonlinear.radius_integrals(r, horizon, a, eta) for r in radii])
        exact_radius, exact_angle = independent[:, 0], independent[:, 1]
        solution = nonlinear.reference(radii, horizon, a, eta, **config["reference_tolerance"])
        ode = nonlinear.reference_values(solution, [0, horizon], radii)
        radius_residual = float(np.sqrt(weights @ (ode["radius"][-1] - exact_radius)**2))
        angle_residual = float(np.sqrt(weights @ (exact_radius * (ode["angle"][-1] - exact_angle))**2))
        np.savez_compressed(raw / f"reference_q{order}.npz", source_radius=radii, weights=weights,
                            exact_radius=exact_radius, exact_angle=exact_angle,
                            ode_radius=ode["radius"][-1], ode_angle=ode["angle"][-1])
        for budget in budgets:
            row = {"quadrature_order": order, "field_evaluations": budget}
            terminal_arrays = {}
            for method in METHODS:
                numerical = terminal_polar(radii, budget, method, horizon, a, eta)
                corrected = method == "corrected_euler"
                w2, particle = nonlinear.radial_errors(numerical["radius"], exact_radius, weights,
                    numerical["angle"] if corrected else None, exact_angle if corrected else None)
                row[method + "_steps"] = numerical["steps"]
                row[method + "_w2"] = float(w2)
                row[method + "_particle_rms"] = float(particle)
                if numerical["field_evaluations"] != budget:
                    raise AssertionError("The field evaluation count is incorrect.")
                fraction = (radius_residual + (angle_residual if corrected else 0)) / min(w2, particle)
                reference_rows.append({"quadrature_order": order, "field_evaluations": budget,
                    "method": method, "radius_rms_difference": radius_residual,
                    "angle_rms_difference": angle_residual, "reference_error_fraction": float(fraction)})
                if fraction >= config["reference_fraction_allowance"]:
                    raise AssertionError("The exact reference check exceeds the error tolerance.")
                terminal_arrays[method + "_radius"] = numerical["radius"]
                terminal_arrays[method + "_angle"] = numerical["angle"]
            row["corrected_euler_over_heun_w2"] = row["corrected_euler_w2"] / row["base_heun_w2"]
            row["corrected_over_base_particle"] = row["corrected_euler_particle_rms"] / row["base_euler_particle_rms"]
            run_rows.append(row)
            np.savez_compressed(raw / f"q{order}_evaluations_{budget}.npz", **terminal_arrays)
    refinement = []
    primary_order, finest = config["primary_quadrature_order"], config["quadrature_orders"][-1]
    for coarse_order, fine_order in zip(config["quadrature_orders"][:-1], config["quadrature_orders"][1:]):
        for budget in budgets:
            coarse = next(row for row in run_rows if row["quadrature_order"] == coarse_order and row["field_evaluations"] == budget)
            fine = next(row for row in run_rows if row["quadrature_order"] == fine_order and row["field_evaluations"] == budget)
            for method in METHODS:
                for metric in ("w2", "particle_rms"):
                    name = method + "_" + metric
                    change = abs(coarse[name] / fine[name] - 1)
                    refinement.append({"coarse_order": coarse_order, "fine_order": fine_order,
                        "field_evaluations": budget, "quantity": name, "relative_change": change})
                    if coarse_order == primary_order and fine_order == finest and change >= config["quadrature_relative_allowance"]:
                        raise AssertionError("The primary quadrature check exceeds the refinement tolerance.")
    radii = np.array([0, 1e-6, 0.07, 0.4, 1, 3, 8, 100])
    angles = np.linspace(0, 5, len(radii))
    initial = radii[:, None] * np.column_stack((np.cos(angles), np.sin(angles)))
    cartesian_rows = []
    for budget in budgets:
        for method in METHODS:
            polar = terminal_polar(radii, budget, method, horizon, a, eta)
            direct = terminal_cartesian(initial, budget, method, horizon, a, eta)
            angle = angles + polar["angle"]
            predicted = polar["radius"][:, None] * np.column_stack((np.cos(angle), np.sin(angle)))
            difference = float(np.max(np.abs(direct - predicted)))
            allowance = 300 * np.finfo(float).eps * (budget + 1) * max(1, np.max(np.abs(predicted)))
            if difference > allowance:
                raise AssertionError("The Cartesian check exceeds the roundoff tolerance.")
            cartesian_rows.append({"field_evaluations": budget, "method": method,
                "maximum_absolute_difference": difference, "roundoff_allowance": allowance})
    gaussian_rows = [{"field_evaluations": n, **gaussian_identity(n)} for n in budgets]
    write_csv(output / "equal_evaluation.csv", run_rows)
    write_csv(output / "equal_evaluation_refinement.csv", refinement)
    write_csv(output / "equal_evaluation_reference.csv", reference_rows)
    write_csv(output / "equal_evaluation_cartesian.csv", cartesian_rows)
    write_csv(output / "equal_evaluation_gaussian.csv", gaussian_rows)
    primary = [row for row in run_rows if row["quadrature_order"] == primary_order]
    table = [r"\begin{tabular}{rccccc}", r"\toprule",
             r"NFE & \shortstack{Base field\\Euler $W_2$} & \shortstack{Corrected field\\Euler $W_2$} & \shortstack{Base field\\Heun $W_2$} & \shortstack{Corrected field\\Euler particle error} & \shortstack{Corrected Euler /\\base-field Heun $W_2$} \\", r"\midrule"]
    for row in primary:
        if row["field_evaluations"] > 16:
            continue
        values = []
        for quantity in ("base_euler_w2", "corrected_euler_w2", "base_heun_w2", "corrected_euler_particle_rms"):
            mantissa, exponent = f"{row[quantity]:.2e}".split("e")
            values.append("$" + mantissa + r"\times10^{" + str(int(exponent)) + "}$")
        values.append(f"{row['corrected_euler_over_heun_w2']:.4f}")
        table.append(str(row["field_evaluations"]) + " & " + " & ".join(values) + r" \\")
    (output / "equal_evaluation_table.tex").write_text("\n".join(table + [r"\bottomrule", r"\end{tabular}"]) + "\n")
    summary = {"primary_quadrature_order": primary_order, "rows": len(run_rows),
        "horizon": horizon, "a": a, "eta": eta, "evaluation_budgets": budgets,
        "maximum_primary_relative_refinement": max(row["relative_change"] for row in refinement if row["coarse_order"] == primary_order),
        "maximum_reference_error_fraction": max(row["reference_error_fraction"] for row in reference_rows),
        "maximum_cartesian_absolute_difference": max(row["maximum_absolute_difference"] for row in cartesian_rows),
        "gaussian_maximum_scale_difference": max(float(row["absolute_difference"]) for row in gaussian_rows),
        "nonlinear_corrected_over_heun_w2_range": [min(row["corrected_euler_over_heun_w2"] for row in primary), max(row["corrected_euler_over_heun_w2"] for row in primary)]}
    (output / "equal_evaluation_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Compare nonlinear solvers at equal field evaluation counts.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--nonlinear-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(generate(args.config, args.nonlinear_config, args.output), sort_keys=True))
