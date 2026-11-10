"""Compare fixed Gaussian fields with Heun and Kutta's third-order method."""

import csv
import json
from pathlib import Path

import mpmath as mp


METHODS = {"corrected_euler": ("euler", 1), "base_heun": ("heun", 2),
           "corrected_heun": ("heun", 2), "base_rk3": ("rk3", 3)}


def _finite(value, name):
    try:
        number = mp.mpf(str(value))
    except (ValueError, TypeError) as error:
        raise ValueError(f"The {name} must be a finite real number.") from error
    if not mp.isfinite(number):
        raise ValueError(f"The {name} must be a finite real number.")
    return number


def _parameters(budget, method, rate, horizon, digits):
    if not isinstance(method, str) or method not in METHODS:
        raise ValueError("Use a known Gaussian comparison method.")
    if type(budget) is not int or budget < 6 or budget % 6:
        raise ValueError("The NFE value must be a positive integer divisible by six.")
    if type(digits) is not int or digits < 30:
        raise ValueError("The precision must be an integer of at least 30 digits.")
    a, time = _finite(rate, "rate"), _finite(horizon, "horizon")
    if a <= 0 or time <= 0:
        raise ValueError("The rate and horizon must be positive.")
    b = a if method == "corrected_euler" else a / mp.sqrt(3) if method == "corrected_heun" else mp.mpf(0)
    return a, b, time, budget // METHODS[method][1]


def _covariance(values):
    if not isinstance(values, (list, tuple)) or len(values) != 2 or any(
            not isinstance(row, (list, tuple)) or len(row) != 2 for row in values):
        raise ValueError("The covariance must be a two-dimensional matrix.")
    matrix = mp.matrix([[_finite(value, "covariance entry") for value in row] for row in values])
    if matrix[0, 1] != matrix[1, 0] or matrix[0, 0] <= 0 or mp.det(matrix) <= 0:
        raise ValueError("The covariance must be symmetric and positive definite.")
    return matrix


def polynomial_terminal(budget, method, rate="0.6", horizon="1", covariance_trace="2.25", digits=80):
    """Return terminal population errors from complex polynomial updates."""
    if type(digits) is not int or digits < 30:
        raise ValueError("The precision must be an integer of at least 30 digits.")
    with mp.workdps(digits):
        a, b, time, steps = _parameters(budget, method, rate, horizon, digits)
        trace = _finite(covariance_trace, "covariance trace")
        if trace <= 0:
            raise ValueError("The covariance trace must be positive.")
        z = time * mp.mpc(a, b) / steps
        solver = METHODS[method][0]
        factor = mp.mpc(1)
        for _ in range(steps):
            update = 1 + z
            if solver in {"heun", "rk3"}:
                update += z**2 / 2
            if solver == "rk3":
                update += z**3 / 6
            factor *= update
        exact = mp.exp(time * mp.mpc(a, b))
        return {"steps": steps, "w2": mp.nstr(mp.sqrt(trace) * abs(abs(factor) - abs(exact)), digits - 5),
                "particle_rms": mp.nstr(mp.sqrt(trace) * abs(factor - exact), digits - 5),
                "factor_real": mp.nstr(factor.real, digits - 5),
                "factor_imaginary": mp.nstr(factor.imag, digits - 5),
                "exact_real": mp.nstr(exact.real, digits - 5),
                "exact_imaginary": mp.nstr(exact.imag, digits - 5)}


def _matrix_step(current, field, step, solver):
    first = field(current)
    if solver == "euler":
        return current + step * first
    if solver == "heun":
        second = field(current + step * first)
        return current + step * (first + second) / 2
    second = field(current + step * first / 2)
    third = field(current - step * first + 2 * step * second)
    return current + step * (first + 4 * second + third) / 6


def matrix_terminal(budget, method, covariance, rate="0.6", horizon="1", digits=80):
    """Check terminal errors with matrix stages and a matrix exponential."""
    if type(digits) is not int or digits < 30:
        raise ValueError("The precision must be an integer of at least 30 digits.")
    # The covariance formula subtracts nearly equal traces to compute a small squared error.
    # Extra precision prevents this subtraction from limiting the check.
    arithmetic_digits = digits + 20
    with mp.workdps(arithmetic_digits):
        a, b, time, steps = _parameters(budget, method, rate, horizon, digits)
        source = _covariance(covariance)
        factor = mp.cholesky(source)
        skew = mp.matrix([[0, -1], [1, 0]])
        generator = a * mp.eye(2) + b * factor * skew * factor**-1
        evaluations = 0

        def field(points):
            nonlocal evaluations
            evaluations += 1
            return generator * points

        current = mp.eye(2)
        for _ in range(steps):
            current = _matrix_step(current, field, time / steps, METHODS[method][0])
        exact = mp.expm(time * generator)
        numerical = current * source * current.T
        target = mp.exp(2 * a * time) * source
        trace = source[0, 0] + source[1, 1]
        scale = mp.sqrt((numerical[0, 0] + numerical[1, 1]) / trace)
        residual = mp.norm(numerical - scale**2 * source)
        if residual > mp.mpf(10)**(-arithmetic_digits + 8):
            raise ArithmeticError("The numerical covariance is not proportional to the source covariance.")
        # The two-dimensional covariance formula checks the scale calculation.
        product = numerical * target
        cross_trace = mp.sqrt(product[0, 0] + product[1, 1] + 2 * mp.sqrt(mp.det(numerical) * mp.det(target)))
        squared_w2 = numerical[0, 0] + numerical[1, 1] + target[0, 0] + target[1, 1] - 2 * cross_trace
        if squared_w2 < 0:
            raise ArithmeticError("The covariance distance has a negative squared value.")
        difference = current - exact
        particle_covariance = difference * source * difference.T
        return {"steps": steps, "field_evaluations": evaluations,
                "arithmetic_precision_digits": arithmetic_digits,
                "w2": mp.nstr(mp.sqrt(squared_w2), digits - 5),
                "particle_rms": mp.nstr(mp.sqrt(particle_covariance[0, 0] + particle_covariance[1, 1]), digits - 5),
                "covariance_residual": mp.nstr(residual, digits - 5),
                "terminal_map": [[mp.nstr(current[i, j], digits - 5) for j in range(2)] for i in range(2)]}


def _write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def generate(config_path, output):
    """Save the comparison and results from each precision level and matrix check."""
    config_path, output = Path(config_path), Path(output)
    config_bytes = config_path.read_bytes()
    config = json.loads(config_bytes)
    required = {"evaluation_budgets", "precision_digits", "rate", "horizon", "covariance", "relative_allowance"}
    if not isinstance(config, dict) or not required.issubset(config):
        raise ValueError("The Gaussian comparison configuration is incomplete.")
    budgets, precisions = config["evaluation_budgets"], config["precision_digits"]
    if not isinstance(budgets, list) or not budgets or len(set(map(str, budgets))) != len(budgets):
        raise ValueError("Use a nonempty list of distinct NFE values.")
    if not isinstance(precisions, list) or len(precisions) < 2 or any(type(p) is not int or p < 30 for p in precisions) or precisions != sorted(set(precisions)):
        raise ValueError("Use at least two distinct increasing precision values of at least 30 digits.")
    with mp.workdps(max(precisions)):
        source = _covariance(config["covariance"])
        trace = mp.nstr(source[0, 0] + source[1, 1], max(precisions))
        allowance = _finite(config["relative_allowance"], "relative tolerance")
        if not 0 < allowance < 1:
            raise ValueError("The relative tolerance must be between zero and one.")
        for budget in budgets:
            _parameters(budget, "corrected_heun", config["rate"], config["horizon"], max(precisions))
        rows, raw, checks = [], [], []
        for budget in budgets:
            row = {"field_evaluations": budget}
            for method in METHODS:
                values = []
                for digits in precisions:
                    polynomial = polynomial_terminal(budget, method, config["rate"], config["horizon"], trace, digits)
                    matrix = matrix_terminal(budget, method, config["covariance"], config["rate"], config["horizon"], digits)
                    if matrix["field_evaluations"] != budget:
                        raise AssertionError("The actual field evaluation count is incorrect.")
                    raw.append({"field_evaluations": budget, "method": method, "precision_digits": digits,
                                **polynomial, "matrix_w2": matrix["w2"], "matrix_particle_rms": matrix["particle_rms"],
                                "actual_field_evaluations": matrix["field_evaluations"],
                                "matrix_arithmetic_precision_digits": matrix["arithmetic_precision_digits"],
                                "covariance_residual": matrix["covariance_residual"],
                                **{f"matrix_{i}{j}": matrix["terminal_map"][i][j] for i in range(2) for j in range(2)}})
                    for quantity in ("w2", "particle_rms"):
                        change = abs(mp.mpf(matrix[quantity]) / mp.mpf(polynomial[quantity]) - 1)
                        checks.append({"field_evaluations": budget, "method": method, "precision_digits": digits,
                                       "check": "matrix_" + quantity, "relative_difference": mp.nstr(change, 20)})
                        if change >= allowance:
                            raise AssertionError("The matrix check exceeds the relative tolerance.")
                    values.append(polynomial)
                for quantity in ("w2", "particle_rms"):
                    for index, coarse in enumerate(values[:-1]):
                        change = abs(mp.mpf(coarse[quantity]) / mp.mpf(values[-1][quantity]) - 1)
                        checks.append({"field_evaluations": budget, "method": method, "precision_digits": precisions[index],
                                       "check": "refinement_" + quantity, "relative_difference": mp.nstr(change, 20)})
                        if change >= allowance:
                            raise AssertionError("The precision check exceeds the relative tolerance.")
                row[method + "_steps"] = values[-1]["steps"]
                row[method + "_w2"] = values[-1]["w2"]
                row[method + "_particle_rms"] = values[-1]["particle_rms"]
            equality = abs(mp.mpf(row["corrected_euler_w2"]) / mp.mpf(row["base_heun_w2"]) - 1)
            if equality >= allowance:
                raise AssertionError("The corrected Euler and base Heun distances differ.")
            row["corrected_heun_over_base_rk3_w2"] = mp.nstr(mp.mpf(row["corrected_heun_w2"]) / mp.mpf(row["base_rk3_w2"]), 20)
            rows.append(row)
        if config_path.read_bytes() != config_bytes:
            raise RuntimeError("The Gaussian comparison configuration changed during calculation.")
        output.mkdir(parents=True, exist_ok=True)
        _write_csv(output / "gaussian_higher_order.csv", rows)
        _write_csv(output / "gaussian_higher_order_raw.csv", raw)
        _write_csv(output / "gaussian_higher_order_checks.csv", checks)
        table = [r"\begin{tabular}{rccc}", r"\toprule",
                 r"NFE & \shortstack{Corrected field\\Heun $W_2$} & \shortstack{Base field, RK3\\$W_2$ and particle error} & \shortstack{Corrected field\\Heun particle error} \\", r"\midrule"]
        for row in rows:
            entries = []
            for quantity in ("corrected_heun_w2", "base_rk3_w2", "corrected_heun_particle_rms"):
                mantissa, exponent = f"{float(row[quantity]):.3e}".split("e")
                entries.append("$" + mantissa + r"\times10^{" + str(int(exponent)) + "}$")
            table.append(str(row["field_evaluations"]) + " & " + " & ".join(entries) + r" \\")
        (output / "gaussian_higher_order_table.tex").write_text("\n".join(table + [r"\bottomrule", r"\end{tabular}"]) + "\n")
        summary = {"rows": len(rows), "evaluation_budgets": budgets,
                   "precision_digits": precisions, "rate": config["rate"], "horizon": config["horizon"],
                   "matrix_guard_digits": 20,
                   "covariance": config["covariance"], "covariance_trace": trace,
                   "maximum_relative_check_difference": max(float(row["relative_difference"]) for row in checks)}
        (output / "gaussian_higher_order_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        return summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Compare Gaussian Heun and Kutta RK3 at equal field evaluations.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(generate(args.config, args.output), sort_keys=True))
