"""Exact Gaussian validation of endpoint cancellation and mesh dependence.

All reported W2 values are exact Gaussian metric formulas evaluated numerically,
not particle couplings or transport proxies. The high-precision checks use an
independent product formula; they are not interval-arithmetic certificates.
"""

from __future__ import annotations

from dataclasses import dataclass
import csv
import json
import math
from pathlib import Path
import platform
import time

import numpy as np
from scipy.optimize import brentq


@dataclass(frozen=True)
class CosineGaussian:
    a: float = 0.4
    epsilon: float = 0.08
    delta: float = 0.2

    def __post_init__(self):
        if not self.a > 0 or not 0 <= self.epsilon < self.a**2:
            raise ValueError("Require a>0 and 0<=epsilon<a^2.")
        if not abs(self.delta) < 1:
            raise ValueError("The quantile density requires |delta|<1.")

    def defect(self, t):
        return self.epsilon*np.cos(2*np.pi*np.asarray(t))

    def angular_speed(self, t):
        return np.sqrt(self.a**2-self.defect(t))

    def matrix(self, t):
        b = float(self.angular_speed(t))
        return np.array([[self.a, -b], [b, self.a]])

    def projected_acceleration_norm(self, t):
        t = np.asarray(t)
        return np.sqrt(2)*np.exp(self.a*t)*np.abs(self.defect(t))

    def density(self, t, schedule):
        delta = self.delta if schedule == "quantile" else 0.
        return 1+delta*np.cos(2*np.pi*np.asarray(t))

    def nodes(self, n, schedule="uniform"):
        if int(n) != n or n < 1:
            raise ValueError("n must be a positive integer.")
        if schedule == "uniform":
            return np.linspace(0., 1., n+1)
        if schedule != "quantile":
            raise ValueError("Unknown mesh schedule.")
        probabilities = np.linspace(0., 1., n+1)
        nodes = probabilities.copy()
        lo = np.zeros(n+1)
        hi = np.ones(n+1)
        for _ in range(60):
            error = (nodes+self.delta*np.sin(2*np.pi*nodes)/(2*np.pi)
                     - probabilities)
            if np.max(np.abs(error)) <= 2*np.finfo(float).eps:
                break
            lo = np.where(error < 0, nodes, lo)
            hi = np.where(error > 0, nodes, hi)
            candidate = nodes-error/self.density(nodes, "quantile")
            nodes = np.where((candidate < lo) | (candidate > hi),
                             (lo+hi)/2, candidate)
        else:
            raise ArithmeticError("Quantile inversion did not converge.")
        nodes[0], nodes[-1] = 0., 1.
        if np.any(np.diff(nodes) <= 0):
            raise ArithmeticError("Quantile nodes are not strictly increasing.")
        return nodes

    def reciprocal_density_integral(self, t, schedule="uniform"):
        """Integral from 0 to t of rho^{-1}; continuous branch on [0,1]."""
        t = np.asarray(t)
        delta = self.delta if schedule == "quantile" else 0.
        if delta == 0:
            return t
        q = np.sqrt((1-delta)/(1+delta))
        angle = np.arctan2(q*np.sin(np.pi*t), np.cos(np.pi*t))
        return angle/(np.pi*np.sqrt(1-delta**2))

    def reciprocal_density_squared_integral(self, t, schedule="uniform"):
        t = np.asarray(t)
        delta = self.delta if schedule == "quantile" else 0.
        if delta == 0:
            return t
        first = self.reciprocal_density_integral(t, schedule)
        return (first-delta*np.sin(2*np.pi*t)/
                (2*np.pi*self.density(t, schedule)))/(1-delta**2)

    def weighted_defect_integral(self, t, schedule="uniform"):
        """Integral from 0 to t of f/rho, independent of the Euler solver."""
        t = np.asarray(t)
        delta = self.delta if schedule == "quantile" else 0.
        if delta == 0:
            return self.epsilon*np.sin(2*np.pi*t)/(2*np.pi)
        return self.epsilon/delta*(
            t-self.reciprocal_density_integral(t, schedule))

    def first_order_profile(self, t, schedule="uniform"):
        t = np.asarray(t)
        return (np.exp(self.a*t)/np.sqrt(2)
                * np.abs(self.weighted_defect_integral(t, schedule)))

    def ap_second_order_profile(self, t, schedule="uniform"):
        """Second-order coefficient for the AP control epsilon=0."""
        t = np.asarray(t)
        return (2*np.sqrt(2)*self.a**3/3*np.exp(self.a*t)
                * self.reciprocal_density_squared_integral(t, schedule))

    def first_order_maximum(self, schedule="uniform"):
        def stationary(t):
            return (self.a*self.weighted_defect_integral(t, schedule)
                    + self.defect(t)/self.density(t, schedule))
        grid = np.linspace(0., 1., 257)
        candidates = [0., 1.]
        values = stationary(grid)
        for left, right, fl, fr in zip(grid[:-1], grid[1:], values[:-1], values[1:]):
            if fl*fr < 0:
                candidates.append(brentq(stationary, left, right, xtol=2e-15))
        result = max(candidates, key=lambda t: float(self.first_order_profile(t, schedule)))
        return {"time": float(result),
                "coefficient": float(self.first_order_profile(result, schedule))}

    def constants(self, schedule="uniform"):
        ap = self.epsilon == 0
        endpoint_order = 2 if ap or schedule == "uniform" else 1
        endpoint = (self.ap_second_order_profile(1., schedule)
                    if endpoint_order == 2 else self.first_order_profile(1., schedule))
        maximum = ({"time": 1., "coefficient": float(endpoint)}
                   if ap else self.first_order_maximum(schedule))
        interior = (self.ap_second_order_profile(.25, schedule) if ap
                    else self.first_order_profile(.25, schedule))
        return {"endpoint_order": endpoint_order, "endpoint_coefficient": float(endpoint),
                "interior_order": 2 if ap else 1, "interior_time": .25,
                "interior_coefficient": float(interior),
                "max_node_order": 2 if ap else 1,
                "max_node_coefficient": maximum["coefficient"],
                "max_node_asymptotic_time": maximum["time"]}


def _log1pmx(x):
    """Stable log(1+x)-x, retaining cubic Euler defects on fine meshes."""
    x = np.asarray(x, dtype=float)
    polynomial = np.zeros_like(x)
    for k in range(24, 1, -1):
        polynomial = polynomial*x+(-1.)**(k+1)/k
    small = x*x*polynomial
    large = np.zeros_like(x)
    mask = np.abs(x) > .1
    large[mask] = np.log1p(x[mask])-x[mask]
    return np.where(mask, large, small)


def _compensated_cumsum(values):
    out = np.zeros(len(values)+1)
    total = correction = 0.
    for i, value in enumerate(values, start=1):
        summand = float(value)-correction
        updated = total+summand
        correction = (updated-total)-summand
        total = updated
        out[i] = total
    return out


def euler_profile(path, n, schedule="uniform"):
    times = path.nodes(n, schedule)
    steps = np.diff(times)
    defect = path.defect(times[:-1])
    q = 2*path.a*steps+(2*path.a**2-defect)*steps**2
    log_increments = .5*_log1pmx(q)+(path.a**2-.5*defect)*steps**2
    log_error = _compensated_cumsum(log_increments)
    exact_scale = np.exp(path.a*times)
    w2 = np.sqrt(2)*exact_scale*np.abs(np.expm1(log_error))
    return {"time": times, "log_scale_error": log_error, "w2": w2,
            "euler_scale": exact_scale*np.exp(log_error)}


def covariance_reference(path, n, schedule="uniform"):
    """Independent 2x2 covariance recurrence; no scalar scale recursion."""
    times = path.nodes(n, schedule)
    covariance = np.eye(2)
    for t, h in zip(times[:-1], np.diff(times)):
        matrix = np.eye(2)+h*path.matrix(t)
        covariance = matrix@covariance@matrix.T
    eigenvalues = np.linalg.eigvalsh(covariance)
    w2 = np.linalg.norm(np.sqrt(eigenvalues)-math.exp(path.a))
    return {"covariance": covariance, "w2": float(w2)}


def high_precision_reference(path, n, schedule="uniform", digits=100):
    """Independent arbitrary-precision product, with freshly solved mesh nodes."""
    import mpmath as mp
    with mp.workdps(digits):
        a, epsilon, delta = (mp.mpf(str(v)) for v in (path.a, path.epsilon, path.delta))
        nodes = [mp.mpf(k)/n for k in range(n+1)]
        if schedule == "quantile":
            for k in range(1, n):
                probability = mp.mpf(k)/n
                t = probability
                for _ in range(30):
                    error = t+delta*mp.sin(2*mp.pi*t)/(2*mp.pi)-probability
                    update = error/(1+delta*mp.cos(2*mp.pi*t))
                    t -= update
                    if abs(update) < mp.eps*10:
                        break
                else:
                    raise ArithmeticError("High-precision quantile inversion failed.")
                nodes[k] = t
        elif schedule != "uniform":
            raise ValueError("Unknown mesh schedule.")
        increments = []
        for left, right in zip(nodes[:-1], nodes[1:]):
            h = right-left
            b_squared = a*a-epsilon*mp.cos(2*mp.pi*left)
            increments.append(mp.log((1+a*h)**2+b_squared*h*h)/2-a*h)
        log_error = mp.fsum(increments)
        w2 = mp.sqrt(2)*mp.exp(a)*abs(mp.expm1(log_error))
        return {"digits": digits, "log_scale_error": mp.nstr(log_error, digits),
                "w2": mp.nstr(w2, digits)}


def _write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot(output, rows, profiles, paths):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 7.6, "axes.titlesize": 8,
                         "axes.labelsize": 7.6, "legend.fontsize": 6.7,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.35), constrained_layout=True)
    selected = [r for r in rows if r["field"] == "cosine" and r["schedule"] == "uniform"]
    ns = np.array([r["n"] for r in selected])
    for column, label, color, marker in [
            ("endpoint_w2", "Endpoint", "#2667a6", "o"),
            ("quarter_w2", "$t=1/4$", "#b34d28", "^"),
            ("max_node_w2", "Maximum over nodes", "#3e7d54", "s")]:
        axes[0].loglog(ns, [r[column] for r in selected], marker=marker,
                       markersize=3, label=label, color=color, linewidth=1.1)
    axes[0].set(title="(a) Uniform mesh: different orders", xlabel="Euler steps $N$", ylabel="Exact Gaussian $W_2$")
    axes[0].legend(loc="lower left")
    for field, schedule, label, color, marker in [
            ("cosine", "uniform", "Cosine, uniform", "#2667a6", "o"),
            ("cosine", "quantile", "Cosine, quantile", "#b34d28", "s"),
            ("ap_control", "quantile", "AP control, quantile", "#675096", "^")]:
        selected = [r for r in rows if r["field"] == field and r["schedule"] == schedule]
        axes[1].loglog([r["n"] for r in selected], [r["endpoint_w2"] for r in selected],
                       label=label, color=color, marker=marker, markersize=3, linewidth=1.1)
    axes[1].set(title="(b) Endpoint depends on the mesh", xlabel="Euler steps $N$", ylabel="Endpoint $W_2$")
    axes[1].legend(loc="lower left")
    for n, color in [(64, "#a3bfd7"), (256, "#5e8db7"), (4096, "#154d7b")]:
        profile = euler_profile(paths["cosine"], n)
        axes[2].plot(profile["time"], n*profile["w2"], color=color,
                     linewidth=1., label=f"$N={n}$")
    t = np.linspace(0, 1, 1001)
    axes[2].plot(t, paths["cosine"].first_order_profile(t), color="black",
                 linestyle="--", linewidth=.85, label="Predicted limit")
    axes[2].set(title="(c) Interior first-order error", xlabel="Time $t$", ylabel="$N W_2(t)$", xlim=(0, 1))
    axes[2].legend(loc="lower center", ncol=2, fontsize=6., framealpha=.92)
    for ax in axes:
        ax.grid(alpha=.2, linewidth=.45)
    fig.savefig(output/"endpoint_cancellation.pdf")
    plt.close(fig)


def generate(output):
    """Write exact-metric convergence data, independent checks, and one figure."""
    started = time.perf_counter()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    paths = {"cosine": CosineGaussian(), "ap_control": CosineGaussian(epsilon=0.)}
    counts = [2**k for k in range(4, 13)]
    rows, profile_rows, covariance_checks, precision_checks = [], [], [], []
    constants = {}
    profiles = {}
    for name, path in paths.items():
        for schedule in ("uniform", "quantile"):
            key = name+"_"+schedule
            constants[key] = path.constants(schedule)
            previous = None
            for n in counts:
                profile = euler_profile(path, n, schedule)
                quarter = int(np.argmin(np.abs(profile["time"]-.25)))
                peak = int(np.argmax(profile["w2"]))
                row = {"field": name, "schedule": schedule, "n": n,
                       "endpoint_w2": float(profile["w2"][-1]),
                       "quarter_node_time": float(profile["time"][quarter]),
                       "quarter_w2": float(profile["w2"][quarter]),
                       "max_node_time": float(profile["time"][peak]),
                       "max_node_w2": float(profile["w2"][peak]),
                       "endpoint_log_scale_error": float(profile["log_scale_error"][-1])}
                for quantity in ("endpoint", "quarter", "max_node"):
                    error = row[quantity+"_w2"]
                    row[quantity+"_observed_order"] = (math.log2(previous[quantity+"_w2"]/error)
                                                          if previous else "")
                endpoint_order = constants[key]["endpoint_order"]
                row["endpoint_scaled_error"] = n**endpoint_order*row["endpoint_w2"]
                row["endpoint_limit"] = constants[key]["endpoint_coefficient"]
                row["max_node_scaled_error"] = n**constants[key]["max_node_order"]*row["max_node_w2"]
                row["max_node_limit"] = constants[key]["max_node_coefficient"]
                rows.append(row)
                previous = row
                if n == counts[-1]:
                    profiles[key] = profile
                    for i, t in enumerate(profile["time"]):
                        profile_rows.append({"field": name, "schedule": schedule, "n": n,
                                             "time": float(t), "w2": float(profile["w2"][i]),
                                             "log_scale_error": float(profile["log_scale_error"][i]),
                                             "first_order_prediction": float(path.first_order_profile(t, schedule)/n),
                                             "ap_second_order_prediction": (float(path.ap_second_order_profile(t, schedule)/n**2)
                                                                            if path.epsilon == 0 else "")})
                if n in (16, 64, 256):
                    reference = covariance_reference(path, n, schedule)
                    scalar_covariance = profile["euler_scale"][-1]**2*np.eye(2)
                    covariance_checks.append({"field": name, "schedule": schedule, "n": n,
                                              "covariance_max_absolute_difference": float(np.max(np.abs(reference["covariance"]-scalar_covariance))),
                                              "w2_absolute_difference": abs(reference["w2"]-row["endpoint_w2"])})
                if n in (16, 256, 4096):
                    reference = high_precision_reference(path, n, schedule)
                    exact_w2 = float(reference["w2"])
                    precision_checks.append({"field": name, "schedule": schedule, "n": n,
                                             **reference, "double_w2": row["endpoint_w2"],
                                             "relative_w2_difference": abs(row["endpoint_w2"]-exact_w2)/exact_w2,
                                             "absolute_log_difference": abs(row["endpoint_log_scale_error"]-float(reference["log_scale_error"]))})
    _write_csv(output/"convergence.csv", rows)
    _write_csv(output/"profiles.csv", profile_rows)
    _write_csv(output/"covariance_checks.csv", covariance_checks)
    _write_csv(output/"high_precision_checks.csv", precision_checks)
    _plot(output, rows, profiles, paths)
    max_covariance = max(r["covariance_max_absolute_difference"] for r in covariance_checks)
    max_covariance_w2 = max(r["w2_absolute_difference"] for r in covariance_checks)
    max_precision = max(r["relative_w2_difference"] for r in precision_checks)
    validation = {"covariance_max_absolute_difference": max_covariance,
                  "covariance_w2_max_absolute_difference": max_covariance_w2,
                  "high_precision_digits": 100, "high_precision_max_relative_w2_difference": max_precision,
                  "checks_passed": max_covariance < 2e-12 and max_covariance_w2 < 2e-12 and max_precision < 2e-9,
                  "arithmetic_scope": "Independent 100-digit agreement; not an interval enclosure."}
    if not validation["checks_passed"]:
        raise ArithmeticError("Independent validation thresholds failed.")
    finest = [r for r in rows if r["n"] == counts[-1]]
    summary = {"parameters": {"a": .4, "epsilon": .08, "quantile_delta": .2},
               "metric": "Full W2 between centered isotropic two-dimensional Gaussians: sqrt(2)*abs(scale difference).",
               "reference_path": "N(0, exp(2*a*t) I_2), t in [0,1]",
               "mesh": "Uniform or inverse CDF of rho(t)=1+delta*cos(2*pi*t).",
               "constants": constants, "finest_mesh_results": finest, "validation": validation,
               "scope": "Analytic Gaussian counterexample and AP control; no learned model or image-generation benchmark.",
               "runtime_seconds": time.perf_counter()-started,
               "environment": {"python": platform.python_version(), "platform": platform.platform(), "numpy": np.__version__}}
    (output/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    (output/"validation.json").write_text(json.dumps(validation, indent=2)+"\n")
    return summary

