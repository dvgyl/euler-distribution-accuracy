#!/usr/bin/env python3
"""Measure population W2 along the three-dimensional mixture path."""
import csv
import json
from math import exp, pi, sqrt
from pathlib import Path

import numpy as np
from scipy.special import ndtr, roots_legendre

SIGMA = 0.5
MU = 1.5
RADIUS = 9
STEPS = (8, 16, 32, 64, 128, 256)
METHODS = ("conditional", "planar_skew")
ORDERS = (512, 1024)
OUTPUT = Path(__file__).resolve().parents[1] / "results" / "population_check"


def scale(time):
    return np.hypot(1 - time, SIGMA * time)


def target_quantiles(times, source):
    values = np.empty((len(times), len(source)))
    values[0] = source
    survival = ndtr(-source)
    largest_residual = 0.0
    for index, time in enumerate(times[1:], 1):
        spread = scale(time)
        center = MU * time
        low = spread * source - center
        high = spread * source + center
        for _ in range(54):
            midpoint = (low + high) / 2
            mass = (ndtr((center - midpoint) / spread)
                    + ndtr((-center - midpoint) / spread)) / 2
            too_low = mass > survival
            low = np.where(too_low, midpoint, low)
            high = np.where(too_low, high, midpoint)
        quantile = (low + high) / 2
        values[index] = quantile
        mass = (ndtr((center - quantile) / spread)
                + ndtr((-center - quantile) / spread)) / 2
        largest_residual = max(largest_residual, np.max(abs(mass / survival - 1)))
    return values, float(largest_residual)


def euler(steps, source):
    h = 1 / steps
    times = np.arange(steps + 1) * h
    mixture = np.empty((steps + 1, len(source)))
    conditional = np.empty(steps + 1)
    skew = np.empty(steps + 1, dtype=complex)
    mixture[0], conditional[0], skew[0] = source, 1, 1
    for k, t in enumerate(times[:-1]):
        variance = scale(t) ** 2
        linear = ((1 + SIGMA**2) * t - 1) / variance
        amplitude = MU * (1 - t) / variance
        frequency = MU * t / variance
        multiplier = 1 + h * linear
        assert multiplier > 0
        mixture[k + 1] = multiplier * mixture[k] + h * amplitude * np.tanh(frequency * mixture[k])
        conditional[k + 1] = multiplier * conditional[k]
        skew[k + 1] = (multiplier + 1j * h * SIGMA / variance) * skew[k]
    return mixture, conditional, abs(skew)


def tail_squared_bound():
    # At each step |linear| is at most L and |amplitude| is at most b.
    lip = (1 + SIGMA**2) / (2 * SIGMA)
    amplitude = MU * (1 + SIGMA**2) / SIGMA**2
    a = exp(lip) + 1
    b = amplitude * exp(lip) + MU
    normal_tail = ndtr(-RADIUS)
    density = exp(-RADIUS**2 / 2) / sqrt(2 * pi)
    return float(2 * (a**2 * (RADIUS * density + normal_tail)
                      + 2 * a * b * density + b**2 * normal_tail))


def measure(order):
    nodes, weights = roots_legendre(order)
    source = (nodes + 1) * RADIUS / 2
    weights *= RADIUS * np.exp(-source**2 / 2) / sqrt(2 * pi)
    # The source law, target quantiles and scalar Euler maps are symmetric.
    base = STEPS[-1]
    target, residual = target_quantiles(np.arange(base + 1) / base, source)
    rows = []
    maxima = {method: [] for method in METHODS}
    peak_planar_error = 0.0
    for steps in STEPS:
        expected = target[::base // steps]
        times = np.arange(steps + 1) / steps
        spread = scale(times)
        by_method = {method: [] for method in METHODS}
        for factor in (1, 2):
            mixture, conditional, skew = (x[::factor] for x in euler(steps * factor, source))
            scalar_sq = (mixture - expected)**2 @ weights
            peak_planar_error = max(peak_planar_error, float(max(abs(skew - spread))))
            for method, planar in zip(METHODS, (conditional, skew)):
                planar_sq = 2 * (planar - spread)**2
                error = np.sqrt(scalar_sq + planar_sq)
                by_method[method].append(float(max(error)))
                for k, t in enumerate(times):
                    rows.append((order, method, steps, steps * factor, t,
                                 scalar_sq[k], planar_sq[k], error[k]))
        for method, (coarse, refined) in by_method.items():
            maxima[method].append({"steps": steps, "coarse": coarse,
                                   "refined": refined, "sum": coarse + refined})
    return rows, maxima, residual, peak_planar_error


def main():
    first, second = [measure(order) for order in ORDERS]
    maximum_relative_difference = 0.0
    for method in METHODS:
        for a, b in zip(first[1][method], second[1][method]):
            maximum_relative_difference = max(maximum_relative_difference,
                                              abs(a["sum"] / b["sum"] - 1))
        for a, b in zip(second[1][method], second[1][method][1:]):
            b["local_order"] = float(np.log2(a["sum"] / b["sum"]))
    assert maximum_relative_difference < 1e-5
    assert max(first[2], second[2]) < 1e-10
    assert max(first[3], second[3]) < 5e-14
    tail = tail_squared_bound()
    min_error = min(row["refined"] for method in METHODS for row in second[1][method])
    assert tail < 1e-6 * min_error**2
    OUTPUT.mkdir(exist_ok=True)
    with (OUTPUT / "node_errors.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("quadrature_order", "field", "coarse_steps", "actual_steps",
                         "time", "scalar_w2_squared", "planar_w2_squared", "w2"))
        writer.writerows(first[0])
        writer.writerows(second[0])
    summary = {
        "dimension": 3, "mu": MU, "sigma": SIGMA, "max_initial_coordinate": RADIUS,
        "quadrature_orders": ORDERS, "largest_relative_quantile_residual": max(first[2], second[2]),
        "largest_relative_sum_difference": maximum_relative_difference,
        "largest_planar_control_scale_error": max(first[3], second[3]),
        "scalar_squared_tail_bound": tail, "maxima_at_finer_quadrature": second[1],
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
