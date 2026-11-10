"""Bound the derivative of each scalar Euler update from the saved weights."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


def affine_bounds(weights, bias, lower, upper):
    positive = np.maximum(weights, 0)
    negative = np.minimum(weights, 0)
    return (
        lower @ positive.T + upper @ negative.T + bias,
        upper @ positive.T + lower @ negative.T + bias,
    )


def tanh_derivative_bounds(lower, upper):
    nearest = np.minimum(np.abs(lower), np.abs(upper))
    nearest[(lower <= 0) & (upper >= 0)] = 0
    farthest = np.maximum(np.abs(lower), np.abs(upper))
    return 1 - np.tanh(farthest) ** 2, 1 - np.tanh(nearest) ** 2


def velocity_derivative_lower_bound(left, right, time, weights):
    first, first_bias, second, second_bias, output = weights
    x_weight, t_weight = first.T
    offset = t_weight * time + first_bias
    left, right = left[:, None], right[:, None]
    with np.errstate(invalid="ignore"):
        lower = np.where(x_weight >= 0, left, right) * x_weight
        upper = np.where(x_weight >= 0, right, left) * x_weight
    lower = np.where(x_weight == 0, 0, lower) + offset
    upper = np.where(x_weight == 0, 0, upper) + offset
    slope_low, slope_high = tanh_derivative_bounds(lower, upper)
    derivative_low = np.minimum(slope_low * x_weight, slope_high * x_weight)
    derivative_high = np.maximum(slope_low * x_weight, slope_high * x_weight)

    second_low, second_high = affine_bounds(
        second, second_bias, np.tanh(lower), np.tanh(upper)
    )
    slope_low, slope_high = tanh_derivative_bounds(second_low, second_high)
    derivative_low, derivative_high = affine_bounds(
        second, 0, derivative_low, derivative_high
    )
    derivative_low = np.minimum(
        derivative_low * slope_low, derivative_low * slope_high
    )
    derivative_high = np.maximum(
        derivative_high * slope_low, derivative_high * slope_high
    )
    return affine_bounds(output, 0, derivative_low, derivative_high)[0].ravel()


def euler_update_derivative_lower_bound(steps, weights):
    edges = np.r_[-np.inf, np.linspace(-64, 64, 513), np.inf]
    minimum = np.inf
    for node in range(steps):
        left, right = edges[:-1], edges[1:]
        for _ in range(15):
            derivative_lower = velocity_derivative_lower_bound(
                left, right, node / steps, weights
            )
            margin = 1 + derivative_lower / steps
            safe = margin > 0.01
            minimum = min(minimum, margin[safe].min(initial=np.inf))
            if safe.all():
                break
            left, right = left[~safe], right[~safe]
            if not (np.isfinite(left).all() and np.isfinite(right).all()):
                raise ValueError("An unbounded interval did not pass the check.")
            middle = (left + right) / 2
            left, right = np.r_[left, middle], np.r_[middle, right]
        else:
            raise ValueError("Interval refinement did not pass the check.")
    return float(minimum)


def check_checkpoint(checkpoint):
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    weights = [
        state[key].numpy()
        for key in (
            "net.0.weight", "net.0.bias", "net.2.weight", "net.2.bias", "net.4.weight"
        )
    ]
    first, _, second, _, output = weights
    path_products = output.T * second * first[:, 0]
    global_velocity_derivative_lower = float(np.minimum(path_products, 0).sum())
    margins = {
        str(steps): euler_update_derivative_lower_bound(steps, weights)
        for steps in (4, 8, 16)
    }
    margins["32_or_more"] = 1 + global_velocity_derivative_lower / 32
    if min(margins.values()) <= 0.01:
        raise ValueError(f"The derivative bound for {checkpoint.name} is too low.")
    return {
        "checkpoint": checkpoint.name,
        "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "velocity_derivative_lower_bound_all_times": global_velocity_derivative_lower,
        "euler_update_derivative_lower_bounds": margins,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    args = parser.parse_args()
    directory = args.source / "checkpoints" / "scalar_population"
    checkpoints = sorted(
        directory / f"{objective}_s{seed}.pt"
        for objective in ("cfm", "one_grid", "two_grid")
        for seed in range(10)
    )
    rows = [check_checkpoint(checkpoint) for checkpoint in checkpoints]
    print(json.dumps({
        "arithmetic": "float64 with a 0.01 acceptance margin; no directed rounding",
        "checked_fields": len(rows),
        "minimum_update_derivative_lower_bound": min(
            min(row["euler_update_derivative_lower_bounds"].values()) for row in rows
        ),
        "rows": rows,
    }, indent=2))


if __name__ == "__main__":
    main()
