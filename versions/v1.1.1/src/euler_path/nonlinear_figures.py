"""Render the nonlinear and solver figures from primary saved outputs."""

import csv
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .figures import BASE, CORRECTED, save


def rows(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def render(output):
    destination = output / "figures"
    destination.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 10, "font.family": "DejaVu Sans", "pdf.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False})
    summary = json.loads((output / "nonlinear_summary.json").read_text())
    runs = [row for row in rows(output / "nonlinear_runs.csv")
            if int(row["quadrature_order"]) == summary["primary_quadrature_order"] and row["grid"] != "nonuniform"]
    counts = np.array([int(row["steps"]) for row in runs])
    figure, axes = plt.subplots(1, 2, figsize=(8.3, 3.45), layout="constrained")
    for quantity, color, label, marker in (("base_w2", BASE, "Base field", "o"),
                                           ("corrected_w2", CORRECTED, "Corrected field", "s")):
        axes[0].loglog(counts, [float(row[quantity]) for row in runs], color=color, marker=marker, ms=3.5, label=label)
    axes[1].loglog(counts, [float(row["base_w2"]) for row in runs], color=BASE,
                   marker="o", ms=3.5, label="Base field")
    axes[1].loglog(counts, [float(row["corrected_particle_rms"]) for row in runs], color=CORRECTED,
                   marker="s", ms=3.5, label="Corrected field")
    for order in summary["orders"]:
        axis = axes[1] if order["quantity"] == "corrected_particle_rms" else axes[0]
        axis.loglog(counts, order["independent_coefficient"] * (summary["horizon"] / counts)**order["expected_order"],
                    color="0.4", ls=":" if order["expected_order"] == 1 else "--", lw=1,
                    label=f"$N^{{-{order['expected_order']}}}$")
    axes[0].set(ylabel="Terminal Wasserstein distance $W_2$", title="Distribution error")
    axes[1].set(ylabel="Particle error (RMS)", title="Particle error, same start")
    for axis in axes:
        axis.set_xlabel("Euler steps $N$")
        axis.set_xticks([8, 64, 512, 4096], ["8", "64", "512", "4096"])
        axis.grid(which="major", color="0.9", lw=0.6)
        axis.legend(fontsize=8.5, loc="lower left")
    save(figure, destination, "nonlinear_convergence")

    with np.load(output / "nonlinear_shape_data.npz") as data:
        figure, axes = plt.subplots(1, 2, figsize=(8.3, 3.45), layout="constrained")
        colors = plt.colormaps["viridis"](np.linspace(0.1, 0.9, len(data["times"])))
        styles = ("-", "--", "-.", ":")
        markers = ("o", "s", "^", "D", "v")
        for index, (time, color) in enumerate(zip(data["times"], colors)):
            axes[0].plot(data["probabilities"], 100 * (data["normalized_quantiles"][index] / data["normalized_quantiles"][0] - 1), color=color, lw=1.6, ls=styles[index % len(styles)],
                         marker=markers[index % len(markers)], markevery=24, ms=3, label=f"$t={time:g}$")
            axes[1].semilogx(data["source_radii"], data["dilation_ratios"][index], color=color, lw=1.6, ls=styles[index % len(styles)],
                         marker=markers[index % len(markers)], markevery=24, ms=3, label=f"$t={time:g}$")
        axes[0].set(xlabel="Radius quantile probability", ylabel="Change in normalized radius quantile (%)",
                    title="Radial shape change from $t=0$")
        axes[1].set(xlabel="Source radius $r$", ylabel="Exact radius ratio $R(t,r)/r$",
                    title="Expansion depends on radius")
        for axis in axes:
            axis.grid(color="0.9", lw=0.6)
            axis.legend(fontsize=8)
        save(figure, destination, "nonlinear_shape")

    sensitivity = rows(output / "gaussian_sensitivity.csv")
    a = float(json.loads((output / "parameters.json").read_text())["config"]["gaussian"]["rate"])
    figure, axes = plt.subplots(1, 2, figsize=(8.3, 3.65), layout="constrained")
    ratios = sorted(set(float(row["rotation_ratio"]) for row in sensitivity))
    colors = plt.colormaps["tab10"](np.arange(len(ratios)))
    markers = ("o", "s", "^", "D", "v", "P", "X", "<", ">", "h")
    for index, (ratio, color) in enumerate(zip(ratios, colors)):
        marker = markers[index % len(markers)]
        data = [row for row in sensitivity if float(row["rotation_ratio"]) == ratio and row["grid"] != "nonuniform"]
        counts = np.array([int(row["steps"]) for row in data])
        error = np.array([float(row["terminal_w2"]) for row in data])
        axes[0].loglog(counts, error, color=color, lw=1.5, marker=marker, ms=3.5, label=rf"$\lambda={ratio:g}$")
        axes[1].semilogx(counts, error * counts / summary["horizon"], color=color, lw=1.5, marker=marker, ms=3.5)
        if ratio != 1:
            limit = float(data[-1]["predicted_coefficient"])
            axes[1].axhline(limit, color=color, ls=":", lw=0.8)
            axes[1].plot(counts[-1], limit, color=color, marker=marker, ms=3.5)
        if 0 < abs(ratio - 1) <= 0.03:
            crossover = 4 * a * summary["horizon"] / (3 * abs(1 - ratio**2))
            style = ":" if ratio < 1 else "--"
            axes[0].axvline(crossover, color=color, ls=style, lw=0.8, alpha=0.8,
                            label=rf"Approx. $N_*$, $\lambda={ratio:g}$")
            axes[1].axvline(crossover, color=color, ls=style, lw=0.8, alpha=0.8)
    axes[0].set(ylabel="Terminal Wasserstein distance $W_2$", title="Coefficient sensitivity, $\\lambda=b/a$")
    axes[1].set(ylabel="$W_2/h$", title="First-order coefficients")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", fontsize=8, ncol=4)
    for axis in axes:
        axis.set(xlabel="Euler steps $N$")
        axis.set_xticks([8, 64, 512, 8192], ["8", "64", "512", "8192"])
        axis.grid(color="0.9", lw=0.6)
    save(figure, destination, "gaussian_sensitivity")

    solver_runs = rows(output / "solver_dependence.csv")
    figure, axes = plt.subplots(1, 2, figsize=(8.3, 3.65), layout="constrained")
    for axis, solver in zip(axes, ("euler", "heun")):
        for method, color, label, style, marker in (("base", BASE, "$b=0$", "-", "o"),
                                                   ("corrected", CORRECTED, "$b=a$", "-.", "s"),
                                                   ("heun_cubic_cancellation", "#52835b", "$b=a/\\sqrt{3}$", "--", "^")):
            data = [row for row in solver_runs if row["solver"] == solver and row["method"] == method and row["grid"] != "nonuniform"]
            axis.loglog([int(row["steps"]) for row in data], [float(row["terminal_w2"]) for row in data],
                        color=color, ls=style, marker=marker, ms=3.5, lw=1.6, label=label)
        axis.set(xlabel="Steps $N$", ylabel="Terminal Wasserstein distance $W_2$",
                 title="Euler, one NFE per step" if solver == "euler" else "Heun, two NFE per step")
        axis.set_xticks([8, 64, 512, 4096], ["8", "64", "512", "4096"])
        axis.legend(fontsize=9)
        axis.grid(color="0.9", lw=0.6)
    save(figure, destination, "solver_dependence")

    render_equal_evaluation(output)
    return 5


def render_equal_evaluation(output):
    """Plot both metrics and retain all NFE values."""
    (output / "figures").mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "font.family": "DejaVu Sans", "pdf.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False})
    summary = json.loads((output / "equal_evaluation_summary.json").read_text())
    data = [row for row in rows(output / "equal_evaluation.csv")
            if int(row["quadrature_order"]) == summary["primary_quadrature_order"]]
    counts = np.array([int(row["field_evaluations"]) for row in data])
    figure, axes = plt.subplots(1, 2, figsize=(8.3, 3.8), layout="constrained")
    methods = (("base_euler", BASE, "Base field, Euler", "o", "-"),
               ("corrected_euler", CORRECTED, "Corrected field, Euler", "s", "-"),
               ("base_heun", "#52835b", "Base field, Heun", "^", "--"))
    for axis, metric in zip(axes, ("w2", "particle_rms")):
        for method, color, label, marker, style in methods:
            axis.loglog(counts, [float(row[method + "_" + metric]) for row in data],
                        color=color, marker=marker, ms=3.5, ls=style, lw=1.5, label=label)
        axis.set_xlabel("NFE")
        axis.set_xticks([2, 8, 64, 512, 4096], ["2", "8", "64", "512", "4096"])
        axis.grid(which="major", color="0.9", lw=0.6)
        axis.legend(fontsize=8, loc="lower left")
    axes[0].set(ylabel="Terminal Wasserstein distance $W_2$", title="Distribution error")
    axes[1].set(ylabel="Particle error (RMS)", title="Particle error, same start")
    save(figure, output / "figures", "equal_evaluation")
