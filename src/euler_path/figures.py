"""Render figures from saved numerical data."""

import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


BASE = "#2166ac"
CORRECTED = "#c66b24"


def save(figure, output, name):
    figure.savefig(output / (name + ".pdf"), metadata={"Creator": "Matplotlib", "Author": "", "CreationDate": None, "ModDate": None})
    plt.close(figure)


def render(output):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "font.family": "DejaVu Sans", "pdf.fonttype": 42, "axes.labelsize": 11,
                         "legend.frameon": False})
    with (output / "runs.csv").open(newline="") as handle:
        runs = list(csv.DictReader(handle))

    figure, axes = plt.subplots(1, 2, figsize=(8.0, 3.45), layout="constrained")
    for method, color, label, marker in (("base", BASE, "Base field", "o"), ("corrected", CORRECTED, "Corrected field", "s")):
        rows = [row for row in runs if row["family"] == "gaussian" and row["dimension"] == "2" and row["method"] == method and row["grid"] != "nonuniform"]
        rows.sort(key=lambda row: int(row["steps"]))
        counts = np.array([int(row["steps"]) for row in rows])
        axes[0].loglog(counts, [float(row["terminal_w2"]) for row in rows], color=color, marker=marker, ms=3.5, lw=1.5, label=label)
        axes[1].loglog(counts, [float(row["terminal_particle_rms"]) for row in rows], color=color, marker=marker, ms=3.5, lw=1.5, label=label)
    base_rows = [row for row in runs if row["family"] == "gaussian" and row["dimension"] == "2" and row["method"] == "base" and row["grid"] != "nonuniform"]
    base_rows.sort(key=lambda row: int(row["steps"]))
    corrected_rows = [row for row in runs if row["family"] == "gaussian" and row["dimension"] == "2" and row["method"] == "corrected" and row["grid"] != "nonuniform"]
    corrected_rows.sort(key=lambda row: int(row["steps"]))
    guide_counts = np.array([int(row["steps"]) for row in base_rows])
    axes[0].loglog(guide_counts, float(base_rows[-1]["terminal_w2"]) * (guide_counts[-1] / guide_counts), color="0.4", ls=":", lw=1, label="$N^{-1}$")
    axes[0].loglog(guide_counts, float(corrected_rows[-1]["terminal_w2"]) * (guide_counts[-1] / guide_counts)**2, color="0.4", ls="--", lw=1, label="$N^{-2}$")
    axes[1].loglog(guide_counts, float(base_rows[-1]["terminal_particle_rms"]) * (guide_counts[-1] / guide_counts), color="0.4", ls=":", lw=1, label="$N^{-1}$")
    axes[0].set_ylabel("Terminal Wasserstein distance $W_2$")
    axes[1].set_ylabel("Particle error (RMS)")
    axes[0].set_title("Distribution error")
    axes[1].set_title("Particle error, same start")
    for axis in axes:
        axis.set_xlabel("Euler steps $N$")
        axis.set_xticks([1, 16, 256, 4096], ["1", "16", "256", "4096"])
        axis.grid(which="major", color="0.9", lw=0.6)
        axis.legend(fontsize=8.5, loc="lower left")
    save(figure, output, "gaussian_convergence")
