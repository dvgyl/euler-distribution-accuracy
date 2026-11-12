#!/usr/bin/env python3
"""Draw the exact Gaussian contour and particle comparison in Figure 1."""

from pathlib import Path

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "paper" / "figures" / "particle_distribution.pdf"
A = 0.6
H = 1.0


def render_contours(output):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "pdf.fonttype": 42, "axes.spines.top": False,
                         "axes.spines.right": False})
    angles = np.linspace(-0.1, np.pi / 2, 500)
    unit = np.column_stack((np.cos(angles), np.sin(angles)))
    exact_scale = np.exp(A * H)
    time = np.linspace(0, H, 200)
    source = np.array([1.0, 0.0])
    label_box = {"facecolor": "white", "edgecolor": "none", "pad": 1.5}
    figure, axes = plt.subplots(1, 2, figsize=(8.1, 3.05), sharex=True, sharey=True)
    for axis, rotation, title, color in zip(
            axes, (0.0, A), ("Expansion", "Expansion with rotation"),
            ("#2166ac", "#c66b24")):
        numerical = np.array([1 + A * H, rotation * H])
        exact = exact_scale * np.array([np.cos(rotation * H), np.sin(rotation * H)])
        paired = exact_scale * numerical / np.linalg.norm(numerical)
        trajectory = np.exp(A * time[:, None]) * np.column_stack(
            (np.cos(rotation * time), np.sin(rotation * time)))
        numerical_scale = np.linalg.norm(numerical)
        contour_gap = exact_scale - numerical_scale
        particle_gap = np.linalg.norm(numerical - exact)
        axis.plot(*(exact_scale * unit).T, color="0.15", lw=1.8)
        axis.plot(*(np.linalg.norm(numerical) * unit).T, color=color, lw=1.6, ls="--")
        axis.plot(*trajectory.T, color="0.6", lw=1.1, ls=":")
        axis.annotate("", numerical, source,
                      arrowprops={"arrowstyle": "->", "color": color, "lw": 1.4})
        axis.plot(*np.column_stack((numerical, exact)), color="#71478c", lw=2)
        if rotation:
            axis.plot(*np.column_stack((numerical, paired)), color="#287660", lw=3)
            axis.scatter(*paired, facecolors="white", edgecolors="#287660", s=28, zorder=6)
            axis.annotate(f"Contour gap {contour_gap:.3f}", (numerical + paired) / 2, (2.43, 0.08),
                          fontsize=9, ha="center", color="#287660", bbox=label_box,
                          arrowprops={"arrowstyle": "-", "color": "#287660", "lw": 0.7})
            axis.annotate(f"Particle gap {particle_gap:.3f}", (numerical + exact) / 2, (2.43, 1.31),
                          fontsize=9, ha="center", color="#71478c", bbox=label_box,
                          arrowprops={"arrowstyle": "-", "color": "#71478c", "lw": 0.7})
        else:
            axis.annotate(f"Particle gap =\ncontour gap {particle_gap:.3f}", (numerical + exact) / 2, (2.43, 0.18),
                          fontsize=9, ha="center", color="#71478c", bbox=label_box,
                          arrowprops={"arrowstyle": "-", "color": "#71478c", "lw": 0.7})
        axis.annotate(f"Exact radius {exact_scale:.3f}", exact_scale * np.array([np.cos(0.75), np.sin(0.75)]),
                      (2.43, 1.02), fontsize=9, ha="center", color="0.15", bbox=label_box,
                      arrowprops={"arrowstyle": "-", "color": "0.15", "lw": 0.7})
        axis.annotate(f"Euler radius {numerical_scale:.3f}", numerical_scale * np.array([np.cos(0.35), np.sin(0.35)]),
                      (2.43, 0.76), fontsize=9, ha="center", color=color, bbox=label_box,
                      arrowprops={"arrowstyle": "-", "color": color, "lw": 0.7})
        axis.scatter(*source, facecolors="white", edgecolors="0.3", s=29, zorder=6)
        axis.scatter(*numerical, color=color, s=28, zorder=7)
        axis.scatter(*exact, color="0.15", marker="s", s=25, zorder=7)
        axis.text(0.96, -0.12, "Start", ha="center", fontsize=9)
        axis.set(title=title, xlim=(0.68, 3.02), ylim=(-0.2, 1.52), aspect="equal")
        axis.set_xticks([])
        axis.set_yticks([])
        axis.spines[["left", "bottom"]].set_visible(False)
    figure.legend([axes[0].lines[0], axes[0].lines[1]],
                  ["Exact distribution contour", "Euler distribution contour"],
                  loc="lower center", ncol=2, frameon=False, fontsize=9)
    figure.subplots_adjust(left=0.025, right=0.975, bottom=0.19, top=0.86, wspace=0.2)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, metadata={"Creator": "Matplotlib", "Author": "",
                                    "CreationDate": None, "ModDate": None})
    plt.close(figure)



def render_coupling(output):
    """Draw the probability-preserving pairing used in the local bound."""
    figure, axis = plt.subplots(figsize=(8.1, 2.6))
    axis.set(xlim=(0, 1), ylim=(0, 1))
    axis.axis("off")
    blue, orange, green = "#2166ac", "#c66b24", "#287660"
    box = {"boxstyle": "round,pad=0.5", "facecolor": "white", "edgecolor": "0.8"}
    axis.text(0.12, 0.78, "$X\\sim p_t$", ha="center", va="center", bbox=box, fontsize=12)
    axis.text(0.12, 0.18, "$\\Psi_t^{h^2/2}X\\sim p_t$", ha="center", va="center", bbox=box, fontsize=12)
    axis.text(0.84, 0.78, "$F_{t,t+h}(X)$", ha="center", va="center", bbox=box, fontsize=12)
    axis.text(0.84, 0.18, "$E_{t,h}(\\Psi_t^{h^2/2}X)$", ha="center", va="center", bbox=box, fontsize=12)
    for start,end,color in (((0.24,0.78),(0.72,0.78),blue),((0.28,0.18),(0.68,0.18),orange),((0.12,0.65),(0.12,0.32),green)):
        axis.annotate("", end, start, arrowprops={"arrowstyle": "->", "lw": 1.5, "color": color})
    axis.text(0.47, 0.87, "Exact flow", ha="center", color=blue, fontsize=11)
    axis.text(0.47, 0.27, "Euler step", ha="center", color=orange, fontsize=11)
    axis.text(0.18, 0.49, "Shift along fixed-time acceleration\nSame starting distribution", va="center", color=green, fontsize=10)
    axis.annotate("", (0.84,0.64), (0.84,0.32), arrowprops={"arrowstyle": "<->", "lw": 1.5, "color": "0.3"})
    axis.text(0.79, 0.48, "Root mean squared\ngap $O(h^3)$", ha="right", va="center", fontsize=10)
    figure.subplots_adjust(left=0.02, right=0.98, bottom=0.04, top=0.98)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, metadata={"Creator": "Matplotlib", "Author": "", "CreationDate": None, "ModDate": None})
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description="Regenerate the two explanatory figures.")
    parser.add_argument("--output", type=Path, default=ROOT / "build" / "illustrations")
    args = parser.parse_args()
    render_contours(args.output / "particle_distribution.pdf")
    render_coupling(args.output / "acceleration_coupling.pdf")
    print("Rendered the contour comparison and acceleration coupling.")


if __name__ == "__main__":
    main()
