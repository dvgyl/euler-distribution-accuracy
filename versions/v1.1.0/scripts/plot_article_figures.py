"""Draw the figures of the article from saved results.

Inputs: results/verification/claims.json, results/scalar_population/*.json,
results/evaluation_set_variability/*.json. Outputs: paper/figures/*.pdf and *.png.
"""
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / 'paper' / 'figures'
BLUE, ORANGE, AQUA, INK, MUTED = '#2a78d6', '#eb6834', '#1baf7a', '#1a1a19', '#6b6a64'

plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['Times New Roman', 'STIXGeneral', 'DejaVu Serif'],
    'mathtext.fontset': 'stix', 'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8,
    'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5, 'legend.fontsize': 7.5,
    'axes.linewidth': 0.6, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK,
    'xtick.color': MUTED, 'ytick.color': MUTED, 'xtick.labelcolor': INK, 'ytick.labelcolor': INK,
    'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'xtick.minor.width': 0.4, 'ytick.minor.width': 0.4,
    'axes.spines.top': False, 'axes.spines.right': False,
    'grid.color': '#d9d8d2', 'grid.linewidth': 0.4, 'lines.linewidth': 1.3, 'lines.markersize': 4,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'savefig.dpi': 600,
})


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f'{name}.pdf', bbox_inches='tight', pad_inches=0.02, metadata={'CreationDate': None, 'ModDate': None})
    fig.savefig(FIG / f'{name}.png', bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)


def panel_label(ax, letter, text):
    ax.set_title(f'({letter}) {text}', loc='left', pad=4, color=INK)


def figure_rates():
    claims = json.loads((ROOT / 'results' / 'verification' / 'claims.json').read_text())
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.25), gridspec_kw={'wspace': 0.36})

    # (a) One Gaussian coordinate: a field fitted to one step of 1/2, then run with two steps of 1/4.
    ax = axes[0]
    g = lambda t: math.sqrt((1 - t) ** 2 + t ** 2)
    coef = lambda t, h=0.5: (g(t + h) - g(t)) / (h * g(t))
    t = np.linspace(0, 0.5, 101)
    ax.plot(t, [math.exp(quad(coef, 0, s)[0]) for s in t], color=MUTED, lw=1.0, ls=(0, (4, 2)),
            label='exact flow of the field')
    half = 1 + 0.25 * coef(0.0)
    ax.plot([0, 0.25, 0.5], [1, half, half * (1 + 0.25 * coef(0.25))], color=ORANGE, marker='s',
            label='Euler, two steps')
    ax.plot([0, 0.5], [1, 1 + 0.5 * coef(0.0)], color=BLUE, marker='o', label='Euler, one step')
    ax.plot(t, [g(s) for s in t], color=INK, lw=1.0, label='target path')
    ax.set_xlabel('time $t$')
    ax.set_ylabel('standard deviation')
    ax.set_xticks([0, 0.25, 0.5])
    ax.set_xticklabels(['0', '0.25', '0.5'])
    ax.set_xlim(-0.02, 0.84)
    ax.set_ylim(0.66, 1.08)
    ax.text(0.30, 0.722, 'target path', ha='right', va='top', color=INK)
    for y, text in ((math.exp(quad(coef, 0, 0.5)[0]), 'exact flow\nof the field'), (half, 'Euler,\ntwo steps'),
                    (g(0.5), 'Euler, one step\n(on the target)')):
        ax.text(0.525, y, text, ha='left', va='center', color=INK, linespacing=0.95)
    panel_label(ax, 'a', 'a fit at one step size')

    # (b) Scalar Gaussian path: scaled two-grid error and its limits.
    ax = axes[1]
    limit = (2 + math.pi) / 8
    rows = claims['sharp_constant_rows']['2']
    ax.axhline(limit, color=MUTED, lw=0.6, ls=(0, (2, 2)))
    ax.axhline(3 * limit, color=MUTED, lw=0.6, ls=(0, (2, 2)))
    ax.plot([r['N'] for r in claims['exact_generator_rows']], [r['N_times_sum'] for r in claims['exact_generator_rows']],
            color=ORANGE, marker='s')
    ax.plot([r['N'] for r in rows], [r['N_times_sum'] for r in rows], color=BLUE, marker='o')
    ax.set_xscale('log', base=2)
    ax.set_xticks([8, 64, 512, 4096])
    ax.set_xticklabels(['8', '64', '512', '4096'])
    ax.set_ylim(0, 2.45)
    ax.set_xlabel('coarse step count $N$')
    ax.set_ylabel('$N\\,(E_c+E_f)$')
    ax.text(4096, 3 * limit + 0.07, 'exact generator: $3(2+\\pi)/8$', ha='right', va='bottom', color=INK)
    ax.text(4096, limit + 0.07, 'secant field: $(2+\\pi)/8$', ha='right', va='bottom', color=INK)
    panel_label(ax, 'b', 'scalar Gaussian path')

    # (c) First-order and second-order decay of the two-grid error.
    ax = axes[2]
    mix = claims['mixture_rows']
    rot = [r for r in claims['rotation_rows'] if r['N'] <= 256]
    n = np.array([r['N'] for r in mix])
    cond = np.array([r['conditional_E_c'] + r['conditional_E_f'] for r in mix])
    skew = np.array([r['skew_E_c'] + r['skew_E_f'] for r in mix])
    rotv = np.array([r['E_c'] + r['E_f'] for r in rot])
    ax.plot(n, cond, color=ORANGE, marker='s')
    ax.plot(n, skew, color=AQUA, marker='^')
    ax.plot(n, rotv, color=BLUE, marker='o')
    ax.plot(n, 0.6 * skew[0] * n[0] / n, color=MUTED, lw=0.6, ls=(0, (2, 2)))
    ax.plot(n, 0.5 * rotv[0] * (n[0] / n) ** 2, color=MUTED, lw=0.6, ls=(0, (2, 2)))
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.set_yticks([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1])
    ax.set_xticks([8, 16, 32, 64, 128, 256])
    ax.set_xticklabels(['8', '16', '32', '64', '128', '256'])
    ax.set_xlabel('coarse step count $N$')
    ax.set_ylabel('$E_c+E_f$')
    ax.set_ylim(3e-6, 60)
    handles = [plt.Line2D([], [], color=ORANGE, marker='s', label='mixture, conditional field'),
               plt.Line2D([], [], color=AQUA, marker='^', label='mixture, skew field'),
               plt.Line2D([], [], color=BLUE, marker='o', label='rotation field')]
    ax.legend(handles=handles, frameon=False, loc='upper right', handlelength=2.0, borderaxespad=0.0, labelspacing=0.25)
    ax.text(256, 0.6 * skew[0] * n[0] / 256 * 0.6, 'slope $-1$', ha='right', va='top', color=MUTED)
    ax.text(30, 0.5 * rotv[0] * (n[0] / 30) ** 2 * 0.75, 'slope $-2$', ha='right', va='top', color=MUTED)
    panel_label(ax, 'c', 'first and second order')
    save(fig, 'fig1_rates')


def figure_learned():
    models = json.loads((ROOT / 'results' / 'scalar_population' / 'models.json').read_text())
    sets = json.loads((ROOT / 'results' / 'evaluation_set_variability' / 'sets.json').read_text())
    steps = [4, 8, 16, 32, 64, 128, 256]
    style = {'cfm': (BLUE, 'o', 'CFM'), 'one_grid': (ORANGE, 's', 'one-grid fit'), 'two_grid': (AQUA, '^', 'two-grid fit')}
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.25), gridspec_kw={'wspace': 0.36})
    for ax, key, title, letter in ((axes[0], 'target_w2', 'target error', 'a'),
                                   (axes[1], 'own_flow_rms', 'integration error', 'b')):
        exact = next(m for m in models if m['objective'] == 'exact')
        ax.plot(steps, [exact['shared'][str(n)][key] for n in steps], color=INK, lw=1.0, ls=(0, (4, 2)),
                label='exact conditional field')
        for objective, (color, marker, label) in style.items():
            group = [m for m in models if m['objective'] == objective]
            values = np.array([[m['shared'][str(n)][key] for n in steps] for m in group])
            for row in values:
                ax.plot(steps, row, color=color, lw=0.5, alpha=0.3)
            ax.plot(steps, values.mean(axis=0), color=color, marker=marker, label=label)
        ax.set_xscale('log', base=2)
        ax.set_yscale('log')
        ax.set_xticks(steps)
        ax.set_xticklabels([str(n) for n in steps])
        ax.set_xlabel('Euler steps $N$')
        ax.set_ylabel(title + r' ($W_2$)' if key == 'target_w2' else title + ' (RMS)')
        panel_label(ax, letter, 'learned scalar fields: ' + title if letter == 'a' else title)
    axes[1].legend(frameon=False, loc='upper right', handlelength=2.2, borderaxespad=0.1, labelspacing=0.25)

    # (c) Change of the empirical SW2 between 8 and 16 steps on independent evaluation sets.
    ax = axes[2]
    fitted = [f'd2_{method}_s{seed}_w48' for method in ('discretization', 'transfer') for seed in (17, 29, 43)]
    sizes = sorted({r['size'] for r in sets})
    rng = np.random.default_rng(0)
    for position, size in enumerate(sizes):
        change = np.array([np.mean([r[f'{n}_16'] - r[f'{n}_8'] for n in fitted]) for r in sets if r['size'] == size])
        ax.scatter(position + rng.uniform(-0.22, 0.22, len(change)), change, s=7, color=BLUE, alpha=0.55,
                   linewidths=0, zorder=2)
        ax.plot([position - 0.3, position + 0.3], [change.mean()] * 2, color=INK, lw=1.0, zorder=3)
    table = {}
    for line in (ROOT / 'results' / 'large_bank' / 'large_bank_metrics.csv').read_text().splitlines()[1:]:
        c = line.split(',')
        if c[0] != 'fm':
            table[c[0], c[1]] = (float(c[4]) - float(c[2]), float(c[5]) - float(c[3]))
    small = np.mean([v[0] for v in table.values()])
    large = np.mean([v[1] for v in table.values()])
    sizes = sorted(set(sizes) | {262144})
    initial = [(sizes.index(size), value) for size, value in ((1024, small), (262144, large))]
    ax.scatter([x for x, _ in initial], [y for _, y in initial], s=26, marker='D', color=ORANGE, edgecolors='white',
               linewidths=0.6, zorder=4, label='separate sets')
    ax.axhline(0, color=MUTED, lw=0.6, ls=(0, (2, 2)), zorder=1)
    ax.set_xticks(range(len(sizes)))
    ax.set_xticklabels([f'{size:,}' for size in sizes])
    ax.set_xlim(-0.5, len(sizes) - 0.5)
    ax.set_xlabel('samples in one evaluation set')
    ax.set_ylabel('change of SW2, 8 to 16 steps')
    ax.legend(frameon=False, loc='lower right', handletextpad=0.2, borderaxespad=0.1)
    panel_label(ax, 'c', 'sample-based comparison')
    save(fig, 'fig2_learned')


if __name__ == '__main__':
    figure_rates()
    figure_learned()
