"""Recompute large-bank plot and table values from per-direction results."""
from pathlib import Path
import json
import os

os.environ.setdefault('MPLCONFIGDIR', '/tmp/euler-matplotlib')
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/large_bank'


def main():
    steps = [8, 16, 64]
    seeds = [17, 29, 43]
    labels = {'fm': 'CFM', 'discretization': 'One-grid fit', 'transfer': 'Two-grid fit'}
    groups = []
    for method, label in labels.items():
        values = []
        for seed in seeds:
            record = json.loads((OUT / f'd2_{method}_s{seed}_w48.json').read_text())
            row = []
            for n in steps:
                result = record['resolutions'][str(n)]
                assert [t['time'] for t in result['times']] == [0, .25, .5, .75, 1]
                time_values = []
                for time in result['times']:
                    projected = np.asarray(time['projection_squared_errors'])
                    assert projected.shape == (128,)
                    value = float(np.sqrt(projected.mean()))
                    assert np.isclose(value, time['sw2'], rtol=0, atol=1e-14)
                    time_values.append(value)
                value = max(time_values)
                assert np.isclose(value, result['maximum_over_times']['sw2'], rtol=0, atol=1e-14)
                row.append(value)
            values.append(row)
        values = np.asarray(values)
        groups.append({'method': method, 'label': label, 'seeds': seeds,
                       'steps': steps, 'per_seed': values.tolist(),
                       'mean': values.mean(axis=0).tolist(),
                       'sample_std': values.std(axis=0, ddof=1).tolist()})

    (OUT / 'plot_summary.json').write_text(json.dumps({
        'samples': 262144, 'directions': 128, 'times': [0, .25, .5, .75, 1],
        'aggregation': 'Maximum over times for each model, then mean and sample standard deviation across three training seeds.',
        'groups': groups}, indent=2) + '\n')
    lines = [r'\begin{table}[t]', r'\centering',
             r'\caption{Two-dimensional comparisons on the 262,144-sample evaluation set. Entries are means with sample standard deviations in parentheses below, across three training seeds per objective. Subscripts give Euler step counts. SW2 uses 128 directions and five diagnostic times. Lower values are better.}',
             r'\label{tab:sample-only}', r'\small',
             r'\begin{tabular}{@{}lrrr@{}}', r'\toprule',
             r'Training & SW2$_8$ & SW2$_{16}$ & SW2$_{64}$\\', r'\midrule']
    for group in groups:
        cells = [r'\shortstack{$' + f'{mean:.5f}' + r'$\\$(' + f'{sd:.5f}' + r')$}'
                 for mean, sd in zip(group['mean'], group['sample_std'])]
        lines.append(group['label'] + ' & ' + ' & '.join(cells) + r'\\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    (OUT / 'main_table.tex').write_text('\n'.join(lines) + '\n')

    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.1), sharey=True)
    for ax, group in zip(axes, groups):
        for seed, values, marker in zip(group['seeds'], group['per_seed'], ('o', 's', '^')):
            ax.plot(steps, values, marker=marker, markersize=3, label=str(seed))
        ax.set_xscale('log', base=2)
        ax.set_xticks(steps, labels=[str(n) for n in steps])
        ax.set_title(group['label'], fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(axis='y', color='0.9')
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].set_ylabel('Empirical SW2', fontsize=9)
    axes[1].set_xlabel('Euler steps', fontsize=9)
    axes[2].legend(title='Training seed', fontsize=7, title_fontsize=7, frameon=False)
    fig.tight_layout(pad=0.5)
    fig.savefig(OUT / 'large_bank_d2.pdf', bbox_inches='tight',
                metadata={'CreationDate': None, 'ModDate': None})
    fig.savefig(OUT / 'large_bank_d2.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print('Verified 135 per-time metrics from 128 projection errors each; generated the large-bank figure and main table.')


if __name__ == '__main__':
    main()
