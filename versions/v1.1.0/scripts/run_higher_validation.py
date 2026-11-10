#!/usr/bin/env python3
"""Reproduce the uniform-barrier, loss-cost and model-bias sharpness checks."""
from __future__ import annotations

import csv
from fractions import Fraction
import json
from pathlib import Path
import platform
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial.legendre import leggauss

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from euler_path.higher_barriers import (
    complex_structure, gaussian_coefficients, gaussian_euler_covariance,
    even_product_errors, gaussian_risk_frontier, distorted_grid,
    odd_budget_field, secant_coefficient, secant_terminal_bias,
    rational_gaussian_certificate, mixture_loss_check, mixture_euler_errors,
)
OUT = ROOT / 'results' / 'higher'


def write_csv(name, rows):
    with (OUT / name).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def save_plot(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / (name + '.pdf'), bbox_inches='tight')
    fig.savefig(OUT / (name + '.png'), dpi=180, bbox_inches='tight')
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    grids = [16, 32, 64, 128, 256, 512, 1024, 2048]
    rows = []
    for beta in [0., .5, 1.]:
        for n in grids:
            alpha = 1 - n**(-beta)
            errors = even_product_errors(np.linspace(0, 1, n + 1), alpha)
            rows.append({'beta': beta, 'N': n, 'alpha_N': alpha,
                         'maximum_node_W2': float(errors.max()),
                         'terminal_W2': float(errors[-1]),
                         'N_power_scaled_maximum': float(errors.max() * n**(1+beta)),
                         'metric': 'population Gaussian W2; floating point'})
    write_csv('even_budget_rates.csv', rows)
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    for beta in [0., .5, 1.]:
        data = [r for r in rows if r['beta'] == beta]
        ax.loglog(grids, [r['maximum_node_W2'] for r in data], marker='o',
                  label=rf'$\beta={beta:g}$: order ${1+beta:g}$')
    ax.set(xlabel='Euler steps N', ylabel='Maximum-node population W2',
           title=r'Even dimension: $\alpha_N=1-N^{-\beta}$')
    ax.legend(fontsize=9); ax.grid(True, which='both', alpha=.2)
    save_plot(fig, 'even_budget_orders')

    odd_rows = []
    for n in grids:
        for name, distortion in [('uniform', 0.), ('fixed_distortion', .45),
                                  ('budget_dependent', .4*np.cos(np.log(n)))]:
            grid = distorted_grid(n, distortion)
            errors, _ = gaussian_euler_covariance(odd_budget_field(n), grid, 3)
            odd_rows.append({'schedule': name, 'N': n, 'distortion': float(distortion),
                             'maximum_node_W2': float(errors.max()),
                             'terminal_W2': float(errors[-1]),
                             'N_scaled_maximum': float(n*errors.max()),
                             'largest_step': float(np.diff(grid).max()),
                             'metric': 'population Gaussian W2; covariance recursion'})
    write_csv('odd_budget_grids.csv', odd_rows)
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    for name in ['uniform', 'fixed_distortion', 'budget_dependent']:
        data = [r for r in odd_rows if r['schedule'] == name]
        ax.semilogx(grids, [r['N_scaled_maximum'] for r in data], marker='o',
                    label=name.replace('_', ' '))
    ax.set(xlabel='Euler steps N', ylabel='N × maximum-node population W2',
           title='Odd dimension: fields and grids vary with N')
    ax.legend(fontsize=9); ax.grid(True, alpha=.2)
    save_plot(fig, 'odd_budget_scaled')

    bias_rows = []
    for n in grids:
        grid = np.linspace(0, 1, n+1)
        errors, _ = gaussian_euler_covariance(
            lambda t: secant_coefficient(t, 1/n) * np.eye(3), grid, 3)
        bias, coefficient, quad_estimate = secant_terminal_bias(n)
        canonical, _ = gaussian_euler_covariance(
            lambda t: gaussian_coefficients(t)[0] * np.eye(3), grid, 3)
        bias_rows.append({'N': n, 'secant_target_maximum_roundoff': float(errors.max()),
                          'secant_target_exact_error': 0,
                          'secant_continuous_terminal_bias': bias,
                          'N_scaled_terminal_bias': n*bias,
                          'terminal_bias_limit_coefficient': coefficient,
                          'quad_error_estimate_not_certificate': quad_estimate,
                          'canonical_maximum_target_error': float(canonical.max()),
                          'canonical_exact_path_bias': 0,
                          'scope': 'terminal continuous bias; maximum-node target error'})
    write_csv('path_bias_frontier.csv', bias_rows)
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.loglog(grids, [r['secant_continuous_terminal_bias'] for r in bias_rows], marker='o',
              label='Grid-fitted field: continuous terminal bias')
    ax.loglog(grids, [r['canonical_maximum_target_error'] for r in bias_rows], marker='s',
              label='Path-faithful field: maximum-node Euler error')
    ax.set(xlabel='Euler steps N', ylabel='Population W2',
           title='Odd dimension: exact grid outputs cost path bias')
    ax.legend(fontsize=8); ax.grid(True, which='both', alpha=.2)
    save_plot(fig, 'path_bias_frontier')

    cert_rows = []
    for n in [16, 32, 64, 128]:
        for alpha in [Fraction(0), Fraction(1, 2), Fraction(1)]:
            certificate = rational_gaussian_certificate(n, alpha)
            out = {'N': n, 'alpha': str(alpha), 'd': 2, 'sigma': 1,
                   'digits': certificate['digits'],
                   'method': 'exact rational covariance recursion; integer square-root enclosures'}
            for kind in ['terminal', 'maximum_node']:
                lo, hi = certificate[kind]
                out[kind + '_lower_rational'] = str(lo)
                out[kind + '_upper_rational'] = str(hi)
                out[kind + '_approximate_display'] = float((lo+hi)/2)
                out[kind + '_interval_width'] = float(hi-lo)
            cert_rows.append(out)
    (OUT / 'rational_certificates.json').write_text(json.dumps(cert_rows, indent=2) + '\n')

    mixture_rows = []
    for mu in [.8, 1.2]:
        for t in [0., .1, .2, .3, .5, .8, 1.]:
            old = mixture_loss_check(t, mu, .5, 128)
            new = mixture_loss_check(t, mu, .5, 192)
            mixture_rows.append({'target_mean_amplitude': mu, 'within_component_variance': .5,
                                 't': t, 'Bayes_loss_192': new[0],
                                 'acceleration_moment_192': new[1],
                                 'identity_discrepancy': abs(new[0]-new[1]),
                                 'quadrature_refinement_change': max(abs(a-b) for a,b in zip(old,new)),
                                 'metric': 'population Gauss-Hermite quadrature; not interval certified'})
    write_csv('non_gaussian_risk_identity.csv', mixture_rows)
    quadrature, weights = leggauss(128)
    action_rows = []
    for mu in [0., .8, 1.2]:
        action = sum(w * mixture_loss_check(float((t+1)/2), mu, .5, 192)[0]/2
                     for t, w in zip(quadrature, weights))
        upper = np.pi/2*np.sqrt(mu**2+.5)
        action_rows.append({'mu': mu, 'within_variance': .5,
                            'integrated_bayes_risk': float(action),
                            'Gaussian_same_covariance_upper_bound': float(upper),
                            'difference': float(upper-action),
                            'scope': 'conditional action interpretation; no AP existence claim'})
    write_csv('mmse_action.csv', action_rows)

    mixture_order_rows = []
    for n in [16, 32, 64, 128, 256, 512]:
        err = mixture_euler_errors(n, quadrature_nodes=32)
        refined = mixture_euler_errors(n, quadrature_nodes=48)
        mixture_order_rows.append({'N': n, 'd': 3, 'target_component_mean': 1.2,
                                  'target_component_sd': .7,
                                  'maximum_node_W2_32': float(err.max()),
                                  'maximum_node_W2_48': float(refined.max()),
                                  'quadrature_change': float(abs(err.max()-refined.max())),
                                  'metric': 'full W2 product-quantile formula; non-certified quadrature'})
    write_csv('non_gaussian_whole_path.csv', mixture_order_rows)

    orders = {}
    for beta in [0., .5, 1.]:
        data = [r for r in rows if r['beta'] == beta]
        orders[str(beta)] = float(np.log2(data[-2]['maximum_node_W2']/data[-1]['maximum_node_W2']))
    summary = {
        'even_observed_orders_finest_pair': orders,
        'odd_scaled_errors_range': [min(r['N_scaled_maximum'] for r in odd_rows),
                                    max(r['N_scaled_maximum'] for r in odd_rows)],
        'secant_maximum_roundoff_across_runs': max(r['secant_target_maximum_roundoff'] for r in bias_rows),
        'secant_terminal_bias_N2048': bias_rows[-1]['secant_continuous_terminal_bias'],
        'secant_bias_order_finest_pair': float(np.log2(bias_rows[-2]['secant_continuous_terminal_bias']/bias_rows[-1]['secant_continuous_terminal_bias'])),
        'mixture_identity_largest_discrepancy': max(r['identity_discrepancy'] for r in mixture_rows),
        'mixture_quadrature_largest_change': max(r['quadrature_refinement_change'] for r in mixture_rows),
        'rational_certificates': len(cert_rows),
        'non_gaussian_whole_path_order': float(np.log2(mixture_order_rows[-2]['maximum_node_W2_48']/mixture_order_rows[-1]['maximum_node_W2_48'])),
        'non_gaussian_whole_path_quadrature_change': max(r['quadrature_change'] for r in mixture_order_rows),
        'largest_certificate_interval_width': max(r['maximum_node_interval_width'] for r in cert_rows),
        'verification_scope': 'population examples and rational enclosures, not theorem formalization',
        'Python': platform.python_version(), 'NumPy': np.__version__,
    }
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    table = [r'\begin{tabular}{lrr}', r'\toprule',
             r'Case & Predicted order & Observed order \\', r'\midrule']
    for beta in [0., .5, 1.]:
        table.append(rf'Even, $\beta={beta:g}$ & ${1+beta:g}$ & ${orders[str(beta)]:.5f}$ \\')
    table.append(rf'Non-Gaussian canonical field & $1$ & ${summary["non_gaussian_whole_path_order"]:.5f}$ \\')
    table.append(rf'Grid-fit continuous bias & $1$ & ${summary["secant_bias_order_finest_pair"]:.5f}$ \\')
    table.extend([r'\bottomrule', r'\end{tabular}'])
    (OUT / 'orders_table.tex').write_text('\n'.join(table)+'\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
