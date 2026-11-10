#!/usr/bin/env python3
"""Generate deterministic independent-interpolation parity validation files."""
import csv
import json
import math
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from euler_path.independent_interpolation import (
    gaussian_diagonal_errors, even_straight_complex_error,
    canonical_terminal_coefficient, independent_mixture_cosine_second_derivative,
    source_residual_for_gaussian_target,
)
OUT = ROOT / 'results' / 'independent'


def write_csv(path, rows):
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    grids = [8,16,32,64,128,256,512,1024,2048]
    cases = [
        ('isotropic_d2_base', [1.5625,1.5625]),
        ('isotropic_d3_base', [1.5625,1.5625,1.5625]),
        ('anisotropic_d3_base', [0.64,1.0,2.25]),
        ('anisotropic_d5_base', [0.49,0.81,1.0,1.44,2.25]),
    ]
    rows = []
    for name, eig in cases:
        for n in grids:
            term, maximum = gaussian_diagonal_errors(eig, n)
            rows.append({'case': name,'N':n,'terminal_W2':format(term,'.16g'),
                         'maximum_node_W2':format(maximum,'.16g'),
                         'scaled_terminal_NW2':format(n*term,'.16g'),
                         'limiting_terminal_coefficient':format(canonical_terminal_coefficient(eig),'.16g'),
                         'metric':'exact_population_Gaussian_W2'})
    write_csv(OUT/'gaussian_population.csv',rows)

    even_rows=[]
    for n in grids:
        term,maxi=even_straight_complex_error(1.25,n)
        even_rows.append({'N':n,'terminal_roundoff_W2':format(term,'.16g'),
                          'max_node_roundoff_W2':format(maxi,'.16g'),
                          'theoretical_exact_W2':0,
                          'note':'floating_point_check_of_exact_algebraic_identity'})
    write_csv(OUT/'even_straight_check.csv',even_rows)
    jet_rows=[]
    for direction, shift in [('parallel',1.1),('orthogonal',0.0),('oblique',0.55)]:
        for k in [0.3,0.75,1.5,2.0]:
            a,b=independent_mixture_cosine_second_derivative(k,1.0,shift,0.49)
            jet_rows.append({'direction':direction,'wave_number':k,'shift_projection':shift,
                             'closed_marginal_derivative':format(a,'.16g'),
                             'straight_coupling_derivative':format(b,'.16g'),
                             'absolute_discrepancy':format(abs(a-b),'.16g'),
                             'note':'analytic_weak_two_jet_for_nonGaussian_target'})
    write_csv(OUT/'mixture_two_jet.csv',jet_rows)

    # A strict analytic (not interval-numerical) certificate for a controlled
    # case: r_0=0, source I_3, target I_3, lambda=1, L_r=0:
    # epsilon_0 >= exp(-2)/(2 sqrt(2pi)) > 1/54,
    # using e<3, pi<4, sqrt(8)<3.
    certificate={'case':'independent_standard_gaussian_d3_canonical',
                 'dimension':3,'source_covariance':'I','target_covariance':'I',
                 'r0_lipschitz':0,'lambda_min':1,
                 'exact_residual':source_residual_for_gaussian_target(np.eye(3)),
                 'lower_bound_formula':math.exp(-2)/(2*math.sqrt(2*math.pi)),
                 'rational_certified_lower_bound':'1/54',
                 'proof':'e<3 and pi<4 imply exp(-2)/(2 sqrt(2pi))>1/54',
                 'status':'analytic_rational_certificate_not_interval_quadrature'}
    (OUT/'analytic_certificate.json').write_text(json.dumps(certificate,indent=2)+'\n')

    fig, axes=plt.subplots(1,2,figsize=(8.8,3.1))
    for case, title, ax in [('isotropic_d3_base','Isotropic Gaussian data (3D)',axes[0]),
                            ('anisotropic_d3_base','Anisotropic Gaussian data (3D)',axes[1])]:
        filtered=sorted((r for r in rows if r['case']==case),key=lambda r:r['N'])
        ns=np.array([int(r['N']) for r in filtered]); errs=np.array([float(r['maximum_node_W2']) for r in filtered]);
        ax.loglog(ns,errs,'o-',label='Canonical Euler: exact $W_2$')
        ax.loglog(ns, errs[-1]*ns[-1]/ns, '--',alpha=.7,label='$N^{-1}$ reference')
        ax.set(xlabel='Euler steps $N$',ylabel='Maximum-node $W_2$',title=title)
        ax.grid(True,which='both',alpha=.18)
        ax.legend(fontsize=7,loc='best')
    fig.tight_layout(); fig.savefig(OUT/'independent_gaussian_parity.pdf',bbox_inches='tight'); plt.close(fig)

    # Publish a small machine-readable numerical receipt.
    summary={'case_count':4,'grid_count':len(grids),
             'max_analytic_mixture_jet_discrepancy':max(float(r['absolute_discrepancy']) for r in jet_rows),
             'max_even_roundoff_W2':max(float(r['max_node_roundoff_W2']) for r in even_rows),
             'd3_isotropic_terminal_limit':canonical_terminal_coefficient([1.5625]*3),
             'd3_anisotropic_terminal_limit':canonical_terminal_coefficient([0.64,1.,2.25]),
             'numerical_claim_scope':'exact_W2_only_for_diagonal_gaussian_laws; roundoff checks are not a positive error; all-general-field impossibility is analytical'}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))

if __name__ == '__main__':
    main()
