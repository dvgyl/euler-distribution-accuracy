#!/usr/bin/env python3
"""Run deterministic step-count-transfer controls (no neural training)."""
from pathlib import Path
import sys, csv, json, platform
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.integrate import quad
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from euler_path.transfer import gaussian_transfer, stable_secant, gaussian_scale
OUT = ROOT/'results'/'transfer'


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer=csv.DictWriter(stream, fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows=[]
    for n in [16,32,64,128,256,512,1024,2048]:
        for d in [1,2,3]:
            for theta in [0.,.75,1.]:
                rows.append(gaussian_transfer(n,theta,d=d))
    write_csv(OUT/'gaussian_transfer.csv',rows)
    fitted=[]
    for n in [16,32,64,128,256,512,1024]:
        opt=minimize_scalar(lambda x:gaussian_transfer(n,x)['two_grid_max'],bounds=(0,1),
                            method='bounded',options={'xatol':1e-11})
        row=gaussian_transfer(n,opt.x)
        row['optimizer_success']=bool(opt.success)
        row['optimizer_nfev']=opt.nfev
        fitted.append(row)
    write_csv(OUT/'balanced_two_grid_fit.csv',fitted)
    fixed=[]
    for n in [16,32,64,128,256,512,1024]:
        for power in [1.,1.3,1.7]:
            fixed.append(gaussian_transfer(n,0,d=3,power=power))
    write_csv(OUT/'nonuniform_transfer.csv',fixed)
    ratios=[]
    from euler_path.transfer import scalar_euler_scale
    for n in [16,32,64,128,256]:
        for m in [1,2,4,8]:
            grid=np.linspace(0,1,m*n+1)
            scale=scalar_euler_scale(lambda t:float(stable_secant(t,1/n)),grid)
            e=np.abs(scale-gaussian_scale(grid))
            own=np.exp(quad(lambda t:float(stable_secant(t,1/n)),0,1,
                           epsabs=1e-12,epsrel=1e-12)[0])
            ratios.append({'fitted_N':n,'evaluated_N':m*n,'refinement':m,
                           'max_W2':float(e.max()),'terminal_W2':float(e[-1]),
                           'terminal_model_bias':float(abs(own-1))})
    write_csv(OUT/'frozen_secant_transfer.csv',ratios)
    def order(rs,key):
        rs=rs[-4:]
        return float(-np.polyfit(np.log([r['N'] for r in rs]),np.log([r[key] for r in rs]),1)[0])
    summary={'python':platform.python_version(),
             'secant_transfer_order':order([r for r in rows if r['d']==1 and r['theta']==1.],'fine_common_max'),
             'balanced_fit_order':order(fitted,'two_grid_max'),
             'balanced_theta_last':fitted[-1]['theta'],
             'balanced_N_scaled_max_last':fitted[-1]['N']*fitted[-1]['two_grid_max'],
             'max_fitted_grid_roundoff':max(r['coarse_max'] for r in rows if r['theta']==1.),
             'scope':'population Gaussian formulas, floating point; finite-N nonlinear lower bounds are not certified by these examples'}
    (OUT/'analytic_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
