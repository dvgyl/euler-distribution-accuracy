#!/usr/bin/env python3
"""Independent numerical checks of the new analytic controls. Not proof certification."""
from pathlib import Path
import sys,json,math
import numpy as np
from scipy.integrate import quad
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from euler_path.transfer import stable_secant,gaussian_scale,scalar_euler_scale
from euler_path.observability import spectral_residual_bound,refinement_energy,tanh_network_budget
OUT=ROOT/'results/audit';OUT.mkdir(exist_ok=True)

def equal_refine(n,m):return np.linspace(0,1,m*n+1)
def secant_error(n,grid,common=None):
    q=scalar_euler_scale(lambda t:float(stable_secant(t,1/n)),grid)
    if common is not None:q=q[common];grid=grid[common]
    return float(np.max(abs(q-gaussian_scale(grid))))
def run():
    rows=[]
    for n in [32,64,128,256,512,1024]:
        coarse=np.linspace(0,1,n+1)
        deg=np.empty(2*n+1);deg[::2]=coarse;deg[1::2]=coarse[:-1]+1/n**2
        for m in [2,3,4]:
            e=secant_error(n,equal_refine(n,m),slice(None,None,m));rows.append({'kind':'balanced','N':n,'factor':m,'error':e,'scaled_error':n*e,'limit':.5*(1-1/m)*(1+math.pi/2)})
        rows.append({'kind':'degenerate','N':n,'beta':refinement_energy([1/n,1-1/n]),'error':secant_error(n,deg,slice(None,None,2))})
        rows.append({'kind':'N_plus_1','N':n,'error':secant_error(n,np.linspace(0,1,n+2))})
        c=lambda t:-5*t*t;g=np.exp(-5*coarse**3/3);q=scalar_euler_scale(c,coarse)
        rows.append({'kind':'endpoint_cancel','N':n,'terminal':abs(q[-1]-g[-1]),'max_node':float(max(abs(q-g)))})
        # Distinct first-order RK scheme with exactly the same leading defect as Euler.
        x=1.;eta=.1;err=0.
        for k,t in enumerate(coarse[:-1]):
            h=1/n;f=lambda tt,xx:float(stable_secant(tt,h))*xx
            a=f(t,x);b=f(t+h/2,x+h*a/2);cc=f(t+h,x+h*a)
            x+=h*((1+eta)*a-2*eta*b+eta*cc);err=max(err,abs(x-float(gaussian_scale(t+h))))
        rows.append({'kind':'equal_defect_RK','N':n,'error':err})
        def alias(t):
            h=1/n;aa=float(stable_secant(t,h));prev=float(stable_secant(t-h/2,h))
            bb=prev/(1+h*prev/2);return aa+(bb-aa)*math.sin(math.pi*t/h)**2
        aq=scalar_euler_scale(alias,np.linspace(0,1,2*n+1))[::2]
        rows.append({'kind':'temporal_alias','N':n,'error':float(max(abs(aq-gaussian_scale(coarse))))})
    radial=[]
    for eps in [.1,1.]:
        for y in [0.,1.,3.]:
            # d=3 so transverse radius density is r exp(-r²/2-eps*(y²+r²)²/4).
            f=lambda r:r*math.exp(-r*r/2-eps*((y*y+r*r)**2-y**4)/4)
            z=quad(f,0,np.inf,epsabs=1e-12)[0]
            radial.append({'eps':eps,'y':y,'second':quad(lambda r:r*r*f(r),0,np.inf)[0]/z,'fourth':quad(lambda r:r**4*f(r),0,np.inf)[0]/z,'bounds':[2.,8.]})
    spectral=[{'q':q,**spectral_residual_bound(1e-100,10**50,1e-50,1,1.,q,uniform=True)} for q in [0,1,2,4,8]]
    import torch
    budgets=[]
    for p in sorted((ROOT/'checkpoints/extended').rglob('*.pt')):
        d=torch.load(p,weights_only=True,map_location='cpu');s=d['state_dict'];w=[s[f'net.{i}.weight'].numpy() for i in [0,2,4]]
        budgets.append({'checkpoint':str(p.relative_to(ROOT)),**tanh_network_budget(w,s['net.4.bias'].numpy(),d['width'])})
    result={'refinements':rows,'radial_conditional_moments':radial,'spectral_formula_checks':spectral,'network_budgets':budgets,'interpretation':'floating numerical sanity checks except the exact-rational network coefficient norm enclosures'}
    (OUT/'extension_validation.json').write_text(json.dumps(result,indent=2))
    print('Saved',len(rows),'refinement controls;',len(budgets),'network budgets.')
if __name__=='__main__':run()
