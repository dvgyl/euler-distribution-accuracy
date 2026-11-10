#!/usr/bin/env python3
"""Predeclared one-seed sensitivity checks; not multi-seed efficacy evidence."""
from pathlib import Path
import json
import run_sample_only_transfer as m
ROOT=Path(__file__).resolve().parents[1]
cases=[]
for w in [24,96]:
    for method in ['fm','discretization','transfer']:
        cases.append((f'capacity_w{w}',method,w,30.,0.))
for reg in [3.,100.]:
    for method in ['discretization','transfer']:
        cases.append((f'weight_{reg:g}',method,48,reg,0.))
for jac in [.01,.1]:
    cases.append((f'jacobian_{jac:g}','transfer',48,30.,jac))
for label,method,width,reg,jac in cases:
    m.OUT=ROOT/'results/extended/ablations'/label
    m.CK=ROOT/'checkpoints/extended/ablations'/label
    m.OUT.mkdir(parents=True,exist_ok=True);m.CK.mkdir(parents=True,exist_ok=True)
    name=m.train(2,17,method,1800,width,reg,jac)
    m.evaluate(name)
(ROOT/'results/extended/ablation_design.json').write_text(json.dumps(cases,indent=2))
