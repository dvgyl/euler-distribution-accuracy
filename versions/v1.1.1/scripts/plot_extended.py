#!/usr/bin/env python3
"""Separate figures; default matplotlib colors; no style preset."""
from pathlib import Path
import os,json
os.environ.setdefault('MPLCONFIGDIR','/tmp/euler-matplotlib')
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['pdf.fonttype']=42
matplotlib.rcParams['ps.fonttype']=42
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/extended'
def save(fig,name):
    fig.tight_layout(pad=.4);fig.savefig(OUT/(name+'.pdf'),bbox_inches='tight');fig.savefig(OUT/(name+'.png'),dpi=200,bbox_inches='tight');plt.close(fig)
def main():
    j=json.loads((OUT/'summary.json').read_text());labels={'fm':'CFM','discretization':'One-grid fit','transfer':'Two-grid fit'}
    for d in [2,8]:
        fig,ax=plt.subplots(figsize=(3.5,2.55))
        for r in j['groups']:
            if r['d']!=d:continue
            ns=[4,8,16,32,64,128];ys=[r[f'euler{n}']['target_sw2']['mean'] for n in ns];err=[r[f'euler{n}']['target_sw2']['std'] for n in ns]
            ax.errorbar(ns,ys,yerr=err,marker='o',markersize=3,capsize=2,label=labels[r['training']])
        ax.set_xscale('log',base=2);ax.set_xlabel('Euler steps',fontsize=9);ax.set_ylabel('Sampled target SW2',fontsize=9);ax.tick_params(labelsize=8);ax.legend(fontsize=7,frameon=False);save(fig,f'sample_only_d{d}')
    fig,ax=plt.subplots(figsize=(3.5,2.55))
    for r in j['groups']:
        if r['d']!=2:continue
        ns=[4,8,16,32,64,128];ys=[r[f'euler{n}']['own_flow_rms']['mean'] for n in ns]
        ax.loglog(ns,ys,marker='o',markersize=3,label=labels[r['training']])
    ax.set_xlabel('Euler steps',fontsize=9);ax.set_ylabel('Own-flow paired RMS',fontsize=9);ax.legend(fontsize=7,frameon=False);ax.tick_params(labelsize=8);save(fig,'own_flow_d2')
    checks=json.loads((ROOT/'results/audit/extension_validation.json').read_text())['refinements']
    fig,ax=plt.subplots(figsize=(3.5,2.55))
    for m in [2,3,4]:
        z=[x for x in checks if x['kind']=='balanced' and x['factor']==m];ax.semilogx([x['N'] for x in z],[x['scaled_error']/x['limit'] for x in z],marker='o',markersize=3,label=f'm={m}')
    ax.axhline(1,linestyle=':',linewidth=1);ax.set_xlabel('Coarse steps N',fontsize=9);ax.set_ylabel('N error / sharp limiting constant',fontsize=8);ax.tick_params(labelsize=8);ax.legend(fontsize=7,frameon=False);save(fig,'sharp_minimax_control')
    fig,ax=plt.subplots(figsize=(3.5,2.55))
    for kind,label in [('balanced','Equal bisection'),('degenerate','Unbalanced 2N'),('equal_defect_RK','Same-defect RK')]:
        z=[x for x in checks if x['kind']==kind and (kind!='balanced' or x['factor']==2)];ax.loglog([x['N'] for x in z],[x['error'] for x in z],marker='o',markersize=3,label=label)
    ax.set_xlabel('Coarse steps N',fontsize=9);ax.set_ylabel('Target error',fontsize=9);ax.tick_params(labelsize=8);ax.legend(fontsize=7,frameon=False);save(fig,'contrast_boundary')
    print('Saved five separate plot PDFs and PNGs.')
if __name__=='__main__':main()
