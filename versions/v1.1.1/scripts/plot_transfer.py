#!/usr/bin/env python3
"""Render one chart per figure and write numerical tables from frozen receipts."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results'/'transfer'

def save(fig,name):
    fig.tight_layout(pad=.4)
    fig.savefig(OUT/f'{name}.pdf',bbox_inches='tight')
    fig.savefig(OUT/f'{name}.png',dpi=220,bbox_inches='tight')
    plt.close(fig)

def main():
    a=pd.read_csv(OUT/'gaussian_transfer.csv');b=pd.read_csv(OUT/'balanced_two_grid_fit.csv')
    fig,ax=plt.subplots(figsize=(3.5,2.65))
    x=a[(a.d==1)&(a.theta==0)]
    ax.loglog(x.N,x.two_grid_max,'o-',markersize=3,linewidth=1.3,label='Canonical: two-grid max')
    x=a[(a.d==1)&(a.theta==1)]
    ax.loglog(x.N,x.fine_common_max,'s--',markersize=3,linewidth=1.3,label='Secant: refined error')
    ax.loglog(b.N,b.two_grid_max,'^-.',markersize=3,linewidth=1.3,label='Balanced scalar fit')
    ax.set_xlabel('Coarse steps N',fontsize=9);ax.set_ylabel('Maximum-node W2 error',fontsize=9)
    ax.tick_params(labelsize=8);ax.legend(fontsize=7.2,frameon=False)
    save(fig,'gaussian_transfer_orders')
    e=pd.read_csv(OUT/'neural'/'euler.csv')
    fig,ax=plt.subplots(figsize=(3.5,2.65))
    for teacher,label,fmt in [('canonical','Bayes teacher','o-'),('secant','Fixed-16-step teacher','s--')]:
        g=e[e.teacher==teacher].groupby('N').target_max.agg(['mean','min','max'])
        ax.errorbar(g.index,g['mean'],yerr=np.array([g['mean']-g['min'],g['max']-g['mean']]),fmt=fmt,markersize=3,linewidth=1.3,capsize=2,label=label)
    ax.set_xscale('log',base=2);ax.set_yscale('log');ax.set_xticks([8,16,32,64,128,256],labels=['8','16','32','64','128','256'])
    ax.set_xlabel('Euler steps (checkpoint fixed)',fontsize=9);ax.set_ylabel('Target max-node W2 estimate',fontsize=9)
    ax.tick_params(labelsize=8);ax.legend(fontsize=7.2,frameon=False)
    save(fig,'neural_target_transfer')
    fig,ax=plt.subplots(figsize=(3.5,2.65))
    for teacher,label,fmt in [('canonical','Bayes teacher','o-'),('secant','Fixed-16-step teacher','s--')]:
        g=e[e.teacher==teacher].groupby('N').own_max.agg(['mean','min','max'])
        ax.errorbar(g.index,g['mean'],yerr=np.array([g['mean']-g['min'],g['max']-g['mean']]),fmt=fmt,markersize=3,linewidth=1.3,capsize=2,label=label)
    ax.set_xscale('log',base=2);ax.set_yscale('log');ax.set_xticks([8,16,32,64,128,256],labels=['8','16','32','64','128','256'])
    ax.set_xlabel('Euler steps (checkpoint fixed)',fontsize=9);ax.set_ylabel('Own-flow max-node W2 estimate',fontsize=9)
    ax.tick_params(labelsize=8);ax.legend(fontsize=7.2,frameon=False);save(fig,'neural_own_integration')
    # Tables are generated, not copied by hand from rounded graph coordinates.
    lines=[r'\begin{table}[t]',r'\centering',r'\caption{Neural diagnostic: mean over three seeds; brackets give the seed range. Each checkpoint is frozen across all step counts. The metrics are quantile approximations, not certified intervals.}',r'\label{tab:neural}',r'\small',r'\begin{tabular}{@{}llrr@{}}',r'\toprule',r'Teacher & Steps & Target error & Own-flow error\\',r'\midrule']
    for teacher,label in [('canonical','Bayes'),('secant','Secant')]:
      for n in [16,32,128]:
        z=e[(e.teacher==teacher)&(e.N==n)]
        def form(k):return f'{z[k].mean():.4f}'+r'\newline'+f'[{z[k].min():.4f}, {z[k].max():.4f}]'
        # Two-line cells use shortstack, so no paragraph-width clipping.
        def cell(k):return r'\shortstack[r]{'+f'{z[k].mean():.4f}'+r'\\'+r'\scriptsize '+f'[{z[k].min():.4f}, {z[k].max():.4f}]'+r'}'
        lines.append(f'{label} & {n} & {cell("target_max")} & {cell("own_max")}'+r'\\[3pt]')
    lines.extend([r'\bottomrule',r'\end{tabular}',r'\end{table}'])
    (OUT/'neural_main_table.tex').write_text('\n'.join(lines)+'\n')
    r=pd.read_csv(OUT/'neural'/'reference.csv');tr=pd.read_csv(OUT/'neural'/'training.csv');ref=pd.read_csv(OUT/'neural'/'quantile_refinement.csv')
    ll=[r'\begin{center}\small',r'\begin{tabular}{@{}llrrrrr@{}}',r'\toprule',r'Teacher & Seed & Val. MSE & Selected step & Train sec. & Sampled model bias & Source excess\\',r'\midrule']
    for _,row in tr.iterrows():
        rr=r[(r.teacher==row.teacher)&(r.seed==row.seed)].iloc[0]
        ll.append(f'{row.teacher} & {row.seed} & {row.validation_mse:.3g} & {row.selected_step} & {row.training_seconds:.2f} & {rr.model_bias_max:.6f} & {rr.source_excess:.6f}'+r'\\')
    ll.extend([r'\bottomrule',r'\end{tabular}\end{center}']);(OUT/'neural_all_seeds.tex').write_text('\n'.join(ll)+'\n')
    summary={'checkpoints':len(tr),'parameters':int(tr.parameters.iloc[0]),'reference_max_refinement':r.reference_tolerance_refinement.max(),'quantile_max_checked_change':ref.abs_change.max(),'mean_secant_target_16':e[(e.teacher=='secant')&(e.N==16)].target_max.mean(),'mean_secant_target_32':e[(e.teacher=='secant')&(e.N==32)].target_max.mean(),'mean_secant_own_16':e[(e.teacher=='secant')&(e.N==16)].own_max.mean(),'mean_secant_own_32':e[(e.teacher=='secant')&(e.N==32)].own_max.mean(),'source_target_variance':4.,'scope':'small analytic-teacher diagnostic; no certified neural W2 bound or practical global derivative budget'}
    (OUT/'neural_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print('Saved three separate figures, two LaTeX tables, and the neural result summary.')
if __name__=='__main__':main()
