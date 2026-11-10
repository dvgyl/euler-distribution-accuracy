#!/usr/bin/env python3
"""Aggregate every declared run. Seed variation is not a confidence interval."""
from pathlib import Path
import json,csv,math
import numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/extended'
def scalar(a):return {'mean':float(np.mean(a)),'std':float(np.std(a,ddof=1)) if len(a)>1 else 0.,'min':float(min(a)),'max':float(max(a))}
def main():
    runs=[json.loads(p.read_text()) for p in sorted(OUT.glob('*_evaluation.json')) if not p.name.startswith('._')];summ=[];cost=[]
    for d in [2,8]:
        for method in ['fm','discretization','transfer']:
            group=[r for r in runs if r['d']==d and r['training_method']==method];z={'d':d,'training':method,'seeds':[r['seed'] for r in group]}
            for n in [4,8,16,32,64,128]:
                rows=[next(x for x in r['rows'] if x['method']=='euler' and x['steps']==n and x['jitter']==0) for r in group]
                z[f'euler{n}']={'target_sw2':scalar([x['max_sw2'] for x in rows]),'own_flow_rms':scalar([x['paired_own_flow_rms'] for x in rows]),'seconds':scalar([x['seconds'] for x in rows])}
            z['reference_bias']=scalar([r['sampled_reference_sw2'] for r in group]);z['refinement_check']=max(r['reference_refinement_rms'] for r in group)
            z['train_seconds']=scalar([json.loads((OUT/(r['name']+'_training.json')).read_text())['seconds'] for r in group])
            z['count_target_worse_8_to_16']=sum(next(x for x in r['rows'] if x['method']=='euler' and x['steps']==16 and x['jitter']==0)['max_sw2']>next(x for x in r['rows'] if x['method']=='euler' and x['steps']==8 and x['jitter']==0)['max_sw2'] for r in group)
            summ.append(z)
            for budget in [24,48,96,192]:
                for solver in ['euler','heun','rk3']:
                    rows=[next(x for x in r['rows'] if x['method']==solver and x['nfe']==budget and x['jitter']==0) for r in group]
                    cost.append({'d':d,'training':method,'nfe':budget,'solver':solver,'target_sw2':scalar([x['max_sw2'] for x in rows]),'own_flow_rms':scalar([x['paired_own_flow_rms'] for x in rows]),'seconds':scalar([x['seconds'] for x in rows])})
    ab=[]
    for p in sorted((OUT/'ablations').rglob('*_evaluation.json')):
        if p.name.startswith('._'):continue
        j=json.loads(p.read_text());rr={n:next(x for x in j['rows'] if x['method']=='euler' and x['steps']==n and x['jitter']==0) for n in [8,16,64]}
        ab.append({'setting':str(p.parent.relative_to(OUT)),'training':j['training_method'],'sw2_8':rr[8]['max_sw2'],'sw2_16':rr[16]['max_sw2'],'sw2_64':rr[64]['max_sw2'],'own8':rr[8]['paired_own_flow_rms'],'own16':rr[16]['paired_own_flow_rms']})
    (OUT/'summary.json').write_text(json.dumps({'groups':summ,'equal_nfe':cost,'single_seed_ablations':ab,'scope':'fixed-projection empirical SW2 at five diagnostic times; RMS is an empirical common-start displacement against a numerical own-flow reference'},indent=2))
    with (OUT/'all_metrics.csv').open('w') as f:
        names=['model','d','training','seed','solver','steps','nfe','jitter','sampled_max_sw2','terminal_sw2','own_flow_rms','seconds'];wr=csv.DictWriter(f,fieldnames=names);wr.writeheader()
        for r in runs:
            for x in r['rows']:wr.writerow(dict(zip(names,[r['name'],r['d'],r['training_method'],r['seed'],x['method'],x['steps'],x['nfe'],x['jitter'],x['max_sw2'],x['terminal_sw2'],x['paired_own_flow_rms'],x['seconds']])))
    labels={'fm':'CFM','discretization':'One-grid fit','transfer':'Two-grid fit'}
    lines=[r'\begin{table}[t]',r'\centering',r'\caption{Descriptive three-seed comparisons on the 1,024-sample evaluation set. Entries are means with sample standard deviations in parentheses below. Subscripts give Euler step counts. Lower values are better. SW2 uses 128 directions and five times; RMS uses a common-start numerical reference.}',r'\label{tab:small-bank}',r'\small',r'\begin{tabular}{@{}clrrrr@{}}',r'\toprule',r'$d$ & Training & SW2$_8$ & SW2$_{16}$ & RMS$_8$ & RMS$_{16}$\\',r'\midrule']
    for z in summ:
        values=[z['euler8']['target_sw2'],z['euler16']['target_sw2'],z['euler8']['own_flow_rms'],z['euler16']['own_flow_rms']]
        cells=[r'\shortstack{$'+f"{v['mean']:.3f}"+r'$\\$('+f"{v['std']:.3f}"+r')$}' for v in values]
        lines.append(f"{z['d']} & {labels[z['training']]} & "+' & '.join(cells)+r'\\')
    lines +=[r'\bottomrule',r'\end{tabular}',r'\end{table}'];(OUT/'main_table.tex').write_text('\n'.join(lines)+'\n')
    lines=[r'\begin{table}[ht]',r'\centering',r'\caption{Equal 96-field-evaluation comparisons, means of three seeds. All target metrics use the same held-out evaluation set.}',r'\label{tab:equal-nfe-new}',r'\begin{tabular}{@{}cllrr@{}}',r'\toprule',r'$d$ & Training & Solver & Target SW2 & Own-flow RMS\\',r'\midrule']
    for x in cost:
        if x['nfe']==96:lines.append(f"{x['d']} & {labels[x['training']]} & {x['solver']} & {x['target_sw2']['mean']:.5f} & {x['own_flow_rms']['mean']:.5f}\\\\")
    lines +=[r'\bottomrule',r'\end{tabular}',r'\end{table}'];(OUT/'equal_nfe_table.tex').write_text('\n'.join(lines)+'\n')
    lines=[r'\begin{table}[ht]',r'\centering',r'\caption{One-seed sensitivity checks ($d=2$, seed 17). These are ablations, not three-seed efficacy comparisons.}',r'\label{tab:ablations-new}',r'\begin{tabular}{@{}llrrr@{}}',r'\toprule',r'Setting & Training & SW2$_8$ & SW2$_{16}$ & SW2$_{64}$\\',r'\midrule']
    for z in ab:lines.append(f"{z['setting'].split('/')[-1].replace('_',' ')} & {labels[z['training']]} & {z['sw2_8']:.4f} & {z['sw2_16']:.4f} & {z['sw2_64']:.4f}\\\\")
    lines +=[r'\bottomrule',r'\end{tabular}',r'\end{table}'];(OUT/'ablation_table.tex').write_text('\n'.join(lines)+'\n')
    print('summarized',len(runs),'primary models and',len(ab),'ablations')
if __name__=='__main__':main()
