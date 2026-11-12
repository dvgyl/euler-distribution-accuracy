"""Write the numbers and tables that the article and the supplement quote.

Every quoted value of the new experiments comes from this script, so the text
and the saved results cannot differ. Outputs: paper/generated_numbers.tex,
paper/generated_table_*.tex, results/article_numbers.json.
"""
import json
import math
from pathlib import Path

import numpy as np
from scipy.integrate import quad
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / 'paper'
LABEL = {'exact': 'Exact field', 'cfm': 'CFM', 'one_grid': 'One-grid fit', 'two_grid': 'Two-grid fit'}
NAME = {'exact': 'Exact', 'cfm': 'Cfm', 'one_grid': 'One', 'two_grid': 'Two'}
STEP_NAME = {8: 'Eight', 16: 'Sixteen', 64: 'SixtyFour', 256: 'Large'}
MODEL_LABEL = {'fm': 'CFM', 'discretization': 'One-grid fit', 'transfer': 'Two-grid fit'}


def f3(x):
    return f'{x:.3f}'


def sci(x):
    exponent = math.floor(math.log10(abs(x)))
    return f'{x / 10 ** exponent:.1f}\\times10^{{{exponent}}}'


def main():
    numbers, macros = {}, {}

    def put(name, value, text):
        numbers[name] = value
        macros[name] = text

    # Introductory example: exact flow of the field fitted to one step of 1/2.
    g = lambda t: math.sqrt((1 - t) ** 2 + t ** 2)
    flow = math.exp(quad(lambda s: (g(s + 0.5) - g(s)) / (0.5 * g(s)), 0, 0.5)[0])
    put('IntroExactFlow', flow, f'${flow:.2f}$')

    # Explicit risk-gap threshold for d = T = K = 1 and gap one.
    risk = next(r for r in json.loads((ROOT / 'results/audit/risk_constant_scale.json').read_text())
                if (r['d'], r['K'], r['gap']) == (1, 1, 1))
    put('RiskMeshThreshold', 10 ** risk['log10_H'], sci(10 ** risk['log10_H']))
    put('RiskStepCount', 10 ** risk['log10_required_N'], sci(10 ** risk['log10_required_N']))

    # Learned scalar fields.
    models = json.loads((ROOT / 'results/scalar_population/models.json').read_text())
    by = {o: sorted((m for m in models if m['objective'] == o), key=lambda m: m['seed']) for o in LABEL}
    w = lambda m, n: m['shared'][str(n)]['target_w2']
    own = lambda m, n: m['shared'][str(n)]['own_flow_rms']
    rows = []
    for o, group in by.items():
        for n, step in STEP_NAME.items():
            v = np.array([w(m, n) for m in group])
            put(f'Scalar{NAME[o]}{step}', float(v.mean()), f3(v.mean()))
        bias = np.array([m['model_bias'] for m in group])
        put(f'Scalar{NAME[o]}Bias', float(bias.mean()), f3(bias.mean()))
        for n, step in ((8, 'Eight'), (256, 'Large')):
            v = np.array([own(m, n) for m in group])
            put(f'Scalar{NAME[o]}Own{step}', float(v.mean()), f3(v.mean()))
        cell = lambda v: f3(v.mean())
        sd_max = max(float(np.array([w(m, n) for m in group]).std(ddof=1)) for n in STEP_NAME) if len(group) > 1 else 0.0
        put(f'Scalar{NAME[o]}SdMax', sd_max, f3(sd_max))
        rows.append(' & '.join([LABEL[o]] + [cell(np.array([w(m, n) for m in group])) for n in STEP_NAME]
                               + [cell(bias), cell(np.array([own(m, 8) for m in group])),
                                  cell(np.array([own(m, 256) for m in group]))]) + r'\\')
        if len(group) > 1:
            for n, step in ((16, 'Sixteen'), (64, 'SixtyFour'), (256, 'Large')):
                diff = np.array([w(m, n) - w(m, 8) for m in group])
                put(f'Scalar{NAME[o]}Increase{step}', int((diff > 0).sum()), str(int((diff > 0).sum())))
                p = float(wilcoxon(diff, alternative='two-sided', method='exact').pvalue)
                put(f'Scalar{NAME[o]}Wilcoxon{step}', p, f'{p:.3f}')
            for n, step in ((64, 'SixtyFour'), (256, 'Large')):
                diff = np.array([w(m, n) - w(m, 16) for m in group])
                put(f'Scalar{NAME[o]}FromSixteenIncrease{step}', int((diff > 0).sum()), str(int((diff > 0).sum())))
                p = float(wilcoxon(diff, alternative='two-sided', method='exact').pvalue)
                put(f'Scalar{NAME[o]}FromSixteenWilcoxon{step}', p, f'{p:.3f}')
            best = [min(m['shared'], key=lambda n: m['shared'][n]['target_w2']) for m in group]
            put(f'Scalar{NAME[o]}ArgminEight', best.count('8'), str(best.count('8')))
            put(f'Scalar{NAME[o]}ArgminSixteen', best.count('16'), str(best.count('16')))
            put(f'Scalar{NAME[o]}ArgminLast', best.count('256'), str(best.count('256')))
            put(f'Scalar{NAME[o]}BiasMin', float(bias.min()), f3(bias.min()))
            put(f'Scalar{NAME[o]}BiasMax', float(bias.max()), f3(bias.max()))
            pc = np.array([m['pair']['8']['E_c'] for m in group])
            pf = np.array([m['pair']['8']['E_f'] for m in group])
            put(f'Scalar{NAME[o]}PairCoarse', float(pc.mean()), f3(pc.mean()))
            put(f'Scalar{NAME[o]}PairFine', float(pf.mean()), f3(pf.mean()))
    numbers['scalar_all_maps_increasing'] = all(m['increasing'] for m in models)
    put('ScalarOneGridFactor', numbers['ScalarOneLarge'] / numbers['ScalarOneEight'],
        f"{numbers['ScalarOneLarge'] / numbers['ScalarOneEight']:.1f}")
    put('ScalarModelCount', len(models) - 1, str(len(models) - 1))
    (PAPER / 'generated_table_scalar.tex').write_text('\n'.join([
        r'\begin{tabular}{@{}l@{\hspace{5pt}}c@{\hspace{5pt}}c@{\hspace{5pt}}c@{\hspace{5pt}}c@{\hspace{7pt}}c@{\hspace{7pt}}c@{\hspace{5pt}}c@{}}', r'\toprule',
        r' & \multicolumn{4}{c}{Target error} & Model & \multicolumn{2}{c@{}}{Integration error}\\',
        r'\cmidrule(lr){2-5}\cmidrule(l){7-8}',
        r'Field & $N{=}8$ & $16$ & $64$ & $256$ & bias & $N{=}8$ & $256$\\', r'\midrule', *rows,
        r'\bottomrule', r'\end{tabular}']) + '\n')
    detail = [r'\begin{tabular}{@{}lrcccccc@{}}', r'\toprule',
              r'Field & Seed & $N=8$ & $16$ & $64$ & $256$ & Model bias & $E_c+E_f$ at $N=8$\\', r'\midrule']
    for o in ('cfm', 'one_grid', 'two_grid'):
        for m in by[o]:
            detail.append(' & '.join([LABEL[o], str(m['seed'])] + [f3(w(m, n)) for n in STEP_NAME]
                                     + [f3(m['model_bias']), f3(m['pair']['8']['E_c'] + m['pair']['8']['E_f'])]) + r'\\')
        detail.append(r'\addlinespace')
    detail[-1:] = [r'\bottomrule', r'\end{tabular}']
    (PAPER / 'generated_table_scalar_models.tex').write_text('\n'.join(detail) + '\n')

    # Variability across evaluation sets.
    var_path = ROOT / 'results/evaluation_set_variability/summary.json'
    if var_path.exists():
        var = json.loads(var_path.read_text())
        sizes = sorted(int(k) for k in var['design'])
        size_name = dict(zip(sizes, ('Small', 'Medium', 'Big')))
        fitted = [n for n in var['models'] if '_fm_' not in n]
        put('VarSetCount', sum(var['design'].values()), str(sum(var['design'].values())))
        for size in sizes:
            tag = size_name[size]
            frac = [var['models'][n][str(size)]['change_8_to_16']['fraction_of_sets_with_increase'] for n in fitted]
            sd = [var['models'][n][str(size)]['change_8_to_16']['sd'] for n in fitted]
            mean = [var['models'][n][str(size)]['change_8_to_16']['mean'] for n in fitted]
            put(f'Var{tag}Sets', var['design'][str(size)], str(var['design'][str(size)]))
            put(f'Var{tag}FractionMin', min(frac), f'{100 * min(frac):.0f}')
            put(f'Var{tag}FractionMax', max(frac), f'{100 * max(frac):.0f}')
            put(f'Var{tag}ChangeSd', float(np.mean(sd)), f3(float(np.mean(sd))))
            put(f'Var{tag}ChangeMean', float(np.mean(mean)), f3(float(np.mean(mean))))
            six = var['six_fitted_models_8_to_16'][str(size)]
            put(f'Var{tag}AllSix', six['fraction_of_sets_where_all_six_increase'],
                f"{100 * six['fraction_of_sets_where_all_six_increase']:.1f}")
            long = var['models']['d2_discretization_s17_w48'][str(size)]['change_8_to_64']
            put(f'Var{tag}GridLongFraction', long['fraction_of_sets_with_increase'],
                f"{100 * long['fraction_of_sets_with_increase']:.0f}")
            put(f'Var{tag}GridLongMean', long['mean'], f3(long['mean']))
            for group_name, method in (('One', 'discretization'), ('Two', 'transfer')):
                for letter, seed in zip('ABC', (17, 29, 43)):
                    value = var['models'][f'd2_{method}_s{seed}_w48'][str(size)]['change_8_to_64']['fraction_of_sets_with_increase']
                    put(f'Var{tag}Long{group_name}{letter}', value, f'{100 * value:.0f}')
            cfm = [var['models'][n][str(size)]['change_8_to_16']['fraction_of_sets_with_increase']
                   for n in var['models'] if '_fm_' in n]
            put(f'Var{tag}CfmFractionMax', max(cfm), f'{100 * max(cfm):.0f}')
            put(f'Var{tag}AnySix', six['fraction_of_sets_where_any_increases'],
                f"{100 * six['fraction_of_sets_where_any_increases']:.0f}")
        lines = [r'\begin{tabular}{@{}llrccccccc@{}}', r'\toprule',
                 r'Training & Seed & Set size & SW2$_8$ & SW2$_{16}$ & SW2$_{64}$ & Change 8 to 16 & Increase & Change 8 to 64 & Increase\\',
                 r'\midrule']
        for name in var['models']:
            _, method, seed, _ = name.split('_')
            for size in sizes:
                e = var['models'][name][str(size)]
                c = e['change_8_to_16']
                lines.append(' & '.join([
                    MODEL_LABEL[method], seed[1:], f'{size:,}',
                    *[f"{e[f'sw2_{k}']['mean']:.3f} ({e[f'sw2_{k}']['sd']:.3f})" for k in (8, 16, 64)],
                    f"${c['mean']:+.3f}$ ({c['sd']:.3f})", f"{100 * c['fraction_of_sets_with_increase']:.0f}\\%",
                    f"${e['change_8_to_64']['mean']:+.3f}$ ({e['change_8_to_64']['sd']:.3f})",
                    f"{100 * e['change_8_to_64']['fraction_of_sets_with_increase']:.0f}\\%"]) + r'\\')
            lines.append(r'\addlinespace')
        lines[-1:] = [r'\bottomrule', r'\end{tabular}']
        (PAPER / 'generated_table_variability.tex').write_text('\n'.join(lines) + '\n')

    (PAPER / 'generated_numbers.tex').write_text(
        ''.join(f'\\newcommand{{\\{k}}}{{{v}}}\n' for k, v in sorted(macros.items())))
    (ROOT / 'results/article_numbers.json').write_text(json.dumps(numbers, indent=1, sort_keys=True) + '\n')
    for k in sorted(macros):
        print(f'{k:32s} {macros[k]}')


if __name__ == '__main__':
    main()
