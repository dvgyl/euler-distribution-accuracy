"""Population Wasserstein errors of learned scalar velocity fields.

The target is a scalar two-component Gaussian mixture. The path is the
independent linear interpolation from a standard Gaussian. In one dimension
the quantile pairing attains the Wasserstein distance, so the errors E_c and
E_f of the article are computed by quadrature. No evaluation samples are used.

Usage:
    python scripts/run_scalar_population.py            # train and evaluate
    python scripts/run_scalar_population.py --eval-only  # evaluate saved fields
"""
import argparse
import json
import math
import multiprocessing as mp
from pathlib import Path

import numpy as np
import torch
from scipy.special import ndtr, ndtri

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results' / 'scalar_population'
CKPT = ROOT / 'checkpoints' / 'scalar_population'

MU, SIGMA = 1.5, 0.5
SHARED_TIMES = (0.25, 0.5, 0.75, 1.0)
STEP_COUNTS = (4, 8, 16, 32, 64, 128, 256)
OBJECTIVES = ('cfm', 'one_grid', 'two_grid')
SEEDS = tuple(range(10))
CONFIG = {
    'target': {'mu': MU, 'sigma': SIGMA},
    'width': 48,
    'updates': 8000,
    'batch': 1024,
    'learning_rate': 1e-3,
    'grid_weight': 10.0,
    'fit_steps': {'one_grid': [8], 'two_grid': [8, 16]},
    'seeds': list(SEEDS),
    'quadrature': {'z_max': 9.0, 'nodes': 8001},
    'reference': {'method': 'rk4', 'steps': 1024},
}


def variance(t):
    return (1.0 - t) ** 2 + SIGMA ** 2 * t ** 2


class Field(torch.nn.Module):
    def __init__(self, width):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(2, width), torch.nn.Tanh(),
            torch.nn.Linear(width, width), torch.nn.Tanh(),
            torch.nn.Linear(width, 1))

    def forward(self, t, x):
        t = torch.as_tensor(t, dtype=x.dtype).expand_as(x)
        return self.net(torch.stack([x, t], dim=-1)).squeeze(-1)


def exact_field(t, x):
    """Conditional velocity of the independent interpolation path."""
    t = torch.as_tensor(t, dtype=x.dtype)
    q = variance(t)
    return ((1 + SIGMA ** 2) * t - 1) / q * x + MU * (1 - t) / q * torch.tanh(MU * t * x / q)


def euler_states(field, x0, steps):
    h = 1.0 / steps
    x, states = x0, []
    for k in range(steps):
        x = x + h * field(k * h, x)
        states.append(x)
    return states


def rk4_states(field, x0, steps):
    h = 1.0 / steps
    x, states = x0, []
    for k in range(steps):
        t = k * h
        k1 = field(t, x)
        k2 = field(t + h / 2, x + h / 2 * k1)
        k3 = field(t + h / 2, x + h / 2 * k2)
        k4 = field(t + h, x + h * k3)
        x = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        states.append(x)
    return states


def sample_target(n, gen):
    sign = torch.randint(0, 2, (n,), generator=gen).to(torch.float64) * 2 - 1
    return MU * sign + SIGMA * torch.randn(n, generator=gen, dtype=torch.float64)


def grid_loss(field, steps, batch, gen):
    """Mean squared distance of sorted Euler outputs and sorted path samples."""
    states = euler_states(field, torch.randn(batch, generator=gen, dtype=torch.float64), steps)
    total = 0.0
    for t in SHARED_TIMES:
        noise = torch.randn(batch, generator=gen, dtype=torch.float64)
        ref = (1 - t) * noise + t * sample_target(batch, gen)
        out = states[round(t * steps) - 1]
        total = total + ((torch.sort(out).values - torch.sort(ref).values) ** 2).mean()
    return total / len(SHARED_TIMES)


def train(objective, seed):
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(100_000 + seed)
    field = Field(CONFIG['width']).double()
    opt = torch.optim.Adam(field.parameters(), lr=CONFIG['learning_rate'])
    batch = CONFIG['batch']
    fit_steps = CONFIG['fit_steps'].get(objective, [])
    history = []
    for update in range(CONFIG['updates']):
        noise = torch.randn(batch, generator=gen, dtype=torch.float64)
        data = sample_target(batch, gen)
        t = torch.rand(batch, generator=gen, dtype=torch.float64)
        loss_cfm = ((field(t, (1 - t) * noise + t * data) - (data - noise)) ** 2).mean()
        loss = loss_cfm
        loss_grid = torch.zeros((), dtype=torch.float64)
        if fit_steps:
            loss_grid = sum(grid_loss(field, n, batch, gen) for n in fit_steps) / len(fit_steps)
            loss = loss + CONFIG['grid_weight'] * loss_grid
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (update + 1) % 1000 == 0:
            history.append({'update': update + 1, 'cfm': float(loss_cfm.detach()), 'grid': float(loss_grid.detach())})
    CKPT.mkdir(parents=True, exist_ok=True)
    torch.save(field.state_dict(), CKPT / f'{objective}_s{seed}.pt')
    return field, history


_QUANTILE_CACHE = {}


def target_quantile_at_normal(t, z, cache_key=None):
    """Quantile of p_t at probability Phi(z), by bisection on z <= 0 and symmetry."""
    if cache_key is not None and (cache_key, t) in _QUANTILE_CACHE:
        return _QUANTILE_CACHE[cache_key, t]
    s, m = math.sqrt(variance(t)), MU * t
    neg = -np.abs(z)
    p = ndtr(neg)
    lo = np.full_like(neg, -m - 10.0 * s - 10.0)
    hi = np.zeros_like(neg)
    for _ in range(90):
        mid = 0.5 * (lo + hi)
        below = 0.5 * (ndtr((mid + m) / s) + ndtr((mid - m) / s)) < p
        lo = np.where(below, mid, lo)
        hi = np.where(below, hi, mid)
    value = np.sign(z) * np.abs(0.5 * (lo + hi))
    if cache_key is not None:
        _QUANTILE_CACHE[cache_key, t] = value
    return value


def quadrature():
    z = np.linspace(-CONFIG['quadrature']['z_max'], CONFIG['quadrature']['z_max'],
                    CONFIG['quadrature']['nodes'])
    w = np.exp(-0.5 * z ** 2) / math.sqrt(2 * math.pi) * (z[1] - z[0])
    w[0] *= 0.5
    w[-1] *= 0.5
    return z, w


def w2_to_target(x, t, z, w):
    """W2 between the law of x(G) and p_t; returns (distance, map is increasing)."""
    increasing = bool(np.all(np.diff(x) > 0))
    if increasing:
        return math.sqrt(float(np.sum(w * (x - target_quantile_at_normal(float(t), z, 'grid')) ** 2))), True
    order = np.argsort(x, kind='stable')
    ws = w[order]
    level = np.clip((np.cumsum(ws) - 0.5 * ws) / np.sum(w), 1e-300, 1 - 1e-16)
    return math.sqrt(float(np.sum(ws * (x[order] - target_quantile_at_normal(t, ndtri(level))) ** 2))), False


def evaluate(field):
    z, w = quadrature()
    x0 = torch.tensor(z)
    with torch.no_grad():
        ref_steps = CONFIG['reference']['steps']
        ref = [s.numpy() for s in rk4_states(field, x0, ref_steps)]
        runs = {n: [s.numpy() for s in euler_states(field, x0, n)] for n in STEP_COUNTS + (512,)}
    at = lambda states, steps, t: states[round(t * steps) - 1]
    result = {'increasing': True, 'shared': {}, 'pair': {}}

    def dist(x, t):
        d, inc = w2_to_target(x, t, z, w)
        result['increasing'] = result['increasing'] and inc
        return d

    result['model_bias'] = max(dist(ref[round(t * ref_steps) - 1], t)
                               for t in np.arange(1, 65) / 64)
    for n in STEP_COUNTS:
        result['shared'][n] = {
            'target_w2': max(dist(at(runs[n], n, t), t) for t in SHARED_TIMES),
            'own_flow_rms': max(math.sqrt(float(np.sum(w * (at(runs[n], n, t) - at(ref, ref_steps, t)) ** 2)))
                                for t in SHARED_TIMES)}
        nodes = np.arange(1, n + 1) / n
        result['pair'][n] = {
            'E_c': max(dist(runs[n][k], t) for k, t in enumerate(nodes)),
            'E_f': max(dist(runs[2 * n][2 * k + 1], t) for k, t in enumerate(nodes))}
    return result


def job(args):
    objective, seed, eval_only = args
    torch.set_num_threads(1)
    history = []
    if objective == 'exact':
        field = exact_field
    elif eval_only:
        field = Field(CONFIG['width']).double()
        field.load_state_dict(torch.load(CKPT / f'{objective}_s{seed}.pt'))
    else:
        field, history = train(objective, seed)
    return {'objective': objective, 'seed': seed, 'training': history, **evaluate(field)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval-only', action='store_true')
    parser.add_argument('--workers', type=int, default=6)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [('exact', -1, True)] + [(o, s, args.eval_only) for o in OBJECTIVES for s in SEEDS]
    with mp.get_context('spawn').Pool(args.workers) as pool:
        rows = pool.map(job, jobs, chunksize=1)
    (OUT / 'config.json').write_text(json.dumps(CONFIG, indent=2) + '\n')
    (OUT / 'models.json').write_text(json.dumps(rows, indent=1) + '\n')
    summary = {}
    for objective in ('exact',) + OBJECTIVES:
        group = [r for r in rows if r['objective'] == objective]
        entry = {'models': len(group), 'all_maps_increasing': all(r['increasing'] for r in group)}
        bias = np.array([r['model_bias'] for r in group])
        entry['model_bias'] = {'mean': float(bias.mean()), 'sd': float(bias.std(ddof=1)) if len(group) > 1 else 0.0}
        for n in STEP_COUNTS:
            for block, keys in (('shared', ('target_w2', 'own_flow_rms')), ('pair', ('E_c', 'E_f'))):
                for key in keys:
                    v = np.array([r[block][n][key] for r in group])
                    entry[f'{key}_{n}'] = {'mean': float(v.mean()),
                                           'sd': float(v.std(ddof=1)) if len(group) > 1 else 0.0,
                                           'min': float(v.min()), 'max': float(v.max())}
        if len(group) > 1:
            for a, b in ((8, 16), (8, 64), (8, 256)):
                diff = np.array([r['shared'][b]['target_w2'] - r['shared'][a]['target_w2'] for r in group])
                entry[f'change_{a}_to_{b}'] = {'mean': float(diff.mean()), 'sd': float(diff.std(ddof=1)),
                                               'models_with_increase': int((diff > 0).sum())}
        summary[objective] = entry
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=1) + '\n')
    for objective, entry in summary.items():
        print(objective, 'bias %.4f' % entry['model_bias']['mean'],
              ' '.join('N%d %.4f' % (n, entry[f'target_w2_{n}']['mean']) for n in STEP_COUNTS))


if __name__ == '__main__':
    main()
