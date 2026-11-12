"""Variability of empirical sliced Wasserstein comparisons across evaluation sets.

The nine two-dimensional fields and the 128 projection directions are fixed.
Only the evaluation samples change. For each evaluation-set size the script
draws independent sets and records the empirical SW2 at 8, 16, and 64 Euler
steps (maximum over the five diagnostic times), so that the spread of a
step-count comparison across evaluation sets can be reported.
"""
import argparse
import importlib.util
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results' / 'evaluation_set_variability'
STEPS = (8, 16, 64)
TIMES = (0.0, 0.25, 0.5, 0.75, 1.0)
NAMES = [f'd2_{method}_s{seed}_w48' for method in ('fm', 'discretization', 'transfer') for seed in (17, 29, 43)]
DESIGN = {1024: 200, 16384: 30}  # add 262144: n for larger sets
LARGE_SET_WORKERS = 2  # one large set needs about 1.5 GB of memory
BASE_SEED = 2026110900


def load_protocol():
    spec = importlib.util.spec_from_file_location('sample_protocol', ROOT / 'scripts' / 'run_sample_only_transfer.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def published_directions(protocol):
    """The 128 directions of the published evaluation, regenerated from its seed."""
    rng = np.random.default_rng(701002)
    rng.normal(size=(1024, 2))
    rng.normal(size=(1024, 2))
    protocol.target(1024, 2, rng)
    directions = rng.normal(size=(2, 128))
    return directions / np.linalg.norm(directions, axis=0)


def sorted_projections(x, directions):
    return np.sort(x @ directions, axis=0)


def evaluate_set(args):
    size, index = args
    torch.set_num_threads(1)
    protocol = load_protocol()
    directions = published_directions(protocol)
    rng = np.random.default_rng([BASE_SEED, size, index])
    start = rng.normal(size=(size, 2))
    noise = rng.normal(size=(size, 2))
    data = protocol.target(size, 2, rng).astype('float64')
    targets = [sorted_projections((1 - t) * noise + t * data, directions) for t in TIMES]
    row = {'size': size, 'set': index}
    for name in NAMES:
        state = torch.load(ROOT / 'checkpoints' / 'extended' / (name + '.pt'), weights_only=True)
        field = protocol.Field(2, 48).double()
        field.load_state_dict(state['state_dict'])
        field.eval()
        for steps in STEPS:
            nodes = [np.empty_like(start) for _ in TIMES]
            with torch.no_grad():
                for lo in range(0, size, 8192):
                    out, _, _ = protocol.rollout(field, torch.from_numpy(start[lo:lo + 8192]), steps)
                    for j, node in enumerate(out):
                        nodes[j][lo:lo + 8192] = node.numpy()
            row[f'{name}_{steps}'] = max(
                float(np.sqrt(np.mean((sorted_projections(node, directions) - target) ** 2)))
                for node, target in zip(nodes, targets))
    return row


def summarize(rows):
    sizes = sorted({r['size'] for r in rows})
    summary = {'design': {str(size): sum(r['size'] == size for r in rows) for size in sizes},
               'base_seed': BASE_SEED, 'steps': STEPS, 'times': TIMES,
               'directions': 'the 128 published directions (seed 701002)', 'models': {},
               'six_fitted_models_8_to_16': {}}
    for name in NAMES:
        entry = {}
        for size in sizes:
            group = [r for r in rows if r['size'] == size]
            values = {steps: np.array([r[f'{name}_{steps}'] for r in group]) for steps in STEPS}
            stats = {f'sw2_{steps}': {'mean': float(v.mean()), 'sd': float(v.std(ddof=1))} for steps, v in values.items()}
            for a, b in ((8, 16), (8, 64)):
                diff = values[b] - values[a]
                stats[f'change_{a}_to_{b}'] = {'mean': float(diff.mean()), 'sd': float(diff.std(ddof=1)),
                                               'fraction_of_sets_with_increase': float((diff > 0).mean())}
            entry[str(size)] = stats
        summary['models'][name] = entry
    six = [n for n in NAMES if '_fm_' not in n]
    for size in sizes:
        group = [r for r in rows if r['size'] == size]
        summary['six_fitted_models_8_to_16'][str(size)] = {
            'sets': len(group),
            'fraction_of_sets_where_all_six_increase':
                float(np.mean([all(r[f'{n}_16'] > r[f'{n}_8'] for n in six) for r in group])),
            'fraction_of_sets_where_any_increases':
                float(np.mean([any(r[f'{n}_16'] > r[f'{n}_8'] for n in six) for r in group]))}
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=1) + '\n')
    print(json.dumps(summary['six_fitted_models_8_to_16'], indent=1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--summarize-only', action='store_true', help='summarize the saved sets.json')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.summarize_only:
        summarize(json.loads((OUT / 'sets.json').read_text()))
        return
    rows = []
    for size, count in sorted(DESIGN.items()):
        workers = LARGE_SET_WORKERS if size > 100_000 else args.workers
        with mp.get_context('spawn').Pool(workers) as pool:
            rows += pool.map(evaluate_set, [(size, index) for index in range(count)], chunksize=1)
        (OUT / 'sets.json').write_text(json.dumps(rows) + '\n')
        print('finished size', size, flush=True)
    summarize(rows)


if __name__ == '__main__':
    main()
