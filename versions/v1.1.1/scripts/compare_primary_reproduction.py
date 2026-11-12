#!/usr/bin/env python3
"""Compare primary checkpoints and non-timing evaluations in two workspaces."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import torch

def scientific(x):
    if isinstance(x,dict): return {k:scientific(v) for k,v in x.items() if k!='seconds'}
    if isinstance(x,list): return [scientific(v) for v in x]
    return x

def main():
    ap=argparse.ArgumentParser();ap.add_argument('original',type=Path);ap.add_argument('reproduced',type=Path);args=ap.parse_args()
    rows=[]
    for p in sorted((args.original/'checkpoints/extended').glob('*.pt')):
        q=args.reproduced/'checkpoints/extended'/p.name
        if not q.exists():raise SystemExit(f'Missing reproduced checkpoint: {q}')
        a=torch.load(p,map_location='cpu',weights_only=True);b=torch.load(q,map_location='cpu',weights_only=True)
        ok=all(k in b['state_dict'] and torch.equal(v,b['state_dict'][k]) for k,v in a['state_dict'].items())
        ra=json.loads((args.original/'results/extended'/f'{p.stem}_evaluation.json').read_text())
        rb=json.loads((args.reproduced/'results/extended'/f'{p.stem}_evaluation.json').read_text())
        rows.append({'model':p.stem,'parameters_identical':ok,'nontiming_metrics_identical':scientific(ra)==scientific(rb)})
    if len(rows)!=18:raise SystemExit(f'Expected 18 primary models, found {len(rows)}')
    print(json.dumps(rows,indent=2))
    if not all(r['parameters_identical'] and r['nontiming_metrics_identical'] for r in rows):raise SystemExit('Reproduction mismatch; inspect numerical/platform differences.')
if __name__=='__main__':main()
