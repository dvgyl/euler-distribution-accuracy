"""Evaluate saved fields on a fixed larger held-out sample bank."""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import math
import platform
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT
OUT = ROOT / 'results/large_bank'
spec = importlib.util.spec_from_file_location('sample_protocol', SOURCE / 'scripts/run_sample_only_transfer.py')
protocol = importlib.util.module_from_spec(spec)
spec.loader.exec_module(protocol)


def primary_directions():
    rng = np.random.default_rng(701002)
    rng.normal(size=(1024, 2))
    rng.normal(size=(1024, 2))
    protocol.target(1024, 2, rng)
    directions = rng.normal(size=(2, 128))
    return directions / np.linalg.norm(directions, axis=0)


def feature_banks():
    training = np.random.default_rng(157).normal(size=(2, 64)).astype('float32') / math.sqrt(2)
    independent = np.random.default_rng(2026110202).normal(size=(2, 16))
    independent /= np.maximum(1, np.linalg.norm(independent, axis=0))
    rng = np.random.default_rng(2026110204)
    tanh = rng.normal(size=(2, 32))
    tanh /= np.maximum(1, np.linalg.norm(tanh, axis=0))
    offsets = rng.uniform(-2, 2, size=32)
    return training.astype('float64'), independent, tanh, offsets


def feature_values(x, bank, kind, offsets=None):
    z = x @ bank
    if kind == 'tanh':
        return np.tanh(z - offsets)
    return np.concatenate((np.cos(z), np.sin(z)), axis=1)


def feature_means(x, banks, batch):
    result = {}
    for kind, bank, offsets in [('training', banks[0], None), ('independent_trig', banks[1], None), ('independent_tanh', banks[2], banks[3])]:
        sums = np.zeros(bank.shape[1] * (1 if kind == 'independent_tanh' else 2))
        for lo in range(0, len(x), batch):
            sums += feature_values(x[lo:lo+batch], bank, 'tanh' if kind == 'independent_tanh' else 'trig', offsets).sum(axis=0)
        result[kind] = sums / len(x)
    return result


def infer_nodes(field, x, steps, batch):
    result = [np.empty_like(x) for _ in range(5)]
    with torch.no_grad():
        for lo in range(0, len(x), batch):
            nodes, _, _ = protocol.rollout(field, torch.from_numpy(x[lo:lo+batch]), steps)
            for j, node in enumerate(nodes):
                result[j][lo:lo+batch] = node.numpy()
    return result


def sorted_projections(x, directions, path):
    result = np.lib.format.open_memmap(path, mode='w+', dtype='float64', shape=(directions.shape[1], len(x)))
    for lo in range(0, directions.shape[1], 16):
        result[lo:lo+16] = np.sort(x @ directions[:, lo:lo+16], axis=0).T
    result.flush()
    return result


def projection_squared_errors(x, target_sorted, directions):
    result = np.empty(directions.shape[1])
    for lo in range(0, directions.shape[1], 16):
        projected = np.sort(x @ directions[:, lo:lo+16], axis=0).T
        result[lo:lo+16] = np.mean((projected - target_sorted[lo:lo+16]) ** 2, axis=1)
    return result


def load_field(name, double=True):
    checkpoint = SOURCE / 'checkpoints/extended' / (name + '.pt')
    data = torch.load(checkpoint, weights_only=True)
    field = protocol.Field(2, 48)
    if double:
        field = field.double()
    field.load_state_dict(data['state_dict'])
    field.eval()
    return field


def primary_spread():
    records = []
    for d in (2, 8):
        for method in ('fm', 'discretization', 'transfer'):
            entries = []
            for seed in (17, 29, 43):
                data = json.loads((SOURCE / 'results/extended' / f'd{d}_{method}_s{seed}_w48_evaluation.json').read_text())
                chosen = [next(r for r in data['rows'] if r['method']=='euler' and r['steps']==steps and r['jitter']==0) for steps in (8,16,64)]
                entries.append([seed, *[r['max_sw2'] for r in chosen], *[r['paired_own_flow_rms'] for r in chosen[:2]]])
            values = np.array([row[1:] for row in entries])
            records.append({'dimension':d,'method':method,'columns':['sw2_8','sw2_16','sw2_64','rms_8','rms_16'],'seeds':entries,'means':values.mean(axis=0).tolist(),'sample_standard_deviations':values.std(axis=0,ddof=1).tolist()})
    (OUT/'primary_spread.json').write_text(json.dumps(records,indent=2)+'\n')


def precision_check(names, directions):
    rng = np.random.default_rng(701002)
    x = rng.normal(size=(1024,2)).astype('float32')
    g = rng.normal(size=(1024,2)).astype('float32')
    y = protocol.target(1024,2,rng)
    records = []
    for name in names:
        record={'name':name,'resolutions':{}}
        for steps in (8,16,64):
            with torch.no_grad():
                single,_,times=protocol.rollout(load_field(name,False),torch.from_numpy(x),steps)
                double,_,_=protocol.rollout(load_field(name,True),torch.from_numpy(x.astype('float64')),steps)
            sm=[protocol.sw2(z.numpy(),(1-t)*g+t*y,directions) for z,t in zip(single,times)]
            dm=[protocol.sw2(z.numpy(),(1-t)*g.astype('float64')+t*y.astype('float64'),directions) for z,t in zip(double,times)]
            stored=json.loads((SOURCE/'results/extended'/(name+'_evaluation.json')).read_text())
            prior=next(r['max_sw2'] for r in stored['rows'] if r['method']=='euler' and r['steps']==steps and r['jitter']==0)
            assert abs(max(sm)-prior)<1e-6,(name,steps,max(sm),prior)
            record['resolutions'][str(steps)]={'stored_primary':prior,'reevaluated_float32':max(sm),'reevaluation_difference':max(sm)-prior,'float64':max(dm),'precision_difference':max(dm)-max(sm)}
        records.append(record)
    (OUT/'precision_check.json').write_text(json.dumps(records,indent=2)+'\n')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--n',type=int,default=262144)
    parser.add_argument('--batch',type=int,default=8192)
    args=parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/'raw').mkdir(exist_ok=True)
    (OUT/'projection_cache').mkdir(exist_ok=True)
    names=[f'd2_{method}_s{seed}_w48' for method in ('fm','discretization','transfer') for seed in (17,29,43)]
    directions=primary_directions();banks=feature_banks()
    config={'samples':args.n,'inference_batch':args.batch,'projection_count':128,'sample_seed':2026110201,'times':[0,.25,.5,.75,1.],'steps':[8,16,64],'dtype':'float64','torch_threads':1,'python':platform.python_version(),'numpy':np.__version__,'torch':torch.__version__,'metric':'sqrt(mean of squared sorted projection displacement over samples and directions)','checkpoint_hashes':{name:hashlib.sha256((SOURCE/'checkpoints/extended'/(name+'.pt')).read_bytes()).hexdigest() for name in names},'independent_trig_seed':2026110202,'independent_tanh_seed':2026110204,'scope':'Descriptive fixed-model diagnostic. Not a population Wasserstein or causal feature-family guarantee.'}
    (OUT/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    primary_spread();precision_check(names,directions)
    rng=np.random.default_rng(2026110201)
    x=rng.normal(size=(args.n,2));g=rng.normal(size=(args.n,2));y=protocol.target(args.n,2,rng).astype('float64')
    np.savez_compressed(OUT/'draws_and_features.npz',source=x,target_source=g,target_endpoint=y,directions=directions,training_features=banks[0],independent_trig=banks[1],independent_tanh=banks[2],tanh_offsets=banks[3])
    target_sorted=[];target_means=[]
    st=time.perf_counter()
    for j,t in enumerate(config['times']):
        target=(1-t)*g+t*y
        path=OUT/'projection_cache'/f'target_{j}.npy'
        target_sorted.append(np.load(path,mmap_mode='r') if path.exists() else sorted_projections(target,directions,path))
        target_means.append(feature_means(target,banks,args.batch))
    config['target_preparation_seconds']=time.perf_counter()-st
    (OUT/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    records=[]
    for name in names:
        record_path=OUT/(name+'.json')
        if record_path.exists():
            previous=json.loads(record_path.read_text())
            if set(previous.get('resolutions',{}))==set(map(str,config['steps'])):
                records.append(previous);continue
        field=load_field(name)
        record={'name':name,'resolutions':{}}
        for steps in config['steps']:
            st=time.perf_counter()
            nodes=infer_nodes(field,x,steps,args.batch)
            inference_seconds=time.perf_counter()-st
            np.savez_compressed(OUT/'raw'/f'{name}_euler{steps}.npz',**{f't{j}':node for j,node in enumerate(nodes[1:],1)})
            values=[]
            for j,(node,t) in enumerate(zip(nodes,config['times'])):
                per_direction=projection_squared_errors(node,target_sorted[j],directions)
                means=feature_means(node,banks,args.batch)
                difference={family:(means[family]-target_means[j][family]).tolist() for family in means}
                values.append({'time':t,'sw2':float(np.sqrt(per_direction.mean())),'projection_squared_errors':per_direction.tolist(),'feature_mean_differences':difference,'feature_maximum_absolute_difference':{family:float(np.max(np.abs(v))) for family,v in difference.items()}})
            maxima={'sw2':max(v['sw2'] for v in values)}
            for family in target_means[0]:maxima[family]=max(v['feature_maximum_absolute_difference'][family] for v in values[1:])
            record['resolutions'][str(steps)]={'times':values,'maximum_over_times':maxima,'inference_seconds':inference_seconds,'total_seconds':time.perf_counter()-st}
            print(name,steps,maxima,'seconds',round(time.perf_counter()-st,2),flush=True)
            record_path.write_text(json.dumps(record,indent=2)+'\n')
        records.append(record)
    (OUT/'large_bank_summary.json').write_text(json.dumps({'config':config,'models':records},indent=2)+'\n')
    print('Complete',flush=True)

if __name__=='__main__':
    main()
