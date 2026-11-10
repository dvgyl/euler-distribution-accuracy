#!/usr/bin/env python3
"""CPU diagnostic; analytic Bayes/secant teachers, separate validation, frozen test.
Not training from an unknown data distribution. No certified W2 intervals.
"""
from __future__ import annotations
import argparse,csv,json,sys,time,platform
from pathlib import Path
import numpy as np
from scipy.special import ndtri
from scipy.integrate import solve_ivp
import torch
from torch import nn
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from euler_path.neural_diagnostic import MixturePath,NumpyField,integrate_fixed,quantile_w2
class Field(nn.Module):
    def __init__(self,sigma,width=48):
        super().__init__();self.sigma=sigma;self.net=nn.Sequential(nn.Linear(2,width),nn.Tanh(),nn.Linear(width,width),nn.Tanh(),nn.Linear(width,1))
    def forward(self,t,x):
        z=torch.stack([x/3,2*t-1],dim=-1);a=((1+self.sigma**2)*t-1)/((1-t)**2+self.sigma**2*t*t)
        return a*x+self.net(z).squeeze(-1)
def dataset(path,n,seed,h):
    rng=np.random.default_rng(seed);t=rng.uniform(0,1,n);t[:n//20]=0;t[n//20:n//10]=1;q=rng.uniform(1e-7,1-1e-7,n);x=path.quantile(t,q)
    return {'t':t,'x':x,'canonical':path.canonical(t,x),'secant':(path.quantile(t+h,q)-x)/h}
def writecsv(path,rows):
    with path.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--steps',type=int,default=6000);ap.add_argument('--seeds',type=int,nargs='+',default=[17,29,43]);ap.add_argument('--width',type=int,default=48);ap.add_argument('--test-points',type=int,default=512);ap.add_argument('--eval-only',action='store_true');args=ap.parse_args()
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    out=ROOT/'results'/'transfer'/'neural';out.mkdir(parents=True,exist_ok=True);p=MixturePath();hfit=1/16
    config={'args':vars(args),'path':p.__dict__,'hfit':hfit,'train_seed':811,'validation_seed':1823,'training_count':32768,'validation_count':8192,'teacher':'analytic Bayes velocity or fixed-h quantile secant','time_sampling':'90% uniform, 5% t=0, 5% t=1','dtype_training':'float32','dtype_evaluation':'float64','torch':torch.__version__,'python':platform.python_version(),'cpu_threads':1,'test_metric':'equal-weight quantile approximation of full 1D W2, not certified'}
    (out/('evaluation_config.json' if args.eval_only else 'config.json')).write_text(json.dumps(config,indent=2));train=dataset(p,32768,811,hfit);val=dataset(p,8192,1823,hfit)
    np.savez_compressed(out/'data_splits.npz',**{f'train_{k}':v for k,v in train.items()},**{f'val_{k}':v for k,v in val.items()})
    training=[];curves=[];rows=[];baselines=[];refs=[];transfer=[];refinement=[]
    q=(np.arange(args.test_points)+.5)/args.test_points;initial=ndtri(q);times=np.linspace(0,1,257);targets=p.quantile(times[:,None],q[None,:])
    for label in ['canonical','secant']:
      for seed in args.seeds:
        file=out/f'{label}_seed{seed}.npz'
        if not args.eval_only:
            torch.manual_seed(seed);model=Field(p.sigma,args.width);opt=torch.optim.Adam(model.parameters(),lr=.003)
            tr={k:torch.tensor(v,dtype=torch.float32) for k,v in train.items()};va={k:torch.tensor(v,dtype=torch.float32) for k,v in val.items()}
            best=float('inf');selected=0;state=None;started=time.perf_counter()
            for step in range(args.steps):
                lr=.00015+.00285*.5*(1+np.cos(np.pi*step/args.steps))
                for group in opt.param_groups:group['lr']=lr
                idx=torch.randint(len(tr['x']),(512,));loss=(model(tr['t'][idx],tr['x'][idx])-tr[label][idx]).square().mean();opt.zero_grad();loss.backward();opt.step()
                if (step+1)%100==0 or step==args.steps-1:
                    with torch.no_grad():vl=float((model(va['t'],va['x'])-va[label]).square().mean())
                    curves.append({'teacher':label,'seed':seed,'step':step+1,'train_batch_mse':float(loss.detach()),'validation_mse':vl})
                    if vl<best:best=vl;selected=step+1;state={k:v.detach().clone() for k,v in model.state_dict().items()}
            model.load_state_dict(state);train_seconds=time.perf_counter()-started;arrays={'sigma':np.array(p.sigma)}
            for i,j in enumerate([0,2,4]):arrays[f'w{i}']=model.net[j].weight.detach().numpy();arrays[f'b{i}']=model.net[j].bias.detach().numpy()
            np.savez_compressed(file,**arrays);field=NumpyField(file);exx=np.linspace(-5,5,100);ext=np.linspace(0,1,100)
            with torch.no_grad():exy=model(torch.tensor(ext,dtype=torch.float32),torch.tensor(exx,dtype=torch.float32)).numpy()
            export=float(np.max(np.abs(exy-field(ext,exx))));training.append({'teacher':label,'seed':seed,'selected_step':selected,'validation_mse':best,'training_seconds':train_seconds,'parameters':sum(x.numel() for x in model.parameters()),'export_max_discrepancy':export})
            print(f'TRAIN {label} seed{seed}: val {best:.5g}, step {selected}, seconds {train_seconds:.2f}',flush=True)
        field=NumpyField(file);start=time.perf_counter();sol=solve_ivp(field,(0,1),initial,method='DOP853',rtol=2e-10,atol=2e-12,dense_output=True);elapsed=time.perf_counter()-start
        if not sol.success:raise RuntimeError(sol.message)
        exact=sol.sol(times).T;modelbias=quantile_w2(exact,targets);tighter=solve_ivp(field,(0,1),initial,method='DOP853',rtol=2e-12,atol=2e-14,dense_output=True);
        if not tighter.success:raise RuntimeError(tighter.message)
        reference_delta=float(quantile_w2(tighter.sol(times).T,exact).max());source_excess=float(np.mean((field(0,initial)+initial)**2))
        refs.append({'teacher':label,'seed':seed,'model_bias_max':float(modelbias.max()),'model_bias_terminal':float(modelbias[-1]),'source_excess':source_excess,'bayes_source_loss':p.target_second_moment,'reference_nfe':sol.nfev,'reference_seconds':elapsed,'reference_tolerance_refinement':reference_delta})
        predictions={}
        for n in [8,16,32,64,128,256]:
            start=time.perf_counter();states=integrate_fixed(field,initial,n);duration=time.perf_counter()-start;predictions[n]=states;target=targets[::256//n];ref=exact[::256//n];et=quantile_w2(states,target);ei=quantile_w2(states,ref)
            rows.append({'teacher':label,'seed':seed,'N':n,'nfe':n,'target_max':float(et.max()),'target_terminal':float(et[-1]),'own_max':float(ei.max()),'own_terminal':float(ei[-1]),'wall_seconds':duration})
        for n in [8,16,32,64,128]:
            fine=predictions[2*n][::2];coarse=predictions[n];tar=targets[::256//n];transfer.append({'teacher':label,'seed':seed,'N':n,'coarse_target':float(quantile_w2(coarse,tar).max()),'fine_common_target':float(quantile_w2(fine,tar).max()),'cross_grid':float(quantile_w2(coarse,fine).max())})
        for budget in [24,48,96,192]:
          for method,cost in [('euler',1),('heun',2),('rk3',3)]:
            n=budget//cost;grid=np.linspace(0,1,n+1);start=time.perf_counter();states=integrate_fixed(field,initial,n,method);duration=time.perf_counter()-start;tar=p.quantile(grid[:,None],q);ref=sol.sol(grid).T
            baselines.append({'teacher':label,'seed':seed,'method':method,'nfe':budget,'steps':n,'target_max':float(quantile_w2(states,tar).max()),'own_max':float(quantile_w2(states,ref).max()),'target_terminal':float(quantile_w2(states[-1],tar[-1])),'wall_seconds':duration})
        qq=(np.arange(2*args.test_points)+.5)/(2*args.test_points);xx=ndtri(qq);s2=solve_ivp(field,(0,1),xx,method='DOP853',rtol=2e-10,atol=2e-12,dense_output=True);
        if not s2.success:raise RuntimeError(s2.message)
        tt=p.quantile(times[:,None],qq[None,:]);bb=float(quantile_w2(s2.sol(times).T,tt).max())
        for n in [16,64,256]:
            ss=integrate_fixed(field,xx,n);ee=float(quantile_w2(ss,tt[::256//n]).max());old=next(r for r in rows if r['teacher']==label and r['seed']==seed and r['N']==n)['target_max'];refinement.append({'teacher':label,'seed':seed,'N':n,'base_quantiles':args.test_points,'refined_quantiles':2*args.test_points,'base_target_max':old,'refined_target_max':ee,'abs_change':abs(ee-old),'refined_model_bias':bb})
        print(f'EVAL {label} seed{seed}: bias {modelbias.max():.5g}, source_excess {source_excess:.5g}',flush=True)
    for name,data in [('training',training),('training_curves',curves),('euler',rows),('baselines',baselines),('reference',refs),('two_grid',transfer),('quantile_refinement',refinement)]:
        if data:writecsv(out/f'{name}.csv',data)
    print('Completed all selected checkpoints and test runs.',flush=True)
if __name__=='__main__':main()
