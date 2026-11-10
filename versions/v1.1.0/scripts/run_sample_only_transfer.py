#!/usr/bin/env python3
"""Sample-only frozen-field experiment. No analytical teacher. Empirical SW2 is
not full W2; common-source RMS is only a coupling upper bound."""
from __future__ import annotations
import argparse,json,math,time,platform
from pathlib import Path
import numpy as np
import torch
from torch import nn
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/extended';CK=ROOT/'checkpoints/extended'
torch.set_num_threads(1)

def target(n,d,rng):
    weights=np.array([.1,.2,.3,.4]);mu=np.array([[-2.,-1.],[1.8,-1.5],[2.,1.6],[-1.6,2.]])
    if d>2:
        a,b=mu[:,0],mu[:,1]
        mu=np.column_stack([a,b,.35*a*b,.3*a*a,.3*b*b,.4*np.sin(a),.5*np.cos(b),.2*(a+b)])[:,:d]
    mu-=weights@mu;comp=rng.choice(4,size=n,p=weights);z=rng.normal(size=(n,d));y=np.empty_like(z)
    for k in range(4):
        q,_=np.linalg.qr(np.random.default_rng(90210+100*d+k).normal(size=(d,d)))
        a=q@np.diag(np.linspace(.30+.04*k,.70+.06*k,d));ix=comp==k;y[ix]=mu[k]+z[ix]@a.T
    return y.astype('float32')

class Field(nn.Module):
    def __init__(self,d,width=48):
        super().__init__();self.d=d
        self.net=nn.Sequential(nn.Linear(d+1,width),nn.Tanh(),nn.Linear(width,width),nn.Tanh(),nn.Linear(width,d))
    def forward(self,t,x):
        if not torch.is_tensor(t):t=torch.full((len(x),1),float(t),dtype=x.dtype,device=x.device)
        elif t.ndim==0:t=t.expand(len(x),1)
        return self.net(torch.cat((x,t),1))

def rollout(f,x,n,method='euler',jitter=0.):
    if n%4:raise ValueError('steps must be divisible by four')
    raw=1+jitter*np.sin(2*np.pi*(np.arange(n)+.5)/n)
    grid=np.r_[0.,np.cumsum(raw/raw.sum())] if jitter else np.linspace(0,1,n+1)
    nodes={0:x};nfe=0
    for k in range(n):
        t=float(grid[k]);h=float(grid[k+1]-grid[k]);a=f(t,x);nfe+=1
        if method=='euler':x=x+h*a
        elif method=='heun':b=f(t+h,x+h*a);x=x+h*(a+b)/2;nfe+=1
        elif method=='rk3':
            b=f(t+h/2,x+h*a/2);c=f(t+h,x-h*a+2*h*b);x=x+h*(a+4*b+c)/6;nfe+=2
        elif method=='rk4':
            b=f(t+h/2,x+h*a/2);c=f(t+h/2,x+h*b/2);e=f(t+h,x+h*c);x=x+h*(a+2*b+2*c+e)/6;nfe+=3
        else:raise ValueError(method)
        if (k+1)%(n//4)==0:nodes[k+1]=x
    return list(nodes.values()),nfe,[float(grid[k]) for k in sorted(nodes)]

def features(x,w):
    z=x@w;return torch.cat((z.cos(),z.sin()),1)
def mmd_loss(x,y,w):
    return ((features(x,w).mean(0)-features(y,w).mean(0))**2).mean()
def cfm_batch(f,g,y,rng,b=256):
    x0=g[torch.from_numpy(rng.integers(len(g),size=b))];x1=y[torch.from_numpy(rng.integers(len(y),size=b))]
    t=torch.from_numpy(rng.random((b,1)).astype('float32'));return ((f(t,(1-t)*x0+t*x1)-(x1-x0))**2).mean()

def train(d,seed,method,steps=1800,width=48,reg=30.,jacobian_penalty=0.):
    torch.manual_seed(seed);rng=np.random.default_rng(seed+100*d)
    g=torch.from_numpy(rng.normal(size=(32768,d)).astype('float32'));y=torch.from_numpy(target(32768,d,rng))
    vg=torch.from_numpy(rng.normal(size=(8192,d)).astype('float32'));vy=torch.from_numpy(target(8192,d,rng))
    w=torch.from_numpy(np.random.default_rng(155+d).normal(size=(d,64)).astype('float32')/math.sqrt(d))
    f=Field(d,width);opt=torch.optim.Adam(f.parameters(),lr=.001);log=[];st=time.perf_counter()
    for it in range(steps):
        opt.zero_grad();loss=cfm_batch(f,g,y,rng);base=float(loss.detach());extra=torch.tensor(0.)
        if method!='fm' and it%4==0:
            x0=g[torch.from_numpy(rng.integers(len(g),size=128))]
            tx0=g[torch.from_numpy(rng.integers(len(g),size=128))];tx1=y[torch.from_numpy(rng.integers(len(y),size=128))]
            co,_,times=rollout(f,x0,8);targ=[(1-t)*tx0+t*tx1 for t in times]
            lc=sum(mmd_loss(co[j],targ[j],w) for j in range(1,5))/4
            if method=='discretization':extra=lc
            elif method=='transfer':
                fi,_,_=rollout(f,x0,16);lf=sum(mmd_loss(fi[j],targ[j],w) for j in range(1,5))/4
                con=sum(mmd_loss(co[j],fi[j],w) for j in range(1,5))/4;extra=(lc+lf)/2+con
            else:raise ValueError(method)
            loss=loss+reg*extra
        if jacobian_penalty and it%4==0:
            # Hutchinson penalty on spatial Jacobian at independent minibatch states.
            xx=g[torch.from_numpy(rng.integers(len(g),size=64))].clone().requires_grad_(True)
            tt=torch.from_numpy(rng.random((64,1)).astype('float32'))
            vv=torch.from_numpy(rng.choice([-1.,1.],size=(64,d)).astype('float32'))
            jj=torch.autograd.grad((f(tt,xx)*vv).sum(),xx,create_graph=True)[0]
            loss=loss+jacobian_penalty*(jj*jj).sum(1).mean()
        loss.backward();torch.nn.utils.clip_grad_norm_(f.parameters(),10.);opt.step()
        if it in (0,steps-1) or (it+1)%300==0:
            with torch.no_grad():val=cfm_batch(f,vg,vy,np.random.default_rng(10000+seed),4096).item()
            log.append({'update':it+1,'cfm_train':base,'extra':float(extra.detach()),'val_cfm':val})
    elapsed=time.perf_counter()-st;name=f'd{d}_{method}_s{seed}_w{width}'
    torch.save({'state_dict':f.state_dict(),'d':d,'width':width,'seed':seed,'method':method,'steps':steps,'reg':reg,'jacobian_penalty':jacobian_penalty},CK/(name+'.pt'))
    (OUT/(name+'_training.json')).write_text(json.dumps({'seconds':elapsed,'log':log,'n_parameters':sum(p.numel() for p in f.parameters()),'checkpoint_rule':'final fixed update; no test-based selection'},indent=2))
    print('TRAIN',name,'seconds',round(elapsed,2),'val',log[-1]['val_cfm'],flush=True);return name

def sw2(x,y,dirs):return float(np.sqrt(np.mean((np.sort(x@dirs,axis=0)-np.sort(y@dirs,axis=0))**2)))
def evaluate(name,n=1024):
    data=torch.load(CK/(name+'.pt'),map_location='cpu',weights_only=True);d=data['d'];f=Field(d,data['width']);f.load_state_dict(data['state_dict']);f.eval()
    rng=np.random.default_rng(701000+d);x0=rng.normal(size=(n,d)).astype('float32');tg=rng.normal(size=(n,d)).astype('float32');ty=target(n,d,rng)
    dirs=rng.normal(size=(d,128));dirs/=np.linalg.norm(dirs,axis=0);wit=rng.normal(size=(d,16));wit/=np.maximum(1,np.linalg.norm(wit,axis=0))
    def witness(x):z=x@wit;return np.concatenate([np.cos(z),np.sin(z)],axis=1)
    with torch.no_grad():
        ref,refnfe,rt=rollout(f,torch.from_numpy(x0),256,'rk4');ref=[z.numpy() for z in ref]
        check,_,_=rollout(f,torch.from_numpy(x0[:256]),512,'rk4')
        refcheck=max(float(np.sqrt(np.mean(np.sum((z.numpy()-ref[j][:256])**2,axis=1)))) for j,z in enumerate(check))
        configs=[('euler',k,0.) for k in [4,8,16,32,64,128]]
        configs +=[(m,b//c,0.) for b in [24,48,96,192] for m,c in [('euler',1),('heun',2),('rk3',3)]]
        configs +=[('euler',k,.8) for k in [8,16,32,64]];rows=[]
        for method,steps,jit in configs:
            st=time.perf_counter();nodes,nfe,times=rollout(f,torch.from_numpy(x0),steps,method,jit);elapsed=time.perf_counter()-st
            terr=[];own=[];ipms=[];raw=[]
            for j,(z,t) in enumerate(zip(nodes,times)):
                a=z.numpy();y=(1-t)*tg+t*ty;terr.append(sw2(a,y,dirs));dm=witness(a).mean(0)-witness(y).mean(0);ipms.append(float(np.max(np.abs(dm))));raw.append(dm.tolist())
                if jit==0:own.append(float(np.sqrt(np.mean(np.sum((a-ref[j])**2,axis=1)))))
            rows.append({'method':method,'steps':steps,'nfe':nfe,'jitter':jit,'seconds':elapsed,'times':times,'sw2_nodes':terr,'max_sw2':max(terr),'terminal_sw2':terr[-1],'paired_own_flow_rms':max(own) if own else None,'witness_ipm_nodes':ipms,'witness_differences':raw})
        bias=max(sw2(ref[j],(1-t)*tg+t*ty,dirs) for j,t in enumerate(rt))
    result={'name':name,'d':d,'seed':data['seed'],'training_method':data['method'],'test_n':n,'directions':128,'witnesses':32,'reference':'RK4 256; checked 512 on first 256 iid sources','reference_nfe':refnfe,'reference_refinement_rms':refcheck,'sampled_reference_sw2':bias,'rows':rows,'metric_warning':'Empirical fixed-projection SW2 is not full W2. RMS is a coupling upper bound. Float32 refinement agreement is not a certified error bound.'}
    (OUT/(name+'_evaluation.json')).write_text(json.dumps(result,indent=2));print('EVAL',name,'refcheck',refcheck,flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--eval-only',action='store_true');ap.add_argument('--steps',type=int,default=1800);ap.add_argument('--dims',type=int,nargs='+',default=[2,8]);ap.add_argument('--seeds',type=int,nargs='+',default=[17,29,43]);ap.add_argument('--methods',nargs='+',default=['fm','discretization','transfer']);ap.add_argument('--width',type=int,default=48);args=ap.parse_args();OUT.mkdir(exist_ok=True,parents=True);CK.mkdir(exist_ok=True,parents=True)
    for d in args.dims:
        for seed in args.seeds:
            for method in args.methods:
                name=f'd{d}_{method}_s{seed}_w{args.width}'
                if not args.eval_only:train(d,seed,method,args.steps,args.width)
                evaluate(name)
    (OUT/'environment.json').write_text(json.dumps({'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,'threads':1,'cuda':False,'command_args':vars(args)},indent=2))
if __name__=='__main__':main()
