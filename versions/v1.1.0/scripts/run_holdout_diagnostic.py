#!/usr/bin/env python3
"""Fresh, predeclared bounded-witness diagnostic on all nine d=2 models.
Conditional statistical intervals assume iid draws/exact evaluation; no numerical
rounding or PRNG certificate is asserted. All models/resolutions are retained.
"""
from pathlib import Path
import json,math,time
import numpy as np
import torch
from run_sample_only_transfer import Field,target,rollout
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/extended';CK=ROOT/'checkpoints/extended'
torch.set_num_threads(1)
def main():
    n=262144;batch=4096;delta=.05;J=32;steps=[8,16,64]
    names=[f'd2_{m}_s{s}_w48' for m in ['fm','discretization','transfer'] for s in [17,29,43]]
    rng=np.random.default_rng(2026110201)
    x=rng.normal(size=(n,2));tg=rng.normal(size=(n,2));ty=target(n,2,rng).astype('float64')
    # Separate fixed witness bank, selected before this fresh holdout evaluation.
    w=np.random.default_rng(2026110202).normal(size=(2,16));w/=np.maximum(1,np.linalg.norm(w,axis=0))
    def feats(z):
        a=z@w;return np.column_stack((np.cos(a),np.sin(a)))
    targetmeans=np.stack([feats((1-t)*tg+t*ty).mean(0) for t in [.25,.5,.75,1.]])
    qmeans=(len(names)*len(steps)+1)*4*J
    r=math.sqrt(2*math.log(2*qmeans/(delta/2))/n)
    qdiff=len(names)*len(steps)*4*J
    zlog=math.log(4*qdiff/(delta/2))
    records=[]
    with torch.no_grad():
        for name in names:
            ck=torch.load(CK/(name+'.pt'),weights_only=True);f=Field(2,48).double();f.load_state_dict(ck['state_dict']);f.eval()
            rec={'name':name,'resolutions':{}};st=time.perf_counter()
            for ns in steps:
                sums=np.zeros((4,J),dtype=np.float64);diffsq=np.zeros((4,J),dtype=np.float64)
                for lo in range(0,n,batch):
                    zs,_,_=rollout(f,torch.from_numpy(x[lo:lo+batch]),ns)
                    for ji,z in enumerate(zs[1:]):
                        ff=feats(z.numpy());tt=[.25,.5,.75,1.][ji]
                        tf=feats((1-tt)*tg[lo:lo+batch]+tt*ty[lo:lo+batch])
                        sums[ji]+=ff.sum(0);diffsq[ji]+=((ff-tf)**2).sum(0)
                means=sums/n;dv=means-targetmeans;node=np.max(abs(dv),axis=1);val=float(node.max())
                var=np.maximum(0.,(diffsq-n*dv**2)/(n-1))
                eb=np.sqrt(2*var*zlog/n)+28*zlog/(3*(n-1))
                low=float(np.maximum(0,abs(dv)-eb).max());up=float(np.minimum(2,abs(dv)+eb).max())
                rec['resolutions'][str(ns)]={'max_witness_ipm':val,'nodes':node.tolist(),'differences':dv.tolist(),'lower':max(0.,val-2*r),'upper':min(2.,val+2*r),'bernstein_lower':low,'bernstein_upper':up,'paired_variances':var.tolist(),'bernstein_radii':eb.tolist()}
            a=rec['resolutions']['8']
            rec['significant_worsening_vs_8']={str(k):rec['resolutions'][str(k)]['lower']>a['upper'] for k in [16,64]}
            rec['bernstein_worsening_vs_8']={str(k):rec['resolutions'][str(k)]['bernstein_lower']>a['bernstein_upper'] for k in [16,64]}
            rec['seconds']=time.perf_counter()-st;records.append(rec)
            print(name,[(k,round(v['max_witness_ipm'],5)) for k,v in rec['resolutions'].items()],rec['bernstein_worsening_vs_8'],flush=True)
    result={'n':n,'delta':delta,'witnesses':J,'times':[.25,.5,.75,1.],'Qmeans':qmeans,'single_mean_radius':r,'comparison_radius':2*r,'models':records,'bounds_failure_budget':{'Hoeffding':delta/2,'paired_empirical_Bernstein':delta/2},'Qdifferences':qdiff,'sample_seed':2026110201,'witness_seed':2026110202,'conditioning':'all nine frozen d2 checkpoints; all three resolutions; fresh holdout; no post-evaluation selection','scope':'finite-witness metric, not full W2 or asymptotic order; assumes ideal iid sampling and exact witness evaluation; floating-point/PRNG error not certified'}
    (OUT/'holdout_diagnostic.json').write_text(json.dumps(result,indent=2))
if __name__=='__main__':main()
