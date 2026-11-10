"""Small 1D neural diagnostic; quantile metrics are not certified enclosures."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from scipy.special import ndtr, logsumexp

@dataclass(frozen=True)
class MixturePath:
    weight: float = .35
    mean_left: float = -2.6
    mean_right: float = 1.4
    sigma: float = .6
    def variance(self,t):
        t=np.asarray(t);return (1-t)**2+self.sigma**2*t*t
    @property
    def target_second_moment(self):
        return self.weight*self.mean_left**2+(1-self.weight)*self.mean_right**2+self.sigma**2
    def cdf(self,t,x):
        t,x=np.broadcast_arrays(t,x);s=np.sqrt(self.variance(t))
        return self.weight*ndtr((x-t*self.mean_left)/s)+(1-self.weight)*ndtr((x-t*self.mean_right)/s)
    def quantile(self,t,q):
        t,q=np.broadcast_arrays(np.asarray(t,dtype=float),np.asarray(q,dtype=float))
        if np.any((q<=0)|(q>=1)):raise ValueError('quantiles must lie in (0,1)')
        s=np.sqrt(self.variance(t));lo=np.minimum(t*self.mean_left,t*self.mean_right)-14*s;hi=np.maximum(t*self.mean_left,t*self.mean_right)+14*s
        for _ in range(54):
            mid=(lo+hi)/2;left=self.cdf(t,mid)<q;lo=np.where(left,mid,lo);hi=np.where(left,hi,mid)
        return (lo+hi)/2
    def canonical(self,t,x):
        t,x=np.broadcast_arrays(t,x);v=self.variance(t)
        l=np.stack([np.log(self.weight)-(x-t*self.mean_left)**2/(2*v),np.log(1-self.weight)-(x-t*self.mean_right)**2/(2*v)],axis=-1)
        w=np.exp(l-logsumexp(l,axis=-1,keepdims=True));mean=w[...,0]*self.mean_left+w[...,1]*self.mean_right
        return ((1+self.sigma**2)*t-1)/v*x+(1-t)*mean/v
    def secant(self,t,x,h):
        q=np.clip(self.cdf(t,x),1e-15,1-1e-15)
        return (self.quantile(np.asarray(t)+h,q)-x)/h

class NumpyField:
    """Frozen instantaneous velocity; exported parameters, float64 evaluation."""
    def __init__(self,path:Path|str):
        with np.load(path) as f:
            self.weights=[f[f'w{i}'].astype(float) for i in range(3)];self.biases=[f[f'b{i}'].astype(float) for i in range(3)];self.sigma=float(f['sigma'])
    def __call__(self,t,x):
        x=np.asarray(x,dtype=float);z=np.stack([x/3,np.broadcast_to(2*t-1,x.shape)],axis=-1)
        for i in range(2):z=np.tanh(z@self.weights[i].T+self.biases[i])
        residual=(z@self.weights[2].T+self.biases[2])[...,0]
        a=((1+self.sigma**2)*t-1)/((1-t)**2+self.sigma**2*t*t)
        return a*x+residual

def integrate_fixed(field,initial,n,method='euler'):
    if method not in {'euler','heun','rk3'}:raise ValueError('unknown method')
    h=1/n;x=np.asarray(initial,dtype=float).copy();states=[x.copy()]
    for k in range(n):
        t=k*h;v=field(t,x)
        if method=='euler':x=x+h*v
        elif method=='heun':x=x+h*(v+field(t+h,x+h*v))/2
        else:
            k2=field(t+h/2,x+h*v/2);k3=field(t+h,x-h*v+2*h*k2);x=x+h*(v+4*k2+k3)/6
        states.append(x.copy())
    return np.asarray(states)

def quantile_w2(x,y):
    x=np.asarray(x);y=np.asarray(y)
    if x.shape!=y.shape:raise ValueError('equal sized quantile arrays required')
    return np.sqrt(np.mean((np.sort(x,axis=-1)-np.sort(y,axis=-1))**2,axis=-1))
