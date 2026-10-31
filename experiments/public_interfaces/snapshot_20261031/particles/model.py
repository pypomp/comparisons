"""Dacca parameter broadcasting for the new IF2 interface; PF remains particles.SMC.

The original scalar Euler step, interval, observation law and StateSpaceModel are
imported unchanged. Only parameter conversion, initial-state broadcasting and
lazy coefficient computation are new. No monthly Gaussian approximation.
"""
from pathlib import Path
import os
import sys
import numpy as np

ROOT = Path(__file__).resolve().parent
os.environ['PFB_DACCA_INPUTS'] = str(ROOT/'input/frozen.npz')
sys.path.insert(0, str(ROOT/'adapter'))
from dacca_numpy import Dacca
from bench_particles import DaccaSSM
import dacca_common as dc

INPUT = dc.load_inputs()
for _key in ['cov0','cov_blocks','dt_blocks']:
    INPUT[_key] = np.asarray(INPUT[_key], dtype=np.float64)
NAMES = INPUT['paramnames']
FIXED = ['rho', 'c', 'alpha', 'delta', 'Y_0']
IVPS = ['S_0', 'I_0', 'R1_0', 'R2_0', 'R3_0']
POSITIVE = ['gamma', 'epsilon', 'm', 'sigma', 'tau']
FREE = [n for n in NAMES if n not in FIXED]
STARTS = np.load(ROOT/'input/paper_starts_estimation_free.npy')

def natural(z):
    z = np.asarray(z, dtype=np.float64)
    if z.shape[-1] != len(FREE):
        raise ValueError('Expected 23 estimation coordinates')
    values = {n:z[...,i] for i,n in enumerate(FREE)}
    for n in POSITIVE: values[n] = np.exp(values[n])
    values['beta_trend'] = values['beta_trend']/100
    ivp = np.stack([np.exp(values[n]) for n in IVPS], axis=-1)
    ivp = ivp/ivp.sum(axis=-1, keepdims=True)
    values.update({n:ivp[...,i] for i,n in enumerate(IVPS)})
    values.update({n:np.full(z.shape[:-1], v) for n,v in
                   dict(rho=0., c=1., alpha=1., delta=.02, Y_0=0.).items()})
    if not all(np.isfinite(v).all() for v in values.values()):
        raise FloatingPointError('Nonfinite natural parameters')
    return {n:values[n] for n in NAMES}

class _Coefficients:
    def __init__(self, th, inp, family):
        self.th, self.inp, self.family = th, inp, family
        self.last_i, self.values = None, None
    def __getitem__(self, key):
        i,k = key
        if i != self.last_i:
            cb = self.inp['cov_blocks'][i]
            prefix = 'bs' if self.family == 'beta' else 'omegas'
            co = np.stack([self.th[f'{prefix}{j}'] for j in range(1,7)])
            value = cb[:,3:9] @ co
            if self.family == 'beta':
                value = value + cb[:,0,None]*self.th['beta_trend'][None,:]
            self.values = np.exp(value)
            self.last_i = i
        return self.values[k]

class VectorDacca(Dacca):
    def __init__(self, theta, inp):
        self.th = theta
        self.nstep, self.cov0 = inp['nstep_const'], np.asarray(inp['cov0'],dtype=np.float64)
        self.cov_blocks = np.asarray(inp['cov_blocks'],dtype=np.float64)
        self.dt_blocks = np.asarray(inp['dt_blocks'],dtype=np.float64)
        # Preserve the scalar adapter's float64 covariates and sqrt(dt).
        prepared=dict(inp,cov_blocks=self.cov_blocks)
        self.beta, self.omega = (_Coefficients(theta,prepared,k) for k in ['beta','omega'])
    def rinit(self, n):
        ivp = np.stack([self.th[k] for k in ['S_0','I_0','Y_0','R1_0','R2_0','R3_0']])
        assert ivp.shape == (6,n)
        vals = self.cov0[2]*ivp/ivp.sum(axis=0,keepdims=True)
        out = dict(zip(['S','I','Y','R1','R2','R3'], vals))
        return dict(out, Mn=np.zeros(n), count=np.zeros(n))

def factory(z, inp=None):
    inp = INPUT if inp is None else inp
    z = np.asarray(z)
    return DaccaSSM(VectorDacca(natural(z),inp) if z.ndim==2 else Dacca(natural(z),inp))

def scales(T=600):
    result = np.full((T,len(FREE)), .02)
    for i,n in enumerate(FREE):
        if n in IVPS: result[:,i]=0;result[0,i]=.16
    return result

def native_pf(z,J,seed,T=600,vector=False):
    import particles
    from particles.state_space_models import Bootstrap
    inp = INPUT if T==600 else dc.load_inputs(T)
    np.random.seed(int(seed))
    p = np.tile(z,(J,1)) if vector else z
    alg = particles.SMC(fk=Bootstrap(ssm=factory(p,inp),data=inp['ys']),N=J,
                        resampling='systematic',ESSrmin=1.,collect='off',
                        store_history=False,verbose=False)
    alg.run()
    return alg
