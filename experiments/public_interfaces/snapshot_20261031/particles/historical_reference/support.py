"""Dacca preflight for the official particles CPU implementation.

Nelder-Mead follows the optimizer combination in book/mle/mle_neuro.py.
This is a finite-call feasibility test, not a completed optimizer comparison.
"""
from pathlib import Path
import os
ROOT = Path(__file__).resolve().parent
os.environ['PFB_DACCA_INPUTS'] = str(ROOT/'input/frozen.npz')
os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import sys
sys.path.insert(0, str(ROOT/'adapter'))
import argparse
import ast
import hashlib
import importlib.metadata
import json
import platform
import socket
import time
import numpy as np
from scipy import optimize
from scipy.special import logsumexp, expit
import particles
from particles import datasets, distributions as dists, state_space_models as ssm
from bench_particles import DaccaSSM
from dacca_numpy import Dacca
import dacca_common as dc

FIXED = ['rho','c','alpha','delta','Y_0']
IVPS = ['S_0','I_0','R1_0','R2_0','R3_0']
POSITIVE = ['gamma','epsilon','m','sigma','tau']
DATA = dict(np.load(ROOT/'input/frozen.npz'))
NAMES = DATA['paramnames'].tolist()
FREE = [n for n in NAMES if n not in FIXED]
STARTS = np.load(ROOT/'input/paper_starts_estimation_free.npy')

def safe(value):
    if isinstance(value,dict):return {k:safe(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [safe(v) for v in value]
    if isinstance(value,np.ndarray):return safe(value.tolist())
    if isinstance(value,(np.bool_,bool)):return bool(value)
    if isinstance(value,np.integer):return int(value)
    if isinstance(value,(float,np.floating)):return float(value) if np.isfinite(value) else None
    return value

class RecordedSMC(particles.SMC):
    """Native SMC with diagnostics after its unchanged summary computation."""
    def __init__(self,*args,**kwargs):
        self.recorded_times=[];self.recorded_increments=[]
        super().__init__(*args,**kwargs)

    def compute_summaries(self):
        super().compute_summaries()
        self.recorded_times.append(int(self.t))
        self.recorded_increments.append(float(self.loglt))

def write(path, value):
    path = ROOT/path
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    temp.replace(path)

def event(phase, **kwargs):
    obj=dict(time=time.time(),phase=phase,**kwargs)
    with (ROOT/'logs/events.jsonl').open('a') as f: f.write(json.dumps(obj)+'\n')
    write('status.json',obj)
    print(json.dumps(obj),flush=True)

def natural(z):
    z=np.asarray(z,dtype=np.float64)
    values=dict(zip(FREE,z))
    for n in POSITIVE: values[n]=np.exp(values[n])
    values['beta_trend']/=100
    ivp=np.exp(np.asarray([values[n] for n in IVPS]))
    values.update(dict(zip(IVPS,ivp/ivp.sum())))
    values.update(rho=0.,c=1.,alpha=1.,delta=.02,Y_0=0.)
    return {n:float(values[n]) for n in NAMES}

def pf(theta,T,J,seed):
    begin=time.perf_counter()
    seed=int(seed)
    inp=dc.load_inputs(T)
    np.random.seed(seed)
    fk=ssm.Bootstrap(ssm=DaccaSSM(Dacca(theta,inp)),data=inp['ys'])
    alg=RecordedSMC(fk=fk,N=J,resampling='systematic',ESSrmin=1.,
                      collect='off',store_history=False,verbose=False)
    setup=time.perf_counter()-begin
    t=time.perf_counter();alg.run();elapsed=time.perf_counter()-t
    np.testing.assert_array_equal(alg.recorded_times,np.arange(T))
    np.testing.assert_allclose(sum(alg.recorded_increments),alg.logLt,rtol=1e-12,atol=1e-10)
    return dict(loglik=float(alg.logLt),seconds=elapsed,seed=seed,requested_seed=seed,
        setup_seconds=setup,total_seconds=time.perf_counter()-begin,
        observations_weighted=int(alg.t),incremental_loglik=alg.recorded_increments,
        native_observation_indices=alg.recorded_times)

def stats(values):
    x=np.asarray(values,float)
    finite=np.isfinite(x)
    if not np.all(finite):
        return dict(valid=False,finite=int(finite.sum()),total=len(x))
    w=np.exp(x-x.max()); wn=w/w.sum()
    return dict(valid=True,logmeanlik=float(logsumexp(x)-np.log(len(x))),
                mcse=float(w.std(ddof=1)/(np.sqrt(len(w))*w.mean())),
                likelihood_ess=float(1./np.sum(wn**2)),repetitions=len(x))

def reference(theta,T,J,R,seed):
    import jax
    import jax.numpy as jnp
    import pypomp as pp
    model=pp.models.dhaka(dt=None,nstep=20).to_struct()
    model=model._replace(ys=model.ys[:T],times=model.times[:T],
         covars_extended=model.covars_extended[:20*T+1],
         dt_array_extended=model.dt_array_extended[:20*T],nstep_array=model.nstep_array[:T])
    p=jnp.asarray([[theta[n] for n in model.param_names]]*R,dtype=jnp.float32)
    keys=jax.random.split(jax.random.key(seed),(R,1))
    fn=jax.jit(lambda p,k:pp.functional.pfilter(model,p,J=J,keys=k)['logLik'].reshape(R))
    t=time.perf_counter();compiled=fn.lower(p,keys).compile();compile_s=time.perf_counter()-t
    t=time.perf_counter();ll=np.asarray(compiled(p,keys).block_until_ready());elapsed=time.perf_counter()-t
    return dict(raw_loglik=ll.tolist(),summary=stats(ll),compile_seconds=compile_s,
                seconds=elapsed,J=J,T=T,R=R,dtype='float32',seed=seed,backend='original PyPOMP 17f8798')

def nm(z0,T,J,maxfev,seed,path):
    rows=[]
    def objective(z):
        call=len(rows)
        result=pf(natural(z),T,J,int(np.random.SeedSequence([seed,call]).generate_state(1)[0]))
        rows.append(dict(call=call,parameters=np.asarray(z).tolist(),**result))
        write(path+'/calls.json',rows)
        return -result['loglik']
    t=time.perf_counter()
    res=optimize.minimize(objective,z0,method='Nelder-Mead',
        options=dict(maxfev=maxfev,maxiter=maxfev,xatol=1e-2))
    elapsed=time.perf_counter()-t
    out=dict(method='particles PF + scipy.optimize.minimize Nelder-Mead',
      nfev=res.nfev,nit=res.nit,success=bool(res.success),status=int(res.status),message=str(res.message),
      J=J,T=T,parameter_names=FREE,initial_parameters=z0.tolist(),
      final_parameters=res.x.tolist(),best_training_loglik=-float(res.fun),seconds=elapsed,
      purpose='finite-call implementation and cost probe; not convergence or final quality evidence',
      official_example='book/mle/mle_neuro.py',qmc=False,
      adaptation='Dacca interval simulator has no ppf, so use bootstrap SMC rather than the Neuro SQMC example')
    write(path+'/summary.json',out)
    return out

class NeuroXp(ssm.StateSpaceModel):
    default_params={'M':50,'rho':.99,'sig2':.0121}
    def PX0(self): return dists.Normal()
    def PX(self,t,xp): return dists.Normal(loc=self.rho*xp,scale=np.sqrt(self.sig2))
    def PY(self,t,xp,x): return dists.Binomial(n=self.M,p=expit(x))

def neuro():
    data=datasets.Neuro().data
    rows=[]
    def objective(x):
        if x[0]>1 or x[0]<0 or x[1]<=0:return np.inf
        np.random.seed(83000+len(rows))
        alg=particles.SMC(fk=ssm.Bootstrap(ssm=NeuroXp(rho=x[0],sig2=x[1]),data=data),N=64,qmc=True)
        alg.run();rows.append(dict(parameters=x.tolist(),loglik=float(alg.logLt)))
        return -alg.logLt
    t=time.perf_counter()
    result=optimize.minimize(objective,[.1,.5],method='Nelder-Mead',options=dict(maxfev=8,xatol=1e-2))
    out=dict(T=len(data),J=64,qmc=True,nfev=result.nfev,seconds=time.perf_counter()-t,
         status=int(result.status),message=str(result.message),final_parameters=result.x.tolist(),calls=rows,
         source='book/mle/mle_neuro.py',purpose='official case implementation smoke, reduced particle and function-call counts')
    write('results/neuro_smoke.json',out)
    return out

def smoke():
    event('local_smoke_started')
    a=pf(natural(STARTS[0]),24,64,81001)
    fit=nm(STARTS[0],24,64,26,81100,'results/local_nm')
    n=neuro()
    out=dict(passed=bool(np.isfinite(a['loglik']) and np.isfinite(fit['best_training_loglik'])),pf=a,nm=fit,neuro=n)
    write('results/local_smoke.json',out)
    event('local_smoke_complete',passed=bool(out['passed']))

def preflight():
    event('preflight_started',host=socket.gethostname())
    versions={n:importlib.metadata.version(n) for n in ['numpy','scipy','numba','joblib','jax']}
    versions['particles']=particles.__version__
    write('results/environment.json',dict(versions=versions,python=platform.python_version(),host=socket.gethostname(),
       affinity=sorted(os.sched_getaffinity(0)),loadavg=os.getloadavg(),SLURM_JOB_ID=os.getenv('SLURM_JOB_ID')))
    published=dict(zip(NAMES,DATA['theta_published'][0].astype(float)))
    comparisons=[]
    for label,theta,R in [('published',published,32),('paper_start_0',natural(STARTS[0]),16)]:
        event('pf_validation',point=label,R=R)
        vals=[]
        for r in range(R):
            vals.append(pf(theta,600,5000,82000+1000*len(comparisons)+r))
            write('results/'+label+'_particles_raw.json',vals)
        ref=reference(theta,600,5000,R,84000+len(comparisons))
        write('results/'+label+'_reference.json',ref)
        summary=stats([v['loglik'] for v in vals]);other=ref['summary']
        delta=summary['logmeanlik']-other['logmeanlik']
        se=np.hypot(summary['mcse'],other['mcse'])
        row=dict(point=label,particles=summary,pypomp=other,delta=delta,combined_mcse=float(se),
          z=float(delta/se),J=5000,T=600,
          first_filter_seconds=vals[0]['seconds'],warm_median_filter_seconds=float(np.median([x['seconds'] for x in vals[1:]])),
          precision_passed=summary['mcse']<=1 and other['mcse']<=1 and summary['likelihood_ess']>=10 and other['likelihood_ess']>=10)
        comparisons.append(row);write('results/pf_comparison.json',comparisons)
    event('full_data_optimizer_probe',maxfev=26)
    fit=nm(STARTS[0],600,5000,26,85000,'results/full_nm')
    final=reference(natural(fit['final_parameters']),600,5000,16,86000)
    write('results/full_nm/independent_reference_score.json',final)
    out=dict(completed=True,full_model=True,PF_comparison=comparisons,optimizer_probe=fit,
             independent_final_score=final,
             scope='CPU feasibility and cost preflight only; not final optimization comparison',
             projected_100_starts_same_26_calls_hours=100*fit['seconds']/3600,
             maximum_likelihood_baseline='official-example-style Nelder-Mead over native particles PF',
             EM_for_Dacca=False,DMOP_in_particles=False)
    write('results/COMPLETE.json',out);event('preflight_complete')

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('phase',choices=['smoke','preflight']);args=ap.parse_args()
    try: (smoke if args.phase=='smoke' else preflight)()
    except Exception as error:
        event('failed',error=repr(error));raise
