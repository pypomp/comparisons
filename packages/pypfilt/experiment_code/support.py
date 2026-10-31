"""pypfilt correctness/cost preflight and an explicitly external MLE wrapper."""
from pathlib import Path
import os
ROOT = Path(__file__).resolve().parent
os.environ['PFB_DACCA_INPUTS'] = str(ROOT/'input/frozen.npz')
os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import sys
sys.path.insert(0,str(ROOT/'adapter'))
import argparse
import importlib.metadata
import io
import json
import platform
import socket
import time
import traceback
import numpy as np
import scipy
from scipy import optimize
from scipy.special import logsumexp
import pypfilt
import dacca_common as dc
import pypfilt_dacca as adapter
from bench_pypfilt import toml, write_obs

FIXED=['rho','c','alpha','delta','Y_0']
IVPS=['S_0','I_0','R1_0','R2_0','R3_0']
POSITIVE=['gamma','epsilon','m','sigma','tau']
DATA=dict(np.load(ROOT/'input/frozen.npz'))
NAMES=DATA['paramnames'].tolist()
FREE=[n for n in NAMES if n not in FIXED]
STARTS=np.load(ROOT/'input/paper_starts_estimation_free.npy')
_OBS={}

def safe(value):
    if isinstance(value,dict):return {k:safe(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [safe(v) for v in value]
    if isinstance(value,np.ndarray):return safe(value.tolist())
    if isinstance(value,(np.bool_,bool)):return bool(value)
    if isinstance(value,np.integer):return int(value)
    if isinstance(value,(float,np.floating)):return float(value) if np.isfinite(value) else None
    return value

def write(name,value):
    path=ROOT/name;path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(safe(value),indent=2,allow_nan=False)+'\n');tmp.replace(path)

def event(phase,**kwargs):
    obj=dict(time=time.time(),phase=phase,**kwargs)
    (ROOT/'logs').mkdir(exist_ok=True)
    with (ROOT/'logs/events.jsonl').open('a') as f:f.write(json.dumps(safe(obj))+'\n')
    write('status.json',obj);print(json.dumps(safe(obj)),flush=True)

def natural(z):
    values=dict(zip(FREE,np.asarray(z,dtype=np.float64)))
    for n in POSITIVE:values[n]=np.exp(values[n])
    values['beta_trend']/=100
    ivp=np.exp([values[n] for n in IVPS]);values.update(dict(zip(IVPS,ivp/ivp.sum())))
    values.update(rho=0.,c=1.,alpha=1.,delta=.02,Y_0=0.)
    return {n:float(values[n]) for n in NAMES}

def pf(theta,T,J,seed):
    begin=time.perf_counter()
    # NumPy 2 scalar repr is not a TOML number; use Python numeric scalars.
    theta={name:float(value) for name,value in theta.items()}
    requested_seed=int(seed)
    # pypfilt metadata serializes all configuration integers as int32.
    seed=requested_seed % (2**31-1)
    inp=dc.load_inputs(T)
    if T not in _OBS:_OBS[T]=write_obs(inp)
    config=toml(J,seed,theta,T,20,_OBS[T],'ministeps',-1)
    (instance,)=pypfilt.load_instances(io.StringIO(config))
    ctx=instance.build_context()
    adapter.LOGLIK[0]=0.;adapter.INCREMENTS.clear()
    setup=time.perf_counter()-begin
    start=time.perf_counter();pypfilt.fit(ctx,filename=None);elapsed=time.perf_counter()-start
    increments=list(adapter.INCREMENTS)
    assert len(increments)==T,(len(increments),T)
    np.testing.assert_array_equal([x['time'] for x in increments],np.arange(1,T+1))
    np.testing.assert_allclose([x['weight_sum'] for x in increments],1.,rtol=0,atol=1e-10)
    assert np.isfinite(adapter.LOGLIK[0])
    return dict(loglik=float(adapter.LOGLIK[0]),seconds=elapsed,
        setup_seconds=setup,total_seconds=time.perf_counter()-begin,seed=seed,requested_seed=requested_seed,
        observations_weighted=len(increments),incremental_loglik=[x['value'] for x in increments])

def stats(values):
    x=np.asarray(values,float)
    if not np.isfinite(x).all():return dict(valid=False,finite=int(np.isfinite(x).sum()),total=len(x))
    w=np.exp(x-x.max());wn=w/w.sum()
    return dict(valid=True,logmeanlik=float(logsumexp(x)-np.log(len(x))),
        mcse=float(w.std(ddof=1)/(np.sqrt(len(w))*w.mean())),
        likelihood_ess=float(1/np.sum(wn**2)),repetitions=len(x))

def reference(theta,T,J,R,seed):
    import jax
    import jax.numpy as jnp
    import pypomp as pp
    st=pp.models.dhaka(dt=None,nstep=20).to_struct()
    st=st._replace(ys=st.ys[:T],times=st.times[:T],covars_extended=st.covars_extended[:20*T+1],
        dt_array_extended=st.dt_array_extended[:20*T],nstep_array=st.nstep_array[:T])
    p=jnp.asarray([[theta[n] for n in st.param_names]]*R,dtype=jnp.float32)
    keys=jax.random.split(jax.random.key(seed),(R,1))
    fn=jax.jit(lambda p,k:pp.functional.pfilter(st,p,J=J,keys=k)['logLik'].reshape(R))
    start=time.perf_counter();exe=fn.lower(p,keys).compile();compile_s=time.perf_counter()-start
    start=time.perf_counter();ll=np.asarray(exe(p,keys).block_until_ready());elapsed=time.perf_counter()-start
    return dict(raw_loglik=ll.tolist(),summary=stats(ll),compile_seconds=compile_s,seconds=elapsed,
        J=J,T=T,R=R,seed=seed,dtype='float32',backend='original PyPOMP 17f8798')

def nm(z0,T,J,maxfev,seed,path):
    rows=[]
    def objective(z):
        call=len(rows);rseed=int(np.random.SeedSequence([seed,call]).generate_state(1)[0])
        try:
            with np.errstate(over='raise',invalid='raise',divide='raise'):theta=natural(z)
            result=pf(theta,T,J,rseed);loss=-result['loglik'];error=None
        except (ValueError,FloatingPointError,OverflowError) as exc:
            result=dict(loglik=None,seed=rseed);error=repr(exc);loss=np.inf
        rows.append(dict(call=call,parameters=np.asarray(z).tolist(),error=error,**result))
        write(path+'/calls.json',rows);return loss
    begin=time.perf_counter()
    res=optimize.minimize(objective,z0,method='Nelder-Mead',
        options=dict(maxfev=maxfev,maxiter=maxfev,xatol=.01))
    out=dict(method='pypfilt PF + external SciPy Nelder-Mead (custom wrapper)',
        native_optimizer=False,nfev=res.nfev,nit=res.nit,success=bool(res.success),status=int(res.status),
        message=str(res.message),J=J,T=T,parameter_names=FREE,initial_parameters=z0.tolist(),
        final_parameters=res.x.tolist(),best_training_loglik=-float(res.fun),seconds=time.perf_counter()-begin,
        purpose='finite-call execution and cost probe, not a converged optimization comparison',
        native_filter='pypfilt.fit',regularisation=False)
    write(path+'/summary.json',out);return out

def native_example():
    """Run the shipped joint state/parameter inference example, with J=128."""
    import pypfilt.examples.predation as example
    examples=Path(example.__file__).parent
    text=(examples/'predation.toml').read_text().replace('particles = 1000','particles = 128')
    text=text.replace('input_directory = "."','input_directory = "'+str(examples)+'"')
    text=text.replace('output_directory = "."','output_directory = "'+str(ROOT/'results')+'"')
    (instance,)=pypfilt.load_instances(io.StringIO(text))
    # Retain the shipped prior and post-regularisation; this is not MLE.
    instance.settings['filter']['results']={'save_history':True}
    ctx=instance.build_context()
    start=time.perf_counter();result=pypfilt.fit(ctx,filename=None);seconds=time.perf_counter()-start
    snapshot=result.estimation.history.snapshot(ctx,expect_time=15)
    weights=snapshot.vec['weight']
    estimates={n:dict(weighted_mean=float(np.sum(weights*snapshot.state_vec[n])),
                       finite=bool(np.isfinite(snapshot.state_vec[n]).all()))
               for n in ['alpha','beta','gamma','delta']}
    out=dict(passed=all(x['finite'] for x in estimates.values()),J=128,T=15,seconds=seconds,
        model='official pypfilt.examples.predation.LotkaVolterra',parameter_summary=estimates,
        regularisation=True,resample_threshold=.25,
        target='joint state and parameter particle inference with priors, not MLE',
        source='pypfilt/examples/predation.toml and predation.py from wheel 0.8.7',
        scope='native API execution check, not a posterior accuracy or optimizer comparison')
    write('results/native_predation.json',out);return out

def smoke():
    event('local_smoke_started')
    first=pf(natural(STARTS[0]),24,64,97001)
    repeat=pf(natural(STARTS[0]),24,64,97001)
    assert first['loglik']==repeat['loglik']
    fit=nm(STARTS[0],24,64,26,97002,'results/local_nm')
    assert not any(r['error'] for r in json.loads((ROOT/'results/local_nm/calls.json').read_text()))
    ref=reference(natural(fit['final_parameters']),24,64,4,97003)
    native=native_example()
    out=dict(passed=True,first_pf=first,repeated_seed_equal=True,external_NM=fit,
             independent_reference=ref,native_example=native,
             note='Small execution checks; no full Dacca accuracy or speed conclusion')
    write('results/local_smoke.json',out);event('local_smoke_complete')

def preflight():
    event('preflight_started',host=socket.gethostname())
    versions={n:importlib.metadata.version(n) for n in ['pypfilt','numpy','scipy','h5py','lhs','tomli','tomli-w','jax']}
    write('results/environment.json',dict(versions=versions,python=platform.python_version(),
        host=socket.gethostname(),affinity=sorted(os.sched_getaffinity(0)),loadavg=os.getloadavg(),
        SLURM_JOB_ID=os.getenv('SLURM_JOB_ID')))
    published=dict(zip(NAMES,DATA['theta_published'][0].astype(float)))
    comparisons=[]
    for label,theta,R in [('published',published,32),('paper_start_0',natural(STARTS[0]),16)]:
        event('PF_validation',point=label,R=R)
        rows=[]
        for rep in range(R):
            rows.append(pf(theta,600,5000,98000+1000*len(comparisons)+rep))
            write('results/'+label+'_pypfilt_raw.json',rows)
        ref=reference(theta,600,5000,R,101000+len(comparisons))
        write('results/'+label+'_reference.json',ref)
        a=stats([row['loglik'] for row in rows]);c=ref['summary']
        delta=a['logmeanlik']-c['logmeanlik'];se=float(np.hypot(a['mcse'],c['mcse']))
        precision=bool(a['mcse']<=1 and c['mcse']<=1 and a['likelihood_ess']>=10 and c['likelihood_ess']>=10)
        equiv=bool(abs(delta)+1.96*se<=5)
        row=dict(point=label,pypfilt=a,pypomp=c,difference=delta,combined_mcse=se,
            standardized_difference=delta/se,equivalence_margin_nats=5,
            precision_passed=precision,agreement_passed=equiv and precision,
            first_filter_seconds=rows[0]['seconds'],
            warm_median_filter_seconds=float(np.median([r['seconds'] for r in rows[1:]])),
            warm_median_total_seconds=float(np.median([r['total_seconds'] for r in rows[1:]])),
            scope='CPU, pypfilt float64 versus original PyPOMP float32',J=5000,T=600)
        comparisons.append(row);write('results/PF_comparison.json',comparisons)
    event('full_data_optimizer_probe',maxfev=26)
    fit=nm(STARTS[0],600,5000,26,102000,'results/full_nm')
    final=reference(natural(fit['final_parameters']),600,5000,16,103000)
    write('results/full_nm/independent_reference_score.json',final)
    out=dict(completed=True,full_model=True,PF_comparison=comparisons,optimizer_probe=fit,
        independent_final_score=final,native_optimizer=False,
        scope='CPU capability, correctness and cost preflight only; formal comparison not run',
        original_model_retained=True,projected_100_starts_same_26_calls_hours=100*fit['seconds']/3600)
    write('results/COMPLETE.json',out);event('preflight_complete')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['smoke','preflight']);args=parser.parse_args()
    try:(smoke if args.phase=='smoke' else preflight)()
    except BaseException as exc:
        write('results/failure_'+str(int(time.time()))+'.json',dict(error=repr(exc),traceback=traceback.format_exc()))
        event('failed',error=repr(exc));raise
