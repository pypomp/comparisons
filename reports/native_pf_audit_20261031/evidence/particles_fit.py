"""Two real package routes; CPU fitting with explicit wall-clock deadlines.

NM uses particles' unchanged PF and SciPy's external optimizer. IF2+NM calls
the candidate public if2_step(vectorized=True), carrying the full particle
cloud and NumPy RNG state, then the identical external NM implementation.
No score is used to choose an endpoint. No failed fit is retried.
"""
from pathlib import Path
import sys,os,json,time,hashlib,signal,traceback,argparse
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent))
import numpy as np
from scipy.optimize import minimize
import model as m
from particles.iterated_filtering import if2_step

class Deadline(Exception):pass
class Interrupted(Exception):pass
def sha_bytes(x):return hashlib.sha256(x).hexdigest()
def parameter_sha(z):return sha_bytes(np.asarray(z,dtype='<f8').tobytes())
def json_sha(x):return sha_bytes(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode())
def safe(x):
    if isinstance(x,dict):return {k:safe(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [safe(v) for v in x]
    if isinstance(x,np.ndarray):return safe(x.tolist())
    if isinstance(x,(np.bool_,bool)):return bool(x)
    if isinstance(x,np.integer):return int(x)
    if isinstance(x,(np.floating,float)):return float(x) if np.isfinite(x) else None
    return x
def write(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(safe(obj),indent=2,allow_nan=False)+'\n');tmp.replace(path)
def swarm(path,cloud,rng):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.with_suffix('.tmp').open('wb') as f:
        np.savez(f,cloud=cloud,rng_name=rng[0],rng_keys=rng[1],rng_position=rng[2],
            rng_has_gauss=rng[3],rng_cached_gaussian=rng[4])
    path.with_suffix('.tmp').replace(path)
    return sha_bytes(path.read_bytes())
def seed(config,start,rep,arm,stage,call=0):
    # Disjoint integer seed blocks avoid collisions from truncating hierarchical
    # SeedSequence outputs to the uint32 required by legacy NumPy/particles.
    assert stage in (0,1) and 0<=call<50000
    task=(start*config['repeats']+rep)*2+arm
    value=config['train_root']+task*100000+stage*50000+call
    assert 0<=value<2**32
    return int(value)

def fit(config,task_id,out,pf=m.native_pf,public_if2=if2_step,clock=time.perf_counter):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    if (out/'endpoint.json').exists():
        old=json.loads((out/'endpoint.json').read_text());assert old['protocol_sha256']==json_sha(config)
        return old
    if (out/'state.json').exists():
        raise RuntimeError('Partial or interrupted fit exists. No automatic retry or replay of failed work.')
    pair,arm=divmod(task_id,2);start,rep=divmod(pair,config['repeats'])
    start=config['starts'][start];method=config['methods'][arm]
    z0=m.STARTS[start].astype(float);z=z0.copy();calls=[];passes=[];best=None
    inp=m.INPUT if config['T']==600 else m.dc.load_inputs(config['T'])
    factory=lambda p:m.factory(p,inp)
    budget=config['budget_seconds'];prefix=config['IF2_prefix_seconds'] if arm else 0.
    begin=clock();elapsed=lambda:clock()-begin
    failed=False;reason=None;stage='IF2' if arm else 'NM';termination=None
    state=dict(task_id=task_id,start_id=start,replicate=rep,method=method,protocol_sha256=json_sha(config))
    write(out/'initial.json',dict(state,parameters=z0.tolist(),parameter_names=m.FREE,parameter_sha256=parameter_sha(z0)))
    def checkpoint(phase):write(out/'state.json',dict(state,phase=phase,stage=stage,charged_fit_seconds=elapsed(),
        completed_PF_calls=len(calls),accepted_IF2_passes=sum(p['accepted'] for p in passes),time_unix=time.time()))
    cloud=None;IF2_seconds=0.;NM_seconds=0.;if2_initial_seed=None;nm_initial=None
    try:
        if arm:
            if2_initial_seed=seed(config,start,rep,arm,0)
            np.random.seed(if2_initial_seed);cloud=np.tile(z0,(config['J'],1))
            swarm(out/'IF2_initial.npz',cloud,np.random.get_state())
            iteration=0
            while elapsed()<prefix:
                checkpoint('running');before=elapsed()
                step=public_if2(factory,inp['ys'],cloud,m.scales(config['T'])*config['IF2_cooling50']**(iteration/50),
                    vectorized=True,resampling='multinomial')
                finish=elapsed();accepted=finish<=prefix
                row=dict(iteration=iteration,start_seconds=before,finish_seconds=finish,seconds=finish-before,
                    accepted=accepted,perturbed_loglik=step.loglik,mean=step.theta_mean.tolist(),
                    mean_sha256=parameter_sha(step.theta_mean),actual_SMC_class=type(step.smc).__module__+'.'+type(step.smc).__name__)
                # Preserve late results for accounting/provenance; they never seed NM.
                path=out/'IF2_swarms'/f'pass_{iteration:04d}.npz'
                row['swarm_file']=str(path.relative_to(out));row['swarm_sha256']=swarm(path,step.theta_particles,np.random.get_state())
                passes.append(row);write(out/'IF2_passes.json',passes)
                if not accepted:break
                cloud=step.theta_particles;z=step.theta_mean.copy();iteration+=1
            IF2_seconds=elapsed();write(out/'IF2_endpoint.json',dict(parameters=z.tolist(),parameter_sha256=parameter_sha(z),
                accepted_passes=sum(p['accepted'] for p in passes),charged_seconds=IF2_seconds,
                requested_budget_seconds=prefix,selection='mean of last full swarm completed by IF2 deadline',
                full_swarm_used=True,initial_seed=if2_initial_seed))
        stage='NM';nm_begin=elapsed();nm_initial=z.copy()
        write(out/'NM_initial.json',dict(parameters=z.tolist(),parameter_sha256=parameter_sha(z),
            remaining_total_budget_seconds=max(0.,budget-nm_begin),optimizer='external scipy.optimize.minimize(Nelder-Mead)'))
        def objective(candidate):
            nonlocal best
            if elapsed()>=budget:raise Deadline()
            checkpoint('running');before=elapsed();index=len(calls)
            call_seed=seed(config,start,rep,arm,1,index)
            # Native original SMC/Bootstrap remains the objective for both methods.
            alg=pf(candidate,config['J'],call_seed,T=config['T'])
            ll=float(alg.logLt)
            if not np.isfinite(ll):raise FloatingPointError('Nonfinite native PF objective')
            finish=elapsed();eligible=finish<=budget
            row=dict(call=index,parameters=np.asarray(candidate).tolist(),parameter_sha256=parameter_sha(candidate),
                seed=call_seed,loglik=ll,observations=int(alg.t),start_seconds=before,finish_seconds=finish,
                seconds=finish-before,eligible=eligible)
            calls.append(row);write(out/'NM_calls'/f'call_{index:05d}.json',row)
            if eligible and (best is None or ll>best['loglik']):best=row
            if not eligible:raise Deadline()
            return -ll
        try:
            result=minimize(objective,z,method='Nelder-Mead',options=dict(xatol=.01,fatol=.0001,
                adaptive=False,maxiter=1000000,maxfev=1000000))
            termination=str(result.message)
            write(out/'NM_result.json',dict(success=bool(result.success),status=int(result.status),
                message=str(result.message),nit=int(result.nit),nfev=int(result.nfev),parameters=result.x.tolist()))
        except Deadline:termination='total fitting wall-clock budget reached'
        finally:NM_seconds=elapsed()-nm_begin
    except BaseException as exc:
        failed=True;reason=repr(exc);termination='failed without retry'
        if stage=='IF2':IF2_seconds=elapsed()
        swarm(out/'failure_state.npz',np.tile(z,(1,1)) if cloud is None else cloud,np.random.get_state())
        write(out/'failure.json',dict(stage=stage,error=reason,traceback=traceback.format_exc(),charged_fit_seconds=elapsed()))
    if best is not None:z=np.asarray(best['parameters'])
    charged=elapsed();natural=m.natural(z)
    # Last finite checkpoint remains visible and is independently scored even if failed.
    result=dict(state,parameters_estimation=z.tolist(),estimation_parameter_names=m.FREE,
        estimation_sha256=parameter_sha(z),parameters_natural=[float(natural[n]) for n in m.NAMES],parameter_names=list(m.NAMES),
        natural_sha256=parameter_sha([natural[n] for n in m.NAMES]),initial_estimation_sha256=parameter_sha(z0),
        fit_failed=failed,failed_stage=stage if failed else None,failure_reason=reason,termination=termination,
        accepted_IF2_passes=sum(p['accepted'] for p in passes),attempted_IF2_passes=len(passes),
        late_IF2_passes=sum(not p['accepted'] for p in passes),IF2_seed=if2_initial_seed,
        IF2_seconds=IF2_seconds,NM_seconds=NM_seconds,charged_fit_seconds=charged,
        fitting_budget_seconds=budget,budget_overshoot_seconds=max(0.,charged-budget),
        PF_calls=len(calls),accepted_PF_calls=sum(c['eligible'] for c in calls),late_PF_calls=sum(not c['eligible'] for c in calls),
        best_training_call=None if best is None else best['call'],best_training_loglik=None if best is None else best['loglik'],
        NM_initial_sha256=None if nm_initial is None else parameter_sha(nm_initial),
        optimizer_native=False,public_IF2_called=bool(passes),
        endpoint_selection='best eligible NM training objective; if none, last accepted IF2 mean or original start',
        cost_scope='actual calls, optimizer work, cloud/RNG checkpoints and fit records; late calls charged but parameters rejected',
        scoring_used_for_selection=False)
    write(out/'endpoint.json',result);checkpoint('fit_failed' if failed else 'fit_complete')
    write(out/'score_input.json',[result])
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--task-id',type=int,required=True);p.add_argument('--protocol',default='protocol.json');a=p.parse_args()
    cfg=json.loads((ROOT/a.protocol).read_text());assert cfg['status']=='frozen';assert 0<=a.task_id<24
    out=ROOT/'results'/f'task_{a.task_id:02d}'
    signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(Interrupted('SIGTERM')))
    before=time.perf_counter()
    result=fit(cfg,a.task_id,out)
    write(out/'runtime.json',dict(process_seconds=time.perf_counter()-before,job_id=os.environ.get('SLURM_JOB_ID'),
        array_job_id=os.environ.get('SLURM_ARRAY_JOB_ID'),array_task_id=os.environ.get('SLURM_ARRAY_TASK_ID'),
        node=os.environ.get('SLURMD_NODENAME'),CPU_only=True,affinity=sorted(os.sched_getaffinity(0))))
    files={str(p.relative_to(out)):sha_bytes(p.read_bytes()) for p in out.rglob('*') if p.is_file() and p.name!='terminal_manifest.json'}
    write(out/'terminal_manifest.json',dict(sha256=files,source_manifest_sha256=sha_bytes((ROOT/'manifest.json').read_bytes())))
    print(json.dumps(safe(result)),flush=True)
if __name__=='__main__':main()
