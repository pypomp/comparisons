"""One CPU job: model/API checks, paired 2+7 PF timing, real IF2 cost."""
from pathlib import Path
import os,sys,json,time,hashlib,subprocess,socket,platform,traceback
import importlib.metadata
import numpy as np
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results/preflight';OUT.mkdir(parents=True,exist_ok=True)
def write(name,value):
    p=OUT/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');t.replace(p)
def status(phase,**kw):
    row=dict(phase=phase,time=time.time(),**kw);write('status.json',row);print(json.dumps(row),flush=True)
def main():
    start=time.perf_counter()
    write('environment.json',dict(host=socket.gethostname(),python=platform.python_version(),
        versions={n:importlib.metadata.version(n) for n in ['numpy','scipy','numba','joblib','jax']},
        job_id=os.environ.get('SLURM_JOB_ID'),affinity=sorted(os.sched_getaffinity(0)),
        CPU_only=True,source_manifest_sha256=hashlib.sha256((ROOT/'manifest.json').read_bytes()).hexdigest()))
    status('adapter_check')
    import check_adapter
    write('adapter.json',check_adapter.run())
    workers={}
    try:
        for variant in ['upstream','candidate']:
            env=dict(os.environ);env['PYTHONPATH']=str(ROOT/'source'/variant)+os.pathsep+env.get('PARTICLES_RUNTIME_DEPS','')
            workers[variant]=subprocess.Popen([sys.executable,str(ROOT/'runtime_worker.py')],cwd=ROOT,env=env,
                stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=(OUT/(variant+'_stderr.log')).open('w'),text=True,bufsize=1)
        rows=[]
        for iteration in range(9):
            # Two independent warmups, then seven paired timed calls. Alternate order.
            status('PF_warmup' if iteration<2 else 'PF_timing',iteration=iteration)
            pair={}
            for variant in (['upstream','candidate'] if iteration%2==0 else ['candidate','upstream']):
                proc=workers[variant];proc.stdin.write(json.dumps(dict(J=5000,T=600,seed=84981000+iteration))+'\n');proc.stdin.flush()
                line=proc.stdout.readline()
                if not line:raise RuntimeError(variant+' PF worker exited; inspect stderr')
                pair[variant]=json.loads(line)
            for k in ['seed','loglik','X_sha256','W_sha256','rng_sha256','rng_position','observations']:
                assert pair['upstream'][k]==pair['candidate'][k],(iteration,k,pair)
            rows.append(dict(iteration=iteration,warmup=iteration<2,**pair));write('PF_raw.json',rows)
        summary={k:dict(median_seconds=float(np.median([r[k]['seconds'] for r in rows[2:]])),
            all_timed_seconds=[r[k]['seconds'] for r in rows[2:]]) for k in workers}
        write('PF_summary.json',dict(passed=True,T=600,J=5000,warmups=2,timed_calls=7,
            same_seed_loglik_states_weights_RNG_exact=True,methods=summary,
            upstream_over_candidate_ratio=summary['upstream']['median_seconds']/summary['candidate']['median_seconds'],
            scope='Native CPU PF unchanged by candidate. Timing includes per-call model setup and run; imports excluded.'))
    finally:
        for proc in workers.values():
            proc.stdin.close();proc.wait(timeout=20)
    status('IF2_J5000_cost')
    import model as m
    from particles.iterated_filtering import if2_step
    cloud=np.tile(m.STARTS[0].astype(float),(5000,1));np.random.seed(84982000);cost=[]
    for iteration in range(2):
        t=time.perf_counter()
        step=if2_step(m.factory,m.INPUT['ys'],cloud,m.scales()*(.5**(iteration/50)),vectorized=True)
        elapsed=time.perf_counter()-t;cloud=step.theta_particles
        rng=np.random.get_state()
        np.savez(OUT/f'IF2_pass{iteration}.npz',cloud=cloud,rng_keys=rng[1],rng_position=rng[2],
            rng_has_gauss=rng[3],rng_cached_gaussian=rng[4])
        cost.append(dict(pass_index=iteration,seconds=elapsed,perturbed_loglik=step.loglik,
            parameter_mean=step.theta_mean.tolist(),smc_class=type(step.smc).__module__+'.'+type(step.smc).__name__))
        write('IF2_cost.json',dict(passed=True,J=5000,T=600,passes=cost,
            API='particles.iterated_filtering.if2_step(vectorized=True)',not_optimization_comparison=True))
    status('complete',passed=True,seconds=time.perf_counter()-start)
    files={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.rglob('*') if p.is_file() and p.name!='terminal_manifest.json'}
    write('terminal_manifest.json',dict(sha256=files))
if __name__=='__main__':
    try:main()
    except BaseException as exc:
        status('failed',error=repr(exc),traceback=traceback.format_exc());raise
