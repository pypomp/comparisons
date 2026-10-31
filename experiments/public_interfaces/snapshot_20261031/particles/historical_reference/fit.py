"""Frozen-budget external Nelder-Mead around the actual particles PF.

Endpoint selection uses training evaluations only. Late calls are charged but
cannot update the endpoint. Interrupted optimization replays saved objectives
through the same SciPy version, without re-running their PFs.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import signal
import socket
import time
import traceback
import numpy as np
from scipy import optimize
import support

ROOT=Path(__file__).resolve().parent

class BudgetReached(Exception): pass
class Interrupted(Exception): pass

def sha(obj):
    return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def write(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(support.safe(obj),indent=2,allow_nan=False)+'\n')
    temp.replace(path)

def read(path): return json.loads(Path(path).read_text())

def endpoint_hash(z):
    return hashlib.sha256(np.asarray(z,dtype='<f8').tobytes()).hexdigest()

def simplex(z):
    result=np.tile(z,(len(z)+1,1))
    for i in range(len(z)):result[i+1,i]=z[i]*1.05 if z[i] else .00025
    return result

def fit_one(config,start,rep,folder,pf=None):
    pf=support.pf if pf is None else pf
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    config_hash=sha(config);state_path=folder/'fit_state.json';endpoint=folder/'endpoint.json'
    if endpoint.exists():
        result=read(endpoint);assert result['config_sha256']==config_hash
        assert result['parameter_sha256']==(endpoint_hash(result['parameters']) if result['parameters'] is not None else None)
        return result
    z0=support.STARTS[start].astype(np.float64)
    old=read(state_path) if state_path.exists() else None
    if old:
        assert old['config_sha256']==config_hash
        if old['phase']=='running':
            raise RuntimeError('Unclean prior process: reconcile its Slurm elapsed allocation and mark fit_state interrupted before resuming; do not discard unmeasured work')
    prior=0. if old is None else float(old['charged_fit_seconds'])
    calls=[read(p) for p in sorted((folder/'calls').glob('call_*.json'))]
    assert all(row['call']==i for i,row in enumerate(calls))
    assert all(row['eligible'] for row in calls[:-1]), 'Only the final call may cross the deadline'
    old_count=len(calls);index=0;active_begin=None;replay_begin=time.perf_counter()
    replay_seconds=0.;last_result=None;interrupted=False
    budget=float(config['budget_seconds']);options=config['optimizer']
    assert options['randomness']=='fresh' and options['objective_repetitions']==1
    segment_id=0 if old is None else int(old['segments'])
    active_call=None;active_pf=False

    def elapsed():
        return prior+(0. if active_begin is None else time.perf_counter()-active_begin)

    def checkpoint(phase):
        write(state_path,dict(phase=phase,config_sha256=config_hash,charged_fit_seconds=elapsed(),
            completed_calls=len(calls),segments=segment_id+1,active_call=active_call,
            replay_preparation_seconds=replay_seconds,time_unix=time.time()))

    def objective(z):
        nonlocal index,active_begin,replay_seconds,active_call,active_pf
        if index<old_count:
            row=calls[index];np.testing.assert_array_equal(np.asarray(z),row['parameters'])
            index+=1
            if not row['eligible']:raise BudgetReached()
            return np.inf if row['error'] else -row['loglik']
        if active_begin is None:
            replay_seconds=time.perf_counter()-replay_begin
            active_begin=time.perf_counter()
        if elapsed()>=budget:raise BudgetReached()
        call=index;active_call=call;checkpoint('running')
        begin=elapsed();requested=int(np.random.SeedSequence([config['seed'],start,rep,call]).generate_state(1)[0])
        row=dict(call=call,parameters=np.asarray(z).tolist(),requested_seed=requested,
                 start_fit_seconds=begin,PF=None,PF_attempted=False,error=None)
        try:
            with np.errstate(over='raise',invalid='raise',divide='raise'):
                theta=support.natural(z)
            if not np.isfinite(list(theta.values())).all():raise ValueError('Non-finite natural parameters')
            row['PF_attempted']=True;active_pf=True
            result=pf(theta,config['T'],config['J'],requested)
            if not math.isfinite(result['loglik']):raise ValueError('Non-finite PF likelihood')
            row.update(PF=result,loglik=result['loglik'])
        except (ValueError,FloatingPointError,OverflowError) as error:
            row.update(error=repr(error),loglik=None)
        finish=elapsed();row.update(finish_fit_seconds=finish,eligible=finish<=budget,
                                   objective_seconds=finish-begin)
        calls.append(row);index+=1
        write(folder/'calls'/f'call_{call:06d}.json',row)
        active_call=None;active_pf=False;checkpoint('running')
        if not row['eligible']:raise BudgetReached()
        return np.inf if row['error'] else -row['loglik']

    try:
        last_result=optimize.minimize(objective,z0,method='Nelder-Mead',options=dict(
            initial_simplex=simplex(z0),xatol=options['xatol'],fatol=options['fatol'],
            adaptive=options['adaptive'],maxiter=options['maxiter'],maxfev=options['maxfev']))
        termination=str(last_result.message)
    except BudgetReached:
        termination='wall-clock budget reached'
    except Interrupted:
        interrupted=True;termination='process interrupted; saved objectives can be replayed'
    finally:
        if active_begin is None:replay_seconds=time.perf_counter()-replay_begin
        charged=elapsed()
        write(folder/'segments'/f'segment_{segment_id:04d}.json',dict(
            interrupted=interrupted,prior_fit_seconds=prior,charged_fit_seconds=charged,
            replay_preparation_seconds=replay_seconds,active_call=active_call,
            interrupted_PF_attempt=interrupted and active_pf))
        checkpoint('interrupted' if interrupted else 'fit_finished')
    if interrupted:raise Interrupted(termination)
    admitted=[row for row in calls if row['eligible'] and row['error'] is None]
    best=max(admitted,key=lambda row:row['loglik']) if admitted else None
    z=None if best is None else best['parameters']
    result=dict(config_sha256=config_hash,start=start,replicate=rep,method=config['method'],
        native_optimizer=False,parameters=z,parameter_names=support.FREE,
        parameter_sha256=None if z is None else endpoint_hash(z),
        initial_parameters=z0.tolist(),initial_simplex=simplex(z0).tolist(),
        best_call=None if best is None else best['call'],
        best_training_loglik=None if best is None else best['loglik'],
        fitting_budget_seconds=budget,charged_fit_seconds=charged,
        budget_overshoot_seconds=max(0.,charged-budget),termination=termination,
        scipy_success=False if last_result is None else bool(last_result.success),
        scipy_iterations=None if last_result is None else int(last_result.nit),
        objective_calls=len(calls),PF_calls=sum(row['PF_attempted'] for row in calls),
        completed_PF_calls=sum(row['PF'] is not None for row in calls),
        interrupted_PF_attempts=sum(read(p)['interrupted_PF_attempt'] for p in (folder/'segments').glob('segment_*.json')),
        failed_calls=sum(row['error'] is not None for row in calls),
        late_calls=sum(not row['eligible'] for row in calls),fit_failed=best is None,
        selection='best finite training objective completed by deadline; independent scores are never consulted',
        cost_scope='all actual objective calls, optimizer work, and checkpoint writes; replay and scoring reported separately')
    write(endpoint,result)
    return result

def score_one(config,start,rep,folder,can_start=lambda _:True,reference=None):
    reference=support.reference if reference is None else reference
    folder=Path(folder);endpoint=read(folder/'endpoint.json');config_hash=sha(config)
    assert endpoint['config_sha256']==config_hash
    score_path=folder/'independent_score.json'
    if score_path.exists():
        saved=read(score_path)
        assert saved['config_sha256']==config_hash
        assert saved['endpoint_sha256']==endpoint['parameter_sha256']
        if saved['finished']:return saved
    if endpoint['fit_failed']:
        saved=dict(config_sha256=config_hash,finished=True,fit_failed=True,endpoint_sha256=None,precision_ok=False)
        write(score_path,saved);return saved
    z=endpoint['parameters'];assert endpoint_hash(z)==endpoint['parameter_sha256']
    scoring=config['score'];batch_size=scoring['batch_repetitions'];cap=max(scoring['stages'])
    raw=[];cost=0.;compile_cost=0.;execution_cost=0.;batch=0;precision=False;stopped=False
    while len(raw)<cap:
        target=folder/'score_batches'/f'batch_{batch:03d}.json'
        if target.exists():
            result=read(target)
            assert result['endpoint_sha256']==endpoint['parameter_sha256']
            assert result['config_sha256']==config_hash
        else:
            # Use the measured previous batch cost for the allocation guard.
            estimate=max(90.,2*cost/max(1,batch))
            if not can_start(estimate):stopped=True;break
            seed=int(np.random.SeedSequence([scoring['seed'],start,rep,batch]).generate_state(1)[0])
            before=time.perf_counter()
            value=reference(support.natural(z),config['T'],scoring['J'],batch_size,seed)
            result=dict(value,config_sha256=config_hash,endpoint_sha256=endpoint['parameter_sha256'],
                batch=batch,wall_seconds=time.perf_counter()-before)
            write(target,result)
        assert len(result['raw_loglik'])==batch_size
        raw.extend(result['raw_loglik']);cost+=result['wall_seconds']
        compile_cost+=result['compile_seconds'];execution_cost+=result['seconds'];batch+=1
        stat=support.stats(raw)
        precision=stat['valid'] and stat['mcse']<=scoring['MCSE_max'] and stat['likelihood_ess']>=scoring['likelihood_ESS_min']
        if precision:break
    stat=support.stats(raw) if raw else None
    saved=dict(config_sha256=config_hash,endpoint_sha256=endpoint['parameter_sha256'],
        finished=precision or len(raw)>=cap,pending_allocation=stopped,precision_ok=precision,
        repetitions=len(raw),raw_loglik=raw,summary=stat,score_wall_seconds=cost,
        compile_seconds=compile_cost,execution_seconds=execution_cost,
        scoring=f'Independent original PyPOMP, append {batch_size} repetitions when precision fails, cap {cap}; never select endpoint',
        endpoint_unchanged=endpoint_hash(read(folder/'endpoint.json')['parameters'])==endpoint['parameter_sha256'])
    assert saved['endpoint_unchanged'];write(score_path,saved);return saved

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--protocol',required=True)
    parser.add_argument('--output',default='results')
    parser.add_argument('--phase',choices=['all','fit','score'],default='all')
    parser.add_argument('--task-id',type=int,help='One start/rep pair, indexed in the unchanged global protocol')
    parser.add_argument('--allocation-seconds',type=float,default=28800)
    parser.add_argument('--prior-allocation-seconds',type=float,default=0)
    parser.add_argument('--allow-draft-smoke',action='store_true')
    args=parser.parse_args();config=read(ROOT/args.protocol)
    assert config['status']=='frozen' or args.allow_draft_smoke
    assert 0<=args.prior_allocation_seconds<args.allocation_seconds
    if config['status']=='frozen':
        assert (config['T'],config['J'],config['Euler_steps_per_month'],config['free_coordinates'])==(600,5000,20,23)
        assert config['score']['stages']==list(range(36,289,36)) and config['score']['batch_repetitions']==36
    pairs=[(s,r) for s in config['starts'] for r in range(config['repeats'])]
    if args.task_id is not None:
        assert 0<=args.task_id<len(pairs)
        pairs=[pairs[args.task_id]]
    output=(ROOT/args.output).resolve();assert output.is_relative_to(ROOT)
    if args.task_id is not None:output=output/'tasks'/str(args.task_id)
    output.mkdir(parents=True,exist_ok=True)
    binding=dict(task_id=args.task_id,pairs=pairs,config_sha256=sha(config))
    binding_path=output/'task.json'
    if binding_path.exists():assert read(binding_path)==json.loads(json.dumps(binding))
    else:write(binding_path,binding)
    versions={name:importlib.metadata.version(name) for name in ['numpy','scipy','jax','numba','joblib']}
    versions['particles']=support.particles.__version__
    if config['status']=='frozen':
        for name,version in config['expected_versions'].items():assert versions[name]==version,(name,versions[name],version)
    write(output/'environment'/f'run_{time.time_ns()}.json',dict(versions=versions,
        host=socket.gethostname(),job_id=os.environ.get('SLURM_JOB_ID'),
        array_job_id=os.environ.get('SLURM_ARRAY_JOB_ID'),array_task_id=os.environ.get('SLURM_ARRAY_TASK_ID'),
        thread_environment={k:os.environ.get(k) for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','NUMBA_NUM_THREADS','TF_NUM_INTRAOP_THREADS','TF_NUM_INTEROP_THREADS','XLA_FLAGS']},
        affinity=sorted(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else None))
    target=output/'protocol.json'
    if target.exists():assert read(target)==config
    else:write(target,config)
    signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(Interrupted('SIGTERM')))
    signal.signal(signal.SIGINT,lambda *_:(_ for _ in ()).throw(Interrupted('SIGINT')))
    allocation_start=float(os.environ.get('TASK_ALLOCATION_START',time.time()))
    can_start=lambda estimate:time.time()-allocation_start+args.prior_allocation_seconds+estimate+60<args.allocation_seconds
    started=time.perf_counter();fits=[];scores=[]
    def recorded_fit_seconds():
        return sum(read(p)['charged_fit_seconds'] for p in output.glob('start_*/fit_state.json'))
    prior_recorded_fit=recorded_fit_seconds();scoring_wall_this_process=0.
    state=dict(phase='preparing',config_sha256=sha(config),prior_allocation_seconds=args.prior_allocation_seconds,task_id=args.task_id)
    write(output/'status.json',state)
    # A tiny PF setup probe is preparation, not used as an optimization score.
    warm=support.pf(support.natural(support.STARTS[pairs[0][0]]),2,32,config['seed']-1)
    preparation=time.perf_counter()-started
    write(output/'preparation'/f'run_{time.time_ns()}.json',dict(warmup=warm,seconds=preparation))
    try:
        if args.phase in ('all','fit'):
            for start,rep in pairs:
                folder=output/f'start_{start}_rep_{rep}'
                if not (folder/'endpoint.json').exists() and not can_start(config['budget_seconds']+30):
                    write(output/'status.json',dict(state,phase='pending_allocation',start=start,replicate=rep));return
                write(output/'status.json',dict(state,phase='fitting',start=start,replicate=rep))
                fits.append(fit_one(config,start,rep,folder))
            write(output/'fits.json',fits)
        if args.phase in ('all','score'):
            for start,rep in pairs:
                folder=output/f'start_{start}_rep_{rep}'
                if not (folder/'endpoint.json').exists():continue
                write(output/'status.json',dict(state,phase='scoring',start=start,replicate=rep))
                score_begin=time.perf_counter()
                try:
                    scores.append(dict(start=start,replicate=rep,**score_one(config,start,rep,folder,can_start)))
                finally:
                    scoring_wall_this_process+=time.perf_counter()-score_begin
            write(output/'scores.json',scores)
        all_endpoints=all((output/f'start_{s}_rep_{r}/endpoint.json').exists() for s,r in pairs)
        all_scores=all((output/f'start_{s}_rep_{r}/independent_score.json').exists() and read(output/f'start_{s}_rep_{r}/independent_score.json')['finished'] for s,r in pairs)
        write(output/'status.json',dict(state,phase='complete' if all_endpoints and all_scores else 'pending',
            complete_endpoints=all_endpoints,complete_scores=all_scores,preparation_seconds=preparation,
            process_seconds=time.perf_counter()-started,allocation_elapsed_seconds=time.time()-allocation_start+args.prior_allocation_seconds))
    except BaseException as error:
        write(output/'status.json',dict(state,phase='interrupted' if isinstance(error,Interrupted) else 'failed',
            error=repr(error),traceback=traceback.format_exc(),process_seconds=time.perf_counter()-started,
            allocation_elapsed_seconds=time.time()-allocation_start+args.prior_allocation_seconds))
        raise
    finally:
        process_seconds=time.perf_counter()-started
        fit_this_process=recorded_fit_seconds()-prior_recorded_fit
        write(output/'accounting'/f'run_{time.time_ns()}.json',dict(
            config_sha256=sha(config),process_seconds=process_seconds,
            charged_fitting_seconds=fit_this_process,scoring_wall_seconds=scoring_wall_this_process,
            preparation_and_reporting_seconds=process_seconds-fit_this_process-scoring_wall_this_process,
            allocation_elapsed_seconds=time.time()-allocation_start+args.prior_allocation_seconds,
            prior_allocation_seconds=args.prior_allocation_seconds,
            note='All process time through final status write partitioned into fit, scoring, and preparation/reporting; Slurm allocation also includes shell setup and imports'))

if __name__=='__main__':main()
