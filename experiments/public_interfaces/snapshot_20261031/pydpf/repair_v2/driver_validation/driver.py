"""New public API experiment. Full IF2 swarm; fixed batch slots; no retries."""
from pathlib import Path
import argparse,copy,hashlib,json,os,signal,sys,time
import numpy as np
import torch
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT.parent))
import api_adapter as api
import dacca_model as m
STOP=False

def cpu(x):
    if isinstance(x,torch.Tensor):return x.detach().cpu().clone()
    if isinstance(x,dict):return {k:cpu(v) for k,v in x.items()}
    if isinstance(x,list):return [cpu(v) for v in x]
    if isinstance(x,tuple):return tuple(cpu(v) for v in x)
    return copy.deepcopy(x)
def clean(x):
    if isinstance(x,torch.Tensor):return clean(x.detach().cpu().tolist())
    if isinstance(x,np.ndarray):return clean(x.tolist())
    if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [clean(v) for v in x]
    if isinstance(x,(float,np.floating)):return float(x) if np.isfinite(x) else None
    if isinstance(x,np.integer):return int(x)
    return x
def read(p):return json.loads(Path(p).read_text())
def atomic(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp');t.write_text(json.dumps(clean(x),indent=2,allow_nan=False)+'\n');t.replace(p)
def checkpoint(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp');torch.save(cpu(x),t);t.replace(p)
def digest(x):return hashlib.sha256(np.asarray(cpu(x),dtype='<f4').tobytes()).hexdigest()
def jsonhash(x):return hashlib.sha256(json.dumps(clean(x),sort_keys=True,separators=(',',':')).encode()).hexdigest()
def sync(device):
    if str(device).startswith('cuda'):torch.cuda.synchronize()
def tensor_hash(x):return hashlib.sha256(cpu(x).numpy().tobytes()).hexdigest()
def signal_stop(*_):
    global STOP;STOP=True

def setup(config,device,smoke):
    torch.set_num_threads(1)
    if not smoke:assert config['status']=='frozen' and torch.cuda.is_available() and str(device).startswith('cuda') and 'A40' in torch.cuda.get_device_name(0)
    starts=np.load(ROOT.parent/'input/paper_starts_estimation_free.npy')
    initial=np.stack([starts[row['start_id']] for row in config['rows']])
    if smoke and config.get('smoke_published'):initial=np.repeat(m.to_est(m.DATA['theta_published']),len(config['rows']),axis=0)
    sync(device);before=time.perf_counter();model=m.Dacca(device,config['T'],True);p=torch.tensor(initial,device=device,dtype=torch.float32);sync(device)
    return model,p,time.perf_counter()-before

def endpoint(config,row,method,params,failed,steps,reason=None):
    z=params.detach();natural=m.natural(z[None])[0]
    return dict(method=method,**row,parameters_natural=cpu(natural),parameter_names=m.NAMES,parameters_estimation=cpu(z),estimation_parameter_names=m.FREE,parameter_sha256=digest(z),natural_parameter_sha256=digest(natural),fit_failed=failed,iterations_completed=steps,failure=reason,initialization_included=method!='A',protocol_sha256=jsonhash(config))

def warm(config,output,device,smoke=False,limit=None):
    prepare_start=time.perf_counter();out=Path(output)/'warm';out.mkdir(parents=True,exist_ok=True)
    if (out/'summary.json').exists():
        prior=read(out/'summary.json');assert prior['phase']=='complete' and prior['protocol_sha256']==jsonhash(config);return
    model,p,_=setup(config,device,smoke);sync(device);preparation=time.perf_counter()-prepare_start
    atomic(out/'runtime_history'/f'run_{time.time_ns()}.json',dict(preparation_seconds=preparation,job=os.environ.get('SLURM_JOB_ID'),protocol_sha256=jsonhash(config)))
    items=[];begin=time.monotonic()
    for position,row in enumerate(config['rows']):
        path=out/f'row_{row["row_id"]:02d}';path.mkdir(exist_ok=True);finished=path/'result.json'
        if finished.exists():
            record=read(finished);assert record['protocol_sha256']==jsonhash(config) and record['initial_sha256']==digest(p[position]);items.append(record);continue
        assert not (path/'started.json').exists(),'An interrupted public IF2 call cannot resume from a mean: preserve state and seek review; no automatic rerun'
        expected=config['resource_estimate'].get('IF2_B1_seconds_per_pass',1.)*config['warm_IF2_iterations']
        if STOP or (limit and time.monotonic()-begin+1.2*expected+60>limit):atomic(out/'status.json',dict(phase='pending_before_next_row',completed=len(items)));return
        random_seed=api.seed(config['train_seed_root'],0,row['row_id']);atomic(path/'started.json',dict(row=row,seed=random_seed,protocol_sha256=jsonhash(config),job=os.environ.get('SLURM_JOB_ID')))
        sync(device);before=time.perf_counter()
        try:result=api.warm_if2(model,p[position:position+1],config['J'],config['warm_IF2_iterations'],random_seed);sync(device)
        except Exception as error:
            sync(device);atomic(path/'unhandled_failure.json',dict(error=repr(error),call_seconds=time.perf_counter()-before,retry=False));raise
        elapsed=time.perf_counter()-before
        state={name:getattr(result,name) for name in result._fields};state.update(generator_state=model.generator.get_state(),protocol_sha256=jsonhash(config),initial_parameters=p[position],seed=random_seed,call_seconds=elapsed)
        checkpoint(path/'state.pt',state)
        record=dict(row=row,seed=random_seed,success=result.success,completed_iterations=result.completed_iterations,failure=result.failure_reason,parameters_estimation=cpu(result.parameters[0]),parameter_sha256=digest(result.parameters[0]),swarm_sha256=digest(result.parameter_swarm),swarm_shape=list(result.parameter_swarm.shape),initial_sha256=digest(p[position]),call_seconds=elapsed,protocol_sha256=jsonhash(config))
        atomic(finished,record);items.append(record)
    atomic(out/'summary.json',dict(phase='complete',rows=items,total_call_seconds=sum(x['call_seconds'] for x in items),preparation_seconds=sum(read(p)['preparation_seconds'] for p in (out/'runtime_history').glob('*.json')),full_swarm_retained=True,protocol_sha256=jsonhash(config),note='One B1 public call per row, preserving its whole swarm across all175 passes. No mid-call restart from the mean.'))

def grad(config,method,output,device,smoke=False,limit=None,stop_after=None):
    prepare_start=time.perf_counter();out=Path(output)/method;out.mkdir(parents=True,exist_ok=True);model,initial,prep=setup(config,device,smoke);B=len(config['rows']);M=config['gradient_iterations'];alpha=config['alpha'][method]
    init=initial.clone();active=torch.ones(B,dtype=torch.bool,device=device);rows=[dict(**row,fit_failed=False,accepted_steps=0,parameters_estimation=cpu(init[i]),failure=None) for i,row in enumerate(config['rows'])];warm_cost=0.;warm_prepare=0.
    if method in ['B','C']:
        w=read(Path(output)/'warm/summary.json');assert w['phase']=='complete' and w['protocol_sha256']==jsonhash(config)
        warm_cost=w['total_call_seconds'];warm_prepare=w['preparation_seconds']
        for i,record in enumerate(w['rows']):
            assert record['row']==config['rows'][i];z=torch.tensor(record['parameters_estimation'],device=device,dtype=torch.float32);assert digest(z)==record['parameter_sha256'];init[i]=z
            rows[i].update(parameters_estimation=cpu(z),warm_completed_iterations=record['completed_iterations'])
            if not record['success']:active[i]=False;rows[i].update(fit_failed=True,failure='public IF2: '+str(record['failure']),failed_phase='warm_IF2')
    coords=[torch.nn.Parameter(init[:,i].clone()) for i in range(23)];opt=torch.optim.Adam([{'params':[v],'lr':.1} for v in coords],betas=(.9,.999),eps=1e-8)
    initial_optimizer_hash=jsonhash(opt.state_dict());initial_parameter_hash=digest(init);iteration=0;compute=0.;prior_wall=0.;prior_init=0.;cp_path=out/'checkpoint.pt'
    if cp_path.exists():
        cp=torch.load(cp_path,map_location='cpu',weights_only=True);assert cp['protocol_sha256']==jsonhash(config) and cp['method']==method and cp['initial_parameter_sha256']==initial_parameter_hash
        if cp['phase']=='complete':return
        assert cp['job_id']==os.environ.get('SLURM_JOB_ID') or smoke,'Resumption in another allocation requires explicit total-cost review'
        for i,p in enumerate(coords):p.data.copy_(cp['parameters'][:,i].to(device))
        opt.load_state_dict(cp['optimizer']);iteration=cp['iteration'];active=cp['active'].to(device);rows=cp['rows'];compute=cp['compute_seconds'];prior_wall=cp['gradient_wall_seconds'];prior_init=cp['initialization_seconds']
    sync(device);begin=time.monotonic();wall_begin=time.perf_counter();initialization=time.perf_counter()-prepare_start+prior_init
    def snapshot(phase):
        params=torch.stack(coords,-1).detach()
        state=dict(phase=phase,method=method,protocol_sha256=jsonhash(config),iteration=iteration,target_iterations=M,parameters=params,optimizer=opt.state_dict(),generator_state=model.generator.get_state(),active=active,rows=rows,compute_seconds=compute,gradient_wall_seconds=prior_wall+time.perf_counter()-wall_begin,initialization_seconds=initialization,warm_IF2_call_seconds=warm_cost,warm_IF2_preparation_seconds=warm_prepare,initial_parameter_sha256=initial_parameter_hash,initial_Adam_state_sha256=initial_optimizer_hash,initial_Adam_state_empty=True,job_id=os.environ.get('SLURM_JOB_ID'),batch_slots=B)
        checkpoint(cp_path,state);atomic(out/'status.json',{k:v for k,v in state.items() if k not in ['parameters','optimizer','generator_state','rows']});return state
    snapshot('running')
    while iteration<M and active.any():
        expected=config['resource_estimate'].get('gradient_B12_seconds',60.)
        if STOP or (stop_after is not None and iteration>=stop_after) or (limit and time.monotonic()-begin+1.5*expected+60>limit):snapshot('pending');return
        params=torch.stack(coords,-1);before=cpu(params);before_opt=cpu(opt.state_dict());before_active=cpu(active);random_seed=api.seed(config['train_seed_root'],2 if method=='A' else 1,iteration)
        opt.zero_grad(set_to_none=True);sync(device);started=time.perf_counter()
        try:
            value=-api.factors(model,params,config['J'],random_seed,alpha=alpha).sum(0).reshape(B)/config['T']
            good_loss=torch.isfinite(value);objective=torch.where(active&good_loss,value,torch.zeros_like(value)).sum();grads=torch.autograd.grad(objective,coords,allow_unused=True)
            grads=[torch.zeros_like(p) if g is None else g for p,g in zip(coords,grads)];good_grad=torch.stack([torch.isfinite(g) for g in grads],-1).all(-1);good=active&good_loss&good_grad
            for p,g in zip(coords,grads):p.grad=torch.where(good,g,torch.zeros_like(g))
            warmup=min(10,M);ramp=.1 if warmup==1 else .1+.9*min(iteration,warmup-1)/(warmup-1);factor=ramp*(.05+.95*.5*(1+np.cos(np.pi*iteration/M)))
            for i,group in enumerate(opt.param_groups):group['lr']=float((.05 if m.FREE[i] in m.IVPS+['gamma','beta_trend','sigma','tau'] else .1)*factor)
            opt.step();proposed=torch.stack(coords,-1).detach();good_parameter=torch.isfinite(proposed).all(-1);good=good&good_parameter
            for i,p in enumerate(coords):
                with torch.no_grad():p.copy_(torch.where(good,p,before[:,i].to(device)))
                # Inactive/failed slots retain their moments; per-failure full prior state is also saved.
                state=opt.state[p];old=before_opt['state'].get(i,{})
                for name in ['exp_avg','exp_avg_sq']:
                    previous=old[name].to(device) if name in old else torch.zeros_like(state[name]);state[name].copy_(torch.where(good,state[name],previous))
            sync(device);elapsed=time.perf_counter()-started;compute+=elapsed
        except Exception as error:
            sync(device);elapsed=time.perf_counter()-started;compute+=elapsed
            checkpoint(out/'unhandled_pre_step.pt',dict(parameters=before,optimizer=before_opt,active=before_active,iteration=iteration,seed=random_seed))
            atomic(out/'unhandled_failure.json',dict(error=repr(error),seconds=elapsed,iteration=iteration,retry=False,note='No row retry or batch resizing; task remains incomplete'));raise
        event=dict(iteration=iteration,seed=random_seed,seconds=elapsed,alpha=alpha,active_before=before_active,accepted=good,training_loss=value,batch_slots=B,generator_after_sha256=tensor_hash(model.generator.get_state()),fallback=False)
        for i,row in enumerate(rows):
            if not bool(before_active[i]):continue
            if bool(good[i]):row.update(parameters_estimation=cpu(torch.stack(coords,-1)[i]),accepted_steps=iteration+1)
            else:
                reason='nonfinite loss' if not bool(good_loss[i]) else 'nonfinite gradient' if not bool(good_grad[i]) else 'nonfinite parameter update'
                row.update(fit_failed=True,failure=reason,failed_iteration=iteration,failed_phase='gradient')
                checkpoint(out/'failures'/f'row_{row["row_id"]:02d}_last_valid.pt',dict(row_id=row['row_id'],row_position=i,parameters=before,optimizer=before_opt,active=before_active,accepted_steps=row['accepted_steps'],failed_iteration=iteration,seed=random_seed,note='Full pre-failure optimizer state; failed slot is not reset or selected from retries'))
        active=good;iteration+=1;atomic(out/'events'/f'step_{iteration-1:05d}.json',event);snapshot('running')
    final=snapshot('complete');params=torch.stack(coords,-1).detach();endpoints=[]
    for i,row in enumerate(rows):
        e=endpoint(config,config['rows'][i],method,params[i],row['fit_failed'],row['accepted_steps'],row['failure']);e.update(warm_IF2_iterations_completed=row.get('warm_completed_iterations',0),initial_parameter_sha256=digest(init[i]),initial_Adam_state_sha256=initial_optimizer_hash,gradient_seed_namespace=2 if method=='A' else 1)
        endpoints.append(e)
    atomic(out/'endpoints.json',endpoints)
    atomic(out/'timing.json',dict(method=method,gradient_wall_seconds=final['gradient_wall_seconds'],compute_seconds=compute,initialization_seconds=initialization,warm_IF2_call_seconds=warm_cost,warm_IF2_preparation_seconds=warm_prepare,logical_method_fit_seconds=final['gradient_wall_seconds']+initialization+warm_cost+warm_prepare,physical_shared_warm_counted_once=True,note='B/C each report full shared warm cost; physical aggregate deduplicates warm job. No multiplication of batch time by12.'))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--protocol',required=True);p.add_argument('--phase',choices=['warm','A','B','C'],required=True);p.add_argument('--output',required=True);p.add_argument('--device',default='cuda');p.add_argument('--smoke',action='store_true');p.add_argument('--limit-seconds',type=float);a=p.parse_args();signal.signal(signal.SIGTERM,signal_stop);signal.signal(signal.SIGINT,signal_stop)
    config=read(a.protocol);warm(config,a.output,a.device,a.smoke,a.limit_seconds) if a.phase=='warm' else grad(config,a.phase,a.output,a.device,a.smoke,a.limit_seconds)
