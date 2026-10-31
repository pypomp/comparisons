"""Fixed-iteration batched Dacca estimation using local PFJAX extensions."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,json,os,time,traceback
import jax
import jax.numpy as jnp
import numpy as np
import bridge as b
import dacca_model as m
import common as c
ROOT=Path(__file__).resolve().parent

def save(p,x):c.atomic_json(p,x)
def finite_rows(x,B):
    flags=np.ones(B,bool)
    for v in jax.tree.leaves(x):
        a=np.asarray(v)
        if a.ndim and a.shape[0]==B:flags &= np.isfinite(a).reshape(B,-1).all(1)
        else:flags &= np.isfinite(a).all()
    return flags

def compile_call(fn,args):
    start=time.perf_counter();exe=jax.jit(fn).lower(*args).compile();elapsed=time.perf_counter()-start
    mem=exe.memory_analysis();peak=mem.argument_size_in_bytes+mem.output_size_in_bytes+mem.temp_size_in_bytes-mem.alias_size_in_bytes
    if jax.devices()[0].platform=='gpu':assert peak<.85*jax.devices()[0].memory_stats()['bytes_limit']
    return exe,elapsed,int(peak)

def if2(cfg,p,keys,steps,cooling,out):
    out.mkdir(parents=True,exist_ok=True)
    if (out/'fit.json').exists():return json.loads((out/'fit.json').read_text()),jnp.asarray(np.load(out/'terminal.npy'))
    fn=jax.vmap(b.if2(cfg['J'],cfg['T'],steps,cooling));exe,comp,mem=compile_call(fn,(p,keys))
    tick=time.perf_counter();trace,ll=jax.block_until_ready(exe(p,keys));seconds=time.perf_counter()-tick
    trace=np.asarray(trace);ll=np.asarray(ll);B=len(p);ok=finite_rows((trace,ll),B);terminal=[];counts=[]
    for i in range(B):
        flags=np.isfinite(trace[i]).all(-1)&np.isfinite(ll[i]);bad=np.where(~flags)[0]
        n=int(bad[0]) if len(bad) else steps
        terminal.append(trace[i,n-1] if n else np.asarray(p[i]));counts.append(n)
    final=np.array(terminal);np.save(out/'terminal.npy',final);np.savez(out/'trace.npz',parameters=trace,perturbed_likelihood=ll,initial=np.asarray(p),keys=np.asarray(jax.random.key_data(keys)))
    info=dict(iterations_requested=steps,iterations_completed=counts,failed=(~ok).tolist(),batch_size=B,batch_fit_seconds=seconds,compile_seconds=comp,estimated_peak_bytes=mem,initial_parameter_sha=c.tree_sha(p),terminal_sha=c.tree_sha(final),cold_execution_in_fit_seconds=True)
    save(out/'fit.json',info);return info,jnp.asarray(final)

def gradients(cfg,p,keys,steps,alpha,out,initial_failed):
    out.mkdir(parents=True,exist_ok=True)
    if (out/'fit.json').exists():return json.loads((out/'fit.json').read_text()),jnp.asarray(np.load(out/'terminal.npy'))
    B=len(p);adam=c.optimizer(cfg);state=adam.init(p);rates=c.lr_array(cfg,steps);objective=b.objective('dmop',cfg['J'],cfg['T'],alpha)
    def update(p,state,keys,eta):
        loss,g=jax.vmap(jax.value_and_grad(lambda q,k:-objective(q,k)/cfg['T']))(p,keys)
        direction,proposed_state=adam.update(g,state,p)
        return p-eta*direction,proposed_state,loss,g
    exe,comp,mem=compile_call(update,(p,state,keys,rates[0]));warmkeys=jax.vmap(lambda k:jax.random.fold_in(k,987654321))(keys)
    tick=time.perf_counter();jax.block_until_ready(exe(p,state,warmkeys,rates[0]));warm=time.perf_counter()-tick
    alive=~np.array(initial_failed,bool);counts=np.zeros(B,int);reasons=['IF2 prefix failed' if f else None for f in initial_failed];seconds=[];losses=[];trace=[np.asarray(p)];first_iteration=0
    if (out/'checkpoint.npz').exists():
        ck=np.load(out/'checkpoint.npz');meta=json.loads(str(ck['metadata']))
        assert meta['steps']==steps and meta['alpha']==alpha and meta['T']==cfg['T'] and meta['J']==cfg['J']
        assert np.array_equal(ck['keys'],np.asarray(jax.random.key_data(keys)))
        assert np.array_equal(ck['trace'][0],np.asarray(p)), 'Checkpoint initial parameters differ'
        p=jnp.asarray(ck['parameters']);state=state._replace(count=jnp.asarray(ck['count']),mu=jnp.asarray(ck['mu']),nu=jnp.asarray(ck['nu']))
        alive=ck['active'];counts=ck['updates_completed'];seconds=ck['seconds'].tolist();losses=list(ck['losses']);trace=list(ck['trace']);first_iteration=int(ck['next_iteration']);reasons=meta['reasons']
    for i in range(first_iteration,steps):
        usekeys=jax.vmap(lambda k:jax.random.fold_in(k,i))(keys);tick=time.perf_counter()
        proposed,newstate,loss,g=jax.block_until_ready(exe(p,state,usekeys,rates[i]));seconds.append(time.perf_counter()-tick)
        finite=finite_rows((proposed,newstate,loss,g),B);newfailed=alive & ~finite
        for row in np.where(newfailed)[0]:reasons[row]=f'Non-finite proposal/loss/gradient/Adam state at update {i}'
        alive &= finite;mask=jnp.asarray(alive)[:,None]
        p=jnp.where(mask,proposed,p)
        state=state._replace(count=newstate.count,mu=jnp.where(mask,newstate.mu,state.mu),nu=jnp.where(mask,newstate.nu,state.nu))
        counts+=alive;losses.append(np.asarray(loss));trace.append(np.asarray(p))
        if (i+1)%10==0 or i==steps-1:
            checkpoint=out/'checkpoint.tmp.npz'
            np.savez(checkpoint,parameters=np.asarray(p),mu=np.asarray(state.mu),nu=np.asarray(state.nu),count=np.asarray(state.count),keys=np.asarray(jax.random.key_data(keys)),active=alive,updates_completed=counts,next_iteration=i+1,seconds=seconds,losses=np.array(losses),trace=np.array(trace),metadata=json.dumps(dict(steps=steps,alpha=alpha,T=cfg['T'],J=cfg['J'],reasons=reasons)))
            checkpoint.replace(out/'checkpoint.npz')
            save(out/'progress.json',dict(iteration=i+1,iterations_completed=counts.tolist(),failed=(~alive).tolist(),batch_fit_seconds=sum(seconds)))
        if not alive.any():break
    final=np.asarray(p);np.save(out/'terminal.npy',final);np.savez(out/'trace.npz',parameters=np.array(trace),training_loss=np.array(losses),keys=np.asarray(jax.random.key_data(keys)))
    info=dict(iterations_requested=steps,iterations_completed=counts.tolist(),failed=(~alive).tolist(),failure_reasons=reasons,batch_size=B,alpha=alpha,batch_fit_seconds=sum(seconds),update_seconds=seconds,compile_seconds=comp,preparation_seconds=warm,estimated_peak_bytes=mem,initial_parameter_sha=c.tree_sha(trace[0]),initial_adam_sha=c.tree_sha(adam.init(jnp.asarray(trace[0]))),terminal_sha=c.tree_sha(final),warmup_state_unchanged=True)
    save(out/'fit.json',info);return info,p

def evaluator(cfg):
    import pypomp as pp
    st=pp.models.dhaka(dt=None,nstep=20).to_struct();T=cfg['T'];J=cfg['evaluation']['J'];R=cfg['evaluation']['stage_repetitions']
    if T!=600:st=st._replace(ys=st.ys[:T],times=st.times[:T],covars_extended=st.covars_extended[:20*T+1],dt_array_extended=st.dt_array_extended[:20*T],nstep_array=st.nstep_array[:T])
    def run(p,key):
        theta=m.from_est(p)[jnp.asarray([m.NAMES.index(n) for n in st.param_names])]
        return pp.functional.pfilter(st,jnp.broadcast_to(theta,(R,len(theta))),J=J,keys=jax.random.split(key,(R,1)))['logLik'].reshape(R)
    return run

def scores(cfg,endpoints,out):
    out.mkdir(exist_ok=True);run=evaluator(cfg);key=jax.random.key(cfg['evaluation']['seed'])
    exe,comp,mem=compile_call(run,(jnp.asarray(endpoints[0]['parameters']),key));save(out/'compilation.json',dict(seconds=comp,estimated_peak_bytes=mem))
    records=[]
    for e in endpoints:
        folder=out/f"{e['method']}_s{e['start_id']}_r{e['replicate']}";folder.mkdir(exist_ok=True)
        if (folder/'score.json').exists():records.append(json.loads((folder/'score.json').read_text()));continue
        p=jnp.asarray(e['parameters']);raw=[];seconds=[];rowkey=jax.random.fold_in(key,e['score_index'])
        for stage in range(cfg['evaluation']['maximum_repetitions']//cfg['evaluation']['stage_repetitions']):
            usekey=jax.random.fold_in(rowkey,stage);tick=time.perf_counter();values=np.asarray(jax.block_until_ready(exe(p,usekey)),float);seconds.append(time.perf_counter()-tick);raw.extend(values.tolist())
            ll=np.asarray(raw);finite=np.isfinite(ll).all();w=np.exp(ll-ll.max()) if finite else np.full(len(ll),np.nan)
            lm=float(ll.max()+np.log(w.mean()));se=float(w.std(ddof=1)/np.sqrt(len(w))/w.mean());ess=float(w.sum()**2/(w*w).sum());precise=bool(finite and se<=1 and ess>=10)
            result={**e,'backend':'original pinned PyPOMP PF','raw':raw,'J':cfg['evaluation']['J'],'repetitions':len(raw),'logmeanlik':lm,'MCSE':se,'likelihood_ESS':ess,'precision_passed':precise,'evaluation_seconds':sum(seconds),'parameter_sha':c.tree_sha(p),'score_key':np.asarray(jax.random.key_data(rowkey)).tolist()}
            save(folder/'partial.json',result)
            if precise:break
        save(folder/'score.json',result);records.append(result);save(out/'all_scores.json',records)
    return records

def execute(cfg,out,smoke=False):
    if not smoke:
        assert cfg['status']=='FROZEN' and cfg['T']==600 and cfg['J']==5000
        assert jax.devices()[0].platform=='gpu' and 'A40' in jax.devices()[0].device_kind
    out.mkdir(parents=True,exist_ok=True)
    if (out/'protocol.json').exists():assert json.loads((out/'protocol.json').read_text())==cfg, 'Cannot replace protocol on resume'
    else:save(out/'protocol.json',cfg)
    started=time.perf_counter()
    save(out/'runtime.json',dict(jax=jax.__version__,GPU=jax.devices()[0].device_kind,node=os.environ.get('SLURMD_NODENAME'),job=os.environ.get('SLURM_JOB_ID'),utc=datetime.now(timezone.utc).isoformat()))
    starts=np.load(ROOT/'input/paper_starts_estimation_free.npy')
    # Validate the actual independent scoring path before spending fit time.
    first=jnp.asarray(starts[cfg['start_ids'][0]]);check_key=jax.random.key(730001)
    check,check_compile,check_memory=compile_call(evaluator(cfg),(first,check_key))
    tick=time.perf_counter();check_raw=np.asarray(jax.block_until_ready(check(first,check_key)));check_seconds=time.perf_counter()-tick
    save(out/'reference_startup_check.json',dict(backend='original PyPOMP',finite=bool(np.isfinite(check_raw).all()),raw=check_raw.tolist(),compile_seconds=check_compile,execution_seconds=check_seconds,estimated_peak_bytes=check_memory,not_used_for_endpoint_selection=True))
    assert np.isfinite(check_raw).all(), 'Independent scoring runtime failed before formal fitting'
    del check
    cells=[(s,r) for s in cfg['start_ids'] for r in range(cfg['repetitions'])];endpoints=[]
    for lo in range(0,len(cells),cfg['batch_size']):
        block=cells[lo:lo+cfg['batch_size']];p=jnp.asarray(np.array([starts[s] for s,r in block]));idx=jnp.asarray([s*100+r for s,r in block],jnp.uint32)
        key=lambda seed:jax.vmap(lambda i:jax.random.fold_in(jax.random.key(seed),i))(idx)
        folder=out/f'batch{lo:03d}';folder.mkdir(exist_ok=True)
        info_if2,q=if2(cfg,p,key(cfg['fit_seed']),cfg['if2_iterations'],cfg['if2_cooling_50'],folder/'IF2')
        info_warm,warm=if2(cfg,p,key(cfg['warm_seed']),cfg['warm_if2_iterations'],.5,folder/'common_IF2')
        info_ifad,a=gradients(cfg,warm,key(cfg['gradient_seed']),cfg['gradient_iterations'],.97,folder/'IFAD',info_warm['failed'])
        info_adpf,d=gradients(cfg,p,key(cfg['gradient_seed']+1),cfg['native_gradient_iterations'],1.,folder/'ADPF_extension',[False]*len(block))
        for ai,(name,info,params,prefix) in enumerate([('IF2',info_if2,q,None),('IFAD',info_ifad,a,info_warm),('ADPF_extension',info_adpf,d,None)]):
            for j,(sid,rep) in enumerate(block):
                endpoints.append(dict(method=name,start_id=sid,replicate=rep,parameters=np.asarray(params[j]).tolist(),failed=info['failed'][j],iterations_completed=info['iterations_completed'][j],batch_index=lo,batch_size=len(block),measured_batch_fit_seconds=info['batch_fit_seconds']+(prefix['batch_fit_seconds'] if prefix else 0),individual_latency_not_measured=True,IF2_prefix_iterations=cfg['warm_if2_iterations'] if prefix else 0,score_index=ai*10000+sid*100+rep))
        save(out/'endpoints.json',endpoints)
    records=scores(cfg,endpoints,out/'scores')
    evidence={str(f.relative_to(ROOT)):c.sha(f) for f in out.rglob('*') if f.is_file() and f.name not in ['COMPLETE.json','terminal_manifest.json']}
    save(out/'terminal_manifest.json',dict(files=evidence,source_manifest_sha256=c.sha(ROOT/'source_manifest.json') if (ROOT/'source_manifest.json').exists() else None))
    save(out/'COMPLETE.json',dict(terminal_manifest_sha256=c.sha(out/'terminal_manifest.json'),complete=True,endpoint_count=len(endpoints),failed=sum(r['failed'] for r in endpoints),all_scores_present=len(records)==len(endpoints),total_wall_seconds=time.perf_counter()-started,native_OT='Full-series gradients previously failed; not silently replaced. ADPF_extension is local correction, not native PFJAX estimator.'))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--protocol',type=Path,default=ROOT/'protocol.json');ap.add_argument('--smoke',action='store_true');a=ap.parse_args();cfg=json.loads(a.protocol.read_text());execute(cfg,ROOT/('local_smoke' if a.smoke else 'results'),a.smoke)
