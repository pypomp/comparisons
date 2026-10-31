"""Checkpointable fixed-iteration Dacca extensions and official Dynamax SGD."""
from pathlib import Path
import hashlib,json,os,signal,time
import jax
import jax.numpy as jnp
import numpy as np
import optax
from dynamax.parameters import ParameterProperties
import bridge as b
import dacca_model as m
STOP=False


def signals():
    def flag(*_):
        global STOP
        STOP=True
    signal.signal(signal.SIGUSR1,flag);signal.signal(signal.SIGTERM,flag)
def ah(x):return hashlib.sha256(np.asarray(x).tobytes()).hexdigest()
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)
def save(path,meta,**arrays):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.npz.tmp')
    with tmp.open('wb') as out:
        np.savez(out,meta=np.asarray(json.dumps(meta)),**{k:np.asarray(v) for k,v in arrays.items()});out.flush();os.fsync(out.fileno())
    tmp.replace(path)
def load(path,identity):
    with np.load(path,allow_pickle=False) as z:
        meta=json.loads(str(z['meta']));arrays={k:jnp.asarray(z[k]) for k in z.files if k!='meta'}
    assert meta['identity']==identity
    if meta.get('in_flight'):raise RuntimeError('Unclean checkpoint; reconcile old Slurm allocation cost before resuming')
    return meta,arrays
def keys(seed,trial_ids,indices):
    return jax.vmap(lambda i:jax.vmap(lambda trial:jax.random.fold_in(jax.random.fold_in(jax.random.key(seed),trial),i))(trial_ids))(indices)
def memory_gate(compiled,cap):
    x=compiled.memory_analysis();parts={k:int(getattr(x,k)) for k in ['argument_size_in_bytes','output_size_in_bytes','temp_size_in_bytes','alias_size_in_bytes']}
    peak=parts['argument_size_in_bytes']+parts['output_size_in_bytes']+parts['temp_size_in_bytes']-parts['alias_size_in_bytes']
    if cap is not None and peak>cap:raise RuntimeError(f'Compiled kernel estimate {peak} exceeds memory gate {cap}')
    return {**parts,'estimated_peak_bytes':peak}


def if2_pass(swarm,key,i,cooling,T):
    J=swarm.shape[0];data=m.inputs(T)
    initial_sd=jnp.asarray([.16 if n in m.IVPS else 0. for n in m.FREE])
    process_sd=jnp.asarray([0. if n in m.IVPS else .02 for n in m.FREE])
    key,pk=jax.random.split(key);swarm=swarm+cooling**(i/50)*initial_sd*jax.random.normal(pk,swarm.shape)
    x=jax.vmap(lambda p:m.initial_state(m.from_est(p)))(swarm)
    def step(carry,item):
        swarm,x,key=carry;n,inp=item;key,qk,xk,rk=jax.random.split(key,4)
        swarm=swarm+cooling**((i+n/T)/50)*process_sd*jax.random.normal(qk,swarm.shape)
        x=jax.vmap(lambda p,k,x:m.factory(p)[1](k,x,inp))(swarm,jax.random.split(xk,J),x)
        w=jax.vmap(lambda p,x:m.observation(m.from_est(p),x,inp['y']))(swarm,x)
        idx=b.systematic(rk,w)
        return (swarm[idx],x[idx],key),jax.scipy.special.logsumexp(w)-jnp.log(J)
    (swarm,_,_),ll=jax.lax.scan(step,(swarm,x,key),(jnp.arange(T),data))
    return swarm,ll.sum()


def if2_chunk(swarm,active,failed_at,ks,indices,cooling,T):
    def one(carry,item):
        sw,ok,fail=carry;k,i=item
        candidate,ll=jax.vmap(lambda s,k:if2_pass(s,k,i,cooling,T))(sw,k)
        good=jnp.isfinite(candidate).all(axis=(1,2)) & jnp.isfinite(ll)
        newfail=ok & ~good;fail=jnp.where(newfail,i+1,fail);ok=ok & good
        sw=jnp.where(ok[:,None,None],candidate,sw)
        return (sw,ok,fail),(jnp.mean(sw,1),ok,ll)
    return jax.lax.scan(one,(swarm,active,failed_at),(ks,indices))


def fit_if2(q0,path,trial_ids,seed,M,cooling,J,T,chunk,identity,cap=None,stop_after=None):
    path=Path(path);B=len(q0);swarm=jnp.broadcast_to(q0[:,None,:],(B,J,q0.shape[-1]));active=jnp.ones(B,bool);failed=jnp.zeros(B,jnp.int32)
    meta=dict(identity=identity,iterations=0,fit_seconds=0.,prepare_seconds=0.,in_flight=False,complete=False,seed=seed,
              trial_ids=np.asarray(trial_ids).tolist(),initial_parameters_hash=ah(q0),M=M,cooling=cooling,T=T,J=J,compile_records=[])
    if path.exists():
        meta,a=load(path,identity);swarm,active,failed=a['swarm'],a['active'],a['failed_at']
        if meta['complete']:return jnp.mean(swarm,1),meta,active,failed
    compiled_by_size={}
    wall_anchor=time.perf_counter();base_fit=meta['fit_seconds'];base_prepare=meta['prepare_seconds']
    def checkpoint(in_flight):
        meta['fit_seconds']=base_fit+time.perf_counter()-wall_anchor-(meta['prepare_seconds']-base_prepare)
        meta.update(in_flight=in_flight,slurm_job=os.environ.get('SLURM_JOB_ID'),recorded_epoch=time.time())
        save(path,meta,swarm=swarm,active=active,failed_at=failed)
    while meta['iterations']<M and bool(jnp.any(active)) and not STOP:
        n=min(chunk,M-meta['iterations']);start=meta['iterations'];indices=jnp.arange(start,start+n);ks=keys(seed,trial_ids,indices)
        if n not in compiled_by_size:
            tick=time.perf_counter();fn=jax.jit(lambda sw,ok,fa,k,i:if2_chunk(sw,ok,fa,k,i,cooling,T))
            compiled=fn.lower(swarm,active,failed,ks,indices).compile();mem=memory_gate(compiled,cap)
            meta['prepare_seconds']+=time.perf_counter()-tick;meta['compile_records'].append(dict(chunk=n,memory=mem));compiled_by_size[n]=compiled
        checkpoint(True);tick=time.perf_counter()
        (swarm,active,failed),history=jax.block_until_ready(compiled_by_size[n](swarm,active,failed,ks,indices))
        meta['kernel_seconds']=meta.get('kernel_seconds',0.)+time.perf_counter()-tick;meta['iterations']+=n
        meta['complete']=meta['iterations']>=M or not bool(jnp.any(active))
        # Every chunk keeps original full swarm, so restart does not collapse it.
        checkpoint(False)
        with (path.parent/'trace.jsonl').open('a') as out:
            out.write(json.dumps(dict(stage='IF2',iterations=meta['iterations'],fit_seconds=meta['fit_seconds'],
                active=np.asarray(active).tolist(),failed_at=np.asarray(failed).tolist(),parameters=np.asarray(jnp.mean(swarm,1)).tolist()))+'\n')
        if stop_after is not None and meta['iterations']>=stop_after:break
    checkpoint(False)
    return jnp.mean(swarm,1),meta,active,failed


class Pause(Exception):pass
ADAM=optax.scale_by_adam(b1=.9,b2=.999,eps=1e-8)
class ScalarCheckpointOptimizer:
    """Unmodified paper Optax updates inside original scalar SSM.fit_sgd."""
    def __init__(self,q,state,base,meta,path,stop_after=None):
        self.q,self.state,self.base=q,state,base;self.meta,self.path=meta,Path(path);self.stop_after=stop_after
        self.start=time.perf_counter();self.anchor=self.start;self.base_seconds=meta['fit_seconds'];self.first=True
    def init(self,params):np.testing.assert_array_equal(params,self.q);return self.state
    def checkpoint(self,in_flight=False):
        self.meta.update(fit_seconds=self.base_seconds+time.perf_counter()-self.start,in_flight=in_flight,
                         slurm_job=os.environ.get('SLURM_JOB_ID'),recorded_epoch=time.time())
        leaves=jax.tree.leaves(self.state)
        save(self.path,self.meta,q=self.q,**{f'state_{i}':v for i,v in enumerate(leaves)})
    def update(self,g,state,params=None):
        if any(isinstance(x,jax.core.Tracer) for x in jax.tree.leaves((g,state))):raise RuntimeError('Checkpoint hook must execute on host')
        jax.block_until_ready(g);i=self.meta['iterations'];self.meta['attempted']+=1
        if not np.isfinite(np.asarray(g)).all():
            self.meta.update(complete=True,failed=True,failed_at=i+1,failure='nonfinite_gradient')
            self.checkpoint(False);raise Pause()
        # Preserve the original Optax call chain, with no jitted replacement.
        updates,new=self.base.update(g,state)
        q=jax.block_until_ready(optax.apply_updates(self.q,updates));jax.block_until_ready(new)
        if not all(np.isfinite(np.asarray(v)).all() for v in jax.tree.leaves((updates,new,q))):
            self.meta.update(complete=True,failed=True,failed_at=i+1,failure='nonfinite_update')
            self.checkpoint(False);raise Pause()
        elapsed=time.perf_counter()-self.anchor
        self.q,self.state=q,new;self.meta['iterations']+=1
        if self.first:
            self.meta['first_updates_including_internal_compile'].append(dict(iteration=i+1,seconds=elapsed))
            self.first=False
        else:self.meta['subsequent_update_seconds']+=elapsed
        self.meta['complete']=self.meta['iterations']>=self.meta['M']
        self.checkpoint(True)
        if STOP or (self.stop_after is not None and self.meta['iterations']>=self.stop_after):
            self.checkpoint(False);raise Pause()
        self.anchor=time.perf_counter();return updates,new


def fit_gradient(q0,path,trial_ids,seed,warm_seed,M,alpha,J,T,chunk,identity,initial_active=None,initial_failed=None,stop_after=None):
    # B1 is mandatory: the product-model batch wrapper failed its GPU pathwise
    # check and is not part of production. All12 fits use independent jobs.
    assert len(q0)==len(trial_ids)==1
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);q=q0[0]
    base=b.paper_optimizer(M);state=base.init(q)
    failed=initial_active is not None and not bool(initial_active[0])
    meta=dict(identity=identity,iterations=0,attempted=0,fit_seconds=0.,prepare_seconds=0.,in_flight=False,complete=failed,
        failed=failed,failed_at=int(initial_failed[0]) if failed else 0,failure='IF2_prefix_failed' if failed else None,
        initial_parameters_hash=ah(q0),initial_Adam_mu_hash=ah(state[0].mu),initial_Adam_nu_hash=ah(state[0].nu),initial_Adam_count=int(state[0].count),
        seed=seed,trial_ids=np.asarray(trial_ids).tolist(),M=M,alpha=alpha,T=T,J=J,
        timing='fit_seconds includes actual official API internal compilation; first update reported separately, not subtracted',
        first_updates_including_internal_compile=[],subsequent_update_seconds=0.,invocations=0)
    if path.exists():
        meta,a=load(path,identity);q=a['q'];leaves=[a[f'state_{i}'] for i in range(len(jax.tree.leaves(state)))]
        state=jax.tree.unflatten(jax.tree.structure(state),leaves)
        assert int(state[0].count)==int(state[1].count)==meta['iterations']
    if meta['complete']:return q[None],meta,jnp.array([not meta['failed']]),jnp.array([meta['failed_at']])
    i=meta['iterations'];remaining=M-i
    # Actual API data has one full observed sequence per fresh PF key. No
    # extra warmup key, product model, or reset of optimizer moments is used.
    train=keys(seed,trial_ids,jnp.arange(i,M))[:,0]
    words=jax.random.key_data(train)
    ys=jnp.broadcast_to(m.inputs(T)['y'][None,:,None],(remaining,T,1));ins=jnp.broadcast_to(words[:,None],(remaining,T,2))
    proxy=ScalarCheckpointOptimizer(q,state,base,meta,path,stop_after);meta['invocations']+=1;proxy.checkpoint(True)
    try:
        result,_=b.DaccaSGD(J,alpha).fit_sgd(q,ParameterProperties(True),ys,ins,optimizer=proxy,batch_size=1,num_epochs=1,shuffle=False)
        jax.block_until_ready(result);np.testing.assert_array_equal(result,proxy.q)
    except Pause:pass
    # Unexpected exceptions retain the in-flight checkpoint for explicit audit.
    proxy.checkpoint(False)
    with (path.parent/'trace.jsonl').open('a') as out:
        out.write(json.dumps(dict(stage='scalar_official_SGD',iterations=meta['iterations'],fit_seconds=meta['fit_seconds'],
            failed=meta['failed'],failed_at=meta['failed_at'],parameters=np.asarray(proxy.q).tolist()))+'\n')
    return proxy.q[None],meta,jnp.array([not meta['failed']]),jnp.array([meta['failed_at']])
