"""A, B, C using public candidate APIs, with a shared IF2 endpoint for B/C.

A: original start + alpha=1 Adam400.
B: IF2(175) + alpha=1 Adam400.
C: same IF2 endpoint + alpha=.97 Adam400.
Each case runs sequentially on one A40. These are three optimization procedures;
A has less total work. The discounted-gradient comparison is B versus C.
"""
from pathlib import Path
import argparse,hashlib,json,os,pickle,time
import jax
import jax.numpy as jnp
import numpy as np
import optax
import dacca_model as m

ROOT=Path(__file__).resolve().parent
def atomic(path,x):
 p=path.with_suffix(path.suffix+'.tmp');p.write_text(json.dumps(x,indent=2)+'\n');p.replace(path)
def tree_sha(x):
 h=hashlib.sha256()
 for v in jax.tree.leaves(x):
  a=np.asarray(v);h.update(str((a.shape,str(a.dtype))).encode());h.update(a.tobytes())
 return h.hexdigest()
def finite(x):return all(np.isfinite(np.asarray(v)).all() for v in jax.tree.leaves(x))
def save_state(path,x):
 p=path.with_suffix('.tmp');p.write_bytes(pickle.dumps(jax.device_get(x),protocol=5));p.replace(path)
def compile_fn(fn,args):
 t=time.perf_counter();exe=jax.jit(fn).lower(*args).compile();seconds=time.perf_counter()-t
 mem=exe.memory_analysis();peak=mem.argument_size_in_bytes+mem.output_size_in_bytes+mem.temp_size_in_bytes-mem.alias_size_in_bytes
 if jax.default_backend()=='gpu':assert peak<.85*jax.devices()[0].memory_stats()['bytes_limit']
 return exe,seconds,peak
def rates(M):
 half=m.IVPS+['gamma','beta_trend','sigma','tau']
 base=np.asarray([.05 if n in half else .1 for n in m.FREE],np.float32)
 warm=np.r_[np.linspace(.1,1,min(10,M)),np.ones(max(0,M-10))]
 cosine=.05+.95*.5*(1+np.cos(np.pi*np.arange(M)/M))
 return jnp.asarray((warm*cosine)[:,None]*base[None],jnp.float32)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--package',choices=['cuthbert','pfjax'],required=True);ap.add_argument('--case',type=int,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--smoke',action='store_true');args=ap.parse_args()
 assert 0<=args.case<12 and not jax.config.jax_enable_x64
 if not args.smoke:assert jax.default_backend()=='gpu' and 'A40' in jax.devices()[0].device_kind
 J=8 if args.smoke else 5000;N=2 if args.smoke else 175;M=2 if args.smoke else 400
 sid,rep=divmod(args.case,2);out=args.output;out.mkdir(parents=True,exist_ok=True)
 if (out/'FIT_COMPLETE.json').exists():return
 # Refuse silent replay after a killed/interrupted fit. Recovery must account
 # for the actual job cost and validate full optimizer/swarm state first.
 assert not (out/'STARTED.json').exists(),'Interrupted attempt exists; do not replay without an explicit state-and-cost audit'
 starts=np.load(ROOT/'input/paper_starts_estimation_free.npy');p=jnp.asarray(starts[sid],jnp.float32);data=m.inputs(600)
 root=24681000 if args.package=='cuthbert' else 25081000
 key=jax.random.fold_in(jax.random.PRNGKey(root),args.case)
 warmkey=jax.random.fold_in(key,0);gradkey=jax.random.fold_in(key,1)
 started=time.perf_counter();atomic(out/'STARTED.json',{'package':args.package,'case':args.case,'J':J,'T':600,'warm_iterations':N,'gradient_iterations':M,'root_seed':root,'warm_key':np.asarray(warmkey).tolist(),'gradient_root_key':np.asarray(gradkey).tolist(),'start_sha256':tree_sha(p),'starts_file_sha256':hashlib.sha256((ROOT/'input/paper_starts_estimation_free.npy').read_bytes()).hexdigest(),'jax':jax.__version__,'device':str(jax.devices()[0]),'device_kind':jax.devices()[0].device_kind,'job':os.environ.get('SLURM_JOB_ID'),'node':os.environ.get('SLURMD_NODENAME')})
 if args.package=='cuthbert':
  from cuthbert.smc import iterated_filtering, dmop
  def warm_fn(p,key):return iterated_filtering.fit(m.factory,p,data,key=key,n_particles=J,n_iterations=N,perturbation_scale=m.PROCESS_SCALE,initial_perturbation_scale=m.INITIAL_SCALE,cooling_rate=.5**(1/50),resampling='multinomial')
  def obj(alpha):return lambda q,k: -dmop.log_likelihood(m.factory,q,data,key=k,n_particles=J,alpha=alpha,resampling='multinomial',checkpoint=False)/600
  warm_exe,wcomp,wmem=compile_fn(warm_fn,(p,warmkey));t=time.perf_counter();w=jax.block_until_ready(warm_exe(p,warmkey));wseconds=time.perf_counter()-t
  warmparams=w.parameters;wfail=not bool(w.success);wcount=int(w.completed_iterations)
  save_state(out/'IF2_full_state.pkl',w)
  warmnote='One public fit call; entire parameter swarm persists; cooling constant within each pass (.5**(iteration/50)).'
 else:
  import pfjax as pf
  import bridge
  model=bridge.Dacca()
  def one(s,k,i):return pf.iterated_filter_step(model,k,data,s,m.PROCESS_SCALE,initial_scale=m.INITIAL_SCALE,iteration=i,cooling_rate=.5,cooling_interval=50.,within_pass_cooling=True)
  def obj(alpha):return lambda q,k:-pf.particle_filter_mop(model,k,data,q,J,alpha=alpha)['loglik']/600
  swarm=jnp.broadcast_to(p,(J,23));warmparams=p;warm_exe,wcomp,wmem=compile_fn(one,(swarm,warmkey,jnp.asarray(0,jnp.int32)));wseconds=0.;wcount=0;wfail=False;trace=[]
  for i in range(N):
   ik=jax.random.fold_in(warmkey,i);t=time.perf_counter();w=jax.block_until_ready(warm_exe(swarm,ik,jnp.asarray(i,jnp.int32)));wseconds+=time.perf_counter()-t
   accepted=bool(w['valid']) and finite(w);trace.append({'iteration':i,'accepted':accepted,'perturbed_loglik':float(w['loglik'])})
   if accepted:swarm=w['parameter_particles'];warmparams=w['theta'];wcount+=1
   else:wfail=True
   if (i+1)%10==0 or wfail or i==N-1:save_state(out/'IF2_full_state.pkl',{'swarm':swarm,'parameters':warmparams,'next_iteration':i+1,'warm_root_key':warmkey,'accepted':wcount,'failed':wfail,'trace':trace,'execution_seconds':wseconds})
   if wfail:break
  warmnote='Public iterated_filter_step; full terminal parameter swarm passed to next call; within-pass geometric cooling enabled.'
 warmrecord={'fit_failed':wfail,'iterations_completed':wcount,'fit_seconds':wseconds,'compile_seconds':wcomp,'estimated_peak_bytes':wmem,'parameters_sha256':tree_sha(warmparams),'original_parameters_sha256':tree_sha(p),'public_api_behavior':warmnote}
 atomic(out/'IF2.json',warmrecord)
 adam=optax.scale_by_adam(b1=.9,b2=.999,eps=1e-8);etas=rates(M);endpoints=[]
 # Rotate ordering by case. B/C always use the same warm endpoint, zero Adam
 # state, learning rates, iteration count and keys, irrespective of order.
 plans=[('A',1.,p,False),('B',1.,warmparams,wfail),('C',.97,warmparams,wfail)]
 offset=args.case%3;plans=plans[offset:]+plans[:offset];compiled={}
 for arm,alpha,q,failed in plans:
  directory=out/arm;directory.mkdir(exist_ok=True);state=adam.init(q);initial_sha=tree_sha(q);adam_sha=tree_sha(state);count=0;times=[];losses=[];trace=[np.asarray(q)];reason='IF2 failed' if failed else None;compile_seconds=0.
  def update(p,state,k,eta):
   val,g=jax.value_and_grad(obj(alpha))(p,k);direction,newstate=adam.update(g,state,p)
   return p-eta*direction,newstate,val,g
  if not failed:
   if alpha not in compiled:
    exe,compile_seconds,peak=compile_fn(update,(q,state,gradkey,etas[0]));compiled[alpha]=(exe,peak)
   else:exe,peak=compiled[alpha]
   for i in range(M):
    ik=jax.random.fold_in(gradkey,i);t=time.perf_counter();proposal,newstate,loss,g=jax.block_until_ready(exe(q,state,ik,etas[i]));times.append(time.perf_counter()-t);losses.append(float(loss))
    if finite((proposal,newstate,loss,g,m.from_est(proposal))):q=proposal;state=newstate;count+=1;trace.append(np.asarray(q))
    else:failed=True;reason=f'Nonfinite loss/gradient/estimation or natural parameter/Adam state at zero-based update {i}'
    if (i+1)%10==0 or failed or i==M-1:
     save_state(directory/'checkpoint.pkl',{'parameters':q,'adam_state':state,'next_iteration':i+1,'accepted_updates':count,'gradient_root_key':gradkey,'trace':trace,'training_loss':losses,'update_seconds':times,'fit_failed':failed,'reason':reason})
     atomic(directory/'progress.json',{'attempted':i+1,'accepted':count,'fit_failed':failed,'fit_seconds':sum(times)})
    if failed:break
  info={'method':arm,'alpha':alpha,'fit_failed':failed,'failure_reason':reason,'iterations_requested':M,'iterations_completed':count,'initial_parameter_sha256':initial_sha,'initial_adam_sha256':adam_sha,'gradient_key':np.asarray(gradkey).tolist(),'gradient_seconds':sum(times),'gradient_compile_seconds_attributed_to_first_use':compile_seconds,'IF2_seconds_attributed':0. if arm=='A' else wseconds,'IF2_compile_seconds_attributed':0. if arm=='A' else wcomp,'logical_fit_seconds':sum(times)+(0. if arm=='A' else wseconds),'B1_individual_latency':True,'initialization_computation_physically_shared_B_C':arm!='A','terminal_estimation_sha256':tree_sha(q)}
  atomic(directory/'fit.json',info);natural=np.asarray(m.from_est(q));np.save(directory/'terminal_estimation.npy',np.asarray(q));np.save(directory/'terminal_natural.npy',natural)
  endpoints.append({**info,'start_id':sid,'replicate':rep,'parameters_natural':[float(x) if np.isfinite(x) else None for x in natural],'parameter_names':m.NAMES,'parameters_estimation':np.asarray(q).tolist(),'IF2_iterations_completed':0 if arm=='A' else wcount,'optimizer':'external Optax scale_by_adam with explicit per-coordinate learning-rate array'})
  atomic(out/'endpoints.json',endpoints)
 b=next(x for x in endpoints if x['method']=='B');c=next(x for x in endpoints if x['method']=='C')
 assert b['initial_parameter_sha256']==c['initial_parameter_sha256'] and b['initial_adam_sha256']==c['initial_adam_sha256'] and b['gradient_key']==c['gradient_key']
 atomic(out/'FIT_COMPLETE.json',{'fit_complete':True,'endpoints':len(endpoints),'failed':sum(e['fit_failed'] for e in endpoints),'B_C_common_initialization_and_keys_verified':True,'elapsed_seconds':time.perf_counter()-started,'physical_IF2_calls':1,'logical_IF2_cost_attributed_to_each_B_C':True,'smoke_only':args.smoke})

if __name__=='__main__':main()
