"""Synchronized native PF timing or public API cost measurement on A40."""
import argparse, hashlib, importlib, inspect, json, os, platform, time
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
import dacca_model as m

p=argparse.ArgumentParser();p.add_argument('package',choices=['cuthbert','pfjax']);p.add_argument('mode',choices=['native','public']);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
assert jax.default_backend()=='gpu' and 'A40' in jax.devices()[0].device_kind
assert not jax.config.jax_enable_x64
params=jnp.asarray(m.PUBLISHED,jnp.float32); data=m.inputs(600); J=5000
record={'package':a.package,'mode':a.mode,'J':J,'T':600,'B':1,'dtype':'float32','device':str(jax.devices()[0]),'device_kind':jax.devices()[0].device_kind,'jax':jax.__version__,'node':platform.node(),'job':os.environ.get('SLURM_JOB_ID'),'fit_comparison':False,'measurements':{}}

def measure(name,fn,inputs,seeds):
 t=time.perf_counter();exe=jax.jit(fn).lower(*inputs,jax.random.PRNGKey(0)).compile();comp=time.perf_counter()-t
 memory=exe.memory_analysis();peak=memory.argument_size_in_bytes+memory.output_size_in_bytes+memory.temp_size_in_bytes-memory.alias_size_in_bytes
 item={'compile_seconds':comp,'estimated_peak_bytes':peak,'calls':[]}
 record['measurements'][name]=item
 assert peak<.85*jax.devices()[0].memory_stats()['bytes_limit'],str(peak)
 for kind,seed in seeds:
  t=time.perf_counter();v=jax.block_until_ready(exe(*inputs,jax.random.PRNGKey(seed)));dt=time.perf_counter()-t
  finite=all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves(v))
  item['calls'].append({'kind':kind,'seed':seed,'seconds':dt,'finite':bool(finite)})
  if name=='native_pf':item['calls'][-1]['loglik']=float(v)
  a.output.write_text(json.dumps(record,indent=2)+'\n')
  assert finite,(name,seed)
 item['median_timed_seconds']=float(np.median([x['seconds'] for x in item['calls'] if x['kind']=='timed']))
 return item

if a.package=='cuthbert':
 from cuthbert import filter as run_filter
 from cuthbert.smc import particle_filter as pf
 from cuthbertlib.resampling import multinomial
 def native(q,k):
  obj=pf.build_filter(*m.factory(q),J,multinomial.resampling);ik,fk=jax.random.split(k)
  return run_filter(obj,data,obj.init_prepare(key=ik),key=fk).log_normalizing_constant[-1]
 record['native_module']=pf.__file__
 if a.mode=='public':
  from cuthbert.smc import dmop, iterated_filtering
  def warm(q,k):
   r=iterated_filtering.fit(m.factory,q,data,key=k,n_particles=J,n_iterations=5,perturbation_scale=m.PROCESS_SCALE,initial_perturbation_scale=m.INITIAL_SCALE,cooling_rate=.5**(1/50),resampling='multinomial')
   return r.parameters,r.parameter_swarm,r.perturbed_log_likelihoods,r.success
  def objective(alpha):return lambda q,k:dmop.log_likelihood(m.factory,q,data,key=k,n_particles=J,alpha=alpha,resampling='multinomial')/600
else:
 import pfjax as pf
 import bridge
 model=bridge.Dacca()
 native=lambda q,k:pf.particle_filter(model,k,data,q,J)['loglik']
 record['native_module']=inspect.getfile(pf.particle_filter)
 if a.mode=='public':
  def warm(q,k):
   def step(s,i):
    r=pf.iterated_filter_step(model,jax.random.fold_in(k,i),data,s,m.PROCESS_SCALE,initial_scale=m.INITIAL_SCALE,iteration=i,cooling_rate=.5,cooling_interval=50.,within_pass_cooling=True)
    return r['parameter_particles'],(r['theta'],r['loglik'],r['valid'])
   return jax.lax.scan(step,jnp.broadcast_to(q,(J,23)),jnp.arange(5))
  def objective(alpha):return lambda q,k:pf.particle_filter_mop(model,k,data,q,J,alpha=alpha)['loglik']/600

if a.mode=='native':
 assert '/upstream/' in record['native_module'],record['native_module']
 measure('native_pf',native,(params,),[('warmup',951000+i) for i in range(2)]+[('timed',952000+i) for i in range(7)])
 record.update(fit_calls=0,gradient_calls=0)
else:
 assert '/candidate/' in record['native_module'],record['native_module']
 measure('public_IF2_5_passes',warm,(params,),[('timed',953000)])
 for alpha in (1.,.97):
  measure(f'public_gradient_alpha_{alpha}',jax.value_and_grad(objective(alpha)),(params,),[('warmup',954000)]+[('timed',954001+i) for i in range(3)])
record['passed']=True;a.output.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record))
