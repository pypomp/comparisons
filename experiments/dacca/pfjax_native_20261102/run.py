"""Only new native basic-PF A/B fits. Existing IF2 and C are read-only inputs."""
import argparse,json,os,time,traceback
import jax
import jax.numpy as jnp
import numpy as np
import optax
import dacca_model as m
from native_api import objective
from common import ROOT,atomic,save,tree_sha,finite,rates,compile_fn

def run(case,out):
 assert not out.exists(),'Attempt exists; no replay'
 out.mkdir(parents=True)
 assert 0<=case<12 and jax.default_backend()=='gpu' and 'A40' in jax.devices()[0].device_kind and not jax.config.jax_enable_x64
 sid,rep=divmod(case,2);p=jnp.asarray(np.load(ROOT/'input/starts.npy')[sid],jnp.float32);warm=jnp.asarray(np.load(ROOT/'input/warm.npy')[case],jnp.float32)
 prefix=json.loads((ROOT/'provenance.json').read_text())['prefix_rows'][case];assert tree_sha(warm)==prefix['parameters_sha256']
 key=jax.random.fold_in(jax.random.PRNGKey(25081000),case);gradkey=jax.random.fold_in(key,1)
 atomic(out/'STARTED.json',dict(case=case,job=os.environ.get('SLURM_JOB_ID'),node=os.environ.get('SLURMD_NODENAME'),jax=jax.__version__,device=jax.devices()[0].device_kind,gradient_root_key=np.asarray(gradkey).tolist(),IF2_calls=0))
 adam=optax.scale_by_adam(b1=.9,b2=.999,eps=1e-8);etas=rates();lossfn=objective()
 def update(q,state,k,eta):
  value,g=jax.value_and_grad(lossfn)(q,k);direction,newstate=adam.update(g,state,q)
  return q-eta*direction,newstate,value,g
 exe,comp,peak=compile_fn(update,(p,adam.init(p),gradkey,etas[0]));np.save(out/'learning_rates.npy',np.asarray(etas))
 endpoints=[];plans=[('A',p),('B',warm)];plans=plans if case%2==0 else plans[::-1]
 started=time.perf_counter()
 for arm,q in plans:
  path=out/arm;path.mkdir();state=adam.init(q);initial_sha=tree_sha(q);adam_sha=tree_sha(state);times=[];losses=[];trace=[np.asarray(q)];failed=False;reason=None;count=0
  save(path/'initial_state.pkl',dict(parameters=q,adam_state=state,gradient_root_key=gradkey,prefix=prefix if arm=='B' else None))
  for i in range(400):
   ik=jax.random.fold_in(gradkey,i);before=time.perf_counter()
   try:proposal,newstate,value,g=jax.block_until_ready(exe(q,state,ik,etas[i]))
   except Exception as error:
    failed=True;reason=repr(error);save(path/'execution_failure.pkl',dict(iteration=i,parameters=q,adam_state=state,key=ik,error=reason,traceback=traceback.format_exc()));break
   times.append(time.perf_counter()-before);losses.append(float(value))
   flags={name:finite(v) for name,v in [('loss',value),('gradient',g),('estimation',proposal),('natural',m.from_est(proposal)),('Adam',newstate)]}
   if all(flags.values()):q=proposal;state=newstate;count+=1;trace.append(np.asarray(q))
   else:
    failed=True;reason=f'Nonfinite proposal at zero-based update {i}'
    save(path/'rejected_proposal.pkl',dict(iteration=i,parameters_before=q,adam_before=state,proposal=proposal,adam_proposal=newstate,loss=value,gradient=g,natural_proposal=m.from_est(proposal),key=ik,learning_rate=etas[i],finite_fields=flags))
   if (i+1)%10==0 or failed or i==399:
    save(path/'checkpoint.pkl',dict(parameters=q,adam_state=state,next_iteration=i+1,accepted_updates=count,gradient_root_key=gradkey,trace=trace,training_loss=losses,update_seconds=times,fit_failed=failed,reason=reason))
    atomic(path/'progress.json',dict(attempted=i+1,accepted=count,fit_failed=failed,fit_seconds=sum(times)))
   if failed:break
  save(path/'final_state.pkl',dict(parameters=q,adam_state=state,accepted_updates=count,trace=trace,training_loss=losses,update_seconds=times,fit_failed=failed,reason=reason))
  natural=np.asarray(m.from_est(q));np.save(path/'terminal_estimation.npy',np.asarray(q));np.save(path/'terminal_natural.npy',natural)
  info=dict(method=arm,start_id=sid,replicate=rep,fit_failed=failed,failure_reason=reason,iterations_requested=400,iterations_completed=count,
   initial_parameter_sha256=initial_sha,initial_adam_sha256=adam_sha,gradient_key=np.asarray(gradkey).tolist(),gradient_seconds=sum(times),gradient_compile_seconds_attributed_to_first_use=comp if len(endpoints)==0 else 0.,
   IF2_seconds_attributed=prefix['IF2_seconds'] if arm=='B' else 0.,IF2_compile_seconds_attributed=prefix['IF2_compile_seconds'] if arm=='B' else 0.,
   parameters_natural=[float(x) if np.isfinite(x) else None for x in natural],parameter_names=m.NAMES,parameters_estimation=np.asarray(q).tolist(),
   native_method='basic_PF_autodiff',native_gradient_biased=True,alpha=None,IF2_iterations_completed=175 if arm=='B' else 0,terminal_estimation_sha256=tree_sha(q))
  atomic(path/'fit.json',info);endpoints.append(info);atomic(out/'endpoints.json',endpoints)
 atomic(out/'FIT_COMPLETE.json',dict(endpoints=2,failed=sum(e['fit_failed'] for e in endpoints),IF2_calls=0,compile_seconds=comp,estimated_peak_bytes=peak,fit_wall_after_compile=time.perf_counter()-started))

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--case',type=int,required=True);ap.add_argument('--output',type=__import__('pathlib').Path,required=True);a=ap.parse_args();run(a.case,a.output)
