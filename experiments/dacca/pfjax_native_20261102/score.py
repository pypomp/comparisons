"""Independent original PyPOMP PF scores for fixed natural-scale endpoints.

This program never imports an optimization package or fits parameters. Launch
with the frozen original PyPOMP source on PYTHONPATH in the reference JAX env.
"""
from pathlib import Path
import argparse,datetime,hashlib,json,math,os,time
import jax
import jax.numpy as jnp
import numpy as np
import pypomp as pp

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def atomic(path,value):
 tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)
def statistics(raw):
 x=np.asarray(raw,float)
 if len(x)<2 or not np.isfinite(x).all():return dict(logmeanlik=None,MCSE=None,likelihood_ESS=None,precision_passed=False,all_finite=False)
 w=np.exp(x-x.max());mcse=float(w.std(ddof=1)/np.sqrt(len(w))/w.mean());ess=float(w.sum()**2/np.dot(w,w))
 return dict(logmeanlik=float(x.max()+np.log(w.mean())),MCSE=mcse,likelihood_ESS=ess,precision_passed=bool(mcse<=1 and ess>=10),all_finite=True)

def main():
 p=argparse.ArgumentParser();p.add_argument('--endpoints',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--allow-cpu',action='store_true');a=p.parse_args()
 if not a.allow_cpu:assert jax.default_backend()=='gpu' and 'A40' in jax.devices()[0].device_kind
 assert not jax.config.jax_enable_x64
 assert a.seed>0
 source=Path(pp.__file__).resolve().parent
 assert source.name=='pypomp'
 endpoints=json.loads(a.endpoints.read_text());identities=[(e['method'],int(e['start_id']),int(e['replicate'])) for e in endpoints]
 assert len(identities)==len(set(identities))
 # Fixed across full lists, per-case lists, and single-arm scoring jobs.
 # Deriving codes from whichever subset arrived would reuse A/B/C keys.
 methods=['A','B','C'];assert all(e['method'] in methods for e in endpoints)
 out=a.output;out.mkdir(parents=True,exist_ok=True)
 protocol={'endpoints_sha256':sha(a.endpoints),'J':5000,'stage_totals':[36,72,144,288],'chunk_size':36,'score_seed':a.seed,'methods':methods,'fit_calls':0,'MCSE_max':1,'likelihood_ESS_min':10,'pp_module':str(source),'pp_source_sha256':{str(p.relative_to(source)):sha(p) for p in sorted(source.rglob('*.py'))},'key_rule':'fold_in(score_root, fixed arm code A=0 B=1 C=2), fold_in(start_id), fold_in(replicate), fold_in(chunk_index); split into (36,1) PF keys'}
 if (out/'protocol.json').exists():assert json.loads((out/'protocol.json').read_text())==protocol
 else:atomic(out/'protocol.json',protocol)
 if (out/'COMPLETE.json').exists():return
 start=time.perf_counter();st=pp.models.dhaka(dt=None,nstep=20).to_struct();names=list(st.param_names)
 assert len(st.ys)==600 and len(names)==28
 def evaluate(theta,key):
  return pp.functional.pfilter(st,jnp.broadcast_to(theta,(36,len(theta))),J=5000,keys=jax.random.split(key,(36,1)))['logLik'].reshape(36)
 example=next((e for e in endpoints if np.isfinite(np.asarray(e['parameters_natural'],float)).all()),None)
 def parameter(e):
  assert set(e['parameter_names'])==set(names)
  return jnp.asarray(np.asarray([e['parameters_natural'][e['parameter_names'].index(n)] for n in names],dtype=np.float32))
 # The all-failed case still emits one unscorable record per endpoint. A shape
 # placeholder is used only for compilation, never as a replacement endpoint.
 example_theta=parameter(example) if example is not None else jnp.zeros((28,),jnp.float32)
 t=time.perf_counter();exe=jax.jit(evaluate).lower(example_theta,jax.random.key(a.seed)).compile();compile_s=time.perf_counter()-t
 attempt=len(list(out.glob('runtime_attempt_*.json')))
 atomic(out/f'runtime_attempt_{attempt:03d}.json',{'compile_seconds':compile_s,'jax':jax.__version__,'device':str(jax.devices()[0]),'device_kind':jax.devices()[0].device_kind,'node':os.environ.get('SLURMD_NODENAME'),'job':os.environ.get('SLURM_JOB_ID'),'started_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
 all_rows=[]
 for e in endpoints:
  rowkey=jax.random.key(a.seed)
  for index in (methods.index(e['method']),int(e['start_id']),int(e['replicate'])):rowkey=jax.random.fold_in(rowkey,index)
  label=f"{e['method']}_s{e['start_id']}_r{e['replicate']}";assert Path(label).name==label
  dest=out/(label+'.json');row={'endpoint':e,'score_key':np.asarray(jax.random.key_data(rowkey)).tolist(),'raw':[],'evaluation_seconds':[],'chunk_keys':[],'score_calls':0,'fit_failed':bool(e['fit_failed'])}
  if dest.exists():
   previous=json.loads(dest.read_text());assert previous['endpoint']==e and previous['score_key']==row['score_key'];all_rows.append(previous);continue
  theta=parameter(e);row['natural_float32_sha256']=hashlib.sha256(np.asarray(theta).tobytes()).hexdigest()
  partial=out/(label+'.partial.json')
  if partial.exists():
   previous=json.loads(partial.read_text());assert previous['endpoint']==e and previous['score_key']==row['score_key'];row=previous
  if np.isfinite(np.asarray(theta)).all():
   for total in protocol['stage_totals']:
    while len(row['raw'])<total:
     key=jax.random.fold_in(rowkey,len(row['raw'])//36);t=time.perf_counter();values=np.asarray(jax.block_until_ready(exe(theta,key)),float);seconds=time.perf_counter()-t
     row['evaluation_seconds'].append(seconds);row['chunk_keys'].append(np.asarray(jax.random.key_data(key)).tolist());row['score_calls']+=36
     row['raw'].extend([float(v) if np.isfinite(v) else None for v in values]);row.update(statistics(row['raw']));atomic(partial,row)
    if row['precision_passed']:break
  else:row.update(statistics([]),unscorable_reason='Nonfinite natural parameters; no replacement or fit retry')
  floor=float(np.float32(-24867.998046875));row['floor_saturated']=bool(row['raw'] and all(x==floor for x in row['raw']));row['repetitions']=len(row['raw']);row['total_evaluation_seconds']=sum(row['evaluation_seconds'])
  atomic(dest,row);all_rows.append(row);atomic(out/'all_scores.json',all_rows)
 atomic(out/'COMPLETE.json',{'complete':True,'endpoint_count':len(endpoints),'score_records':len(all_rows),'fit_calls':0,'elapsed_seconds':time.perf_counter()-start,'compilation_seconds':compile_s,'failed_fits_retained':sum(bool(e['fit_failed']) for e in endpoints),'score_precision_passed':sum(bool(r['precision_passed']) for r in all_rows),'score_seed':a.seed})

if __name__=='__main__':main()
