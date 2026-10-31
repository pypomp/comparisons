"""Run one reviewed phase once. Save incomplete work; never resubmit or refit."""
from pathlib import Path
import argparse,datetime,hashlib,json,os,signal,subprocess,sys,time
ROOT=Path(__file__).resolve().parent
BASE=Path('/scratch/ionides_root/ionides1/hkunyang/model024_20261006')
TORCH=BASE/'competitor_estimation_20261028/pydpf';JAX=BASE/'model024_panel_cute_20261023/venv/bin/python'
child=None

def atomic(path,value):
 path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temp.replace(path)
def stop(*_):
 if child is not None and child.poll() is None:child.send_signal(signal.SIGTERM)
def call(command,env,log):
 global child
 with log.open('w') as handle:
  child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=handle,stderr=subprocess.STDOUT);code=child.wait()
 child=None
 return code

def main():
 p=argparse.ArgumentParser();p.add_argument('--phase',choices=['warm','A','B','C'],required=True);a=p.parse_args();phase=a.phase
 config=json.loads((ROOT/'protocol.json').read_text());assert config['status']=='frozen';allocation=config['allocation_by_phase'][phase]
 out=ROOT/'results';dest=out/phase;dest.mkdir(parents=True,exist_ok=True)
 run_id=os.environ['SLURM_JOB_ID'];begin=time.perf_counter();record=dict(job=run_id,phase=phase,node=os.environ.get('SLURMD_NODENAME'),started_at_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),allocation_seconds=allocation['allocation_seconds'],protocol_sha256=hashlib.sha256((ROOT/'protocol.json').read_bytes()).hexdigest(),automatic_resubmit=False)
 atomic(dest/f'launch_{run_id}.json',record);subprocess.run([sys.executable,'verify_manifest.py'],cwd=ROOT,check=True)
 env=os.environ.copy();env['PYTHONPATH']=f'{ROOT}/candidate/pydpf:{TORCH}/runtime-deps'
 code=call([str(TORCH/'venv/bin/python'),'capture_environment.py'],env,dest/f'environment_torch_{run_id}.json')
 assert code==0
 gate=ROOT.parent/'repair_v2'
 code=call([sys.executable,'check_gate.py','--protocol','protocol.json','--gate',str(gate),'--output',str(dest/'gate')],env,dest/f'gate_{run_id}.log');assert code==0,'GPU correctness/rate/budget gate failed; no fit started'
 t=time.perf_counter();code=call([str(TORCH/'venv/bin/python'),'runtime/check_driver_v2.py','--device','cuda:0','--output-root',str(dest/f'GPU_driver_checks_{run_id}')],env,dest/f'GPU_driver_checks_{run_id}.log');record['GPU_driver_check_seconds']=time.perf_counter()-t;assert code==0,'CUDA optimizer-state regression failed; no fit started'
 fit_limit=allocation['fit_limit_seconds']-(time.perf_counter()-begin)
 code=call([str(TORCH/'venv/bin/python'),'runtime/driver.py','--protocol',str(dest/'gate/resolved_protocol.json'),'--phase',phase,'--output','results','--device','cuda:0','--limit-seconds',str(fit_limit)],env,dest/f'fit_{run_id}.log')
 record['fit_returncode']=code;record['fit_process_elapsed_seconds']=time.perf_counter()-begin
 done=(dest/'summary.json').exists() if phase=='warm' else (dest/'endpoints.json').exists()
 if code==0 and done and phase!='warm':
  env['PYTHONPATH']=f'{ROOT}/reference_source:{BASE}/cuthbert_dacca_20261027/source:{BASE}/cuthbert_dacca_20261027/deps';env['JAX_PLATFORMS']='cuda';env['JAX_ENABLE_X64']='false';env['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
  cuda=Path(os.environ['CUDA_HOME']);env['LD_LIBRARY_PATH']=f'{cuda}/lib64:{cuda}/targets/x86_64-linux/lib';env['LD_PRELOAD']=str(cuda/'lib64/libcusparse.so.12.5.8.93');assert Path(env['LD_PRELOAD']).is_file()
  code=call([str(JAX),'capture_environment.py'],env,dest/f'environment_reference_{run_id}.json');assert code==0
  t=time.perf_counter();code=call([str(JAX),'score.py','--endpoints',str(dest/'endpoints.json'),'--output',str(dest/'scores'),'--seed',str(config['score_seed_root'])],env,dest/f'score_{run_id}.log');record.update(score_returncode=code,score_process_elapsed_seconds=time.perf_counter()-t)
  done=(dest/'scores/COMPLETE.json').exists()
 record.update(phase_complete=bool(done and code==0),launch_elapsed_seconds=time.perf_counter()-begin,finished_at_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat())
 atomic(dest/f'launch_{run_id}.json',record);subprocess.run([sys.executable,'verify_manifest.py'],cwd=ROOT,check=True)
 manifest={str(p.relative_to(dest)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(dest.rglob('*')) if p.is_file() and p.name!='terminal_manifest.json' and 'jax_cache' not in p.parts and 'tmp' not in p.parts}
 atomic(dest/'terminal_manifest.json',dict(job=run_id,phase=phase,phase_complete=record['phase_complete'],files=manifest))
 return 0 if record['phase_complete'] else code or 75

if __name__=='__main__':
 signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop);raise SystemExit(main())
