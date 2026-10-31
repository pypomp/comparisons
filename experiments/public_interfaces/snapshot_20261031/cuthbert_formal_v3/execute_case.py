"""Frozen manifest check and correctness gate before each formal case."""
import argparse,datetime,hashlib,json,os,subprocess,sys,time
from pathlib import Path
r=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('package',choices=['cuthbert']);a=p.parse_args()
case=int(os.environ['SLURM_ARRAY_TASK_ID']);out=r/'results'/a.package/f'case{case:02d}';out.mkdir(parents=True,exist_ok=True)
gate=r.parent/'cuthbert_gate_v3';manifest=json.loads((r/'source_manifest.json').read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
for n,h in manifest['files'].items():assert sha(r/n)==h,n
assert sha(gate/'source_manifest.json')==manifest['gate_source_manifest_sha256']
for n,h in json.loads((gate/'source_manifest.json').read_text())['files'].items():assert sha(gate/n)==h,n
approval=json.loads((r/'release_approval.json').read_text())
assert approval['approved'] and approval['gate_source_manifest_sha256']==sha(gate/'source_manifest.json')
assert approval['formal_driver_sha256']==sha(r/'run.py')
assert approval['gate_complete_sha256']==sha(gate/'results/COMPLETE.json')
checks=json.loads((gate/'results/COMPLETE.json').read_text())
assert checks['core_checks_passed'] and checks['formal_fit_calls']==0
timing=json.loads((gate/'results'/'public_cost_results.json').read_text());assert timing['passed'] and timing['J']==5000 and timing['T']==600
mes=timing['measurements'];predicted=35*mes['public_IF2_5_passes']['median_timed_seconds']+1200*max(mes[f'public_gradient_alpha_{v}']['median_timed_seconds'] for v in (1.0,.97))
forecast=1.5*predicted+3*sum(x['compile_seconds'] for x in mes.values())+300
assert forecast<3120,{'forecast_seconds':forecast,'allocation_seconds':3360,'action':'Revise allocation explicitly before fitting; no shortened optimization'}
atomic=lambda p,x:p.write_text(json.dumps(x,indent=2)+'\n')
attempt=len(list(out.glob('allocation_attempt_*.json')));attempt_file=out/f'allocation_attempt_{attempt:03d}.json';start=time.perf_counter()
record={'job_id':os.environ['SLURM_JOB_ID'],'case':case,'package':a.package,'start_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'forecast_seconds':forecast,'source_manifest_sha256':sha(r/'source_manifest.json')};atomic(attempt_file,record)
base=Path('/scratch/ionides_root/ionides1/hkunyang/model024_20261006');env=os.environ.copy()
candidate=gate/a.package/'candidate';candidate=candidate/'src' if a.package=='pfjax' else candidate
env['PYTHONPATH']=':'.join(map(str,[candidate,gate/a.package/'validation',base/'cuthbert_dacca_20261027/source',base/'cuthbert_dacca_20261027/deps',base/'competitor_estimation_20261028/pfjax/deps']))
fit=subprocess.run([sys.executable,str(r/'run.py'),'--package',a.package,'--case',str(case),'--output',str(out)],env=env)
record['fit_exit_code']=fit.returncode;atomic(attempt_file,record)
if fit.returncode:sys.exit(fit.returncode)
env['PYTHONPATH']=':'.join(map(str,[r/'reference_source',base/'cuthbert_dacca_20261027/source',base/'cuthbert_dacca_20261027/deps']))
score=subprocess.run([sys.executable,str(r/'score.py'),'--endpoints',str(out/'endpoints.json'),'--output',str(out/'scores'),'--seed',str(94681000 if a.package=='cuthbert' else 95081000)],env=env)
record.update(score_exit_code=score.returncode,total_wall_seconds=time.perf_counter()-start);atomic(attempt_file,record)
if score.returncode:sys.exit(score.returncode)
evidence={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name not in ('terminal_manifest.json','COMPLETE.json')}
atomic(out/'terminal_manifest.json',{'files':evidence,'source_manifest_sha256':sha(r/'source_manifest.json')})
atomic(out/'COMPLETE.json',{'complete':True,'endpoints':3,'terminal_manifest_sha256':sha(out/'terminal_manifest.json'),'job':os.environ['SLURM_JOB_ID'],'fit_failed_count':json.loads((out/'FIT_COMPLETE.json').read_text())['failed']})
