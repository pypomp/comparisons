"""One-shot read-only scheduler capture from current local job records."""
from pathlib import Path
import argparse,datetime,json,re,subprocess
ROOT=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--tag');a=p.parse_args()
files=['jax_gate_v2/jobs.json','jax_formal_v2/jobs.json','cuthbert_gpu_diagnosis_v1/jobs.json','cuthbert_gate_v3/jobs.json','cuthbert_formal_v3/jobs.json','tfp/jobs.json','tfp/formal_v2/jobs.json','pydpf/jobs.json','pydpf/formal_v1/jobs.json','particles/formal_v1/jobs.json']
records={};ids=set()
def walk(x,key=''):
 if isinstance(x,dict):
  for k,v in x.items():
   if k in ('history','job_history','superseded_before_start'):continue
   walk(v,k)
 elif isinstance(x,list):
  for v in x:walk(v,key)
 elif ('job' in key.lower() or key in ('prefix','warm','A','B','C','fit','score')) and re.fullmatch(r'65\d{6}',str(x)):ids.add(str(x))
for name in files:
 path=ROOT/name
 if path.exists():records[name]=json.loads(path.read_text());walk(records[name])
assert ids
base=['ssh','-S','/tmp/greatlakes-codex-login2.sock','-o','BatchMode=yes','-o','ConnectTimeout=15','greatlakes']
joblist=','.join(sorted(ids));command=f"squeue -h -j {joblist} -o '%i|%T|%M|%N|%R'; sacct -j {joblist} -X --format=JobID,State,ExitCode,ElapsedRaw,NodeList,AllocTRES -n -P"
r=subprocess.run(base+[command],capture_output=True,text=True,timeout=60)
now=datetime.datetime.now(datetime.timezone.utc);doc={'captured_at_utc':now.isoformat(),'job_records':records,'job_ids':sorted(ids),'ssh_returncode':r.returncode,'scheduler_output':r.stdout,'ssh_stderr':r.stderr}
dest=ROOT/f"scheduler_{a.tag or now.strftime('%Y%m%dT%H%M%S')}.json";assert not dest.exists();dest.write_text(json.dumps(doc,indent=2)+'\n')
print(dest);print(r.stdout);print(r.stderr)
raise SystemExit(r.returncode)
