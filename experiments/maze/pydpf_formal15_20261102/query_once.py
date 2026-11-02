"""One snapshot of only the three new five-task arrays, or the existing bridge."""
from pathlib import Path
import datetime,json,shlex,subprocess,sys
P=Path(__file__).resolve().parent
if not (P/'jobs.json').exists():
    sys.exit(subprocess.run([sys.executable,str(P.parent/'query_once.py')]).returncode)
jobs=json.loads((P/'jobs.json').read_text());ids=[v['array_job_id'] for v in jobs.values()]
assert 1<=len(ids)<=3 and all(v.isdigit() for v in ids)
ssh=['ssh','-S','/tmp/greatlakes-codex-login2.sock','-o','BatchMode=yes','-o','ConnectTimeout=15','greatlakes']
code="""import datetime,json,subprocess
jobs=JOBS
a=subprocess.run(['sacct','-n','-P','-j',','.join(jobs),'--format=JobID%40,State,ExitCode,ElapsedRaw,NodeList,TimelimitRaw,AllocTRES%100'],capture_output=True,text=True)
q=subprocess.run(['squeue','-h','-j',','.join(jobs),'-o','%i|%T|%M|%R'],capture_output=True,text=True)
print(json.dumps(dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),jobs=jobs,sacct=a.stdout,sacct_returncode=a.returncode,sacct_stderr=a.stderr,squeue=q.stdout,squeue_returncode=q.returncode,squeue_stderr=q.stderr)))
""".replace('JOBS',repr(ids))
r=subprocess.run(ssh+['python3 -c '+shlex.quote(code)],capture_output=True,text=True)
tag=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
record=dict(ssh_returncode=r.returncode,stdout=r.stdout,stderr=r.stderr)
with (P/('scheduler_'+tag+'.json')).open('x') as f:json.dump(record,f,indent=2)
assert r.returncode==0,r.stderr
actual=json.loads(r.stdout);assert actual['sacct_returncode']==actual['squeue_returncode']==0
print(json.dumps(actual,indent=2))
