"""Run one bounded command and preserve real exit code and complete output."""
from pathlib import Path
import argparse,datetime,json,os,subprocess,time
p=argparse.ArgumentParser();p.add_argument('--record',required=True);p.add_argument('--cwd');p.add_argument('--timeout',type=int,default=3600);p.add_argument('command',nargs=argparse.REMAINDER);a=p.parse_args()
r=Path(a.record).resolve();r.parent.mkdir(parents=True,exist_ok=True)
assert not r.exists(),str(r)
c=a.command[1:] if a.command and a.command[0]=='--' else a.command
start=time.monotonic();d={'command':c,'cwd':a.cwd,'started_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'timeout_seconds':a.timeout,'state':'running','environment':{n:os.environ.get(n) for n in ['PYTHONPATH','JAX_PLATFORMS','JAX_ENABLE_X64','CUDA_VISIBLE_DEVICES']}}
r.write_text(json.dumps(d,indent=2)+'\n')
with r.with_suffix('.log').open('w') as log:
 try:
  result=subprocess.run(c,cwd=a.cwd,stdout=log,stderr=subprocess.STDOUT,timeout=a.timeout)
  d.update(state='finished',exit_code=result.returncode)
 except subprocess.TimeoutExpired:d.update(state='timeout',exit_code=124)
d.update(elapsed_seconds=time.monotonic()-start,finished_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
r.write_text(json.dumps(d,indent=2)+'\n');print(json.dumps(d))
raise SystemExit(d['exit_code'])
