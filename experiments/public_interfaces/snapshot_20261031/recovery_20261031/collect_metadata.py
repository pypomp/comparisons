from pathlib import Path
import hashlib,json,subprocess
R=Path(__file__).resolve().parent
SSH=['ssh','-S','/tmp/greatlakes-codex-login2.sock','-o','BatchMode=yes','-o','ConnectTimeout=15','greatlakes']
dest=R/'particles/logs_receipt.json'
if not dest.exists():
 code="""from pathlib import Path
import json,hashlib
r=Path('/scratch/ionides_root/ionides1/hkunyang/model024_20261006/implementation_comparison_20261030/particles/logs')
files=sorted(r.glob('formal-65097887_*.log'))+sorted(r.glob('score-65097888.log'))
out={}
for p in files:
 data=p.read_bytes();h=hashlib.sha256(data).hexdigest()
 assert hashlib.sha256(p.read_bytes()).hexdigest()==h
 out[p.name]={'sha256':h,'content':data.decode()}
assert len(files)==25,len(files)
print(json.dumps(out))
"""
 data=json.loads(subprocess.run(SSH+['python3 -'],input=code,text=True,capture_output=True,check=True,timeout=60).stdout)
 out=R/'particles/logs';out.mkdir(exist_ok=True)
 for n,d in data.items():
  (out/n).write_text(d['content']);assert hashlib.sha256((out/n).read_bytes()).hexdigest()==d['sha256']
 dest.write_text(json.dumps({'files':{n:d['sha256'] for n,d in data.items()},'transfer_hashes_verified':True},indent=2)+'\n')
ids='65097906,65097907,65098862,65098863,65098864,65098865,65098962,65098963,65098964,65098965'
p=R/'cancelled_formal_sacct.txt'
if not p.exists():
 s=subprocess.run(SSH+['sacct -X -n -P -j '+ids+' --format=JobID,State,ExitCode,ElapsedRaw,NodeList,Start,End,Reason'],text=True,capture_output=True,check=True,timeout=60).stdout
 p.write_text(s)
print('Recovered 25 particles logs and formal cancellation records')
